"""Gym packages: one format for every gym, whatever runs it.

A package is a directory with a manifest and exercises:

    gyms/<id>/gym.json                manifest (below)
    <problems>/<slug>/problem.json    title, difficulty, summary, kind            public
    <problems>/<slug>/PROBLEM.md      statement                                     public
    <problems>/<slug>/HINTS.md        "## Hint N" sections, revealed one at a time  public
    <problems>/<slug>/EDITORIAL.md    worked solution: approach, complexity,       public, but the board
                                      pitfalls (code exercises)                    shows it only after an attempt
    <problems>/<slug>/starter/*       one starter file per language                 public
    <problems>/<slug>/solution/*      reference solutions, one per language         private
    <problems>/<slug>/...             everything else is the grader: problem.py,   private
                                      C++ harnesses, data

gym.json:
    schema       1
    id, title, description
    runtime      "gymkit" (Python/C++ graders, gymkit/) or "gpu_gym" (CUDA/PyTorch/Triton, gpu_gym/)
    problems     exercise directory, relative to the manifest (default "problems")
    requires     {"gpu": bool}: does any exercise need an NVIDIA GPU
    langs        [{"id", "label", "editor" (Monaco language), "file" (solution file name)}]
    tracks       [{"name", "summary" (optional, one sentence), "exercises": [slug, ...]}]: order and
                 grouping on the board
    warmup       [slug, ...]: worked examples, listed but not counted as exercises
    order        position among gyms on the board
    publish      "upload": only ever published by uploading the package (courses with answer or
                 essay exercises). Runners skip it; their local gyms cannot carry answer keys.
    sources      [{"title", "url", "note" (optional)}]: what the gym is based on and further reading,
                 listed on the gym's page. URLs are http(s).

The board and browsers only ever get the public parts. Machines that grade get the whole package.

A bundle is the package as a .tar.gz: gym.json at the top, then problems/. Authors upload bundles to
the board (gymhub/gymversions.py), which never imports their code; a worker validates each version.

    python -m gymhub.packages validate [gyms/<id> ...]
    python -m gymhub.packages pack gyms/<id> [-o bundle.tar.gz]
"""

import gzip
import hashlib
import io
import json
import re
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

from gymhub.kinds import KINDS, PUBLIC_FIELDS, key_errors  # see kinds.py for every non-code kind

REPO = Path(__file__).resolve().parents[1]
GYMS = REPO / "gyms"
SCHEMA = 1
RUNTIMES = {"gymkit", "gpu_gym"}
DIFFICULTIES = {"easy", "medium", "hard"}
PUBLIC_NAMES = {"problem.json", "PROBLEM.md", "HINTS.md", "EDITORIAL.md"}
SLUG = re.compile(r"^[a-z][a-z0-9_]*$")
GYM_ID = re.compile(r"^[a-z][a-z0-9-]{1,38}[a-z0-9]$")
UPLOAD_RUNTIMES = {"gymkit", "gpu_gym"}
# gpu_gym's languages and the file each one is graded from (gpu_gym/catalog.py LANG_FILES; kept here so
# the board can check an upload without importing torch).
GPU_LANG_FILES = {"cuda": "solution.cu", "pytorch": "solution.py", "triton": "triton_solution.py"}
MAX_SOURCES = 40
SOURCE_URL = re.compile(r"^https?://[^\s<>\"']+$")
MAX_BUNDLE_BYTES = 20 * 1024 * 1024  # compressed
MAX_UNPACKED_BYTES = 64 * 1024 * 1024  # the board unpacks in memory-backed /tmp on Cloud Run
MAX_FILES = 3000
SKIP_PARTS = {"__pycache__", ".git", "solutions", ".pytest_cache"}  # never packed


class BundleError(ValueError):
    pass


@dataclass
class Package:
    root: Path  # the directory holding gym.json
    manifest: dict
    errors: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.manifest["id"]

    @property
    def runtime(self) -> str:
        return self.manifest.get("runtime", "gymkit")

    @property
    def upload_only(self) -> bool:
        return self.manifest.get("publish") == "upload"

    @property
    def problems_dir(self) -> Path:
        return (self.root / self.manifest.get("problems", "problems")).resolve()

    @property
    def langs(self) -> list[dict]:
        return self.manifest.get("langs", [])

    def slugs(self) -> list[str]:
        """Exercises in board order: tracks first, then anything else found on disk."""
        found = sorted(
            path.name
            for path in self.problems_dir.iterdir()
            if path.is_dir() and not path.name.startswith("_") and (path / "problem.json").is_file()
        ) if self.problems_dir.is_dir() else []
        ordered = [slug for track in self.manifest.get("tracks", []) for slug in track["exercises"] if slug in found]
        return ordered + [slug for slug in found if slug not in ordered]

    def track_of(self, slug: str) -> str:
        for track in self.manifest.get("tracks", []):
            if slug in track["exercises"]:
                return track["name"]
        return "Your exercises"

    def problem_dir(self, slug: str) -> Path:
        return self.problems_dir / slug

    def meta(self, slug: str) -> dict:
        return json.loads((self.problem_dir(slug) / "problem.json").read_text())

    def statement(self, slug: str) -> str:
        path = self.problem_dir(slug) / "PROBLEM.md"
        return path.read_text().rstrip() if path.is_file() else ""

    def hints(self, slug: str) -> list[str]:
        path = self.problem_dir(slug) / "HINTS.md"
        if not path.is_file():
            return []
        parts = re.split(r"^## .*$", path.read_text(), flags=re.MULTILINE)
        return [part.strip() for part in parts[1:] if part.strip()]

    def editorial(self, slug: str) -> str:
        """EDITORIAL.md: the worked solution. The board holds it back until the learner has tried."""
        path = self.problem_dir(slug) / "EDITORIAL.md"
        return path.read_text().strip() if path.is_file() else ""

    def starters(self, slug: str) -> dict[str, str]:
        starter = self.problem_dir(slug) / "starter"
        return {
            lang["id"]: (starter / lang["file"]).read_text()
            for lang in self.langs
            if (starter / lang["file"]).is_file()
        }

    def public_files(self, slug: str) -> list[Path]:
        directory = self.problem_dir(slug)
        files = [directory / name for name in sorted(PUBLIC_NAMES) if (directory / name).is_file()]
        if (directory / "starter").is_dir():
            files += sorted(path for path in (directory / "starter").rglob("*") if path.is_file())
        return files

    def grader_files(self, slug: str) -> list[Path]:
        public = set(self.public_files(slug))
        return sorted(
            path
            for path in self.problem_dir(slug).rglob("*")
            if path.is_file() and path not in public and "__pycache__" not in path.parts
        )

    def answer_key(self, slug: str) -> dict | None:
        """answer.json: the private key of a non-code exercise (gymhub/kinds.py)."""
        path = self.problem_dir(slug) / "answer.json"
        try:
            return json.loads(path.read_text()) if path.is_file() else None
        except json.JSONDecodeError:
            return None

    def references(self, slug: str) -> dict[str, Path]:
        """Reference solutions (lang -> file) that validation expects to pass."""
        directory = self.problem_dir(slug) / "solution"
        return {lang["id"]: directory / lang["file"] for lang in self.langs if (directory / lang["file"]).is_file()}

    def tests_hash(self, slug: str) -> str:
        """Changes when anything that decides pass or fail changes (the grader files, not the statement)."""
        digest = hashlib.sha256()
        directory = self.problem_dir(slug)
        for path in self.grader_files(slug):
            if "solution" in path.relative_to(directory).parts[:1]:
                continue  # a new reference solution does not change the tests
            digest.update(str(path.relative_to(directory)).encode() + b"\0" + path.read_bytes() + b"\0")
        return digest.hexdigest()[:16]


def load(path: Path | str) -> Package:
    """`path` is a package directory, or a gym id under gyms/."""
    root = Path(path)
    if not root.is_dir() and (GYMS / str(path)).is_dir():
        root = GYMS / str(path)
    manifest = json.loads((root / "gym.json").read_text())
    manifest.setdefault("id", root.name)
    return Package(root=root.resolve(), manifest=manifest)


def discover(gyms_dir: Path = GYMS, include_uploads: bool = False) -> list[Package]:
    """Gym packages under gyms/. Those marked "publish": "upload" only when `include_uploads`."""
    if not gyms_dir.is_dir():
        return []
    packages = [load(path) for path in sorted(gyms_dir.iterdir()) if (path / "gym.json").is_file()]
    if not include_uploads:
        packages = [package for package in packages if not package.upload_only]
    return sorted(packages, key=lambda package: (package.manifest.get("order", 99), package.id))


def public_catalog(package: Package) -> dict:
    """Everything the board may show: manifest fields and each exercise's public parts. No grader content."""
    warmup = set(package.manifest.get("warmup", []))
    problems = []
    for order, slug in enumerate(package.slugs()):
        meta = package.meta(slug)
        starters = package.starters(slug)
        problems.append(
            {
                "slug": slug,
                "title": meta["title"],
                "difficulty": meta["difficulty"],
                "summary": meta.get("summary", ""),
                "kind": meta.get("kind", "code"),
                "track": package.track_of(slug),
                "order": order,
                "warmup": slug in warmup,
                "statement": package.statement(slug),
                "hints": package.hints(slug),
                "editorial": package.editorial(slug),
                "langs": list(starters),
                "starters": starters,
                "tests_hash": package.tests_hash(slug),
                **{key: meta[key] for key in PUBLIC_FIELDS if key in meta},
            }
        )
    gym = {key: package.manifest.get(key) for key in ("title", "description", "langs", "order", "requires", "runtime")}
    gym["sources"] = sources(package.manifest)
    if not isinstance(gym["order"], (int, float)) or isinstance(gym["order"], bool):
        gym["order"] = 99  # gym.json may leave it out; lists sort by it
    # Track names and summaries for the gym page (which exercise is in which track is on each exercise).
    gym["tracks"] = [{"name": t.get("name", ""), "summary": t.get("summary", "")} for t in package.manifest.get("tracks", [])]
    return {"gym": gym, "problems": problems}


def _kind(package: Package, slug: str) -> str:
    try:
        return package.meta(slug).get("kind", "code")
    except (json.JSONDecodeError, OSError):
        return "code"


def answer_keys(package: Package) -> dict[str, dict]:
    """slug -> answer key, for the non-code exercises. Kept off the public catalog."""
    return {slug: package.answer_key(slug) or {} for slug in package.slugs() if _kind(package, slug) != "code"}


def sources(manifest: dict) -> list[dict]:
    """The manifest's well-formed sources, as the board shows them (title, url, note)."""
    shown = []
    for item in manifest.get("sources") or []:
        if isinstance(item, dict) and not source_errors(item):
            shown.append({"title": item["title"], "url": item["url"], "note": item.get("note", "")})
    return shown[:MAX_SOURCES]


def source_errors(item) -> list[str]:
    if not isinstance(item, dict):
        return ['each source is {"title", "url", "note"}']
    errors = []
    if set(item) - {"title", "url", "note"}:
        errors.append(f"unknown keys {sorted(set(item) - {'title', 'url', 'note'})}")
    title, url, note = item.get("title"), item.get("url"), item.get("note", "")
    if not isinstance(title, str) or not title.strip() or len(title) > 120:
        errors.append("title is 1-120 characters")
    if not isinstance(url, str) or len(url) > 500 or not SOURCE_URL.match(url):
        errors.append("url is an http:// or https:// link")
    if not isinstance(note, str) or len(note) > 240:
        errors.append("note is at most 240 characters")
    return errors


def validate(package: Package) -> list[str]:
    """Problems that would break the board or grading, as messages. Empty when the package is fine."""
    errors: list[str] = []
    manifest = package.manifest
    where = f"{package.root.name}/gym.json"
    if manifest.get("schema") != SCHEMA:
        errors.append(f"{where}: schema must be {SCHEMA}")
    needs_langs = any(_kind(package, slug) == "code" for slug in package.slugs()) if package.problems_dir.is_dir() else True
    for key in ("id", "title", "description", "langs"):
        if key == "langs" and not needs_langs:
            continue
        if not manifest.get(key):
            errors.append(f"{where}: missing {key}")
    if package.runtime not in RUNTIMES:
        errors.append(f"{where}: runtime must be one of {sorted(RUNTIMES)}")
    if not isinstance(manifest.get("requires", {}), dict):
        errors.append(f"{where}: requires must be an object like {{\"gpu\": true}}")
    listed = manifest.get("sources", [])
    if not isinstance(listed, list) or len(listed) > MAX_SOURCES:
        errors.append(f"{where}: sources is a list of at most {MAX_SOURCES} {{\"title\", \"url\", \"note\"}}")
    else:
        for number, item in enumerate(listed, 1):
            errors.extend(f"{where}: source {number}: {error}" for error in source_errors(item))
    lang_ids = []
    for lang in manifest.get("langs", []):
        missing = [key for key in ("id", "label", "editor", "file") if not lang.get(key)]
        if missing:
            errors.append(f"{where}: language {lang.get('id', '?')} is missing {', '.join(missing)}")
        lang_ids.append(lang.get("id"))
    if len(set(lang_ids)) != len(lang_ids):
        errors.append(f"{where}: duplicate language ids")
    if not package.problems_dir.is_dir():
        return errors + [f"{where}: problems directory {package.problems_dir} not found"]
    found = set(package.slugs())
    for track in manifest.get("tracks", []):
        for slug in track.get("exercises", []):
            if slug not in found:
                errors.append(f"{where}: track {track.get('name')!r} lists {slug!r}, which has no problem.json")
        summary = track.get("summary")
        if summary is not None and (not isinstance(summary, str) or len(summary) > 240):
            errors.append(f"{where}: track {track.get('name')!r}: summary is one sentence, at most 240 characters")
    for slug in package.slugs():
        place = f"{package.root.name}/{slug}"
        if not SLUG.match(slug):
            errors.append(f"{place}: exercise names use lowercase letters, digits and _")
        try:
            meta = package.meta(slug)
        except json.JSONDecodeError as exc:
            errors.append(f"{place}/problem.json: not valid JSON ({exc})")
            continue
        for key in ("title", "difficulty"):
            if not meta.get(key):
                errors.append(f"{place}/problem.json: missing {key}")
        if meta.get("difficulty") and meta["difficulty"] not in DIFFICULTIES:
            errors.append(f"{place}/problem.json: difficulty must be easy, medium or hard")
        kind = meta.get("kind", "code")
        if kind not in KINDS:
            errors.append(f"{place}/problem.json: kind {kind!r} is not supported yet")
        if not package.statement(slug):
            errors.append(f"{place}: PROBLEM.md is missing or empty")
        if kind != "code":
            errors.extend(f"{place}: {error}" for error in key_errors(kind, meta, package.answer_key(slug)))
            if (package.problem_dir(slug) / "answer.json").is_file() and package.answer_key(slug) is None:
                errors.append(f"{place}/answer.json: not valid JSON")
        if kind == "code":
            if not package.starters(slug):
                errors.append(f"{place}: no starter file for any language ({', '.join(l['file'] for l in package.langs)})")
            if not (package.problem_dir(slug) / "problem.py").is_file():
                errors.append(f"{place}: no grader (problem.py)")
            if "cpp" in package.starters(slug) and not (package.problem_dir(slug) / "cpp" / "harness.cpp").is_file():
                errors.append(f"{place}: has a C++ starter but no cpp/harness.cpp")
    return errors


def validate_upload(package: Package) -> list[str]:
    """validate() plus what an uploaded bundle must also satisfy."""
    errors = validate(package)
    where = "gym.json"
    if not GYM_ID.match(str(package.manifest.get("id", ""))):
        errors.append(f"{where}: id must be 3-40 characters: lowercase letters, digits and -")
    if package.runtime not in UPLOAD_RUNTIMES:
        errors.append(f"{where}: runtime {package.runtime!r} cannot be uploaded yet (use {', '.join(sorted(UPLOAD_RUNTIMES))})")
    if package.manifest.get("problems", "problems") != "problems":
        errors.append(f"{where}: uploaded gyms keep their exercises in problems/")
    if package.runtime == "gpu_gym":
        if not (package.manifest.get("requires") or {}).get("gpu"):
            errors.append(f'{where}: a gpu_gym gym must declare "requires": {{"gpu": true}}')
        for lang in package.manifest.get("langs", []):
            expected = GPU_LANG_FILES.get(lang.get("id"))
            if expected is None:
                errors.append(f"{where}: gpu_gym languages are {', '.join(GPU_LANG_FILES)}, not {lang.get('id')!r}")
            elif lang.get("file") != expected:
                errors.append(f"{where}: the {lang['id']} file of a gpu_gym gym is {expected}, not {lang.get('file')!r}")
    return errors


def requirements(package: Package) -> dict:
    """What a machine needs to validate or run this package: a GPU if declared (always for gpu_gym),
    and the tools of its languages (routing.LANG_TOOLS, e.g. nvcc for CUDA)."""
    from gymhub import routing

    gpu = package.runtime == "gpu_gym" or bool((package.manifest.get("requires") or {}).get("gpu"))
    tools = sorted({tool for lang in package.langs for tool in routing.LANG_TOOLS.get(lang["id"], [])})
    return {"gpu": gpu, "tools": tools}


def _packable(root: Path) -> list[Path]:
    files = [root / "gym.json"]
    problems = root / "problems"
    if problems.is_dir():
        files += sorted(
            path for path in problems.rglob("*")
            if path.is_file() and not SKIP_PARTS.intersection(path.relative_to(root).parts) and path.suffix != ".pyc"
        )
    return files


def pack(root: Path | str) -> bytes:
    """The package as a .tar.gz. Deterministic: the same files always give the same bytes (and hash)."""
    root = Path(root).resolve()
    if not (root / "gym.json").is_file():
        raise BundleError(f"{root} has no gym.json")
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for path in _packable(root):
                data = path.read_bytes()
                info = tarfile.TarInfo(str(path.relative_to(root)))
                info.size, info.mode, info.mtime = len(data), 0o644, 0
                tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def bundle_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def unpack(data: bytes, dest: Path) -> Package:
    """Extract an uploaded bundle into the empty directory `dest`, refusing anything unsafe:
    links, devices, absolute paths, "..", too many files or too many bytes."""
    if len(data) > MAX_BUNDLE_BYTES:
        raise BundleError(f"the bundle is over {MAX_BUNDLE_BYTES // 1024 // 1024} MB")
    dest = Path(dest).resolve()
    total = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            members = tar.getmembers()
            if len(members) > MAX_FILES:
                raise BundleError(f"the bundle has more than {MAX_FILES} files")
            for member in members:
                name = member.name.lstrip("./") if member.name.startswith("./") else member.name
                parts = Path(name).parts
                if not name or name.startswith("/") or ".." in parts or "\\" in name:
                    raise BundleError(f"unsafe path in the bundle: {member.name!r}")
                if member.isdir():
                    continue
                if not member.isfile():
                    raise BundleError(f"only plain files are allowed in a bundle: {member.name!r}")
                total += member.size
                if total > MAX_UNPACKED_BYTES:
                    raise BundleError(f"the bundle unpacks to more than {MAX_UNPACKED_BYTES // 1024 // 1024} MB")
                target = (dest / name).resolve()
                if dest not in target.parents:
                    raise BundleError(f"unsafe path in the bundle: {member.name!r}")
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                assert source is not None
                target.write_bytes(source.read())
    except (tarfile.TarError, EOFError, OSError, gzip.BadGzipFile) as exc:
        raise BundleError(f"not a .tar.gz bundle: {exc}") from exc
    if not (dest / "gym.json").is_file():
        raise BundleError("the bundle has no gym.json at its top level")
    try:
        manifest = json.loads((dest / "gym.json").read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BundleError(f"gym.json is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict) or not manifest.get("id"):
        raise BundleError("gym.json must name the gym: \"id\": \"my-gym\"")
    return Package(root=dest, manifest=manifest)


def extract_subtree(data: bytes, dest: Path, subdir: str = "") -> Path:
    """A GitHub tarball (one top folder, then the repo): extract only `subdir` of the repo into `dest`.
    Links and other non-files are skipped; paths are checked like unpack(). Returns `dest`."""
    if len(data) > MAX_BUNDLE_BYTES:
        raise BundleError(f"the repository archive is over {MAX_BUNDLE_BYTES // 1024 // 1024} MB")
    subdir = subdir.strip("/")
    if ".." in Path(subdir).parts:
        raise BundleError("the gym path must stay inside the repository")
    dest = Path(dest).resolve()
    total = files = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            for member in tar:
                parts = Path(member.name).parts[1:]  # drop "<repo>-<sha>/"
                if not member.isfile() or not parts or ".." in parts or member.name.startswith("/"):
                    continue
                relative = Path(*parts)
                if subdir:
                    if not str(relative).startswith(subdir + "/"):
                        continue
                    relative = relative.relative_to(subdir)
                files += 1
                total += member.size
                if files > MAX_FILES or total > MAX_UNPACKED_BYTES:
                    raise BundleError("the gym folder in the repository is too large")
                target = (dest / relative).resolve()
                if dest not in target.parents:
                    raise BundleError(f"unsafe path in the archive: {member.name!r}")
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                assert source is not None
                target.write_bytes(source.read())
    except (tarfile.TarError, EOFError, OSError, gzip.BadGzipFile) as exc:
        raise BundleError(f"not a .tar.gz archive: {exc}") from exc
    if not (dest / "gym.json").is_file():
        raise BundleError(f"no gym.json in {subdir or 'the repository root'}")
    return dest


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "pack" and len(args) >= 2:
        package = load(args[1])
        errors = validate_upload(package)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        if errors:
            return 1
        out = Path(args[args.index("-o") + 1]) if "-o" in args else Path(f"{package.id}.tar.gz")
        data = pack(package.root)
        out.write_bytes(data)
        print(f"{out}: {len(package.slugs())} exercises, {len(data) // 1024} KB, sha256 {bundle_hash(data)[:12]}")
        return 0
    if not args or args[0] != "validate":
        print(__doc__)
        return 2
    packages = [load(path) for path in args[1:]] or discover(include_uploads=True)
    bad = 0
    for package in packages:
        errors = validate(package)
        status = "ok" if not errors else f"{len(errors)} problem(s)"
        print(f"{package.id}: {len(package.slugs())} exercises, {status}")
        for error in errors:
            print(f"  - {error}")
        bad += bool(errors)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())


def required_checks(package: "Package") -> set[tuple[str, str, str]]:
    """The (slug, language, what) checks a validation run must report, every one passing, before a
    version publishes: gymhub/runner/validate.py reports exactly these."""
    required = set()
    for slug in package.slugs():
        kind = package.meta(slug).get("kind", "code")
        if kind != "code":
            required.add((slug, "", f"{kind} answer key"))
            continue
        required.add((slug, "", "grader loads"))
        required.update((slug, lang, "starter fails") for lang in package.starters(slug))
        required.update((slug, lang, "reference passes") for lang in package.references(slug))
    return required
