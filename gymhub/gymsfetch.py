"""The gyms live in their own repository (gyms.lock names it and pins a commit). This puts them in gyms/.

    python -m gymhub.gymsfetch              # gyms/ at the pinned commit (does nothing when already there)
    python -m gymhub.gymsfetch --ref main   # another branch, tag or commit
    python -m gymhub.cli gym fetch          # the same

gyms/ is not part of this repository. setup.sh, `gymhub-board` and CI call this when it is missing.
To work on the gyms themselves, clone the gyms repository and link it instead:

    git clone https://github.com/ykirpichev/gymhub-gyms ../gymhub-gyms && ln -s ../gymhub-gyms gyms

A gyms/ that this module did not write (a link, a clone, your own folder) is never touched.
A private gyms repository is read with GITHUB_TOKEN or GH_TOKEN, when set. Standard library only.
"""

import argparse
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOCK = REPO / "gyms.lock"
GYMS = REPO / "gyms"
MARKER = ".fetched-from"  # inside gyms/: what this module fetched, so it knows the folder is its own
MAX_BYTES = 200 * 1024 * 1024


class FetchError(RuntimeError):
    pass


def pinned(lock: Path = LOCK) -> dict:
    data = json.loads(lock.read_text())
    if not data.get("repo") or not data.get("ref"):
        raise FetchError(f"{lock} needs \"repo\" (owner/name) and \"ref\"")
    return data


def state(dest: Path = GYMS) -> str | None:
    """The ref gyms/ was fetched at; "" for a gyms/ this module did not write; None when there is none."""
    if dest.is_symlink() or (dest.exists() and not (dest / MARKER).is_file()):
        return ""
    if not dest.exists():
        return None
    return json.loads((dest / MARKER).read_text()).get("ref", "")


def _download(repo: str, ref: str) -> bytes:
    request = urllib.request.Request(f"https://api.github.com/repos/{repo}/tarball/{ref}",
                                     headers={"Accept": "application/vnd.github+json", "User-Agent": "gymhub-gymsfetch"})
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=120) as reply:
            data = reply.read(MAX_BYTES + 1)
    except OSError as error:
        raise FetchError(f"could not download {repo}@{ref}: {error}") from error
    if len(data) > MAX_BYTES:
        raise FetchError(f"{repo}@{ref} is over {MAX_BYTES >> 20} MB")
    return data


def _extract(data: bytes, into: Path) -> None:
    """GitHub's tarball has one top folder (<owner>-<repo>-<sha>/); its contents go into `into`."""
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts[1:]
            if not parts or member.issym() or member.islnk() or ".." in parts or Path(member.name).is_absolute():
                continue
            target = into.joinpath(*parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, target.open("wb") as out:
                    shutil.copyfileobj(source, out)


def fetch(ref: str | None = None, dest: Path = GYMS, lock: Path = LOCK, force: bool = False,
          download=None) -> str:
    """Put the gyms at `ref` (default: the pinned one) in `dest`. Returns what happened, in words."""
    current = state(dest)
    if current == "":
        return f"{dest} is your own (a link or a folder this did not fetch); left as it is"
    pin = pinned(lock)
    ref = ref or pin["ref"]
    if current == ref and not force:
        return f"{dest} is at {pin['repo']}@{ref}"
    data = (download or _download)(pin["repo"], ref)
    staging = Path(tempfile.mkdtemp(prefix=".gyms-", dir=dest.parent))
    try:
        _extract(data, staging)
        (staging / MARKER).write_text(json.dumps({"repo": pin["repo"], "ref": ref}) + "\n")
        if dest.exists():
            shutil.rmtree(dest)
        staging.rename(dest)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return f"fetched {pin['repo']}@{ref} into {dest}"


def ensure(dest: Path = GYMS, lock: Path = LOCK) -> str | None:
    """Fetch the pinned gyms when gyms/ is missing; otherwise nothing. Errors become a printed warning."""
    if state(dest) is not None or not lock.is_file():
        return None
    try:
        return fetch(dest=dest, lock=lock)
    except (FetchError, OSError, ValueError) as error:
        print(f"warning: no gyms yet ({error}). Run: python -m gymhub.gymsfetch", file=sys.stderr)
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gymhub.gymsfetch", description="Put the gyms repository in gyms/.")
    add_arguments(parser)
    return run(parser.parse_args(argv))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ref", help="branch, tag or commit (default: the one gyms.lock pins)")
    parser.add_argument("--force", action="store_true", help="download again even when gyms/ is at that ref")


def run(args: argparse.Namespace) -> int:
    try:
        print(fetch(ref=args.ref, force=args.force))
    except (FetchError, OSError, ValueError) as error:
        print(f"gymsfetch: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
