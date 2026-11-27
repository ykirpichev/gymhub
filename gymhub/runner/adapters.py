"""How the runner reaches each local gym: files to sync, and the command that tests or benches a solution.

Adding a gym = adding an adapter here. A gymkit gym (gyms/<id>/gym.json) is picked up automatically.
"""

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


@dataclass
class GymAdapter:
    id: str
    lang_files: dict[str, str]  # lang -> file name inside the exercise's solution directory
    solutions_dir: Path
    starters_dir: Path  # <starters_dir>/<slug>/starter/<file>
    progress_file: Path
    watch_dirs: list[Path] = field(default_factory=list)  # changes here mean the catalog changed
    command_prefix: list[str] = field(default_factory=list)
    mounts: list[Path] = field(default_factory=list)  # an uploaded gym's unpacked bundle (validation reads it)
    root: Path | None = None  # the gym's folder: a pool job sees a slice of it without answers (slices.gym_slice)
    runtime: str = "gymkit"  # or "gpu_gym" (packages.py)
    env: dict[str, str] = field(default_factory=dict)  # added to every job's environment

    def solution_path(self, slug: str, lang: str) -> Path:
        if lang not in self.lang_files:
            raise ValueError(f"{self.id} has no language {lang!r}")
        if "/" in slug or slug.startswith(".") or not slug:
            raise ValueError(f"bad exercise name {slug!r}")
        return self.solutions_dir / slug / self.lang_files[lang]

    def starter_path(self, slug: str, lang: str) -> Path:
        return self.starters_dir / slug / "starter" / self.lang_files[lang]

    def current_code(self, slug: str) -> dict[str, str]:
        code = {}
        for lang in self.lang_files:
            path = self.solution_path(slug, lang)
            code[lang] = path.read_text() if path.is_file() else ""
        return code

    def progress(self) -> dict:
        if not self.progress_file.is_file():
            return {"version": 1, "exercises": {}}
        try:
            return json.loads(self.progress_file.read_text())
        except json.JSONDecodeError:
            return {"version": 1, "exercises": {}}

    def command(self, action: str, slug: str, lang: str, options: dict, file: Path | None = None) -> list[str]:
        args = [action, slug, "--lang", lang]
        if file is not None:
            args += ["--file", str(file)]
        if options.get("cases_file"):
            args += ["--cases", str(options["cases_file"])]
            if options.get("only_custom"):
                args.append("--only-custom")
        if action == "bench":
            runs = int(options.get("runs") or 0)
            if 1 <= runs <= 100:
                args += ["--runs", str(runs)]
            if options.get("profile") and self.runtime == "gpu_gym":  # Nsight Compute summary (gpu_gym/profile.py)
                args.append("--profile")
        return self.command_prefix + args

    def load_catalog(self) -> dict:
        result = subprocess.run(
            [sys.executable, "-m", "gymhub.runner.catalog", self.id],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(f"catalog for {self.id} failed:\n{result.stderr[-4000:]}")
        return json.loads(result.stdout)

    def solution_stamps(self) -> dict[str, tuple]:
        """slug -> mtimes of its solution files, to notice edits made in a local editor."""
        stamps: dict[str, tuple] = {}
        if self.solutions_dir.is_dir():
            for directory in sorted(self.solutions_dir.iterdir()):
                if directory.is_dir():
                    stamps[directory.name] = tuple(
                        self.solution_path(directory.name, lang).stat().st_mtime_ns
                        if self.solution_path(directory.name, lang).is_file()
                        else 0
                        for lang in self.lang_files
                    )
        return stamps

    def progress_stamp(self) -> int:
        return self.progress_file.stat().st_mtime_ns if self.progress_file.is_file() else 0

    def catalog_stamp(self) -> tuple:
        stamp = []
        for root in self.watch_dirs:
            if root.is_dir():
                for path in sorted(root.rglob("*")):
                    if path.is_file() and "__pycache__" not in path.parts:
                        stamp.append((str(path), path.stat().st_mtime_ns))
        return tuple(stamp)


def gpu_adapter() -> GymAdapter:
    root = REPO / "gyms" / "gpu"
    return GymAdapter(
        id="gpu",
        lang_files={"cuda": "solution.cu", "pytorch": "solution.py", "triton": "triton_solution.py"},
        solutions_dir=root / "solutions",
        starters_dir=root / "problems",
        progress_file=root / "progress.json",
        watch_dirs=[root / "problems"],
        command_prefix=[sys.executable, "-m", "gpu_gym"],
        root=root,
        runtime="gpu_gym",
    )


def kit_adapter(gym_id: str) -> GymAdapter:
    root = REPO / "gyms" / gym_id
    config = json.loads((root / "gym.json").read_text())
    return GymAdapter(
        id=gym_id,
        lang_files={lang["id"]: lang["file"] for lang in config["langs"]},
        solutions_dir=root / "solutions",
        starters_dir=root / "problems",
        progress_file=root / "progress.json",
        watch_dirs=[root / "problems"],
        command_prefix=[sys.executable, "-m", "gymkit", gym_id],
        root=root,
    )


def bundle_adapter(root: Path) -> GymAdapter:
    """An uploaded gym, unpacked on this machine (gymhub/runner/bundles.py). Always run isolated."""
    config = json.loads((root / "gym.json").read_text())
    gpu = config.get("runtime") == "gpu_gym"
    return GymAdapter(
        id=config["id"],
        lang_files={lang["id"]: lang["file"] for lang in config.get("langs") or []},
        solutions_dir=root / "solutions",
        starters_dir=root / "problems",
        progress_file=root / "progress.json",
        # gpu_gym grades the folder GPU_GYM_ROOT names; gymkit takes the folder as an argument.
        command_prefix=[sys.executable, "-m", "gpu_gym"] if gpu else [sys.executable, "-m", "gymkit", str(root)],
        mounts=[root],
        root=root,
        runtime="gpu_gym" if gpu else "gymkit",
        env={"GPU_GYM_ROOT": str(root)} if gpu else {},
    )


def all_adapters(only: list[str] | None = None) -> dict[str, GymAdapter]:
    """One adapter per gym package found under gyms/, by runtime."""
    from gymhub import packages

    adapters = {}
    for package in packages.discover(REPO / "gyms"):
        adapters[package.id] = gpu_adapter() if package.runtime == "gpu_gym" else kit_adapter(package.id)
    if only:
        adapters = {key: value for key, value in adapters.items() if key in only}
    return adapters
