"""Load a gym directory: gym.json, problems/<slug>/problem.py, solutions/, progress.json."""

import importlib.util
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from gymkit.spec import Problem

STARTER_MARK = "GYM_STARTER"
REPO_ROOT = Path(__file__).resolve().parents[1]
GYMS_DIR = REPO_ROOT / "gyms"


@dataclass
class Lang:
    id: str
    file: str
    label: str
    editor: str  # Monaco language id

    @property
    def kind(self) -> str:
        """How solutions run: "cpp" for compiled C++, "python" otherwise."""
        return "cpp" if self.file.endswith((".cpp", ".cc")) else "python"


@dataclass
class Gym:
    id: str
    title: str
    description: str
    root: Path
    langs: list[Lang]
    order: list[str] = field(default_factory=list)
    tracks: dict[str, str] = field(default_factory=dict)  # slug -> track name
    read_only: bool = False  # an unpacked bundle: never write solutions/ into it

    @property
    def problems_dir(self) -> Path:
        return self.root / "problems"

    @property
    def solutions_dir(self) -> Path:
        return self.root / "solutions"

    @property
    def progress_file(self) -> Path:
        # GYMKIT_PROGRESS_FILE points somewhere else, e.g. a scratch copy for another user's board run.
        override = os.environ.get("GYMKIT_PROGRESS_FILE")
        return Path(override) if override else self.root / "progress.json"

    def lang(self, lang_id: str) -> Lang:
        for lang in self.langs:
            if lang.id == lang_id:
                return lang
        raise SystemExit(f"{self.id} has no language {lang_id!r}. Known: {', '.join(l.id for l in self.langs)}")

    def problem_dirs(self) -> list[Path]:
        if not self.problems_dir.is_dir():
            return []
        return sorted(
            path
            for path in self.problems_dir.iterdir()
            if path.is_dir() and not path.name.startswith("_") and (path / "problem.py").is_file()
        )

    def load(self, slug: str) -> Problem:
        directory = self.problems_dir / slug
        module_path = directory / "problem.py"
        if not module_path.is_file():
            known = ", ".join(path.name for path in self.problem_dirs())
            raise SystemExit(f"Unknown exercise {slug!r} in {self.id}. Known: {known}")
        name = f"gymkit_{self.id}_{slug}"
        spec = importlib.util.spec_from_file_location(name, module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        problem = module.build()
        if problem.slug != slug:
            raise ValueError(f"{module_path} slug is {problem.slug!r}, directory is {slug!r}")
        problem.directory = directory
        if not problem.track:
            problem.track = self.tracks.get(slug, "")
        return problem

    def problems(self) -> list[Problem]:
        loaded = [self.load(path.name) for path in self.problem_dirs()]
        rank = {slug: index for index, slug in enumerate(self.order)}
        return sorted(loaded, key=lambda problem: (rank.get(problem.slug, len(rank)), problem.slug))

    def langs_for(self, slug: str) -> list[str]:
        """Languages this exercise supports: those with a starter file."""
        return [lang.id for lang in self.langs if self.starter_path(slug, lang.id).is_file()]

    def solution_path(self, slug: str, lang_id: str) -> Path:
        return self.solutions_dir / slug / self.lang(lang_id).file

    def starter_path(self, slug: str, lang_id: str) -> Path:
        return self.problems_dir / slug / "starter" / self.lang(lang_id).file

    def ensure_solutions(self) -> None:
        if os.environ.get("GYM_SANDBOX") or self.read_only:
            return  # read-only checkout inside the board's sandbox, or an unpacked bundle
        for directory in self.problem_dirs():
            starter = directory / "starter"
            if not starter.is_dir():
                continue
            target_dir = self.solutions_dir / directory.name
            target_dir.mkdir(parents=True, exist_ok=True)
            for source in starter.iterdir():
                target = target_dir / source.name
                if source.is_file() and not target.exists():
                    shutil.copy2(source, target)

    def is_unsolved(self, slug: str, lang_id: str) -> bool:
        path = self.solution_path(slug, lang_id)
        starter = self.starter_path(slug, lang_id)
        if not path.is_file() or not starter.is_file():
            return False
        text = path.read_text()
        return text == starter.read_text() and STARTER_MARK in text

    def statement(self, slug: str) -> str:
        path = self.problems_dir / slug / "PROBLEM.md"
        return path.read_text() if path.is_file() else f"# {slug}\n"

    def hints(self, slug: str) -> list[str]:
        path = self.problems_dir / slug / "HINTS.md"
        if not path.is_file():
            return []
        parts = re.split(r"^## .*$", path.read_text(), flags=re.MULTILINE)
        return [part.strip() for part in parts[1:] if part.strip()]


def load_gym(gym_id: str | Path) -> Gym:
    """A gym under gyms/ by id, or a package directory by path (an unpacked bundle, read-only)."""
    read_only = False
    root = GYMS_DIR / str(gym_id)
    if "/" in str(gym_id) or isinstance(gym_id, Path):
        root, read_only = Path(gym_id).resolve(), True
    config_path = root / "gym.json"
    if not config_path.is_file():
        known = ", ".join(gym_ids())
        raise SystemExit(f"Unknown gym {gym_id!r}. Known: {known}")
    config = json.loads(config_path.read_text())
    if config.get("runtime", "gymkit") != "gymkit":
        raise SystemExit(f"{gym_id} is a {config['runtime']} gym; use its own command (gpu-gym)")
    tracks = {}
    order = []
    for track in config.get("tracks", []):
        for slug in track["exercises"]:
            tracks[slug] = track["name"]
            order.append(slug)
    return Gym(
        id=config.get("id", root.name) if read_only else str(gym_id),
        read_only=read_only,
        title=config["title"],
        description=config.get("description", ""),
        root=root,
        langs=[Lang(**lang) for lang in config["langs"]],
        order=order,
        tracks=tracks,
    )


def _is_gymkit(path: Path) -> bool:
    manifest = path / "gym.json"
    return manifest.is_file() and json.loads(manifest.read_text()).get("runtime", "gymkit") == "gymkit"


def gym_ids() -> list[str]:
    if not GYMS_DIR.is_dir():
        return []
    return sorted(path.name for path in GYMS_DIR.iterdir() if _is_gymkit(path))
