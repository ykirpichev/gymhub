"""Create exercises, restore starters, and write the CUDA compilation database."""

import json
import os
import re
import shutil
from pathlib import Path

from gpu_gym.catalog import LANG_FILES, LANGS, iter_problem_dirs
from gpu_gym.paths import COMPILE_DB, CUDA_INCLUDE, CUDA_PATH, PROBLEMS, ROOT, SOLUTIONS

_SLUG = re.compile(r"^[a-z][a-z0-9_]*$")


def ensure_solutions() -> None:
    """Copy any missing starter into solutions/ without touching files you already edited."""
    if os.environ.get("GYM_SANDBOX"):
        return  # read-only checkout inside the board's sandbox
    for directory in iter_problem_dirs():
        starter_dir = directory / "starter"
        if not starter_dir.is_dir():
            continue
        destination = SOLUTIONS / directory.name
        destination.mkdir(parents=True, exist_ok=True)
        for source in starter_dir.iterdir():
            if source.is_file():
                target = destination / source.name
                if not target.exists():
                    shutil.copy2(source, target)


def reset_solution(slug: str, lang: str | None, force: bool) -> int:
    directory = PROBLEMS / slug
    starter_dir = directory / "starter"
    if not starter_dir.is_dir():
        print(f"No starter for {slug}")
        return 1
    langs = LANGS if lang is None else (lang,)
    blocked = False
    for item in langs:
        source = starter_dir / LANG_FILES[item]
        target = SOLUTIONS / slug / LANG_FILES[item]
        if not source.is_file():
            print(f"Missing starter {source}")
            return 1
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != source.read_bytes() and not force:
            print(f"{target.relative_to(ROOT)} differs from the starter. Re-run with --force to overwrite it.")
            blocked = True
            continue
        shutil.copy2(source, target)
        print(f"Restored {target.relative_to(ROOT)}")
    return 1 if blocked else 0


def new_problem(slug: str, title: str) -> int:
    if not _SLUG.match(slug):
        print("The exercise name must start with a letter and use only lowercase letters, digits, and underscores.")
        return 1
    if slug.startswith("_"):
        print("The exercise name cannot start with an underscore.")
        return 1
    destination = PROBLEMS / slug
    if destination.exists():
        print(f"{destination} already exists.")
        return 1
    template = PROBLEMS / "_template"
    shutil.copytree(
        template,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    for path in destination.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".cu", ".md", ".json"}:
            continue
        text = path.read_text()
        text = text.replace("__SLUG__", slug).replace("__TITLE__", title)
        path.write_text(text)
    ensure_solutions()
    write_compile_db()
    print(f"Created {destination.relative_to(ROOT)}")
    print(f"Edit gyms/gpu/problems/{slug}/PROBLEM.md and gyms/gpu/problems/{slug}/problem.py")
    print(f"Write the kernel in gyms/gpu/solutions/{slug}/")
    print(f"Check it with: gpu-gym test {slug} --lang cuda")
    return 0


def write_compile_db(arch: str = "sm_90") -> None:
    """Tell clangd how to parse every CUDA file.

    The arch here is only for the editor. nvcc still compiles for the GPU
    that is actually installed. Clang 18, which ships with this project,
    understands sm_90 and does not understand sm_121.
    """
    commands = []
    files = list(PROBLEMS.rglob("*.cu")) + list(SOLUTIONS.rglob("*.cu"))
    for path in sorted(set(files)):
        commands.append(
            {
                "directory": str(ROOT),
                "file": str(path.resolve()),
                "arguments": [
                    "clang",
                    "-std=c++17",
                    "-x",
                    "cuda",
                    f"--cuda-path={CUDA_PATH}",
                    f"--cuda-gpu-arch={arch}",
                    f"-I{CUDA_INCLUDE}",
                    "-c",
                    str(path.resolve()),
                ],
            }
        )
    cuda_count = len(commands)
    # C++ exercises of the other gyms (gyms/<gym>/...): plain host C++ with the gymkit harness header.
    gym_io = ROOT / "gymkit" / "cpp"
    for path in sorted((ROOT / "gyms").rglob("*.cpp")) if (ROOT / "gyms").is_dir() else []:
        commands.append(
            {
                "directory": str(ROOT),
                "file": str(path.resolve()),
                "arguments": ["clang++", "-std=c++20", "-x", "c++", f"-I{gym_io}", "-c", str(path.resolve())],
            }
        )
    COMPILE_DB.write_text(json.dumps(commands, indent=2) + "\n")
    print(
        f"Wrote {COMPILE_DB.relative_to(ROOT)} ({cuda_count} CUDA files, {arch}; "
        f"{len(commands) - cuda_count} C++ files)"
    )
