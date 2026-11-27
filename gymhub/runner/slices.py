"""What a job may see of its gym: a copy without answer keys and reference solutions.

Isolated jobs (other users' code, uploaded gyms) are graded on the compute pool
(gymhub/runner/pooled.py), which ships them this slice as /work/gyms/<id>, never the gym itself.
"""

import hashlib
import os
import shutil
import time
from pathlib import Path

from gymhub.runner.adapters import REPO

SLICES = REPO / ".cache" / "gymhub" / "gym-slices"
SLICE_TTL = 24 * 3600  # an unused slice is removed after a day (a running job may still use an old one)


def _slice_skips(rel: Path) -> bool:
    """Left out of what a job sees: the owner's solutions and progress, answer keys, reference solutions."""
    parts = rel.parts
    if parts[0] in {"solutions", ".cache", "__pycache__"} or "__pycache__" in parts:
        return True
    if rel.name.startswith("progress.json") or rel.name == "answer.json":
        return True
    return len(parts) >= 3 and parts[0] == "problems" and parts[2] == "solution"


def gym_slice(gym: Path) -> Path:
    """A read-only copy of `gym` without the files that exist only to hide the answer.

    Grading needs problem.py (the checker), problem.json, the statement and the starters; it never
    needs answer.json (the board grades those kinds) or problems/<slug>/solution/. The copy is keyed
    by a hash of what it contains, so an edited gym gets a new slice and a running job keeps its own.
    """
    files = sorted(path for path in gym.rglob("*") if path.is_file() and not _slice_skips(path.relative_to(gym)))
    signature = hashlib.sha256()
    for path in files:
        info = path.stat()
        signature.update(f"{path.relative_to(gym)}\0{info.st_size}\0{info.st_mtime_ns}\n".encode())
    target = SLICES / f"{gym.name}-{signature.hexdigest()[:16]}"
    if not target.is_dir():
        SLICES.mkdir(parents=True, exist_ok=True)
        building = SLICES / f".{target.name}-{os.getpid()}"
        shutil.rmtree(building, ignore_errors=True)
        for path in files:
            dest = building / path.relative_to(gym)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        building.mkdir(parents=True, exist_ok=True)  # an empty gym still gets a folder
        try:
            building.rename(target)
        except OSError:  # another job built the same slice first
            shutil.rmtree(building, ignore_errors=True)
    os.utime(target)
    for old in SLICES.glob(f"{gym.name}-*"):
        try:
            if old != target and time.time() - old.stat().st_mtime > SLICE_TTL:
                shutil.rmtree(old, ignore_errors=True)
        except OSError:
            pass
    return target
