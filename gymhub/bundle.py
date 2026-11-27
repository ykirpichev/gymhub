"""The worker bundle: everything a machine needs to run jobs, and nothing personal.

Contains the gym code (gpu_gym, gymkit), the runner (gymhub minus the board), exercise
definitions (gyms/*/gym.json and gyms/*/problems/), and pyproject.toml.
Never contains solutions/, progress.json, .cache/ or anything else from the owner's checkout, and
never what exists only to hide the answer: answer.json keys and problems/*/solution/ references.

    python -m gymhub.bundle out.tar.gz      # built by deploy/gcp.sh into the board image
"""

import io
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INCLUDE = ["gpu_gym", "gymkit", "gymhub", "gyms", "pyproject.toml", "LICENSE"]
SKIP_DIRS = {"__pycache__", "solutions", ".cache", "board"}  # the board is not needed on a worker
SKIP_FILES = {"progress.json", "progress.json.tmp", "progress.json.broken"}


def is_answer(rel: Path) -> bool:
    """An answer key or a reference solution inside an exercise: gyms/<gym>/problems/<slug>/(answer.json|solution/...)."""
    parts = rel.parts
    if len(parts) < 5 or parts[0] != "gyms" or parts[2] != "problems":
        return False
    return parts[4] == "solution" or (len(parts) == 5 and parts[4] == "answer.json")


def files(root: Path = REPO) -> list[Path]:
    selected = []
    for name in INCLUDE:
        path = root / name
        if path.is_file():
            selected.append(path)
            continue
        if not path.is_dir():
            continue
        for item in sorted(path.rglob("*")):
            rel = item.relative_to(root)
            if not item.is_file() or SKIP_DIRS.intersection(rel.parts) or item.name in SKIP_FILES or item.suffix == ".pyc":
                continue
            if is_answer(rel):
                continue
            selected.append(item)
    return selected


def build(root: Path = REPO) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in files(root):
            info = tar.gettarinfo(str(path), arcname=str(path.relative_to(root)))
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0  # reproducible
            with path.open("rb") as handle:
                tar.addfile(info, handle)
    return buffer.getvalue()


if __name__ == "__main__":
    data = build()
    Path(sys.argv[1]).write_bytes(data)
    print(f"wrote {sys.argv[1]}: {len(data) / 1e6:.1f} MB, {len(files())} files")
