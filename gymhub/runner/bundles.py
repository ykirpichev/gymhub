"""Uploaded gyms on a worker: each bundle is fetched once, checked against its hash and unpacked.

    .cache/gymhub/bundles/<sha256>/   gym.json, problems/   (read-only use; never edited)

The newest KEEP bundles are kept; older ones are removed and fetched again if a job needs them.
"""

import shutil
import tempfile
import time
from pathlib import Path

from gymhub import packages

ROOT = Path(__file__).resolve().parents[2] / ".cache" / "gymhub" / "bundles"
KEEP = 20


class BundleUnavailable(RuntimeError):
    pass


def ensure(backend, job: dict, bundle: dict) -> Path:
    sha = bundle["sha"]
    target = ROOT / sha
    if (target / "gym.json").is_file():
        target.touch()
        return target
    data = backend.fetch_bundle(job)
    if packages.bundle_hash(data) != sha:
        raise BundleUnavailable("the downloaded bundle does not match its hash")
    ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT, prefix=".unpack-") as tmp:
        packages.unpack(data, Path(tmp) / "pkg")
        try:
            (Path(tmp) / "pkg").rename(target)
        except OSError:
            if not (target / "gym.json").is_file():  # not just another job unpacking the same bundle
                raise
    _prune()
    return target


def _prune() -> None:
    entries = sorted((path for path in ROOT.iterdir() if path.is_dir() and not path.name.startswith(".")),
                     key=lambda path: path.stat().st_mtime, reverse=True)
    for old in entries[KEEP:]:
        if time.time() - old.stat().st_mtime > 3600:
            shutil.rmtree(old, ignore_errors=True)
