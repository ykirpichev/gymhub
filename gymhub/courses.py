"""The repository's courses on a local board: `python -m gymhub.cli gym seed`.

The courses in gyms/ marked "publish": "upload" (ml-coding, ml-fundamentals, llm-fundamentals,
gpu-interview, pytorch-numpy, ml-system-design) reach a board only as uploaded versions, never from a runner's local files. On a hosted board you
push them (gyms/README.md). On a local board (GYMHUB_STORE=sqlite:...) nobody would, so a direct
runner seeds them at start, and this module is what it calls:

    pack      the course folder, deterministically (the same files give the same hash)
    upload    as an ordinary new version (gymversions.upload), marked source "repo"
    validate  here, in a subprocess (gymhub/runner/validate.py), instead of waiting for a machine:
              these are the repository's own graders, as trusted as the runner's local gyms
    publish   through the same path a machine's report takes (jobqueue.complete), public on first publish

A course is seeded when the store has no gym of that id, or when the live gym was seeded from the
repository and its files changed since (after a `git pull`). A gym of that id uploaded by someone
else, or edited on the board, is left alone.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

from gymhub import gymversions, jobqueue, packages
from gymhub.store import Store, now

SOURCE = "repo"
WORKER = "repo-courses"  # the job's runner and lease while it validates here
OWNER_UID = "local-dev"  # GYMHUB_AUTH=dev's user (gymhub/board/auth.py), so it can manage them
OWNER_EMAIL = "local-dev"


def courses(gyms_dir: Path = packages.GYMS, only: list[str] | None = None) -> list[packages.Package]:
    found = [package for package in packages.discover(gyms_dir, include_uploads=True) if package.upload_only]
    return [package for package in found if not only or package.id in only]


def _live_version(store: Store, gym_id: str, live: dict) -> dict | None:
    number = live.get("version")
    return store.get("gym_versions", gymversions.version_id(gym_id, number)) if number else None


def needs_seed(store: Store, package: packages.Package, sha: str) -> str | None:
    """Why the course should be (re)seeded, or None to leave the store as it is."""
    live = store.get("gyms", package.id)
    if live is None:
        return "not on this board yet"
    if not gymversions.is_bundle_gym(live):
        return None  # a machine publishes a gym of that id from its own files
    version = _live_version(store, package.id, live) or {}
    if version.get("source") != SOURCE:
        return None  # uploaded or edited by someone: theirs now
    if live.get("bundle", {}).get("sha") == sha:
        return None
    return "its files changed"


def _validate(data: bytes, timeout: int) -> dict:
    """What a machine would report for this bundle: {"status", "ok", "output", "result"}."""
    with tempfile.TemporaryDirectory(prefix="gym-seed-") as tmp:
        root = Path(tmp) / "package"
        root.mkdir()
        packages.unpack(data, root)
        out = Path(tmp) / "result.json"
        try:
            done = subprocess.run([sys.executable, "-m", "gymhub.runner.validate", str(root), "--out", str(out)],
                                  capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"status": "error", "ok": False, "output": f"validation took over {timeout} s\n"}
        output = done.stdout + done.stderr
        result = json.loads(out.read_text()) if out.is_file() else {}
        return {"status": "done", "ok": done.returncode == 0, "exit_code": done.returncode,
                "output": output[-jobqueue.MAX_OUTPUT:], "result": result}


def seed_one(store: Store, blobs, package: packages.Package, log: Callable[[str], None] = print,
             timeout: int = gymversions.VALIDATE_TIMEOUT) -> str:
    """Seed one course. Returns "published", "failed" or "skipped"."""
    data = packages.pack(package.root)
    reason = needs_seed(store, package, packages.bundle_hash(data))
    if reason is None:
        return "skipped"
    first = store.get("gyms", package.id) is None
    version = gymversions.upload(store, blobs, data, OWNER_UID, OWNER_EMAIL, may_publish=lambda _existing, _gym: None,
                                 extra={"source": SOURCE})
    job_id = version["job"]
    if version.get("duplicate") or not store.claim(
        "jobs", job_id, {"status": "queued"},
        # Expires like a machine's lease, as its last try: if this process dies mid-way, the sweep fails
        # the version and the next start seeds the course again.
        {"status": "running", "runner": WORKER, "lease_id": WORKER, "started_at": now(),
         "lease_expires_at": now() + timeout + 60, "tries": jobqueue.MAX_TRIES},
    ):
        return "skipped"  # a machine is validating this exact version already
    log(f"seeding course {package.id} ({reason}): validating {version['exercises']} exercises")
    final = _validate(data, timeout)
    jobqueue.complete(store, job_id, WORKER, final, blobs=blobs, lease_id=WORKER)
    doc = store.get("gym_versions", version["id"]) or {}
    if doc.get("status") != "published":
        errors = doc.get("errors") or ["validation failed"]
        log(f"course {package.id} v{version['version']} was not published: " + "; ".join(errors[:3]))
        return "failed"
    if first:  # a local board's own courses: listed for everyone who signs in to it
        store.set("gyms", package.id, {"visibility": "public"}, merge=True)
    log(f"published course {package.id} v{version['version']}: {version['exercises']} exercises")
    return "published"


def seed(store: Store, blobs, only: list[str] | None = None, log: Callable[[str], None] = print,
         gyms_dir: Path = packages.GYMS) -> dict[str, str]:
    """Seed every course (or those in `only`). Returns gym id -> outcome; one course failing does not stop the rest."""
    outcomes = {}
    for package in courses(gyms_dir, only):
        try:
            outcomes[package.id] = seed_one(store, blobs, package, log)
        except Exception as exc:  # noqa: BLE001 - reported, and the next course still gets its turn
            log(f"could not seed course {package.id}: {type(exc).__name__}: {exc}")
            outcomes[package.id] = "failed"
    return outcomes
