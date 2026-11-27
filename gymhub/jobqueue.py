"""The job queue, on top of the document store. Shared by the worker API and a runner's direct mode.

A worker leases a queued job for LEASE_SECONDS and keeps the lease alive while it streams output.
If it disappears, `sweep` puts the job back in the queue (up to MAX_TRIES runs), so a crashed or
unplugged machine never leaves a job stuck in "running".
"""

import os
import re

from gymhub import gymaccess, health, routing
from gymhub.store import EXPIRING, Store, expires_in, new_id, now

LEASE_SECONDS = int(os.environ.get("GYMHUB_LEASE_SECONDS", "90"))
MAX_TRIES = 3
MAX_OUTPUT = 200_000
RUN_JOB_TTL = 15 * 60  # seconds a finished run-mode job stays for the browser to read

# Retention (expire_at, deleted by Firestore's TTL). Only jobs nobody can look back at expire: Run is
# scratch (the sweep deletes it RUN_JOB_TTL after it ends; this catches what the sweep missed, such as
# a run that never got a machine), and save, reset and hint only appear among "recent jobs" for a
# while. Submissions (test, bench, answer) are the history with code (issue #3) behind solves, bests
# and leaderboards, and validate jobs back gym versions: those never expire.
EXPIRES = {"run": 24 * 3600, "save": 30 * 86400, "reset": 30 * 86400, "hint": 30 * 86400}


def expiry(action: str) -> dict:
    """{"expire_at": ...} for a new job of this action, or {} when it is kept."""
    return {"expire_at": expires_in(EXPIRES[action])} if action in EXPIRES else {}


def empty_progress() -> dict:
    return {"version": 1, "exercises": {}}


def _end_without_result(store: Store, doc_id: str, doc: dict, expires, status: str, note: str) -> None:
    """Close a running job whose lease expired, and fail the gym version a validation job was for."""
    end = {"status": status, "ok": False, "finished_at": now(), "lease_expires_at": None}
    claimed = store.claim(
        "jobs", doc_id, {"status": "running", "lease_expires_at": expires}, {**end, "output": (doc.get("output") or "") + note},
    )
    if claimed:
        health.log_finished(doc_id, doc, end)
    if claimed and doc.get("action") == "validate":
        from gymhub import gymversions

        gymversions.fail(store, doc, note.strip().strip("()") or "validation ended without a result")


def sweep(store: Store) -> None:
    """Requeue jobs whose lease expired; drop old run-mode jobs.

    A job that was cancelled is never queued again, and a validation job that runs out of tries
    fails its version, so neither is left running or validating forever.
    """
    for doc_id, doc in store.where("jobs", "status", "running"):
        expires = doc.get("lease_expires_at")
        if not expires or expires > now():
            continue
        if doc.get("cancel"):
            _end_without_result(store, doc_id, doc, expires, "cancelled", "\nCancelled.\n")
        elif doc.get("tries", 1) >= MAX_TRIES:
            _end_without_result(store, doc_id, doc, expires, "error",
                                f"\nThe machine running this job stopped responding ({MAX_TRIES} tries).\n")
        elif store.claim(
            "jobs", doc_id, {"status": "running", "lease_expires_at": expires},
            {"status": "queued", "runner": None, "host": None, "via": None, "lease_expires_at": None, "lease_id": None,
             "output": (doc.get("output") or "") + "\n(The machine running this job stopped responding; it was queued again.)\n"},
        ):
            health.log_requeued(doc_id, doc, "lease expired")
    for doc_id, doc in store.where("jobs", "action", "run"):
        finished = doc.get("finished_at")
        if finished and now() - finished > RUN_JOB_TTL:
            store.delete("jobs", doc_id)
    for collection in EXPIRING:  # a no-op on Firestore, whose TTL policy deletes them
        store.delete_expired(collection)


def lease(
    store: Store,
    gyms: list[str],
    worker: str,
    only_uid: str | None = None,
    capabilities: dict | None = None,
    skip=None,
    name: str | None = None,
) -> dict | None:
    """Claim the oldest queued job this worker can run: one of `gyms`, requirements met by
    `capabilities`, submitted by `only_uid` if given, not sent elsewhere ("Run on"), and not vetoed
    by `skip(doc)`. `name`, the machine's name, is where the job runs (`host`) until the result says
    otherwise (a pool machine)."""
    queued = sorted(store.where("jobs", "status", "queued"), key=lambda item: item[1].get("created_at", 0))
    takes_bundles = bool((capabilities or {}).get("bundles"))
    public: dict[str, bool] = {}
    for doc_id, doc in queued:
        # A machine runs the gyms it has locally, and any uploaded gym (it fetches the bundle).
        if doc.get("gym") not in gyms and not (doc.get("bundle") and takes_bundles):
            continue
        # Whether the gym's content may reach any pool machine (gymhub/gymaccess.py): decided here,
        # from the gym's current visibility, never by the runner; not stored with the job.
        gym = doc.get("gym", "")
        if gym not in public:
            public[gym] = gymaccess.public(store.get("gyms", gym))
        doc["gym_public"] = public[gym]
        if only_uid is not None and doc.get("uid") != only_uid:
            continue
        if routing.misses_target(worker, capabilities, doc):
            continue
        if routing.refuses(capabilities, doc) or routing.unmet(routing.for_job(capabilities, doc), doc.get("requires")):
            continue
        if skip is not None and skip(doc):
            continue
        # Graded on the pool: the pool machine's hardware class arrives with the result (complete), so
        # a best time is never filed under the leasing runner's own GPU.
        hardware = None if routing.on_pool(capabilities, doc) else routing.hardware(capabilities)
        claimed = store.claim(
            "jobs", doc_id, {"status": "queued"},
            {"status": "running", "runner": worker, "host": name or worker, "via": None, "started_at": now(), "hardware": hardware,
             "lease_expires_at": now() + LEASE_SECONDS, "tries": doc.get("tries", 0) + 1, "lease_id": new_id()},
        )
        if not claimed:
            continue  # another worker got it first
        job = store.get("jobs", doc_id) or {}
        job["id"] = doc_id
        job["gym_public"] = public[gym]
        health.log_leased(doc_id, job)
        if job.get("isolated") and job.get("action") not in {"run", "validate"}:
            key = f"{job['uid']}__{job['gym']}"
            job["user_progress"] = (store.get("user_progress", key) or {}).get("data") or empty_progress()
        return job
    return None


def where_ran(report: dict, worker_name: str | None = None) -> dict:
    """`host` (and `via`) from a runner's update or result: the pool machine that runs a bridged job,
    and the runner that sent it there. Short strings only; anything else is ignored."""
    fields = {}
    for key in ("host", "via"):
        value = report.get(key)
        if isinstance(value, str) and 0 < len(value) <= 80:
            fields[key] = value
    if "host" in fields and "via" not in fields and worker_name:
        fields["via"] = worker_name
    return fields


def update(store: Store, job_id: str, worker: str, output: str | None, lease_id: str | None = None,
           host: str | None = None) -> bool:
    """Record progress output and extend the lease (and, once known, the pool machine running it).
    Returns True when the worker should stop."""
    doc = store.get("jobs", job_id)
    if not doc or doc.get("runner") != worker or doc.get("status") != "running" or doc.get("lease_id") != lease_id:
        return True  # cancelled while queued, or the lease moved to another worker or another attempt
    fields: dict = {"lease_expires_at": now() + LEASE_SECONDS}
    if output is not None:
        fields["output"] = output[-MAX_OUTPUT:]
    if host and host != doc.get("host"):
        fields.update(where_ran({"host": host}, doc.get("via") or doc.get("host")))
    store.set("jobs", job_id, fields, merge=True)
    return bool(doc.get("cancel"))


def complete(store: Store, job_id: str, worker: str, final: dict, user_progress: dict | None = None, blobs=None,
             lease_id: str | None = None) -> bool:
    """Store the result. Ignored (False) when this worker no longer holds the job, when the result
    belongs to an earlier lease of it, or when the job already ended (a repeat changes nothing)."""
    doc = store.get("jobs", job_id)
    if not doc or doc.get("runner") != worker or doc.get("lease_id") != lease_id:
        return False
    if doc.get("status") in {"done", "error", "cancelled", "finishing"}:
        return False
    if not store.claim("jobs", job_id, {"status": doc.get("status"), "runner": worker}, {"status": "finishing"}):
        return False  # a second complete for the same attempt got there first
    if doc.get("action") == "validate":
        from gymhub import gymversions
        from gymhub.blobs import open_blobs

        try:
            gymversions.finish(store, blobs or open_blobs(), doc, final)
        except Exception as exc:  # the job itself still completes; the version records why it is not live
            if doc.get("version_doc"):
                store.set("gym_versions", doc["version_doc"],
                          {"status": "failed", "errors": [f"publishing failed: {type(exc).__name__}: {exc}"], "finished_at": now()},
                          merge=True)
    allowed = {"status", "ok", "exit_code", "output"}
    fields = {key: value for key, value in final.items() if key in allowed}
    # Graded on the compute pool (gymhub/runner/pooled.py): the hardware class of the pool machine that
    # ran it, not of the runner that leased it, so best times are grouped by the GPU that measured them.
    if isinstance(final.get("hardware"), str) and 0 < len(final["hardware"]) <= 80:
        fields["hardware"] = final["hardware"]
    # Where it ran: a pool bridge reports the pool machine (`host`) and itself (`via`).
    ran = where_ran(final, doc.get("via") or doc.get("host"))
    if ran.get("host") and ran["host"] != doc.get("host"):
        fields.update(ran)
    if fields.get("status") not in {"done", "error"}:
        fields["status"] = "error"
    fields["output"] = str(fields.get("output") or "")[-MAX_OUTPUT:]
    fields["finished_at"] = now()
    fields["lease_expires_at"] = None
    if doc.get("action") == "run":
        fields["code"] = None  # run mode keeps nothing
    store.set("jobs", job_id, fields, merge=True)
    health.log_finished(job_id, doc, fields)
    if doc.get("isolated") and doc.get("started_at"):
        from gymhub.quotas import add_machine_time

        add_machine_time(store, doc.get("uid", ""), fields["finished_at"] - doc["started_at"])
    if doc.get("action") == "bench" and fields.get("ok"):
        record_best(store, {**doc, "id": job_id, "hardware": fields.get("hardware", doc.get("hardware"))}, fields["output"])
    if doc.get("action") == "test" and fields.get("ok"):
        record_solve(store, {**doc, "id": job_id})
        if (doc.get("options") or {}).get("review") and doc.get("uid"):
            from gymhub import reviews

            reviews.advance(store, doc["uid"], doc["gym"], doc["slug"])
    if user_progress is not None and doc.get("isolated") and doc.get("action") != "run":
        store.set(
            "user_progress", f"{doc['uid']}__{doc['gym']}",
            {"uid": doc["uid"], "gym": doc["gym"], "data": user_progress, "updated_at": now()},
        )
    return True


def record_solve(store: Store, job: dict) -> None:
    """Which tests a submission passed, so the board can say "tests changed since you solved".

    solves/{uid}__{gym}__{slug}__{lang}   tests_hash, version, at
    """
    if not job.get("uid") or not job.get("tests_hash"):
        return
    key = f"{job['uid']}__{job['gym']}__{job['slug']}__{job.get('lang', '')}"
    store.set("solves", key, {
        "uid": job["uid"], "gym": job["gym"], "slug": job["slug"], "lang": job.get("lang"),
        "tests_hash": job["tests_hash"], "version": (job.get("bundle") or {}).get("version"), "at": now(), "job": job.get("id"),
    })


_MEDIAN = re.compile(r"time\s+([0-9.]+) ms median")


def record_best(store: Store, job: dict, output: str) -> None:
    """Keep the user's best median per exercise, language and hardware class.

    bests/{uid}__{gym}__{slug}__{lang}   {hardware: {"ms": best, "at": when}}
    """
    match = _MEDIAN.search(output or "")
    hardware = job.get("hardware")
    if not match or not hardware or not job.get("uid"):
        return
    ms = float(match.group(1))
    from gymhub import leaderboard

    leaderboard.record(store, job, ms)  # only for opted-in gyms and trusted machines
    key = f"{job['uid']}__{job['gym']}__{job['slug']}__{job.get('lang', '')}"
    doc = store.get("bests", key) or {}
    best = doc.get("by_hardware", {}).get(hardware)
    if best is None or ms < best["ms"]:
        by_hardware = {**doc.get("by_hardware", {}), hardware: {"ms": ms, "at": now(), "job": job.get("id")}}
        store.set("bests", key, {"uid": job["uid"], "gym": job["gym"], "slug": job["slug"], "lang": job.get("lang"), "by_hardware": by_hardware})
