"""Per-user limits on running code, so one member cannot take over the owner's machine.

Admins set the limits on the Admin page (settings/limits). Admins themselves are exempt. Only
jobs that run code count (Run, Submit, Bench); saving, resetting and hints are free.

usage/{uid}   hour, jobs_this_hour, day, seconds_today   (rolled over when the hour or day changes)

Both usage and attempts documents expire (expire_at, Firestore TTL) a while after their last write:
by then their counts have rolled over or their delay has passed, so deleting them changes nothing.

Machine time is added when a job finishes, from its real duration, so a long benchmark counts
for what it used. Times use UTC days.
"""

import time

from gymhub.store import Store, expires_in

DEFAULTS = {
    "jobs_per_hour": 60,
    "minutes_per_day": 120,
    "max_waiting": 3,  # queued or running at the same time
}
LIMIT_KEYS = tuple(DEFAULTS)


class QuotaExceeded(Exception):
    pass


def limits(store: Store) -> dict:
    saved = store.get("settings", "limits") or {}
    return {key: int(saved.get(key, default)) for key, default in DEFAULTS.items()}


def _stamp() -> tuple[str, str]:
    now = time.gmtime()
    return time.strftime("%Y%m%d%H", now), time.strftime("%Y%m%d", now)


def usage(store: Store, uid: str) -> dict:
    hour, day = _stamp()
    doc = store.get("usage", uid) or {}
    return {
        "jobs_this_hour": doc.get("jobs_this_hour", 0) if doc.get("hour") == hour else 0,
        "seconds_today": doc.get("seconds_today", 0.0) if doc.get("day") == day else 0.0,
    }


def waiting_jobs(store: Store, uid: str) -> int:
    return sum(
        1 for status in ("queued", "running") for _id, doc in store.where("jobs", "status", status) if doc.get("uid") == uid
    )


HOLD_SECONDS = 60  # a reserved job slot is released if the job document never appears
USAGE_KEEP = 2 * 86400  # past the UTC day it counts, plus a day for waiting jobs' holds


def _live_holds(store: Store, holds: dict) -> dict:
    """Reserved job slots that still count: the job is queued or running, or is about to be created."""
    live = {}
    for job_id, stamp in holds.items():
        job = store.get("jobs", job_id)
        if job is None:
            if time.time() - stamp < HOLD_SECONDS:
                live[job_id] = stamp
        elif job.get("status") in ("queued", "running"):
            live[job_id] = stamp
    return live


def check_and_count(store: Store, uid: str, job_id: str | None = None) -> None:
    """Raise QuotaExceeded with a message for the user, or count one more job this hour.

    The check and the increment are one transaction on the user's usage document, so parallel
    requests cannot all pass. Pass the id the job will get: it is held as a waiting slot until the
    job finishes, which closes the gap before the job document exists.
    """
    rules = limits(store)
    hour, day = _stamp()
    waiting = waiting_jobs(store, uid) if rules["max_waiting"] else 0

    def apply(doc: dict | None) -> dict:
        doc = doc or {}
        jobs_this_hour = doc.get("jobs_this_hour", 0) if doc.get("hour") == hour else 0
        seconds_today = doc.get("seconds_today", 0.0) if doc.get("day") == day else 0.0
        holds = _live_holds(store, doc.get("holds") or {})
        if rules["max_waiting"] and max(waiting, len(holds)) >= rules["max_waiting"]:
            raise QuotaExceeded(
                f"You already have {rules['max_waiting']} jobs waiting or running. Wait for one to finish, or cancel it."
            )
        if rules["jobs_per_hour"] and jobs_this_hour >= rules["jobs_per_hour"]:
            minutes = 60 - time.gmtime().tm_min
            raise QuotaExceeded(f"You reached {rules['jobs_per_hour']} runs this hour. Try again in about {minutes} min.")
        if rules["minutes_per_day"] and seconds_today >= rules["minutes_per_day"] * 60:
            raise QuotaExceeded(
                f"You used today's {rules['minutes_per_day']} minutes of machine time. It resets at midnight UTC."
            )
        if job_id:
            holds[job_id] = time.time()
        return {"hour": hour, "jobs_this_hour": jobs_this_hour + 1, "day": day, "seconds_today": seconds_today,
                "holds": holds, "updated_at": time.time(), "expire_at": expires_in(USAGE_KEEP)}

    store.transact("usage", uid, apply)


def add_machine_time(store: Store, uid: str, seconds: float) -> None:
    if not uid or seconds <= 0:
        return
    hour, day = _stamp()

    def apply(doc: dict | None) -> dict:
        doc = doc or {}
        return {
            "hour": hour, "jobs_this_hour": doc.get("jobs_this_hour", 0) if doc.get("hour") == hour else 0,
            "day": day, "seconds_today": round((doc.get("seconds_today", 0.0) if doc.get("day") == day else 0.0) + seconds, 1),
            "holds": doc.get("holds") or {}, "updated_at": time.time(), "expire_at": expires_in(USAGE_KEEP),
        }

    store.transact("usage", uid, apply)


# ---- attempts at answer and choice questions -------------------------------------------------------

FREE_MISSES = 3  # wrong answers before a delay starts
MAX_DELAY = 300  # seconds
ATTEMPTS_KEEP = 86400  # a day without trying again starts the count of misses afresh


def attempt_delay(misses: int) -> int:
    """Seconds to wait after the nth miss in a row: none for the first few, then 5, 10, 20 ... up to 5 minutes."""
    return 0 if misses <= FREE_MISSES else min(MAX_DELAY, 5 * 2 ** (misses - FREE_MISSES - 1))


def begin_attempt(store: Store, uid: str, problem: str) -> None:
    """Count an attempt at a non-code exercise as a miss until it is graded, or raise QuotaExceeded.

    Counting before grading is what makes this a claim: parallel requests each see the others'
    misses, and the delay applies to all of them. A correct answer clears it (`finish_attempt`).
    """
    def apply(doc: dict | None) -> dict:
        doc = doc or {}
        wait = doc.get("blocked_until", 0) - time.time()
        if wait > 0:
            raise QuotaExceeded(f"Too many wrong answers in a row. Try again in {int(wait) + 1} s.")
        misses = doc.get("misses", 0) + 1
        return {"misses": misses, "blocked_until": time.time() + attempt_delay(misses), "updated_at": time.time(),
                "expire_at": expires_in(ATTEMPTS_KEEP)}

    store.transact("attempts", f"{uid}__{problem}", apply)


def finish_attempt(store: Store, uid: str, problem: str, ok: bool) -> None:
    if ok:
        store.delete("attempts", f"{uid}__{problem}")
