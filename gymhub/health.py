"""Operational numbers: structured logs of each job's life, and a summary for the Admin page.

Every step of a job is one record from the "gymhub.jobs" logger (stdlib logging):

    job_created    the board queued it
    job_leased     a machine took it          queue_wait_s, tries
    job_requeued   its machine went away      reason, tries
    job_finished   done, error or cancelled   status, ok, queue_wait_s, run_s

each with job, uid, gym, slug, lang, action, isolated, runner and hardware where known. On Cloud Run
(K_SERVICE is set) or with GYMHUB_LOG_FORMAT=json a record is one JSON line in Cloud Logging's format,
so the fields land in jsonPayload (Logs Explorer: jsonPayload.event="job_finished"); elsewhere it is
a readable line. Never code, output, emails or tokens: only JOB_FIELDS, the job id and numbers.
"""

import json
import logging
import math
import os
import sys
import time

log = logging.getLogger("gymhub.jobs")
JOB_FIELDS = ("uid", "gym", "slug", "lang", "action", "isolated", "runner", "hardware")


class _Stdout(logging.StreamHandler):
    """Writes to whatever sys.stdout is at the time (pytest swaps it), not the one at start-up."""

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, _value):
        pass


class CloudFormatter(logging.Formatter):
    """One JSON object per line: Cloud Logging takes severity, message and time, the rest is jsonPayload."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "severity": record.levelname, "message": record.getMessage(), "logger": record.name,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
        }
        entry.update(getattr(record, "fields", None) or {})
        if record.exc_info:
            entry["stack_trace"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


class PlainFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        fields = getattr(record, "fields", None) or {}
        extra = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None and key != "event")
        line = f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<7} {record.getMessage()}"
        line = f"{line}  [{extra}]" if extra else line
        return f"{line}\n{self.formatException(record.exc_info)}" if record.exc_info else line


def json_logs() -> bool:
    """JSON on Cloud Run (it sets K_SERVICE) unless GYMHUB_LOG_FORMAT says otherwise."""
    chosen = os.environ.get("GYMHUB_LOG_FORMAT", "")
    return chosen == "json" or (chosen != "plain" and bool(os.environ.get("K_SERVICE")))


def setup_logging() -> None:
    """Give the "gymhub" loggers one stdout handler, once. Records still propagate, so pytest's caplog
    sees them; under uvicorn the root logger has no handler, so nothing is printed twice."""
    logger = logging.getLogger("gymhub")
    if any(isinstance(handler, _Stdout) for handler in logger.handlers):
        return
    handler = _Stdout()
    handler.setFormatter(CloudFormatter() if json_logs() else PlainFormatter())
    logger.addHandler(handler)
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)


def _seconds(later, earlier) -> float | None:
    return round(later - earlier, 3) if later and earlier else None


def job_event(event: str, job_id: str | None, doc: dict, level: int = logging.INFO, **numbers) -> None:
    """Log one step of a job. `doc` is the job document; only JOB_FIELDS are taken from it."""
    setup_logging()
    fields = {"event": event, "job": job_id, **{key: doc.get(key) for key in JOB_FIELDS}, **numbers}
    status = f" {numbers['status']}" if numbers.get("status") else ""
    message = f"{event.replace('_', ' ')}: {doc.get('action')} {doc.get('gym')}/{doc.get('slug')}{status} ({job_id})"
    log.log(level, message, extra={"fields": fields})


def log_created(job_id: str, doc: dict) -> None:
    job_event("job_created", job_id, doc)


def log_leased(job_id: str, doc: dict) -> None:
    job_event("job_leased", job_id, doc, queue_wait_s=_seconds(doc.get("started_at"), doc.get("created_at")),
              tries=doc.get("tries"))


def log_requeued(job_id: str, doc: dict, reason: str) -> None:
    job_event("job_requeued", job_id, doc, level=logging.WARNING, reason=reason, tries=doc.get("tries"))


def log_finished(job_id: str, doc: dict, fields: dict) -> None:
    """`doc` is the job while it ran, `fields` what its end wrote (status, ok, finished_at, hardware)."""
    status, started = fields.get("status"), doc.get("started_at")
    job_event(
        "job_finished", job_id, {**doc, "hardware": fields.get("hardware", doc.get("hardware"))},
        level=logging.WARNING if status == "error" else logging.INFO,
        status=status, ok=bool(fields.get("ok")), cancel_requested=bool(doc.get("cancel")) or None,
        queue_wait_s=_seconds(started, doc.get("created_at")), run_s=_seconds(fields.get("finished_at"), started),
    )


# ---- machine online time ------------------------------------------------------------------------
#
# Utilization needs the time a machine was online, which nothing else records: runners/{id} only
# holds the latest heartbeat. So each heartbeat adds the gap since the previous one (when it is short
# enough that the machine counted as online throughout) to a 10-minute slot:
#
#   runners/{id}.online_slots   {"<slot start, unix s>": seconds online in it}   the last ONLINE_KEEP seconds

ONLINE_SLOT = 600
ONLINE_KEEP = 25 * 3600
ONLINE_GAP = 90  # longest heartbeat gap that still counts as online (RUNNER_ONLINE_SECONDS)


def add_online(previous: dict | None, at: float) -> dict:
    """The `online` slots after a heartbeat at `at`, given the machine's runners doc before it."""
    previous = previous or {}
    slots = {key: value for key, value in (previous.get("online_slots") or {}).items() if float(key) > at - ONLINE_KEEP}
    gap = at - (previous.get("heartbeat_at") or 0)
    if 0 < gap <= ONLINE_GAP:
        key = str(int(at // ONLINE_SLOT * ONLINE_SLOT))
        slots[key] = round(min(ONLINE_SLOT, slots.get(key, 0) + gap), 1)
    return slots


def online_seconds(slots: dict, since: float) -> float:
    """Seconds online after `since`; a slot that straddles it counts in proportion."""
    total = 0.0
    for key, seconds in (slots or {}).items():
        start = float(key)
        if start + ONLINE_SLOT > since:
            total += seconds * min(1.0, (start + ONLINE_SLOT - since) / ONLINE_SLOT)
    return total


# ---- the Admin page summary ----------------------------------------------------------------------


def _pct(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile: a value that really occurred."""
    if not values:
        return None
    values = sorted(values)
    return round(values[max(0, math.ceil(q * len(values)) - 1)], 2)


def _spread(values: list[float]) -> dict:
    return {"p50": _pct(values, 0.5), "p95": _pct(values, 0.95), "n": len(values)}


def _window(jobs: list[dict], runners: dict[str, dict], hours: float, at: float) -> dict:
    since = at - hours * 3600
    created = [doc for doc in jobs if (doc.get("created_at") or 0) >= since]
    leased = [doc for doc in created if doc.get("started_at") and doc.get("runner")]  # answers and saves never reach a machine
    ended = [doc for doc in leased if doc.get("finished_at") and doc.get("status") in {"done", "error"}]
    errors = sum(1 for doc in ended if doc.get("status") == "error")
    ended_ids = {id(doc) for doc in ended}
    by_action: dict[str, int] = {}
    for doc in created:
        by_action[doc.get("action", "?")] = by_action.get(doc.get("action", "?"), 0) + 1

    machines: dict[str, dict] = {}

    def machine(runner_id: str) -> dict:
        doc = runners.get(runner_id) or {}
        return machines.setdefault(runner_id, {"name": doc.get("name") or runner_id, "hardware": None, "jobs": 0, "errors": 0,
                                               "busy_s": 0.0, "online_s": None, "utilization": None})

    # Busy time is each job's running interval clipped to the window, so a job that started before
    # it counts only for its part inside (jobs are sampled by created_at, newest first).
    for doc in jobs:
        start, end = doc.get("started_at"), doc.get("finished_at") or (at if doc.get("status") == "running" else None)
        if not doc.get("runner") or not start or not end or end <= since:
            continue
        entry = machine(doc["runner"])
        entry["busy_s"] += min(end, at) - max(start, since)
        entry["hardware"] = entry["hardware"] or doc.get("hardware")
        if id(doc) in ended_ids:
            entry["jobs"] += 1
            entry["errors"] += doc.get("status") == "error"
    for runner_id, doc in runners.items():
        online = online_seconds(doc.get("online_slots") or {}, since)
        if online or runner_id in machines:
            entry = machine(runner_id)
            entry["online_s"] = round(online, 1)
    for entry in machines.values():
        entry["busy_s"] = round(entry["busy_s"], 1)
        if entry["online_s"]:
            entry["utilization"] = round(entry["busy_s"] / entry["online_s"], 3)
    return {
        "hours": hours, "jobs": len(created), "by_action": by_action,
        "queue_wait_s": _spread([doc["started_at"] - doc["created_at"] for doc in leased]),
        "run_s": _spread([doc["finished_at"] - doc["started_at"] for doc in ended]),
        "finished": len(ended), "errors": errors, "error_rate": round(errors / len(ended), 3) if ended else None,
        "cancelled": sum(1 for doc in created if doc.get("status") == "cancelled"),
        "machines": dict(sorted(machines.items(), key=lambda item: -item[1]["busy_s"])),
    }


def summary(store, hours: float = 24, limit: int = 1000) -> dict:
    """Queue wait, run time (p50/p95), error rate and per-machine utilization over the last hour and
    the last `hours`, from the newest `limit` jobs (a bounded read: Firestore bills each document).

    Error rate is the share of machine jobs that ended in error (cancelled ones not counted).
    Utilization is busy time over online time per machine (see add_online); a pool bridge grades
    several jobs at once, so its utilization can pass 1.
    """
    at = time.time()
    sample = store.recent("jobs", "created_at", limit)
    jobs = [doc for _id, doc in sample]
    runners = {runner_id: doc for runner_id, doc in store.recent("runners", "heartbeat_at", 50)}
    oldest = min((doc.get("created_at") or 0 for doc in jobs), default=0)
    return {
        "windows": [_window(jobs, runners, window, at) for window in sorted({1.0, float(hours)})],
        "queued_now": len(store.where("jobs", "status", "queued")),
        "running_now": len(store.where("jobs", "status", "running")),
        "sampled": len(jobs),
        # False when the day had more jobs than the sample holds: the numbers cover only the newest ones.
        "complete": len(jobs) < limit or oldest < at - hours * 3600,
    }
