"""Refs and queues: the coordination primitives of M3 (#45 sections 4 and 8).

Ref     A named, versioned pointer to an artifact, shared with the pool. `publish` is a
        compare-and-swap: it succeeds only when `parent` is the current version (0 for a new ref),
        or with `force`. Every version is kept: consumers ask for the current one, or wait for one
        newer than what they have.

Queue   Small units of work (tasks), each normally a batch (64 prompts, say). A task carries inline
        JSON inputs and an optional idempotency key, so adding the same work twice is a no-op.
        Jobs lease tasks; the agent's heartbeat keeps their leases alive, and a lease that lapses
        puts the task back. The first result wins:
        - a task running much longer than its queue's typical task (1.5 x p90, at least
          `duplicate_after` seconds) is offered to a second worker when nothing is queued, and
          whichever result arrives first is kept;
        - with `ref` and `max_lag`, a result computed with a version more than `max_lag` behind the
          ref's current one is stale: it is requeued (or dropped, with `stale: drop`) and counted.
        Accepted results get a sequence number, so a driver pages through them with `after`.
        - with `replicate` (a fraction), that share of tasks runs on two machines and is accepted only
          when their results match; a third machine breaks a tie, and pool/trust.py hears of every vote.

Everything here is plain DB code shared by the member API (/v1/refs, /v1/queues) and the job API
that running jobs reach through the agent's unix socket (/v1/agent/jobs/<id>/api/...).
"""

import hashlib
import json
import re

from pool import trust
from pool.db import DB, new_id, now

REF_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")
QUEUE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
TASK_LEASE_SECONDS = 90
MAX_TASKS_PER_ADD = 1000
MAX_INPUTS_BYTES = 64 * 1024
MAX_RESULT_BYTES = 256 * 1024
DUPLICATE_AFTER = 60.0  # seconds before a slow task may be duplicated, whatever the queue's p90
DUPLICATES_MAX = 1


class CoordError(Exception):
    def __init__(self, status: int, message: str, **extra):
        super().__init__(message)
        self.status, self.message, self.extra = status, message, extra


def _json(value, limit: int, what: str) -> str:
    text = json.dumps(value)
    if len(text) > limit:
        raise CoordError(413, f"{what} larger than {limit // 1024} KB: put big data in an artifact and pass its id")
    return text


# ---- refs ---------------------------------------------------------------------------------------

def _ref_row(row: dict | None) -> dict | None:
    if row is None:
        return None
    return {"name": row["name"], "version": row["version"], "artifact": row["artifact"], "parent": row["parent"],
            "meta": json.loads(row["meta"] or "{}"), "created_at": row["created_at"]}


def ref_current(db: DB, pool: str, name: str) -> dict | None:
    return _ref_row(db.one("select * from refs where pool=? and name=? order by version desc limit 1", pool, name))


def ref_get(db: DB, pool: str, name: str, version: int | None = None) -> dict:
    row = (db.one("select * from refs where pool=? and name=? and version=?", pool, name, version) if version
           else db.one("select * from refs where pool=? and name=? order by version desc limit 1", pool, name))
    if not row:
        raise CoordError(404, f"no ref {name}" + (f" version {version}" if version else ""))
    return _ref_row(row)


def ref_history(db: DB, pool: str, name: str, limit: int = 50) -> list[dict]:
    return [_ref_row(row) for row in db.all("select * from refs where pool=? and name=? order by version desc limit ?",
                                            pool, name, max(1, min(limit, 1000)))]


def ref_list(db: DB, pool: str) -> list[dict]:
    rows = db.all("select r.* from refs r join (select name, max(version) as v from refs where pool=? group by name) m "
                  "on r.name=m.name and r.version=m.v where r.pool=? order by r.name", pool, pool)
    return [_ref_row(row) for row in rows]


def ref_publish(db: DB, pool: str, name: str, artifact: str, parent: int | None, member: str,
                force: bool = False, meta: dict | None = None) -> dict:
    """Compare-and-swap: the new version is current+1, accepted only if `parent` is the current
    version (0 or None for a new ref) or `force`. A 409 carries the current version."""
    if not REF_NAME.match(name or "") or "//" in name or name.endswith("/"):
        raise CoordError(400, "ref names: letters, digits, '.', '_', '-' and '/', at most 128")
    meta_text = _json(meta or {}, 16 * 1024, "meta")
    with db.lock:
        current = ref_current(db, pool, name)
        version = current["version"] if current else 0
        if not force and (parent or 0) != version:
            raise CoordError(409, f"{name} is at version {version}, not {parent or 0}: fetch it, then publish on top",
                             current=version)
        db.run("insert into refs (pool, name, version, artifact, parent, meta, created_by, created_at) values (?, ?, ?, ?, ?, ?, ?, ?)",
               pool, name, version + 1, artifact, version, meta_text, member, now())
    return ref_get(db, pool, name, version + 1)


def ref_artifact_in_pools(db: DB, artifact: str, pools: list[str]) -> bool:
    if not pools:
        return False
    marks = ",".join("?" * len(pools))
    return bool(db.one(f"select 1 as x from refs where artifact=? and pool in ({marks})", artifact, *pools))


# ---- queues ---------------------------------------------------------------------------------------

def _queue(db: DB, pool: str, name: str) -> dict:
    row = db.one("select * from queues where pool=? and name=?", pool, name)
    if not row:
        raise CoordError(404, f"no queue {name}")
    return {**row, "settings": json.loads(row["settings"] or "{}")}


def queue_create(db: DB, pool: str, name: str, owner: str, ref: str | None = None, max_lag: int | None = None,
                 stale: str = "requeue", duplicate_after: float | None = None, replicate: float | None = None) -> dict:
    """Create a queue, or update its settings (the same call again is a no-op)."""
    if not QUEUE_NAME.match(name or ""):
        raise CoordError(400, "queue names: letters, digits, '.', '_' and '-', at most 64")
    if ref is not None and not REF_NAME.match(ref):
        raise CoordError(400, "ref: a ref name")
    if max_lag is not None and (ref is None or not isinstance(max_lag, int) or max_lag < 0):
        raise CoordError(400, "max_lag is a whole number of versions, and needs a ref")
    if stale not in ("requeue", "drop"):
        raise CoordError(400, "stale is requeue or drop")
    settings = {"stale": stale}
    if replicate:
        if not 0 < float(replicate) <= 1:
            raise CoordError(400, "replicate is the fraction of tasks run twice to compare, from 0 to 1")
        settings["replicate"] = float(replicate)
    if duplicate_after is not None:
        settings["duplicate_after"] = max(1.0, float(duplicate_after))
    with db.lock:
        existing = db.one("select owner from queues where pool=? and name=?", pool, name)
        if existing and existing["owner"] != owner:
            raise CoordError(403, f"queue {name} belongs to someone else")
        if existing:
            db.run("update queues set ref=?, max_lag=?, settings=? where pool=? and name=?",
                   ref, max_lag, json.dumps(settings), pool, name)
        else:
            db.run("insert into queues (pool, name, owner, ref, max_lag, settings, created_at) values (?, ?, ?, ?, ?, ?, ?)",
                   pool, name, owner, ref, max_lag, json.dumps(settings), now())
    return queue_stats(db, pool, name)


def tasks_add(db: DB, pool: str, name: str, tasks: list) -> list[str]:
    """Add tasks: [{inputs, key?, requires_version?}]. A key already in the queue returns the existing id."""
    _queue(db, pool, name)
    if not isinstance(tasks, list) or not tasks or len(tasks) > MAX_TASKS_PER_ADD:
        raise CoordError(400, f"tasks is a list of 1 to {MAX_TASKS_PER_ADD}")
    ids = []
    with db.lock:
        for item in tasks:
            if not isinstance(item, dict) or set(item) - {"inputs", "key", "requires_version"}:
                raise CoordError(400, "each task is {inputs, key?, requires_version?}")
            key = item.get("key")
            if key is not None and (not isinstance(key, str) or not 1 <= len(key) <= 200):
                raise CoordError(400, "a task key is a string of 1 to 200 characters")
            if key is not None:
                row = db.one("select id from tasks where pool=? and queue=? and task_key=?", pool, name, key)
                if row:
                    ids.append(row["id"])
                    continue
            task_id = new_id("t")
            db.run("insert into tasks (id, pool, queue, task_key, inputs, requires_version, status, created_at) "
                   "values (?, ?, ?, ?, ?, ?, 'queued', ?)", task_id, pool, name, key,
                   _json(item.get("inputs"), MAX_INPUTS_BYTES, "task inputs"), item.get("requires_version"), now())
            ids.append(task_id)
    return ids


def expire_task_leases(db: DB) -> int:
    """Drop lapsed leases; a running task without a lease goes back to the queue."""
    with db.lock:
        lapsed = db.all("select task from task_leases where expires < ?", now())
        if not lapsed:
            return 0
        db.run("delete from task_leases where expires < ?", now())
        back = 0
        for task in {row["task"] for row in lapsed}:
            if not db.one("select 1 as x from task_leases where task=?", task):
                back += db.run("update tasks set status='queued', started_at=null where id=? and status='running'", task)
        return back


def extend_task_leases(db: DB, machine: str, jobs: list[str]) -> None:
    for job in jobs:
        db.run("update task_leases set expires=? where job=? and machine=?", now() + TASK_LEASE_SECONDS, job, machine)


def _slow_after(db: DB, pool: str, name: str, settings: dict) -> float:
    """1.5 x the p90 of recent task times, and at least `duplicate_after` seconds."""
    rows = db.all("select seconds from tasks where pool=? and queue=? and status='done' and seconds is not null "
                  "order by finished_at desc limit 200", pool, name)
    floor = float(settings.get("duplicate_after", DUPLICATE_AFTER))
    if len(rows) < 5:
        return floor
    times = sorted(row["seconds"] for row in rows)
    p90 = times[min(len(times) - 1, int(0.9 * len(times)))]
    return max(floor, 1.5 * p90)


def _public_task(row: dict, duplicate: bool = False) -> dict:
    return {"id": row["id"], "key": row["task_key"], "inputs": json.loads(row["inputs"]),
            "requires_version": row["requires_version"], "tries": row["tries"], "duplicate": duplicate}


def tasks_lease(db: DB, pool: str, name: str, job: str, machine: str, n: int = 1) -> list[dict]:
    """Up to `n` tasks for this job: queued ones oldest first, then, if none are left, a duplicate of
    a task that runs too long elsewhere."""
    queue = _queue(db, pool, name)
    n = max(1, min(int(n), 256))
    out = []
    if trust.excluded(db, pool, machine):
        return out
    with db.lock:
        expire_task_leases(db)
        rows = db.all("select * from tasks where pool=? and queue=? and status='queued' order by created_at, id limit ?",
                      pool, name, n + 16)
        rows = [r for r in rows if not db.one("select 1 as x from task_votes where task=? and machine=?", r["id"], machine)][:n]
        for row in rows:
            if db.run("update tasks set status='running', started_at=?, tries=tries+1, machine=? where id=? and status='queued'",
                      now(), machine, row["id"]):
                db.run("insert into task_leases (task, job, machine, started_at, expires) values (?, ?, ?, ?, ?) "
                       "on conflict (task, job) do update set expires=excluded.expires, started_at=excluded.started_at",
                       row["id"], job, machine, now(), now() + TASK_LEASE_SECONDS)
                out.append(_public_task({**row, "tries": row["tries"] + 1}))
        if not out:
            slow = now() - _slow_after(db, pool, name, queue["settings"])
            for row in db.all("select t.* from tasks t where t.pool=? and t.queue=? and t.status='running' and t.started_at < ? "
                              "and t.duplicates < ? order by t.started_at limit ?", pool, name, slow, DUPLICATES_MAX, n):
                if db.one("select 1 as x from task_leases where task=? and (job=? or machine=?)", row["id"], job, machine):
                    continue  # a copy on the same machine would be just as slow
                db.run("update tasks set duplicates=duplicates+1 where id=?", row["id"])
                db.run("insert into task_leases (task, job, machine, started_at, expires) values (?, ?, ?, ?, ?)",
                       row["id"], job, machine, now(), now() + TASK_LEASE_SECONDS)
                out.append(_public_task(row, duplicate=True))
    return out


def task_result(db: DB, pool: str, name: str, task_id: str, job: str, machine: str, result,
                computed_with: int | None = None, failed: str | None = None) -> dict:
    """The first result wins. A stale one (more than max_lag behind the queue's ref) is requeued or dropped."""
    queue = _queue(db, pool, name)
    text = _json(result, MAX_RESULT_BYTES, "a task result")
    with db.lock:
        row = db.one("select * from tasks where id=? and pool=? and queue=?", task_id, pool, name)
        if not row:
            raise CoordError(404, f"no task {task_id} in {name}")
        lease = db.one("select * from task_leases where task=? and job=?", task_id, job)
        db.run("delete from task_leases where task=? and job=?", task_id, job)
        if row["status"] in ("done", "dropped", "failed"):
            return {"accepted": False, "reason": "already done (a duplicate finished first)"}
        if not lease:
            return {"accepted": False, "reason": "this job no longer holds the task (its lease lapsed)"}
        others = db.one("select 1 as x from task_leases where task=?", task_id)
        if trust.excluded(db, pool, machine):
            if not others:
                db.run("update tasks set status='queued', started_at=null where id=?", task_id)
            return {"accepted": False, "reason": "this machine is excluded from the pool's work (failed checks)"}
        if failed:
            if not others:
                db.run("update tasks set status='queued', started_at=null where id=?", task_id)
            return {"accepted": False, "reason": "requeued after a failure"}
        if queue["ref"] and queue["max_lag"] is not None:
            current = ref_current(db, pool, queue["ref"])
            version = current["version"] if current else 0
            if computed_with is None or version - int(computed_with) > queue["max_lag"]:
                db.run("update tasks set stale=stale+1 where id=?", task_id)
                drop = queue["settings"].get("stale") == "drop"
                if drop:
                    db.run("update tasks set status='dropped', finished_at=? where id=?", now(), task_id)
                elif not others:
                    db.run("update tasks set status='queued', started_at=null where id=?", task_id)
                return {"accepted": False, "reason": f"stale: computed with version {computed_with}, the ref is at {version} "
                                                     f"and the queue allows {queue['max_lag']} behind",
                        "current": version, "dropped": drop}
        if _replicated(queue, task_id):
            verdict = _vote(db, pool, row, job, machine, text, computed_with)
            if verdict is None:  # needs another machine's result first
                if not others:
                    db.run("update tasks set status='queued', started_at=null where id=?", task_id)
                return {"accepted": True, "pending": "waiting for another machine to check this task"}
            text, machine, computed_with = verdict
        seconds = now() - (lease["started_at"] or now())
        db.run("update tasks set status='done', finished_at=?, computed_with=?, machine=?, seconds=? where id=?",
               now(), computed_with, machine, seconds, task_id)
        db.run("delete from task_leases where task=?", task_id)  # a duplicate still running has lost
        db.run("insert into task_results (pool, queue, task, result, computed_with, machine, created_at) values (?, ?, ?, ?, ?, ?, ?)",
               pool, name, task_id, text, computed_with, machine, now())
    return {"accepted": True}


def _replicated(queue: dict, task_id: str) -> bool:
    """A fixed fraction of the queue's tasks, chosen by id, is run twice (the duplicate check, M6)."""
    fraction = float(queue["settings"].get("replicate") or 0)
    if fraction <= 0:
        return False
    return int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < fraction


def _vote(db: DB, pool: str, task: dict, job: str, machine: str, text: str, computed_with):
    """Record one machine's result for a replicated task. Two matching results from different machines
    decide it (a third breaks a tie). Returns (result, machine, computed_with) once decided, else None.
    Every vote is reported: agreeing with the decision passes the check, disagreeing fails it."""
    digest = hashlib.sha256(text.encode()).hexdigest()
    db.run("insert into task_votes (task, machine, job, digest, result, computed_with, created_at) values (?, ?, ?, ?, ?, ?, ?) "
           "on conflict (task, machine) do update set digest=excluded.digest, result=excluded.result", task["id"], machine, job,
           digest, text, computed_with, now())
    votes = db.all("select * from task_votes where task=?", task["id"])
    counts: dict[str, int] = {}
    for vote in votes:
        counts[vote["digest"]] = counts.get(vote["digest"], 0) + 1
    winner = next((d for d, c in counts.items() if c >= 2), None)
    if winner is None:
        if len(votes) >= 3:  # three different answers: nobody to believe
            db.run("update tasks set status='failed', finished_at=? where id=?", now(), task["id"])
            for vote in votes:
                trust.report(db, pool, vote["machine"], False, "duplicate", f"task {task['id']}: no two machines agreed")
        return None
    for vote in votes:
        trust.report(db, pool, vote["machine"], vote["digest"] == winner, "duplicate",
                     "" if vote["digest"] == winner else f"task {task['id']}: result differs from {counts[winner]} other machines")
    chosen = next(v for v in votes if v["digest"] == winner)
    return chosen["result"], chosen["machine"], chosen["computed_with"]


def results(db: DB, pool: str, name: str, after: int = 0, limit: int = 500) -> dict:
    _queue(db, pool, name)
    rows = db.all("select r.*, t.task_key from task_results r join tasks t on t.id=r.task where r.pool=? and r.queue=? and r.seq > ? "
                  "order by r.seq limit ?", pool, name, int(after), max(1, min(int(limit), 5000)))
    out = [{"seq": row["seq"], "task": row["task"], "key": row["task_key"], "result": json.loads(row["result"]),
            "computed_with": row["computed_with"], "machine": row["machine"]} for row in rows]
    return {"results": out, "next": out[-1]["seq"] if out else int(after)}


def queue_stats(db: DB, pool: str, name: str) -> dict:
    queue = _queue(db, pool, name)
    counts = {row["status"]: row["n"] for row in db.all(
        "select status, count(*) as n from tasks where pool=? and queue=? group by status", pool, name)}
    extra = db.one("select coalesce(sum(stale), 0) as stale, coalesce(sum(duplicates), 0) as duplicates from tasks "
                   "where pool=? and queue=?", pool, name)
    times = sorted(row["seconds"] for row in db.all(
        "select seconds from tasks where pool=? and queue=? and status='done' and seconds is not null "
        "order by finished_at desc limit 200", pool, name))
    pick = (lambda q: round(times[min(len(times) - 1, int(q * len(times)))], 3)) if times else (lambda q: None)
    return {"name": name, "ref": queue["ref"], "max_lag": queue["max_lag"], "settings": queue["settings"],
            "counts": {status: counts.get(status, 0) for status in ("queued", "running", "done", "dropped", "failed")},
            "stale_results": extra["stale"], "duplicates": extra["duplicates"], "p50_seconds": pick(0.5), "p90_seconds": pick(0.9)}


def queue_list(db: DB, pool: str) -> list[dict]:
    return [queue_stats(db, pool, row["name"]) for row in db.all("select name from queues where pool=? order by name", pool)]
