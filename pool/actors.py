"""Actors and checkpoints (M3, #45 sections 4, 6.3 and 8).

Actor       A named group of identical long-lived jobs (replicas), usually workers that lease
            tasks from a queue and keep weights loaded between them. The server keeps an actor
            between `min` and `max` replicas: with a queue, about one replica per `per_replica`
            queued tasks; without one, `min`. A replica that ends, fails or is lost is replaced.
            Scaling down removes queued replicas only, never one that is running.
            A replica that fails within FAST_SECONDS of starting counts as a crash; after
            CRASH_LIMIT crashes in CRASH_WINDOW the actor is marked failing and stops relaunching,
            so a broken image does not loop forever (`actor scale` or a new `actor create` resets it).

Checkpoint  A job saves its /checkpoint folder through its agent (or the agent does, when the job is
            preempted). When the job runs again (a lost lease, a preemption), the agent restores the
            newest checkpoint into /checkpoint first and sets POOL_RESUMED=1.

Jobs may subscribe to refs (spec `subscribe: [names]`); the heartbeat reply lists the current
versions of the refs the machine's running jobs subscribe to, so the agent fetches a new version
into its cache as soon as it is published (`warm`).
"""

import json
import math
import re

from pool.db import DB, now

ACTOR_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,47}$")
MAX_REPLICAS = 256
FAST_SECONDS = 60.0
CRASH_LIMIT = 5
CRASH_WINDOW = 600.0


class ActorError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def _actor(db: DB, pool: str, name: str) -> dict:
    row = db.one("select * from actors where pool=? and name=?", pool, name)
    if not row:
        raise ActorError(404, f"no actor {name}")
    return row


def _bounds(min_replicas, max_replicas) -> tuple[int, int]:
    low = int(min_replicas if min_replicas is not None else 1)
    high = int(max_replicas if max_replicas is not None else max(low, 1))
    if not 0 <= low <= high <= MAX_REPLICAS:
        raise ActorError(400, f"replicas: 0 <= min <= max <= {MAX_REPLICAS}")
    return low, high


def create(db: DB, pool: str, name: str, owner: str, spec: dict, min_replicas=1, max_replicas=None,
           queue: str | None = None, per_replica: int = 1) -> dict:
    """Create an actor, or replace its spec and bounds (running replicas keep their spec until replaced)."""
    if not ACTOR_NAME.match(name or ""):
        raise ActorError(400, "actor names: letters, digits, '.', '_' and '-', at most 48")
    low, high = _bounds(min_replicas, max_replicas)
    if spec.get("gang"):
        raise ActorError(400, "an actor's replicas are single jobs (no gang)")
    if queue is not None and not db.one("select 1 as x from queues where pool=? and name=?", pool, queue):
        raise ActorError(404, f"no queue {queue}")
    per = max(1, int(per_replica or 1))
    with db.lock:
        existing = db.one("select owner from actors where pool=? and name=?", pool, name)
        if existing and existing["owner"] != owner:
            raise ActorError(403, f"actor {name} belongs to someone else")
        if existing:
            db.run("update actors set spec=?, min_replicas=?, max_replicas=?, queue=?, per_replica=?, status='running', note=null "
                   "where pool=? and name=?", json.dumps(spec), low, high, queue, per, pool, name)
        else:
            db.run("insert into actors (pool, name, owner, spec, min_replicas, max_replicas, queue, per_replica, status, created_at) "
                   "values (?, ?, ?, ?, ?, ?, ?, ?, 'running', ?)", pool, name, owner, json.dumps(spec), low, high, queue, per, now())
    return status(db, pool, name)


def scale(db: DB, pool: str, name: str, owner: str, min_replicas=None, max_replicas=None) -> dict:
    actor = _actor(db, pool, name)
    if actor["owner"] != owner:
        raise ActorError(403, f"actor {name} belongs to someone else")
    low, high = _bounds(actor["min_replicas"] if min_replicas is None else min_replicas,
                        actor["max_replicas"] if max_replicas is None else max_replicas)
    db.run("update actors set min_replicas=?, max_replicas=?, status='running', note=null where pool=? and name=?",
           low, high, pool, name)
    return status(db, pool, name)


def stop(db: DB, pool: str, name: str, owner: str) -> list[str]:
    """Stop the actor; returns its live jobs, for the caller to cancel."""
    actor = _actor(db, pool, name)
    if actor["owner"] != owner:
        raise ActorError(403, f"actor {name} belongs to someone else")
    db.run("update actors set status='stopped' where pool=? and name=?", pool, name)
    return [row["id"] for row in _live(db, pool, name)]


def _live(db: DB, pool: str, name: str) -> list[dict]:
    return db.all("select j.id, j.status, a.replica from actor_jobs a join jobs j on j.id=a.job "
                  "where a.pool=? and a.actor=? and j.status in ('queued', 'running') order by a.replica", pool, name)


def desired(db: DB, actor: dict) -> int:
    low, high = actor["min_replicas"], actor["max_replicas"]
    if not actor["queue"]:
        return low
    backlog = db.one("select count(*) as n from tasks where pool=? and queue=? and status in ('queued', 'running')",
                     actor["pool"], actor["queue"])["n"]
    return max(low, min(high, math.ceil(backlog / actor["per_replica"])))


def _crashes(db: DB, actor: dict) -> int:
    rows = db.all("select j.started_at, j.finished_at from actor_jobs a join jobs j on j.id=a.job "
                  "where a.pool=? and a.actor=? and j.status='failed' and j.finished_at > ?",
                  actor["pool"], actor["name"], now() - CRASH_WINDOW)
    return sum(1 for row in rows if row["started_at"] and row["finished_at"] - row["started_at"] < FAST_SECONDS)


def reconcile(db: DB, create_jobs, cancel_job) -> dict:
    """Bring every running actor to its desired replica count. `create_jobs(pool, owner, spec, name)`
    queues a job and returns its ids; `cancel_job(job_id)` cancels a queued one."""
    started, removed = 0, 0
    for actor in db.all("select * from actors where status='running'"):
        if _crashes(db, actor) >= CRASH_LIMIT:
            db.run("update actors set status='failing', note=? where pool=? and name=?",
                   f"{CRASH_LIMIT} replicas failed within {FAST_SECONDS:.0f} s of starting; fix the spec, then scale or create it again",
                   actor["pool"], actor["name"])
            continue
        live = _live(db, actor["pool"], actor["name"])
        want = desired(db, actor)
        if len(live) < want:
            spec = json.loads(actor["spec"])
            taken = {row["replica"] for row in live}
            free = (n for n in range(MAX_REPLICAS) if n not in taken)
            for _ in range(want - len(live)):
                replica = next(free)
                (job_id,) = create_jobs(actor["pool"], actor["owner"], spec, f"{actor['name']}.{replica}")
                db.run("insert into actor_jobs (job, pool, actor, replica, created_at) values (?, ?, ?, ?, ?)",
                       job_id, actor["pool"], actor["name"], replica, now())
                started += 1
        elif len(live) > want:
            for row in reversed([row for row in live if row["status"] == "queued"][: len(live) - want]):
                cancel_job(row["id"])
                removed += 1
    return {"started": started, "removed": removed}


def status(db: DB, pool: str, name: str) -> dict:
    actor = _actor(db, pool, name)
    live = _live(db, pool, name)
    spec = json.loads(actor["spec"])
    return {"name": name, "status": actor["status"], "note": actor["note"], "image": spec.get("image"),
            "min": actor["min_replicas"], "max": actor["max_replicas"], "queue": actor["queue"],
            "per_replica": actor["per_replica"], "desired": desired(db, actor) if actor["status"] == "running" else 0,
            "running": sum(1 for row in live if row["status"] == "running"),
            "queued": sum(1 for row in live if row["status"] == "queued"),
            "replicas": [{"job": row["id"], "replica": row["replica"], "status": row["status"]} for row in live],
            "crashes": _crashes(db, actor)}


def listing(db: DB, pool: str) -> list[dict]:
    return [status(db, pool, row["name"]) for row in db.all("select name from actors where pool=? order by name", pool)]


# ---- checkpoints -----------------------------------------------------------------------------

def save_checkpoint(db: DB, job: str, artifact: str, attempt: int | None) -> None:
    db.run("insert into job_checkpoints (job, artifact, attempt, saved_at) values (?, ?, ?, ?) "
           "on conflict (job) do update set artifact=excluded.artifact, attempt=excluded.attempt, saved_at=excluded.saved_at",
           job, artifact, attempt, now())


def checkpoint_of(db: DB, job: str) -> dict | None:
    row = db.one("select artifact, attempt, saved_at from job_checkpoints where job=?", job)
    return dict(row) if row else None


def checkpoint_readable(db: DB, artifact: str, machine: str) -> bool:
    """A machine may read the checkpoint of a job it is running."""
    return bool(db.one("select 1 as x from job_checkpoints c join jobs j on j.id=c.job "
                       "where c.artifact=? and j.machine=? and j.status='running'", artifact, machine))


# ---- ref subscriptions -----------------------------------------------------------------------------

def warm_for(db: DB, machine: str) -> list[dict]:
    """Current versions of the refs this machine's running jobs subscribe to."""
    wanted = {}
    for row in db.all("select pool, spec from jobs where machine=? and status='running'", machine):
        for name in json.loads(row["spec"]).get("subscribe") or []:
            wanted[(row["pool"], name)] = True
    out = []
    for pool, name in wanted:
        ref = db.one("select name, version, artifact from refs where pool=? and name=? order by version desc limit 1", pool, name)
        if ref:
            out.append(dict(ref))
    return out

