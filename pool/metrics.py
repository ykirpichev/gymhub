"""Prometheus metrics for the pool server (#45 section 13): GET /metrics, in the text format 0.0.4.

Server-wide, so the endpoint takes a server admin's member token or the scrape token given with
`pool server --metrics-token` (or POOL_METRICS_TOKEN), for example in prometheus.yml:

    scrape_configs:
      - job_name: pool
        authorization: {credentials_file: /etc/prometheus/pool-token}
        static_configs: [{targets: ["pool.example:8765"]}]

Every value is read from the database at scrape time, so several server processes sharing one
Postgres database report the same numbers.
"""

import json

from pool import scheduler, update
from pool.db import DB, now

JOB_STATES = ("queued", "running", "succeeded", "failed", "cancelled")
TASK_STATES = ("queued", "running", "done", "failed", "dropped")
CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def _escape(value) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


class Exposition:
    def __init__(self):
        self.lines: list[str] = []

    def metric(self, name: str, kind: str, help_text: str, samples) -> None:
        """samples: [(labels dict, value)]; a metric without samples is still described."""
        self.lines.append(f"# HELP {name} {help_text}")
        self.lines.append(f"# TYPE {name} {kind}")
        for labels, value in samples:
            body = ",".join(f'{key}="{_escape(val)}"' for key, val in labels.items())
            number = float(value)
            number = str(int(number)) if number.is_integer() else repr(number)
            self.lines.append(f"{name}{{{body}}} {number}" if body else f"{name} {number}")

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def render(db: DB, version: str = "", schema: int | None = None, agent_digest: str | None = None) -> str:
    t = now()
    out = Exposition()
    pools = {row["id"]: row["name"] for row in db.all("select id, name from pools")}

    def name(pool_id):
        return pools.get(pool_id, pool_id)

    out.metric("pool_server_info", "gauge", "The pool server's version.", [({"version": version}, 1)])
    if schema is not None:
        out.metric("pool_schema_version", "gauge", "The newest database migration applied.", [({}, schema)])

    # ---- jobs and queues ---------------------------------------------------------------------------
    counts = {(row["pool"], row["status"]): row["n"]
              for row in db.all("select pool, status, count(*) as n from jobs group by pool, status")}
    states = list(JOB_STATES) + sorted({status for _, status in counts} - set(JOB_STATES))
    out.metric("pool_jobs", "gauge", "Jobs by pool and state.",
               [({"pool": name(p), "state": s}, counts.get((p, s), 0)) for p in pools for s in states])
    oldest = {row["pool"]: row["t"] for row in db.all("select pool, min(created_at) as t from jobs where status='queued' group by pool")}
    out.metric("pool_queue_depth", "gauge", "Jobs waiting for a machine.",
               [({"pool": name(p)}, counts.get((p, "queued"), 0)) for p in pools])
    out.metric("pool_queue_oldest_seconds", "gauge", "Age of the oldest queued job (0 if none).",
               [({"pool": name(p)}, max(0.0, t - oldest[p]) if oldest.get(p) else 0) for p in pools])
    tasks = db.all("select pool, queue, status, count(*) as n from tasks group by pool, queue, status")
    queues = sorted({(row["pool"], row["queue"]) for row in db.all("select pool, name as queue from queues")}
                    | {(row["pool"], row["queue"]) for row in tasks})
    by_queue = {(row["pool"], row["queue"], row["status"]): row["n"] for row in tasks}
    task_states = list(TASK_STATES) + sorted({row["status"] for row in tasks} - set(TASK_STATES))
    out.metric("pool_tasks", "gauge", "Tasks in coordination queues (pool queue), by state.",
               [({"pool": name(p), "queue": q, "state": s}, by_queue.get((p, q, s), 0)) for p, q in queues for s in task_states])

    # ---- machines ----------------------------------------------------------------------------------
    machines = db.all("select id, pool, name, profile, drain, last_seen from machines order by name")
    busy: dict[str, int] = {}
    for row in db.all("select machine, gpus from jobs where status='running' and machine is not null"):
        busy[row["machine"]] = busy.get(row["machine"], 0) + len(json.loads(row["gpus"] or "[]"))
    tally: dict[tuple, int] = {}
    gpus_online: dict[str, int] = {p: 0 for p in pools}
    gpus_busy: dict[str, int] = {p: 0 for p in pools}
    lag = []
    for m in machines:
        online = bool(m["last_seen"]) and t - m["last_seen"] < scheduler.ONLINE_SECONDS
        state = "draining" if online and m["drain"] else "online" if online else "offline"
        tally[(m["pool"], state)] = tally.get((m["pool"], state), 0) + 1
        if online:
            gpus_online[m["pool"]] = gpus_online.get(m["pool"], 0) + len(json.loads(m["profile"] or "{}").get("gpus") or [])
            gpus_busy[m["pool"]] = gpus_busy.get(m["pool"], 0) + busy.get(m["id"], 0)
        if m["last_seen"]:
            lag.append(({"pool": name(m["pool"]), "machine": m["name"]}, max(0.0, t - m["last_seen"])))
    out.metric("pool_machines", "gauge", f"Machines by state (online: a heartbeat in the last {scheduler.ONLINE_SECONDS} s).",
               [({"pool": name(p), "state": s}, tally.get((p, s), 0)) for p in pools for s in ("online", "draining", "offline")])
    out.metric("pool_machines_online", "gauge", "Machines online, draining ones included.",
               [({"pool": name(p)}, tally.get((p, "online"), 0) + tally.get((p, "draining"), 0)) for p in pools])
    out.metric("pool_gpus_online", "gauge", "GPUs on machines that are online.", [({"pool": name(p)}, n) for p, n in gpus_online.items()])
    out.metric("pool_gpus_busy", "gauge", "GPUs that running jobs hold.", [({"pool": name(p)}, n) for p, n in gpus_busy.items()])
    out.metric("pool_machine_heartbeat_age_seconds", "gauge", "Seconds since each machine was last heard from.", lag)
    if agent_digest:
        # unknown counts as outdated: an agent that does not report its digest predates reporting it
        out.metric("pool_machine_agent_outdated", "gauge",
                   "1 when a machine's agent differs from the server's agent package (or does not report its version).",
                   [({"pool": name(m["pool"]), "machine": m["name"]},
                     int(update.agent_state(json.loads(m["profile"] or "{}").get("agent_digest"), agent_digest) != "current"))
                    for m in machines])

    # ---- GPU-hours ---------------------------------------------------------------------------------
    total: dict[str, float] = {p: 0.0 for p in pools}
    day: dict[str, float] = {p: 0.0 for p in pools}
    since = t - 86400
    for row in db.all("select pool, spec, status, started_at, finished_at from jobs where started_at is not null"):
        res = json.loads(row["spec"]).get("resources") or {}
        gpus = float(res.get("gpus") or 0)
        end = t if row["status"] == "running" or not row["finished_at"] else row["finished_at"]
        total[row["pool"]] = total.get(row["pool"], 0.0) + gpus * max(0.0, end - row["started_at"]) / 3600
        day[row["pool"]] = day.get(row["pool"], 0.0) + gpus * max(0.0, end - max(row["started_at"], since)) / 3600
    out.metric("pool_gpu_hours_total", "counter", "GPU-hours that jobs have used, running jobs up to now.",
               [({"pool": name(p)}, round(v, 6)) for p, v in total.items()])
    out.metric("pool_gpu_hours_24h", "gauge", "GPU-hours used in the last 24 hours.",
               [({"pool": name(p)}, round(v, 6)) for p, v in day.items()])

    # ---- storage -----------------------------------------------------------------------------------
    owned = {row["pool"]: row for row in db.all(
        "select o.pool as pool, count(*) as n, coalesce(sum(a.size), 0) as bytes from "
        "(select distinct pool, artifact from artifact_owners) o join artifacts a on a.id=o.artifact group by o.pool")}
    out.metric("pool_artifacts", "gauge", "Artifacts that members of a pool own.",
               [({"pool": name(p)}, owned[p]["n"] if p in owned else 0) for p in pools])
    out.metric("pool_artifact_bytes", "gauge", "Logical size of the artifacts members of a pool own (shared chunks counted in each).",
               [({"pool": name(p)}, owned[p]["bytes"] if p in owned else 0) for p in pools])
    stored = db.one("select count(*) as n, coalesce(sum(size), 0) as bytes from chunks")
    out.metric("pool_chunks", "gauge", "Chunks in the server's store.", [({}, stored["n"])])
    out.metric("pool_chunk_bytes", "gauge", "Bytes in the server's chunk store (each chunk once).", [({}, stored["bytes"])])
    return out.text()
