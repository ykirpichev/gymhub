"""Placement (#45 section 6), M2: feasibility, the network check, scores with cache affinity,
fair share, quotas, leases and retries.

Machines pull work: a machine asks with what it has free, and the server picks, among the queued
jobs of its pool in fair-share order, one that it may run and that passes the network check. To
get cache affinity without a central planner, a machine passes over a job for a short while
(`affinity_wait_s`) when another online machine with free room would score clearly better on it,
typically because it already holds the inputs.

The network check: for job j on machine m,
    t_compute = cost.flops / (TFLOP/s of m in cost.dtype x GPUs)
    t_net     = bytes of j's inputs m does not hold / m's download rate + bytes_out / m's upload rate
and m is refused when t_net > alpha x t_compute (alpha 0.25 by default; a pool setting), unless
cost.bulk. Jobs without a cost hint skip the check but still prefer machines that hold their inputs.

Gangs: the ranks of a gang are ordinary queued jobs tied together in `gang_members`. Each rank is
claimed like any job, but never by a machine that already runs another rank of the same gang, and
only by machines that advertise a peer address (and allow peer networking, for network: peers). A
rank waits on its machine until every rank is claimed (the rendezvous); a rank that waits too long
is handed back without using a retry, and so are its running siblings. When a rank's machine
disappears, the whole gang goes back to the queue; when a rank fails for good, the gang fails.

With the pool setting `placement: maxflow` (M7), a gang is planned as a pipeline before its ranks
are claimed: pool/planner.py picks which machine runs which rank to maximise throughput, using
the machines' measured speed, the measured links between them and the job's cost hint (flops per
rank, link_bytes sent from each rank to the next). A machine then claims only the rank planned for
it, for up to affinity_wait_s after submission; after that any machine may (as with greedy).
"""

import fnmatch
import json

from pool import island, models, planner
from pool.db import DB, FINAL, load_job, load_machine, now
from pool.spec import image_for

LEASE_SECONDS = 90
ONLINE_SECONDS = 60
PLACEMENTS = ("greedy", "maxflow")
PLAN_SECONDS = 120  # a gang's plan is redrawn when older than this and none of its ranks runs yet
POOL_DEFAULTS = {"alpha": 0.25, "affinity_wait_s": 60, "retention_days": 30, "placement": "greedy",
                 "quota": {"max_running_gpus": None, "max_running_jobs": None, "gpu_hours_per_day": None,
                           "storage_gb": None}}
# Rough TFLOP/s when a machine has not measured itself: by dtype, per GPU; and for a CPU-only box.
GUESS_TFLOPS = {"fp64": 1, "fp32": 15, "tf32": 50, "bf16": 100, "fp16": 100, "fp8": 200, "int8": 200, "fp4": 400}
CPU_TFLOPS = 0.5
GUESS_MBPS = 100


def pool_settings(db: DB, pool_id: str) -> dict:
    row = db.one("select settings from pools where id=?", pool_id)
    saved = json.loads(row["settings"]) if row else {}
    merged = {**POOL_DEFAULTS, **saved}
    merged["quota"] = {**POOL_DEFAULTS["quota"], **(saved.get("quota") or {})}
    return merged


def member_quota(db: DB, pool_id: str, member: str, settings: dict | None = None) -> dict:
    settings = settings or pool_settings(db, pool_id)
    row = db.one("select quota from pool_members where pool=? and member=?", pool_id, member)
    return {**settings["quota"], **(json.loads(row["quota"]) if row and row["quota"] else {})}


def capacity(machine: dict) -> dict:
    """Everything a machine has, as if it were idle: for "could this job ever run here?"."""
    profile = machine.get("profile") or {}
    return {"gpus": [gpu.get("index") for gpu in profile.get("gpus", [])],
            "cpus": profile.get("cpus", 0), "memory_gb": profile.get("memory_gb", 0)}


def _weight(spec: dict) -> float:
    """GPU-equivalents a job occupies, for fair share and quotas: its GPUs, or CPUs/16."""
    res = spec["resources"]
    return float(res["gpus"]) if res["gpus"] else res["cpus"] / 16


# ---- feasibility ----------------------------------------------------------------------------------


def policy_allows(machine: dict, owner: str, owner_name: str | None = None) -> bool:
    who = (machine.get("policy") or {}).get("who", "pool")
    if who == "pool":
        return True
    if who == "owner":
        return owner == machine["owner"]
    return owner == machine["owner"] or owner in who or (owner_name is not None and owner_name in who)


def fits(spec: dict, machine: dict, free: dict, owner: str | None = None, owner_name: str | None = None,
         why: list | None = None) -> list[int] | None:
    """The GPU indices this job would get on this machine, or None if it does not fit. `free` is
    what the agent reported: {"gpus": [indices], "cpus": n, "memory_gb": n}. Reasons go to `why`."""
    def no(reason: str):
        if why is not None:
            why.append(reason)
        return None

    need = spec["resources"]
    profile = machine.get("profile") or {}
    if owner is not None and not policy_allows(machine, owner, owner_name):
        return no("its use policy does not admit you")
    if need.get("machines") == "own" and (owner is None or machine.get("owner") != owner):
        return no("not your machine (resources.machines: own)")  # fail closed: no owner known, no machine
    if need.get("machine") and machine.get("name") != need["machine"]:
        return no(f"not {need['machine']} (resources.machine)")
    labels = machine.get("labels") or {}
    if any(labels.get(key) != value for key, value in need["labels"].items()):
        return no("labels")
    allowed = profile.get("allow_images") or ["*"]
    if not any(fnmatch.fnmatchcase(image_for(spec, labels), pattern) for pattern in allowed):
        return no("image not allowed")
    peer = profile.get("peer") or {}
    if (spec.get("gang") or {}).get("size", 1) > 1 and not peer.get("addr"):
        return no("no peer address (agent too old, or not on a network the others can reach)")
    if spec.get("network") == "peers" and not peer.get("network"):
        return no("peer networking not allowed (pool agent run --allow-peer-network)")
    if spec.get("network") == "island" and not peer.get("island"):
        return no(f"no island networking ({peer.get('island_why') or 'agent too old'})")
    memory = need["memory_gb"]
    gpu_need = need["gpu_memory_gb"]
    unified = any(gpu.get("unified_memory") for gpu in profile.get("gpus", []))
    fit = need.get("fits")
    if fit and fit.get("meta"):
        required = models.need(fit["meta"], fit["mode"], fit["kv_gb"], need["gpus"])
        gpu_need = max(gpu_need, required["gpu_gb"])
        memory = max(memory, required["ram_gb"])
        if unified:  # GPU memory is system memory: count it once, in RAM
            memory = max(memory, required["ram_gb"] + required["gpu_gb"] * max(need["gpus"], 1))
    if need["cpus"] > float(free.get("cpus", 0)):
        return no("CPUs")
    if memory > float(free.get("memory_gb", 0)):
        return no(f"memory ({memory:.0f} GB needed)")
    if need["gpus"] == 0:
        return [] if not fit else no("a model fit needs GPUs")
    gpus = {gpu.get("index"): gpu for gpu in profile.get("gpus", [])}
    if need.get("share_gpu"):  # pack onto a GPU other shared jobs use, else take a free one
        shared = sorted((float(left), int(index)) for index, left in (free.get("shared_gpus") or {}).items()
                        if int(index) in gpus and float(left) >= gpu_need)
        if shared:
            return [shared[0][1]]
    usable = [index for index in free.get("gpus", []) if index in gpus
              and float(gpus[index].get("memory_gb") or 0) >= gpu_need]
    if len(usable) < need["gpus"]:
        return no(f"GPUs ({need['gpus']} with {gpu_need:.0f} GB needed)")
    return sorted(usable)[: need["gpus"]]


# ---- the network check and the score ----------------------------------------------------------------


def missing_bytes(db: DB, spec: dict, machine_id: str) -> tuple[int, int]:
    """(bytes of the job's input chunks this machine does not hold, total input bytes)."""
    artifacts = [a for a in _input_artifacts(spec)]
    if not artifacts:
        return 0, 0
    marks = ",".join("?" * len(artifacts))
    total = db.one(f"select coalesce(sum(c.size), 0) as n from chunks c where c.sha in "
                   f"(select distinct chunk from artifact_chunks where artifact in ({marks}))", *artifacts)["n"]
    held = db.one(f"select coalesce(sum(c.size), 0) as n from chunks c join chunk_locations l on l.chunk=c.sha "
                  f"where l.machine=? and c.sha in (select distinct chunk from artifact_chunks where artifact in ({marks}))",
                  machine_id, *artifacts)["n"]
    return int(total - held), int(total)


def _input_artifacts(spec: dict) -> list[str]:
    return ([spec["code"]] if spec.get("code") else []) + [item["artifact"] for item in spec.get("inputs", []) if item.get("artifact")]


def tflops(machine: dict, dtype: str) -> float:
    profile = machine.get("profile") or {}
    measured = (profile.get("bench") or {}).get("tflops") or {}
    if measured.get(dtype):
        return float(measured[dtype])
    if not profile.get("gpus"):
        return float(measured.get("fp32") or CPU_TFLOPS)
    if measured.get("bf16"):  # scale the measured bf16 rate by the usual ratio between dtypes
        return float(measured["bf16"]) * GUESS_TFLOPS[dtype] / GUESS_TFLOPS["bf16"]
    return float(GUESS_TFLOPS.get(dtype, 50))


def evaluate(db: DB, job: dict, machine: dict, alpha: float) -> dict:
    """Times, goodput and score of `job` on `machine`; ok=False when the link is too slow."""
    spec = job["spec"]
    network = (machine.get("profile") or {}).get("network") or {}
    down = float(network.get("down_mbps") or GUESS_MBPS) * 1e6 / 8
    up = float(network.get("up_mbps") or network.get("down_mbps") or GUESS_MBPS) * 1e6 / 8
    missing, total = missing_bytes(db, spec, machine["id"])
    cost = spec.get("cost") or {}
    t_net = missing / down + float(cost.get("bytes_out", 0)) / up
    dtype = cost.get("dtype", "bf16")
    rate = tflops(machine, dtype) * max(spec["resources"]["gpus"], 1) * 1e12
    t_compute = float(cost["flops"]) / rate if cost.get("flops") else None
    ok = True
    if t_compute is not None and not cost.get("bulk") and t_net > alpha * t_compute:
        ok = False
    basis = t_compute if t_compute is not None else 600.0  # without a hint, judge downloads against 10 minutes
    goodput = basis / (basis + t_net) if basis + t_net > 0 else 1.0
    speed = tflops(machine, dtype) / GUESS_TFLOPS.get(dtype, 100)
    cached = 1 - missing / total if total else 1.0
    score = goodput * min(speed, 4.0) + 0.3 * cached
    return {"ok": ok, "t_net": round(t_net, 2), "t_compute": round(t_compute, 2) if t_compute is not None else None,
            "missing_bytes": missing, "input_bytes": total, "goodput": round(goodput, 3), "score": round(score, 4),
            "down_mbps": round(down * 8 / 1e6, 1)}


# ---- fair share and quotas ----------------------------------------------------------------------------


def usage(db: DB, pool_id: str, hours: float = 24) -> dict[str, float]:
    """GPU-equivalent seconds per member in the last `hours`, running jobs included."""
    since = now() - hours * 3600
    out: dict[str, float] = {}
    rows = db.all("select owner, spec, started_at, finished_at, status from jobs where pool=? and started_at is not null "
                  "and (status='running' or finished_at > ?)", pool_id, since)
    for row in rows:
        start = max(row["started_at"], since)
        end = row["finished_at"] if row["status"] != "running" and row["finished_at"] else now()
        out[row["owner"]] = out.get(row["owner"], 0.0) + max(0.0, end - start) * _weight(json.loads(row["spec"]))
    return out


def within_quota(db: DB, job: dict, settings: dict) -> str | None:
    """None if the owner may start this job now, else the reason."""
    quota = member_quota(db, job["pool"], job["owner"], settings)
    running = [load_job(r) for r in db.all("select * from jobs where pool=? and owner=? and status='running'", job["pool"], job["owner"])]
    if quota.get("max_running_jobs") is not None and len(running) + 1 > quota["max_running_jobs"]:
        return f"at most {quota['max_running_jobs']} running jobs"
    gpus = job["spec"]["resources"]["gpus"]
    if quota.get("max_running_gpus") is not None and gpus and sum(r["spec"]["resources"]["gpus"] for r in running) + gpus > quota["max_running_gpus"]:
        return f"at most {quota['max_running_gpus']} GPUs at once"
    if quota.get("gpu_hours_per_day") is not None and gpus:
        used = usage(db, job["pool"]).get(job["owner"], 0.0) / 3600
        if used >= quota["gpu_hours_per_day"]:
            return f"{quota['gpu_hours_per_day']} GPU-hours a day used up"
    return None


def fair_order(db: DB, pool_id: str, machine_owner: str | None = None, owner_first: bool = True) -> list[dict]:
    """Queued jobs of a pool: the machine owner's first (owner-first policy), then members with the
    least GPU time in the last 24 hours, then priority, then oldest."""
    used = usage(db, pool_id)
    queued = [load_job(row) for row in db.all("select * from jobs where pool=? and status='queued' order by created_at limit 1000", pool_id)]
    return sorted(queued, key=lambda job: (
        0 if owner_first and job["owner"] == machine_owner else 1,
        used.get(job["owner"], 0.0), -job["priority"], job["created_at"]))


# ---- claims and leases ----------------------------------------------------------------------------------


def claim(db: DB, machine: dict, free: dict) -> dict | None:
    """Pick and lease one job for `machine`, or None."""
    if machine.get("drain"):
        return None
    settings = pool_settings(db, machine["pool"])
    policy = machine.get("policy") or {}
    names = {row["id"]: row["name"] for row in db.all("select id, name from members")}
    others = [load_machine(row) for row in db.all(
        "select * from machines where pool=? and id<>? and drain=0 and free_at > ?", machine["pool"], machine["id"], now() - ONLINE_SECONDS)]
    with db.lock:
        for job in fair_order(db, machine["pool"], machine["owner"], policy.get("owner_first", True)):
            gpus = fits(job["spec"], machine, free, job["owner"], names.get(job["owner"]))
            if gpus is None:
                continue
            if within_quota(db, job, settings):
                continue
            gang = gang_of(job)
            if gang and any(s["machine"] == machine["id"] and s["status"] == "running" for s in siblings(db, job)):
                continue  # one rank per machine
            mine = evaluate(db, job, machine, settings["alpha"])
            if not mine["ok"]:
                continue
            planned = None
            if gang and settings.get("placement") == "maxflow":
                planned = gang_plan(db, job, settings, names)
                target = planned["ranks"][gang["rank"]] if planned else None
                if target and target != machine["id"] and now() - job["created_at"] < settings["affinity_wait_s"]:
                    continue  # this rank is planned for another machine
            if not gang and now() - job["created_at"] < settings["affinity_wait_s"]:
                better = [m["name"] for m in others
                          if fits(job["spec"], m, m["free"] or {}, job["owner"], names.get(job["owner"])) is not None
                          and (e := evaluate(db, job, m, settings["alpha"]))["ok"] and e["score"] > mine["score"] * 1.25 + 0.05]
                if better:
                    continue  # leave it to a machine that holds its inputs, for a little while
            placement = {**mine, "machine": machine["name"], "at": now()}
            if planned:
                placement["plan"] = {"throughput": planned["throughput"], "bottleneck": planned["bottleneck"],
                                     "followed": planned["ranks"][gang["rank"]] == machine["id"]}
            claimed = db.run(
                "update jobs set status='running', machine=?, gpus=?, lease_expires=?, tries=tries+1, placement=?, "
                "started_at=coalesce(started_at, ?) where id=? and status='queued'",
                machine["id"], json.dumps(gpus), now() + LEASE_SECONDS, json.dumps(placement), now(), job["id"])
            if claimed:
                return load_job(db.one("select * from jobs where id=?", job["id"]))
    return None


def explain(db: DB, job: dict) -> list[str]:
    """Why a queued job is not running: one line per machine of its pool."""
    settings = pool_settings(db, job["pool"])
    names = {row["id"]: row["name"] for row in db.all("select id, name from members")}
    lines = []
    reason = within_quota(db, job, settings)
    if reason:
        lines.append(f"quota: {reason}")
    for machine in [load_machine(r) for r in db.all("select * from machines where pool=? order by name", job["pool"])]:
        why: list[str] = []
        online = machine["last_seen"] and now() - machine["last_seen"] < ONLINE_SECONDS
        if not online:
            lines.append(f"{machine['name']}: offline")
            continue
        if machine["drain"]:
            lines.append(f"{machine['name']}: draining")
            continue
        if fits(job["spec"], machine, capacity(machine), job["owner"], names.get(job["owner"]), why) is None:
            lines.append(f"{machine['name']}: never fits ({', '.join(why)})")
            continue
        verdict = evaluate(db, job, machine, settings["alpha"])
        if not verdict["ok"]:
            lines.append(f"{machine['name']}: link too slow ({verdict['t_net']:.0f} s to fetch inputs vs "
                         f"{verdict['t_compute']:.0f} s of compute at {verdict['down_mbps']:.0f} Mbit/s)")
            continue
        if fits(job["spec"], machine, machine["free"] or {}, job["owner"], names.get(job["owner"])) is None:
            lines.append(f"{machine['name']}: busy")
        else:
            lines.append(f"{machine['name']}: can run it (score {verdict['score']})")
    gang = gang_of(job)
    if gang and settings.get("placement") == "maxflow":
        planned = gang_plan(db, job, settings, names)
        if planned:
            label = {row["id"]: row["name"] for row in db.all("select id, name from machines where pool=?", job["pool"])}
            lines.append("plan: " + ", ".join(f"rank {r} on {label.get(m, m)}" for r, m in enumerate(planned["ranks"]))
                         + f" ({planned['throughput']:.3g} runs/s; bottleneck: {', '.join(planned['bottleneck']) or 'none'})")
    if gang:
        able = sum(1 for line in lines if ": can run it" in line or line.endswith(": busy"))
        if able < gang["size"]:
            lines.append(f"gang: needs {gang['size']} machines that can run a rank; {able} can")
    return lines or ["no machines in this pool"]


def expire_leases(db: DB) -> int:
    """Jobs whose machine stopped reporting: retry elsewhere while tries remain, else fail. For a
    gang rank, its running siblings go back to the queue too (without using a retry), or fail with it."""
    changed = 0
    with db.lock:
        for row in db.all("select * from jobs where status='running' and lease_expires < ?", now()):
            job = load_job(row)
            found = db.one("select name from machines where id=?", job["machine"])
            name = found["name"] if found else job["machine"]
            if job["tries"] - job["preemptions"] <= job["spec"]["retries"]:
                done = db.run("update jobs set status='queued', machine=null, gpus=null, lease_expires=null "
                              "where id=? and status='running'", job["id"])
                changed += done
                if done:
                    append_log(db, job["id"], f"\n[pool] machine {name} stopped reporting; job requeued\n")
                    changed += requeue_siblings(db, job, f"rank {gang_of(job)['rank']} lost its machine {name}") if gang_of(job) else 0
            else:
                done = db.run("update jobs set status='failed', error=?, finished_at=?, lease_expires=null "
                              "where id=? and status='running'",
                              f"machine {name} stopped reporting (no retries left)", now(), job["id"])
                changed += done
                if done and gang_of(job):
                    changed += fail_siblings(db, job, f"rank {gang_of(job)['rank']} failed: machine {name} stopped reporting")
    return changed


def requeue_preempted(db: DB, job_id: str, machine_id: str, reason: str) -> bool:
    """The machine's owner needs it back: the job goes back to the queue without using a retry."""
    with db.lock:
        done = db.run("update jobs set status='queued', machine=null, gpus=null, lease_expires=null, "
                      "preemptions=preemptions+1 where id=? and machine=? and status='running'", job_id, machine_id)
        if done:
            append_log(db, job_id, f"\n[pool] preempted ({reason}); job requeued\n")
            job = load_job(db.one("select * from jobs where id=?", job_id))
            if gang_of(job):
                requeue_siblings(db, job, f"rank {gang_of(job)['rank']} was handed back ({reason})")
    return bool(done)


# ---- gangs ------------------------------------------------------------------------------------------------


def gang_of(job: dict | None) -> dict | None:
    gang = (job or {}).get("spec", {}).get("gang")
    return gang if gang and gang.get("id") else None


def siblings(db: DB, job: dict) -> list[dict]:
    """The other ranks of a job's gang."""
    gang = gang_of(job)
    if not gang:
        return []
    return [load_job(row) for row in db.all(
        "select j.* from jobs j join gang_members g on g.job=j.id where g.gang=? and j.id<>? order by g.rank", gang["id"], job["id"])]


def requeue_siblings(db: DB, job: dict, reason: str) -> int:
    """Running ranks of the gang go back to the queue without using a retry: a gang starts together."""
    changed = 0
    with db.lock:
        for other in siblings(db, job):
            if other["status"] == "running":
                done = db.run("update jobs set status='queued', machine=null, gpus=null, lease_expires=null, "
                              "preemptions=preemptions+1 where id=? and status='running'", other["id"])
                if done:
                    append_log(db, other["id"], f"\n[pool] {reason}; the gang goes back to the queue\n")
                changed += done
    return changed


def fail_siblings(db: DB, job: dict, reason: str) -> int:
    changed = 0
    with db.lock:
        for other in siblings(db, job):
            if other["status"] not in FINAL:
                done = db.run("update jobs set status='failed', error=?, finished_at=?, lease_expires=null "
                              "where id=? and status in ('queued', 'running')", reason, now(), other["id"])
                if done:
                    append_log(db, other["id"], f"\n[pool] {reason}; stopping this rank\n")
                changed += done
    return changed


def rendezvous(db: DB, job: dict) -> dict:
    """Where a gang's ranks find each other: ready once every rank runs somewhere."""
    gang = gang_of(job)
    ranks = sorted([job] + siblings(db, job), key=lambda j: j["spec"]["gang"]["rank"])
    # A rank that already succeeded met the others (a fast rank can finish before a slow one polls again).
    waiting = [r["spec"]["gang"]["rank"] for r in ranks if r["status"] not in ("running", "succeeded") or not r["machine"]]
    if waiting:
        return {"ready": False, "waiting": waiting}
    peers = []
    for r in ranks:
        machine = load_machine(db.one("select * from machines where id=?", r["machine"]))
        peer = (machine.get("profile") or {}).get("peer") or {}
        peers.append({"rank": r["spec"]["gang"]["rank"], "machine": machine["name"], "addr": peer.get("addr"), "job": r["id"]})
    port = 29500 + int(gang["id"].lstrip("g")[:6] or "0", 16) % 400
    out = {"ready": True, "rank": gang["rank"], "world_size": gang["size"], "master_addr": peers[0]["addr"],
           "master_port": port, "peers": peers, "gang": gang["id"]}
    if job["spec"].get("network") == "island":  # every port the gang uses is in its block (pool/island.py)
        lo, hi = island.port_block(gang["id"])
        out.update({"master_port": lo, "ports": [lo, hi]})
    return out


def bandwidth(db: DB, machines: list[dict]):
    """bytes/s from machine a to b: the measured peer link, else the slower of a's upload and b's download."""
    ids = [m["id"] for m in machines]
    measured = {}
    if ids:
        marks = ",".join("?" * len(ids))
        for row in db.all(f"select src, dst, mbps from peer_links where src in ({marks}) and dst in ({marks})", *ids, *ids):
            measured[(row["src"], row["dst"])] = float(row["mbps"])
    network = {m["id"]: (m.get("profile") or {}).get("network") or {} for m in machines}

    def speed(a: str, b: str) -> float:
        found = [measured[k] for k in ((a, b), (b, a)) if k in measured]
        if found:
            return max(found) * 1e6 / 8
        up = network.get(a, {}).get("up_mbps") or network.get(a, {}).get("down_mbps") or GUESS_MBPS
        down = network.get(b, {}).get("down_mbps") or GUESS_MBPS
        return min(float(up), float(down)) * 1e6 / 8
    return speed


def gang_plan(db: DB, job: dict, settings: dict, names: dict | None = None) -> dict | None:
    """The max-flow plan of a gang (pool/planner.py): {"ranks": [machine id per rank], "throughput",
    "bottleneck"}, drawn over the online machines that could run a rank, and kept while ranks run."""
    gang = gang_of(job)
    with db.lock:
        row = db.one("select plan, planned_at from gang_plans where gang=?", gang["id"])
        ranks = [job] + siblings(db, job)
        running = any(r["status"] == "running" for r in ranks)
        if row and (running or now() - row["planned_at"] < PLAN_SECONDS):
            saved = json.loads(row["plan"])
            online = {r["id"] for r in db.all("select id from machines where pool=? and drain=0 and last_seen > ?",
                                               job["pool"], now() - ONLINE_SECONDS)}
            if running or all(m in online for m in saved["ranks"]):
                return saved
        names = names if names is not None else {r["id"]: r["name"] for r in db.all("select id, name from members")}
        candidates = []
        for m in [load_machine(r) for r in db.all("select * from machines where pool=? and drain=0 and last_seen > ?",
                                                    job["pool"], now() - ONLINE_SECONDS)]:
            if fits(job["spec"], m, capacity(m), job["owner"], names.get(job["owner"])) is None:
                continue
            if not evaluate(db, job, m, settings["alpha"])["ok"]:
                continue
            candidates.append(m)
        cost = job["spec"].get("cost") or {}
        dtype = cost.get("dtype", "bf16")
        gpus = max(job["spec"]["resources"]["gpus"], 1)
        rate = {m["id"]: tflops(m, dtype) * gpus * 1e12 for m in candidates}
        stages = [{"flops": float(cost.get("flops") or 0), "bytes": float(cost.get("link_bytes") or 0)}] * gang["size"]
        if not any(s["flops"] or s["bytes"] for s in stages):
            stages = [{"flops": 1.0, "bytes": 1.0}] * gang["size"]  # no hint: weigh compute and links alike
        found = planner.plan(stages, [m["id"] for m in candidates], rate, bandwidth(db, candidates))
        if found is None:
            return None
        saved = {"ranks": [group[0] for group in found["stages"]], "throughput": found["throughput"],
                 "bottleneck": found["bottleneck"], "searched": found["searched"]}
        db.run("delete from gang_plans where gang=?", gang["id"])
        db.run("insert into gang_plans (gang, pool, plan, planned_at) values (?, ?, ?, ?)", gang["id"], job["pool"],
               json.dumps(saved), now())
        return saved


def extend(db: DB, machine_id: str, job_ids: list[str]) -> None:
    for job_id in job_ids:
        db.run("update jobs set lease_expires=? where id=? and machine=? and status='running'",
               now() + LEASE_SECONDS, job_id, machine_id)


def append_log(db: DB, job_id: str, text: str, key: str | None = None, cap: int = 10 * 1024 * 1024) -> bool:
    """Store a chunk of output, in arrival order. `key` (attempt and chunk number) makes a chunk the
    agent resent after a timeout a no-op. Output past `cap` bytes is dropped."""
    with db.lock:
        job = db.one("select log_bytes from jobs where id=?", job_id)
        if job is None or job["log_bytes"] >= cap:
            return False
        if job["log_bytes"] + len(text) > cap:
            text = text[: cap - job["log_bytes"]] + "\n[pool] output truncated: more than 10 MB\n"
        inserted = db.run("insert into logs (job, key, text) values (?, ?, ?) on conflict do nothing", job_id, key, text)
        if inserted:
            db.run("update jobs set log_bytes=log_bytes+? where id=?", len(text), job_id)
        return bool(inserted)
