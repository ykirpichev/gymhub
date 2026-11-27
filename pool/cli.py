"""pool: run containers on the pool's machines.

    pool server [--data DIR] [--db URL] [--port 8765]   start a server (prints the first admin code)
    pool login <server> --code CODE [--name NAME]       join with an invite code
    pool join --code CODE                               join another pool with an invite code
    pool pools | pool use NAME                          your pools; the default for other commands
    pool pool create NAME | show [NAME] | set NAME key=value ...
                                                        settings: alpha, affinity_wait_s, retention_days, placement
                                                        (greedy | maxflow: plan gangs as pipelines),
                                                        quota.max_running_gpus, quota.max_running_jobs,
                                                        quota.gpu_hours_per_day, quota.storage_gb (none = no limit)
    pool invite [--pool P] [--role admin|member|viewer] (pool admins) a code for someone new
    pool member set NAME [--pool P] [--role R] [quota.key=value ...]
    pool machine add [--name NAME] | ls | drain NAME [--off] | rm NAME
    pool machine cloud NAME --gpus A100:1 [--cloud gcp] [--spot] [--idle-minutes 30] [--write F | --launch | --down]
                                                        a cloud VM through SkyPilot that joins the pool (dry run by default)
    pool machine policy NAME [--who pool|owner|ann,bob] [--hours 22:00-08:00] [--idle-only] [--no-owner-first]
    pool model add name@tag DIR | ls | rm name@tag      ModelRefs shared with the pool
    pool agent register|run|bench                       the agent itself (the installer runs these)
    pool agent status|pause|resume|uninstall            on the machine (also as `pool-agent ...` after the installer)

    pool run [options] -- python train.py               upload this folder, run, stream logs, fetch /outputs
    pool submit job.yaml [-f]                           the same from a spec file
    pool ps [--all]   pool show JOB   pool logs JOB [-f]   pool get JOB [--to DIR]   pool cancel JOB

`pool run` options: --pool, --image, --gpus N, --gpu-mem GB, --cpus N, --mem GB, --code DIR (default .),
--no-code, --data [NAME=]DIR (repeatable, at /inputs/NAME), --model NAME=model@tag (repeatable, at
/inputs/NAME), --fit NAME[:full|offload|stage:k/n[:kv_gb]], --flops N, --dtype bf16, --bytes-out N,
--bulk, --priority N, --env K=V, --label K=V, --timeout MIN, --retries N, --name, -d, --to DIR.
Inside the job: the code is the working directory /work, inputs are under /inputs, and whatever
is written to /outputs ($POOL_OUTPUTS) comes back.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from pool.artifacts import ArtifactError
from pool.client import Client, PoolError, pairs, save_config

DEFAULT_IMAGE = os.environ.get("POOL_IMAGE", "python:3.12-slim")
CODE_LIMIT_MB = 200


def _client(args) -> Client:
    return Client.member(getattr(args, "server", None), getattr(args, "token", None))


def _say(text: str) -> None:
    print(text, file=sys.stderr)


def _upload_dir(client: Client, folder: Path, kind: str, pool: str | None, limit_mb: int | None = None) -> str:
    folder = Path(folder)
    reply = client.upload_folder(folder, pool=pool, kind=kind, follow_links=True,
                                 max_bytes=limit_mb * 1024**2 if limit_mb else None,
                                 say=lambda text: _say(f"{folder}: {text}"))
    if not reply["sent_bytes"]:
        _say(f"{folder}: already uploaded ({reply['size'] / 1024**2:.1f} MB)")
    return reply["artifact"]


def _ago(timestamp: float | None) -> str:
    if not timestamp:
        return "-"
    seconds = time.time() - timestamp
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size:
            return f"{int(seconds // size)}{unit} ago"
    return f"{int(seconds)}s ago"


def _duration(job: dict) -> str:
    if not job.get("started_at"):
        return "-"
    seconds = (job.get("finished_at") or time.time()) - job["started_at"]
    return f"{seconds / 60:.1f}m" if seconds >= 60 else f"{seconds:.0f}s"


def _value(text: str):
    """A command-line setting value: none -> None, numbers -> numbers, else the string."""
    if text.lower() in ("none", "null", ""):
        return None
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def _settings(items: list[str]) -> dict:
    out: dict = {}
    for key, value in pairs(items, "settings").items():
        if key.startswith("quota."):
            out.setdefault("quota", {})[key[6:]] = _value(value)
        else:
            out[key] = _value(value)
    return out


# ---- account, pools and members ----------------------------------------------------------------


def cmd_login(args) -> int:
    client = Client(args.server_url)
    reply = client.call("POST", "/v1/login", {"code": args.code, "name": args.name or os.environ.get("USER", "")})
    path = save_config("member.json", {"server": client.server, "token": reply["token"]})
    member = reply["member"]
    print(f"Logged in to {client.server} as {member['name']} in pool {reply.get('pool')}"
          + (" (server admin)" if member.get("is_admin") else "") + f". Token stored in {path}.")
    print("Next: pool machine add   (prints a one-line installer for a machine)")
    return 0


def cmd_join(args) -> int:
    reply = _client(args).call("POST", "/v1/join", {"code": args.code})
    print(f"Joined pool {reply['pool']}. Make it your default with: pool use {reply['pool']}")
    return 0


def cmd_use(args) -> int:
    reply = _client(args).call("POST", "/v1/use", {"pool": args.pool})
    print(f"Default pool: {reply['pool']}")
    return 0


def cmd_pools(args) -> int:
    pools = _client(args).call("GET", "/v1/pools")["pools"]
    print(f"{'POOL':<20} {'ROLE':<8} {'MEMBERS':>7} {'MACHINES':>8}")
    for item in pools:
        print(f"{item['name'] + (' *' if item['default'] else ''):<20} {item['role']:<8} {item['members']:>7} {item['machines']:>8}")
    return 0


def cmd_pool_create(args) -> int:
    reply = _client(args).call("POST", "/v1/pools", {"name": args.name})
    print(f"Created pool {reply['pool']['name']}; you are its admin. Invite people with: pool invite --pool {args.name}")
    return 0


def cmd_pool_show(args) -> int:
    client = _client(args)
    name = args.name or client.call("GET", "/v1/me")["member"].get("default_pool") or ""
    pool = client.call("GET", f"/v1/pools/{name}")["pool"]
    settings = pool["settings"]
    print(f"pool {pool['name']} (you: {pool['role']})")
    print(f"  alpha {settings['alpha']}  affinity_wait_s {settings['affinity_wait_s']}  retention_days {settings['retention_days']}"
          f"  placement {settings.get('placement', 'greedy')}")
    print("  default quota: " + ", ".join(f"{k}={v}" for k, v in settings["quota"].items()))
    print(f"  {'MEMBER':<16} {'ROLE':<8} {'GPU-H 24H':>9} {'STORAGE GB':>10}  QUOTA")
    for member in pool["members"]:
        quota = ", ".join(f"{k}={v}" for k, v in member["quota"].items() if v is not None) or "none"
        print(f"  {member['name']:<16} {member['role']:<8} {member['gpu_hours_24h']:>9.2f} {member['storage_gb']:>10.2f}  {quota}")
    return 0


def cmd_pool_set(args) -> int:
    reply = _client(args).call("PATCH", f"/v1/pools/{args.name}", {"settings": _settings(args.settings)})
    print(json.dumps(reply["settings"], indent=2))
    return 0


def cmd_invite(args) -> int:
    client = _client(args)
    reply = client.call("POST", "/v1/invites", {"pool": args.pool, "role": args.role})
    print(f"Send this to the new {args.role} of {reply['pool']} (it works once, for 7 days):\n"
          f"  pool login {client.server} --code {reply['code']} --name <their name>\n"
          f"Someone already in another pool joins with: pool join --code {reply['code']}")
    return 0


def cmd_member_set(args) -> int:
    body: dict = {}
    if args.role:
        body["role"] = args.role
    if args.quota:
        settings = _settings(args.quota)
        if set(settings) - {"quota"}:
            raise SystemExit("member settings are quota.<key>=<value>")
        body["quota"] = settings.get("quota", {})
    client = _client(args)
    pool = args.pool or client.call("GET", "/v1/me")["member"].get("default_pool") or ""
    client.call("PUT", f"/v1/pools/{pool}/members/{args.member}", body)
    print(f"{args.member} in {pool}: updated")
    return 0


def cmd_whoami(args) -> int:
    client = _client(args)
    me = client.call("GET", "/v1/me")
    who = me["member"]
    pools = ", ".join(f"{p['name']} ({p['role']})" for p in me["pools"])
    print(f"{who['name']} on {client.server}; pools: {pools}; default {who.get('default_pool')}")
    return 0


# ---- machines ----------------------------------------------------------------------------------


def cmd_machine_add(args) -> int:
    client = _client(args)
    reply = client.call("POST", "/v1/machine-codes", {"pool": args.pool, "name": args.name or ""})
    print(f"Adds a machine to pool {reply['pool']}. On the new machine (Ubuntu 22.04/24.04, Debian 12, DGX Spark or Jetson), run:\n")
    print(f"  {reply['install']}\n")
    if reply.get("sha256"):
        print(f"It installs Docker and the NVIDIA container toolkit if missing, and an agent package with sha256\n  {reply['sha256']}\n"
              "(add --sha256 <that> to the command to insist on it). Without sudo (a user service, no Docker install):\n")
        print(f"  {reply.get('install_no_sudo', '')}\n")
    print("Or, from a checkout of this repository:\n")
    print(f"  python -m pool.cli agent register --server {client.server} --code {reply['code']}"
          + (f" --name {args.name}" if args.name else "") + "\n  python -m pool.cli agent run\n")
    print("The code works once, for 7 days. Jobs run in containers with no network access.")
    return 0


def cmd_machine_ls(args) -> int:
    reply = _client(args).call("GET", "/v1/machines" + (f"?pool={args.pool}" if args.pool else ""))
    machines = reply["machines"]
    if not machines:
        print(f"No machines in {reply['pool']} yet: pool machine add")
        return 0
    print(f"pool {reply['pool']}")
    print(f"{'NAME':<18} {'OWNER':<10} {'STATE':<9} {'GPUS':<16} {'BF16':>6} {'CPUS':>4} {'MEM':>6} {'LINK':>9} {'CACHED':>8} {'RUN':>4}  AGENT")
    for m in machines:
        p = m["profile"] or {}
        gpus = ", ".join(sorted({g["model"].replace("NVIDIA ", "") for g in p.get("gpus", [])})) or "-"
        count = len(p.get("gpus", []))
        state = "excluded" if (m.get("reputation") or {}).get("excluded") else "draining" if m["drain"] else "online" if m["online"] else "offline"
        bf16 = ((p.get("bench") or {}).get("tflops") or {}).get("bf16")
        link = (p.get("network") or {}).get("down_mbps")
        print(f"{m['name']:<18} {m['owner']:<10} {state:<9} {(f'{count}x ' + gpus) if count else '-':<16} "
              f"{(f'{bf16:g}T' if bf16 else '-'):>6} {p.get('cpus', '-'):>4} {str(p.get('memory_gb', '-')) + 'G':>6} "
              f"{(f'{link:g}M' if link else '-'):>9} {str(m['cached_gb']) + 'G':>8} {len(m['running']):>4}  {_agent(m.get('agent'))}")
        error = (m.get("agent") or {}).get("update_error")
        if error:
            print(f"{'':<18} agent self-update failed {_ago(error.get('at'))}: {error.get('message')}")
    return 0


def _agent(info: dict | None) -> str:
    """A machine's agent in one short column: the first 12 characters of its digest, then current,
    outdated (it differs from the server's) or unknown (an agent too old to report its digest)."""
    if not info:
        return "-"  # a server from before agents reported their version
    return f"{info.get('short') or '-':<12} {info.get('state', 'unknown')}"


def _ago(at) -> str:
    if not at:
        return ""
    seconds = max(0, time.time() - float(at))
    return f"{seconds / 60:.0f} min ago" if seconds < 3600 else f"{seconds / 3600:.1f} h ago" if seconds < 86400 else f"{seconds / 86400:.0f} d ago"


def cmd_machine_trust(args) -> int:
    """Clear a machine excluded by failed checks (checked pools, pool/trust.py)."""
    _client(args).call("POST", f"/v1/machines/{args.name}/trust" + (f"?pool={args.pool}" if args.pool else ""))
    print(f"{args.name}: trusted again; its checks start over")
    return 0


def cmd_machine_links(args) -> int:
    reply = _client(args).call("GET", "/v1/links" + (f"?pool={args.pool}" if args.pool else ""))
    if not reply["links"]:
        print(f"No links measured in {reply['pool']} yet: agents measure their peers within a minute of starting")
        return 0
    print(f"pool {reply['pool']}: links measured machine to machine (download at FROM, served by TO)")
    print(f"{'FROM':<18} {'TO':<18} {'MBIT/S':>9} {'RTT MS':>7}  MEASURED")
    for link in reply["links"]:
        print(f"{link['src']:<18} {link['dst']:<18} {link['mbps']:>9g} {link['rtt_ms']:>7g}  {_ago(link['measured_at'])}")
    rule = reply["island_rule"]
    print(f"\nislands (>= {rule['mbps'] / 1000:g} Gbit/s and < {rule['rtt_ms']:g} ms between members):")
    for island in reply["islands"]:
        print("  " + " + ".join(island))
    return 0


def cmd_machine_drain(args) -> int:
    _client(args).call("POST", f"/v1/machines/{args.machine}/drain", {"drain": not args.off})
    print(f"{args.machine}: {'accepting jobs again' if args.off else 'takes no new jobs; running ones finish'}")
    return 0


def cmd_machine_cloud(args) -> int:
    """A SkyPilot VM as a machine of the pool (pool/skypilot.py). Without --write, --launch or --down
    this is a dry run: nothing is created, on the server or in a cloud."""
    from pool import skypilot

    try:
        if args.down:
            reply = skypilot.down(args.name, dry_run=not args.yes)
            if reply["dry_run"]:
                print("Would run: " + " ".join(reply["command"]) + "\n(add --yes to tear it down; then pool machine rm "
                      + args.name + ")")
            else:
                print(f"{args.name} is down; remove it from the pool with: pool machine rm {args.name}")
            return 0
        server, code = args.server_url, skypilot.CODE_PLACEHOLDER
        if args.launch or args.write or not server:  # a dry run with --server-url needs no login
            client = _client(args)
            server = server or client.server
            if args.launch or args.write:
                code = client.call("POST", "/v1/machine-codes", {"pool": args.pool, "name": args.name})["code"]
        config = skypilot.task(server, code, args.name, args.gpus, args.cloud, args.region,
                               args.cpus, args.memory, args.spot, args.disk_gb, pairs(args.label, "--label"), args.image_id)
        if args.write:
            skypilot.write(config, args.write)
            print(f"Wrote {args.write} (it holds a one-time machine code: keep it private). Launch it with:\n  "
                  + " ".join(skypilot.command(args.name, args.write, args.idle_minutes, not args.keep)))
            return 0
        reply = skypilot.launch(config, args.idle_minutes, not args.keep, dry_run=not args.launch)
    except skypilot.SkyError as exc:
        print(f"pool machine cloud: {exc}", file=sys.stderr)
        return 2
    if reply["dry_run"]:
        print("Dry run: nothing was created. The SkyPilot task:\n")
        print(json.dumps(reply["task"], indent=2))
        print("\nWith --write FILE it is saved with a real machine code, for: " + " ".join(reply["command"]))
        print("With --launch, pool launches it through SkyPilot's Python API (this rents a VM and costs money).")
    else:
        print(f"Launched {args.name}; it joins the pool once the agent registers (pool machine ls)."
              + (f" It stops after {args.idle_minutes} idle minutes." if args.idle_minutes else ""))
    return 0


def cmd_machine_rm(args) -> int:
    _client(args).call("DELETE", f"/v1/machines/{args.machine}")
    print(f"{args.machine} removed; its running jobs go back to the queue")
    return 0


def cmd_machine_policy(args) -> int:
    who = args.who if args.who in ("pool", "owner") else [name for name in args.who.split(",") if name]
    reply = _client(args).call("PUT", f"/v1/machines/{args.machine}/policy",
                               {"who": who, "hours": args.hours, "idle_only": args.idle_only, "owner_first": not args.no_owner_first})
    print(f"{args.machine}: {json.dumps(reply['policy'])}")
    return 0


# ---- models ------------------------------------------------------------------------------------


def cmd_model_add(args) -> int:
    from pool import models

    folder = Path(args.dir).expanduser()
    meta = models.describe(folder)
    if not meta["weight_bytes"]:
        raise SystemExit(f"{folder} holds no model weights (safetensors, bin, pt, gguf)")
    _say(f"{args.name}: {meta['architecture'] or 'model'}, {meta['params'] / 1e9:.2f}B parameters "
         f"({meta['active_params'] / 1e9:.2f}B active), {meta['weight_bytes'] / 1024**3:.1f} GB")
    client = _client(args)
    artifact = _upload_dir(client, folder, "model", args.pool)
    reply = client.call("POST", "/v1/models", {"name": args.name, "artifact": artifact, "meta": meta, "pool": args.pool})
    print(f"model {reply['model']} in {reply['pool']}: use it with --model base={reply['model']}")
    return 0


def cmd_model_ls(args) -> int:
    reply = _client(args).call("GET", "/v1/models" + (f"?pool={args.pool}" if args.pool else ""))
    if not reply["models"]:
        print(f"No models in {reply['pool']} yet: pool model add name@tag DIR")
        return 0
    print(f"{'MODEL':<32} {'PARAMS':>8} {'ACTIVE':>8} {'GB':>7}  {'ARCHITECTURE':<36} BY")
    for item in reply["models"]:
        meta = item["meta"]
        print(f"{item['model']:<32} {meta.get('params', 0) / 1e9:>7.2f}B {meta.get('active_params', 0) / 1e9:>7.2f}B "
              f"{meta['weight_bytes'] / 1024**3:>7.1f}  {str(meta.get('architecture') or '-'):<36} {item['by']}")
    return 0


def cmd_model_rm(args) -> int:
    _client(args).call("DELETE", f"/v1/models/{args.name}" + (f"?pool={args.pool}" if args.pool else ""))
    print(f"model {args.name} removed")
    return 0


# ---- jobs --------------------------------------------------------------------------------------


def _resources(args) -> dict:
    out = {"gpus": args.gpus, "cpus": args.cpus, "memory_gb": args.mem}
    for key in ("shm_gb", "tmp_gb", "pids"):
        if getattr(args, key, None) is not None:
            out[key] = getattr(args, key)
    if args.gpu_mem:
        out["gpu_memory_gb"] = args.gpu_mem
    if args.label:
        out["labels"] = pairs(args.label, "--label")
    if args.fit:
        target, _, rest = args.fit.partition(":")
        mode, kv = "full", 0.0
        if rest:
            parts = rest.split(":")
            if parts[0] == "stage" and len(parts) > 1:  # stage:k/n[:kv]
                mode, parts = f"stage:{parts[1]}", parts[2:]
            else:
                mode, parts = parts[0], parts[1:]
            if parts:
                kv = float(parts[0])
        out["fits"] = {"model": target, "mode": mode, "kv_gb": kv}
    return out


# ---- refs and queues (M3, pool/coord.py) -------------------------------------------------------

def _when(seconds: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(seconds))


def _coord(args, op: str, **values) -> dict:
    return _client(args).call("POST", "/v1/coord", {"op": op, "pool": args.pool, "args": values})


def cmd_ref_ls(args) -> int:
    refs = _coord(args, "ref.list")["refs"]
    if not refs:
        print("No refs yet: pool ref publish NAME DIR")
        return 0
    print(f"{'REF':<40} {'VERSION':>7}  {'WHEN':<19} ARTIFACT")
    for ref in refs:
        print(f"{ref['name']:<40} {ref['version']:>7}  {_when(ref['created_at']):<19} {ref['artifact']}")
    return 0


def cmd_ref_get(args) -> int:
    print(json.dumps(_coord(args, "ref.get", name=args.name, version=args.version)["ref"], indent=2))
    return 0


def cmd_ref_log(args) -> int:
    for ref in _coord(args, "ref.get", name=args.name, history=True, limit=args.limit)["versions"]:
        print(f"v{ref['version']:<5} parent v{ref['parent']:<5} {_when(ref['created_at']):<19} {ref['artifact']}")
    return 0


def cmd_ref_publish(args) -> int:
    client = _client(args)
    artifact = _upload_dir(client, Path(args.dir).expanduser(), "data", args.pool)
    parent = args.parent
    if parent is None and not args.force:
        try:  # publish on top of what is there now (compare-and-swap with the version just read)
            parent = _coord(args, "ref.get", name=args.name)["ref"]["version"]
        except PoolError as exc:
            if exc.status != 404:
                raise
            parent = 0
    try:
        ref = _coord(args, "ref.publish", name=args.name, artifact=artifact, parent=parent, force=args.force)["ref"]
    except PoolError as exc:
        if exc.status == 409:
            raise SystemExit(f"{exc.message} (or pass --force)") from None
        raise
    print(f"{ref['name']} v{ref['version']}: {ref['artifact']}")
    return 0


def cmd_ref_fetch(args) -> int:
    ref = _coord(args, "ref.get", name=args.name, version=args.version)["ref"]
    count = _client(args).download_folder(ref["artifact"], Path(args.dest).expanduser())
    print(f"{ref['name']} v{ref['version']}: {count} files in {args.dest}")
    return 0


def cmd_queue_create(args) -> int:
    queue = _coord(args, "queue.create", name=args.name, ref=args.ref, max_lag=args.max_lag, stale=args.stale,
                   duplicate_after=args.duplicate_after, replicate=args.replicate)["queue"]
    print(f"queue {queue['name']}" + (f": results more than {queue['max_lag']} versions behind {queue['ref']} are "
                                      f"{'dropped' if args.stale == 'drop' else 'requeued'}" if queue["ref"] else ""))
    return 0


def _queue_line(queue: dict) -> str:
    c = queue["counts"]
    p90 = f"{queue['p90_seconds']:.1f}s" if queue["p90_seconds"] is not None else "-"
    return (f"{queue['name']:<24} {c['queued']:>7} {c['running']:>7} {c['done']:>7} {c['dropped']:>7} "
            f"{queue['stale_results']:>6} {queue['duplicates']:>5} {p90:>7}")


def cmd_queue_ls(args) -> int:
    queues = _coord(args, "queue.list")["queues"]
    print(f"{'QUEUE':<24} {'QUEUED':>7} {'RUNNING':>7} {'DONE':>7} {'DROPPED':>7} {'STALE':>6} {'DUPS':>5} {'P90':>7}")
    for queue in queues:
        print(_queue_line(queue))
    return 0


def cmd_queue_stats(args) -> int:
    print(json.dumps(_coord(args, "queue.stats", name=args.name)["queue"], indent=2))
    return 0


def cmd_queue_add(args) -> int:
    """Tasks as JSON lines (one {inputs, key?} per line, or bare inputs) from a file or stdin."""
    source = sys.stdin if args.file == "-" else open(args.file)
    tasks = []
    for line in source:
        if line.strip():
            item = json.loads(line)
            tasks.append(item if isinstance(item, dict) and "inputs" in item else {"inputs": item})
    added = 0
    for start in range(0, len(tasks), 1000):
        added += len(_coord(args, "tasks.add", name=args.name, tasks=tasks[start:start + 1000])["ids"])
    print(f"{added} tasks in {args.name}")
    return 0


def cmd_queue_results(args) -> int:
    after = args.after
    while True:
        reply = _coord(args, "results", name=args.name, after=after, wait=25 if args.follow else 0)
        for item in reply["results"]:
            print(json.dumps({"key": item["key"], "result": item["result"], "computed_with": item["computed_with"]}))
        after = reply["next"]
        if not args.follow:
            return 0


def cmd_round_open(args) -> int:
    r = _coord(args, "round.open", ref=args.ref, quorum=args.quorum, min=args.min, deadline_s=args.deadline,
               grace_s=args.grace, late=args.late, decay=args.decay, repeat=args.repeat, mode=args.mode,
               fanout=args.fanout, bias=args.bias, reducer=args.reducer, lighthouse=args.lighthouse,
               reduce=args.reduce, handoff_s=args.handoff)["round"]
    kind = f"gossip round (fanout {args.fanout})" if args.mode == "gossip" else "round"
    print(f"{kind} {r['id']} on {r['ref']} v{r['base_version']}: ready at {r['quorum']} contributions "
          f"(+{r['grace']:g} s), or at the deadline with {r['min']}" + ("; repeats" if args.repeat else "")
          + (f"; reduced by torchft, lighthouse {r['settings']['lighthouse']}" if r["settings"].get("reducer") == "torchft" else "")
          + ("; reduced island first (tree)" if r["settings"].get("reduce") == "tree" else ""))
    return 0


def cmd_round_status(args) -> int:
    print(json.dumps(_coord(args, "round.current", ref=args.ref)["round"], indent=2))
    return 0


def cmd_run(args) -> int:
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        print("Nothing to run: pool run [options] -- python train.py", file=sys.stderr)
        return 2
    client = _client(args)
    name = args.name or Path(command[-1]).stem
    spec: dict = {
        "name": "".join(c if c.isalnum() or c in "._-" else "-" for c in name).strip("-._")[:64] or "job",
        "image": args.image, "command": command if len(command) > 1 else ["sh", "-c", command[0]],
        "resources": _resources(args), "env": pairs(args.env, "--env"), "priority": args.priority,
        "timeout_minutes": args.timeout, "retries": args.retries, "inputs": [],
    }
    if args.flops or args.bytes_out or args.bulk or args.link_bytes:
        spec["cost"] = {"flops": args.flops or 0, "dtype": args.dtype, "bytes_out": args.bytes_out or 0, "bulk": args.bulk}
        if args.link_bytes:
            spec["cost"]["link_bytes"] = args.link_bytes
    if args.gang:
        spec["gang"] = {"size": args.gang}
    if args.network:
        spec["network"] = args.network
    if args.image_for:
        images = {}
        for item in args.image_for:
            match, sep, image = item.rpartition("=") if item.count("=") >= 2 else ("", "", "")
            if not sep or not match or not image:
                raise SystemExit(f"--image-for {item!r}: LABEL=VALUE=IMAGE, for example gpu=tegra=dustynv/l4t-pytorch:r36.4.0")
            images[match] = image
        spec["images"] = images
    if not args.no_code:
        try:
            spec["code"] = _upload_dir(client, Path(args.code), "code", args.pool, CODE_LIMIT_MB)
        except ArtifactError as exc:
            print(f"{exc}. Point --code at the folder with your script, or use --no-code and --data for big files.", file=sys.stderr)
            return 2
    for item in args.data or []:
        name, sep, folder = item.partition("=")
        if not sep:
            name, folder = "", item
        folder_path = Path(folder)
        spec["inputs"].append({"path": name or folder_path.resolve().name, "artifact": _upload_dir(client, folder_path, "data", args.pool)})
    for path, ref in pairs(args.model, "--model").items():
        spec["inputs"].append({"path": path, "model": ref})
    return _after_submit(client, client.call("POST", "/v1/jobs", {"pool": args.pool, "spec": spec}), args)


def _load_spec(path: Path) -> dict:
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise SystemExit("Reading YAML needs PyYAML (pip install pyyaml); or write the spec as JSON") from None
        return yaml.safe_load(text)
    return json.loads(text)


def _spec_with_uploads(client: Client, path: Path, pool_arg: str | None) -> tuple[dict, str | None]:
    """A spec file (YAML or JSON) with its local code and input folders uploaded."""
    spec = _load_spec(path)
    pool = pool_arg or spec.pop("pool", None)
    base = path.resolve().parent
    code = spec.get("code")
    if isinstance(code, str) and not code.startswith("sha256:"):  # a local folder
        spec["code"] = _upload_dir(client, (base / code).resolve(), "code", pool, CODE_LIMIT_MB)
    inputs = []
    for item in spec.get("inputs") or []:
        if isinstance(item, dict) and "dir" in item:
            inputs.append({"path": item.get("path") or Path(item["dir"]).name,
                           "artifact": _upload_dir(client, (base / item["dir"]).resolve(), "data", pool)})
        else:
            inputs.append(item)
    spec["inputs"] = inputs
    return spec, pool


def cmd_submit(args) -> int:
    client = _client(args)
    spec, pool = _spec_with_uploads(client, Path(args.spec), args.pool)
    args.detach = not args.follow
    return _after_submit(client, client.call("POST", "/v1/jobs", {"pool": pool, "spec": spec}), args)


def _actor_line(actor: dict) -> str:
    queue = f"{actor['queue']} (1 per {actor['per_replica']})" if actor["queue"] else "-"
    return (f"{actor['name']:<24} {actor['status']:<9} {actor['running']:>3}/{actor['desired']:<3} "
            f"{actor['min']:>3}-{actor['max']:<4} {queue:<28} {actor['image']}")


def cmd_actor_create(args) -> int:
    client = _client(args)
    spec, pool = _spec_with_uploads(client, Path(args.spec), args.pool)
    actor = client.call("POST", "/v1/coord", {"op": "actor.create", "pool": pool, "args": {
        "name": args.name, "spec": spec, "min": args.min, "max": args.max, "queue": args.queue,
        "per_replica": args.per_replica}})["actor"]
    print(f"actor {actor['name']}: {actor['desired']} replicas wanted, {len(actor['replicas'])} queued or running")
    return 0


def cmd_actor_ls(args) -> int:
    print(f"{'ACTOR':<24} {'STATUS':<9} {'RUN':>7} {'MIN-MAX':>8} {'QUEUE':<28} IMAGE")
    for actor in _coord(args, "actor.list")["actors"]:
        print(_actor_line(actor))
    return 0


def cmd_actor_status(args) -> int:
    print(json.dumps(_coord(args, "actor.status", name=args.name)["actor"], indent=2))
    return 0


def cmd_actor_scale(args) -> int:
    actor = _coord(args, "actor.scale", name=args.name, min=args.min, max=args.max)["actor"]
    print(_actor_line(actor))
    return 0


def cmd_actor_stop(args) -> int:
    actor = _coord(args, "actor.stop", name=args.name)["actor"]
    print(f"actor {actor['name']} stopped; its replicas are cancelled")
    return 0


def _after_submit(client: Client, reply: dict, args) -> int:
    job = reply["job"]
    gang = job.get("gang")
    if gang:
        _say(f"gang {gang['id']} of {gang['size']} queued in {job['pool']}: ranks " + " ".join(gang["jobs"]))
    else:
        _say(f"job {job['id']} queued in {job['pool']} ({job['name']})")
    if reply.get("warning"):
        _say(f"warning: {reply['warning']}")
    if args.detach:
        print(job["id"])
        return 0
    if gang:
        _say("following rank 0; pool logs -f <job> for the others")
    final = _follow(client, job["id"])
    if final is None:
        return 130
    to = getattr(args, "to", None)
    if not gang:
        return _finish(client, final, to, getattr(args, "no_get", False))
    code = 0
    for job_id in gang["jobs"]:  # each rank's outputs in <to>/<job> (default pool-outputs/<job>)
        done = final if job_id == final["id"] else _wait_final(client, job_id)
        code = _finish(client, done, str(Path(to) / job_id) if to else None, getattr(args, "no_get", False)) or code
    return code


def _wait_final(client: Client, job_id: str) -> dict:
    while True:
        job = client.call("GET", f"/v1/jobs/{job_id}")["job"]
        if job["status"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(2)


def _follow(client: Client, job_id: str) -> dict | None:
    """Print the job's output as it arrives; return the finished job (None on Ctrl+C)."""
    after = 0
    try:
        while True:
            try:
                reply = client.call("GET", f"/v1/jobs/{job_id}/logs?after={after}&wait=20", timeout=60)
            except PoolError as exc:
                if exc.status:
                    raise
                _say(f"[pool] {exc}; retrying")
                time.sleep(3)
                continue
            for chunk in reply["chunks"]:
                sys.stdout.write(chunk["text"])
            sys.stdout.flush()
            after = reply["next"]
            if reply["status"] in ("succeeded", "failed", "cancelled") and not reply["chunks"]:
                return client.call("GET", f"/v1/jobs/{job_id}")["job"]
    except KeyboardInterrupt:
        _say(f"\nStopped following; the job keeps running. pool logs -f {job_id}   pool cancel {job_id}")
        return None


def _finish(client: Client, job: dict, to: str | None, no_get: bool) -> int:
    where = f" on {job['machine']}" if job.get("machine") else ""
    line = f"job {job['id']} {job['status']}{where} in {_duration(job)}"
    if job.get("error") and job["status"] != "succeeded":
        line += f": {job['error']}"
    _say(line)
    if job.get("outputs") and not no_get:
        _get(client, job, Path(to or Path("pool-outputs") / job["id"]))
    return 0 if job["status"] == "succeeded" else (job.get("exit_code") or 1)


def _get(client: Client, job: dict, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    count = client.download_folder(job["outputs"], dest)
    _say(f"outputs ({count} files) in {dest}")


def cmd_ps(args) -> int:
    query = f"?all={'true' if args.all else 'false'}&limit={args.n}" + (f"&pool={args.pool}" if args.pool else "")
    jobs = _client(args).call("GET", f"/v1/jobs{query}")["jobs"]
    if not jobs:
        print("No jobs yet: pool run -- python train.py")
        return 0
    print(f"{'ID':<11} {'NAME':<20} {'POOL':<10} {'STATUS':<10} {'MACHINE':<16} {'GPUS':<6} {'TIME':>6}  {'CREATED':<9}"
          + ("  OWNER" if args.all else ""))
    for job in jobs:
        gpus = ",".join(str(g) for g in job["gpus"]) or "-"
        print(f"{job['id']:<11} {job['name'][:20]:<20} {job['pool'][:10]:<10} {job['status']:<10} {(job['machine'] or '-')[:16]:<16} "
              f"{gpus:<6} {_duration(job):>6}  {_ago(job['created_at']):<9}" + (f"  {job['owner']}" if args.all else ""))
    return 0


def cmd_show(args) -> int:
    reply = _client(args).call("GET", f"/v1/jobs/{args.job}")
    job = reply["job"]
    print(f"job {job['id']} ({job['name']}) in {job['pool']}, by {job['owner']}: {job['status']}"
          + (f" on {job['machine']}, GPUs {job['gpus'] or 'none'}" if job.get("machine") else ""))
    print(f"  tries {job['tries']}, preemptions {job['preemptions']}, priority {job['priority']}, time {_duration(job)}"
          + (f", error: {job['error']}" if job.get("error") else ""))
    placement = job.get("placement")
    if placement:
        print(f"  placed on {placement.get('machine')}: score {placement.get('score')}, inputs {placement.get('input_bytes', 0) / 1024**3:.2f} GB "
              f"of which {placement.get('missing_bytes', 0) / 1024**3:.2f} GB to fetch, t_net {placement.get('t_net')} s, "
              f"t_compute {placement.get('t_compute')} s, link {placement.get('down_mbps')} Mbit/s")
    for line in reply.get("why") or []:
        print(f"  {line}")
    for agent in reply.get("agents") or []:
        print(f"  agent on {agent['machine']}: {_agent(agent)}"
              + (f" (self-update failed: {agent['update_error'].get('message')})" if agent.get("update_error") else ""))
    if job.get("outputs"):
        print(f"  outputs {job['outputs']}")
    return 0


def cmd_logs(args) -> int:
    client = _client(args)
    if args.follow:
        final = _follow(client, args.job)
        if final:
            _finish(client, final, None, no_get=True)
        return 0
    after = 0
    while True:
        reply = client.call("GET", f"/v1/jobs/{args.job}/logs?after={after}")
        for chunk in reply["chunks"]:
            sys.stdout.write(chunk["text"])
        if not reply["chunks"]:
            return 0
        after = reply["next"]


def cmd_get(args) -> int:
    client = _client(args)
    job = client.call("GET", f"/v1/jobs/{args.job}")["job"]
    if not job.get("outputs"):
        _say(f"job {job['id']} is {job['status']} and has no outputs" + (" yet" if job["status"] in ("queued", "running") else ""))
        return 1
    _get(client, job, Path(args.to or Path("pool-outputs") / job["id"]))
    return 0


def cmd_cancel(args) -> int:
    job = _client(args).call("POST", f"/v1/jobs/{args.job}/cancel")["job"]
    print(f"job {job['id']} {job['status']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pool", description="Run containers on the pool's machines.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = parser.add_subparsers(dest="command_name", required=True)
    sub.add_parser("server", help="start a pool server (pool server --help)")  # handled in main()

    login = sub.add_parser("login", help="join with an invite code")
    login.add_argument("server_url")
    login.add_argument("--code", required=True)
    login.add_argument("--name")
    login.set_defaults(func=cmd_login)
    join = sub.add_parser("join", help="join another pool with an invite code")
    join.add_argument("--code", required=True)
    join.set_defaults(func=cmd_join)
    use = sub.add_parser("use", help="your default pool")
    use.add_argument("pool")
    use.set_defaults(func=cmd_use)
    sub.add_parser("pools", help="your pools").set_defaults(func=cmd_pools)
    sub.add_parser("whoami").set_defaults(func=cmd_whoami)

    pool_cmd = sub.add_parser("pool", help="create, show and configure pools").add_subparsers(dest="pool_command", required=True)
    create = pool_cmd.add_parser("create", help="(server admin) a new pool")
    create.add_argument("name")
    create.set_defaults(func=cmd_pool_create)
    show_pool = pool_cmd.add_parser("show", help="settings, members, usage")
    show_pool.add_argument("name", nargs="?")
    show_pool.set_defaults(func=cmd_pool_show)
    set_pool = pool_cmd.add_parser("set", help="(pool admin) key=value settings")
    set_pool.add_argument("name")
    set_pool.add_argument("settings", nargs="+")
    set_pool.set_defaults(func=cmd_pool_set)

    invite = sub.add_parser("invite", help="(pool admins) an invite code for someone new")
    invite.add_argument("--pool")
    invite.add_argument("--role", default="member", choices=["admin", "member", "viewer"])
    invite.set_defaults(func=cmd_invite)

    member = sub.add_parser("member", help="(pool admins) roles and quotas").add_subparsers(dest="member_command", required=True)
    member_set = member.add_parser("set")
    member_set.add_argument("member")
    member_set.add_argument("--pool")
    member_set.add_argument("--role", choices=["admin", "member", "viewer"])
    member_set.add_argument("quota", nargs="*", help="quota.<key>=<value>")
    member_set.set_defaults(func=cmd_member_set)

    machine = sub.add_parser("machine", help="machines in the pool").add_subparsers(dest="machine_command", required=True)
    add = machine.add_parser("add", help="a one-line installer for a new machine")
    add.add_argument("--name")
    add.add_argument("--pool")
    add.set_defaults(func=cmd_machine_add)
    ls = machine.add_parser("ls")
    ls.add_argument("--pool")
    ls.set_defaults(func=cmd_machine_ls)
    links = machine.add_parser("links", help="measured machine-to-machine links, and the islands they form")
    links.add_argument("--pool")
    links.set_defaults(func=cmd_machine_links)
    drain = machine.add_parser("drain", help="stop giving a machine new jobs")
    drain.add_argument("machine")
    drain.add_argument("--off", action="store_true", help="accept jobs again")
    drain.set_defaults(func=cmd_machine_drain)
    trust_cmd = machine.add_parser("trust", help="(pool admins) let a machine excluded by failed checks work again")
    trust_cmd.add_argument("name")
    trust_cmd.add_argument("--pool")
    trust_cmd.set_defaults(func=cmd_machine_trust)
    cloud = machine.add_parser("cloud", help="a cloud VM through SkyPilot that joins the pool (dry run unless --launch)")
    cloud.add_argument("name", help="the VM's and the machine's name: lowercase letters, digits, dashes")
    cloud.add_argument("--pool")
    cloud.add_argument("--server-url", help="the server address the VM uses (default: yours)")
    cloud.add_argument("--gpus", help="SkyPilot accelerators, e.g. A100:1, L4:1, H100:8")
    cloud.add_argument("--cloud", help="aws, gcp, azure, lambda, runpod, ... (default: the cheapest SkyPilot finds)")
    cloud.add_argument("--region")
    cloud.add_argument("--cpus", help="e.g. 8 or 8+")
    cloud.add_argument("--memory", help="GB, e.g. 32+")
    cloud.add_argument("--disk-gb", type=int)
    cloud.add_argument("--image-id", help="the VM image (it needs Python 3.10+, Docker and, for GPUs, the NVIDIA toolkit)")
    cloud.add_argument("--spot", action="store_true", help="a spot or preemptible VM")
    cloud.add_argument("--label", action="append", default=[], help="k=v, a machine label for jobs to match")
    cloud.add_argument("--idle-minutes", type=int, default=30, help="stop the VM after this long idle (0: never)")
    cloud.add_argument("--keep", action="store_true", help="when idle, stop the VM instead of deleting it")
    mode = cloud.add_mutually_exclusive_group()
    mode.add_argument("--write", metavar="FILE", help="save the task with a real machine code, for `sky launch`")
    mode.add_argument("--launch", action="store_true", help="launch it now through SkyPilot (rents a VM: costs money)")
    mode.add_argument("--down", action="store_true", help="tear the VM down (with --yes)")
    cloud.add_argument("--yes", action="store_true", help="with --down: really tear it down")
    cloud.set_defaults(func=cmd_machine_cloud)
    rm = machine.add_parser("rm", help="remove a machine")
    rm.add_argument("machine")
    rm.set_defaults(func=cmd_machine_rm)
    policy = machine.add_parser("policy", help="who may use a machine, and when")
    policy.add_argument("machine")
    policy.add_argument("--who", default="pool", help="pool (everyone), owner, or member names: ann,bob")
    policy.add_argument("--hours", help="HH:MM-HH:MM in the machine's local time")
    policy.add_argument("--idle-only", action="store_true", help="only while nobody else uses its GPUs")
    policy.add_argument("--no-owner-first", action="store_true", help="the owner's jobs do not go first")
    policy.set_defaults(func=cmd_machine_policy)

    model = sub.add_parser("model", help="ModelRefs shared with a pool").add_subparsers(dest="model_command", required=True)
    model_add = model.add_parser("add", help="upload a model folder (for example a Hugging Face snapshot)")
    model_add.add_argument("name", help="name@tag")
    model_add.add_argument("dir")
    model_add.add_argument("--pool")
    model_add.set_defaults(func=cmd_model_add)
    model_ls = model.add_parser("ls")
    model_ls.add_argument("--pool")
    model_ls.set_defaults(func=cmd_model_ls)
    model_rm = model.add_parser("rm")
    model_rm.add_argument("name")
    model_rm.add_argument("--pool")
    model_rm.set_defaults(func=cmd_model_rm)

    ref = sub.add_parser("ref", help="named, versioned pointers to artifacts (M3)").add_subparsers(dest="ref_command", required=True)
    for name, func, text in (("ls", cmd_ref_ls, "refs in the pool"), ("get", cmd_ref_get, "one version (default: current)"),
                             ("log", cmd_ref_log, "versions, newest first"), ("publish", cmd_ref_publish, "upload a folder as the next version"),
                             ("fetch", cmd_ref_fetch, "download a version")):
        command = ref.add_parser(name, help=text)
        if name != "ls":
            command.add_argument("name")
        if name in ("get", "fetch"):
            command.add_argument("--version", type=int)
        if name == "log":
            command.add_argument("--limit", type=int, default=20)
        if name == "publish":
            command.add_argument("dir")
            command.add_argument("--parent", type=int, help="the version this builds on (default: the current one)")
            command.add_argument("--force", action="store_true", help="publish even if someone else published meanwhile")
        if name == "fetch":
            command.add_argument("dest")
        command.add_argument("--pool")
        command.set_defaults(func=func)
    queue = sub.add_parser("queue", help="task queues for actors and jobs (M3)").add_subparsers(dest="queue_command", required=True)
    q_create = queue.add_parser("create", help="create a queue, or change its settings")
    q_create.add_argument("name")
    q_create.add_argument("--ref", help="the ref results are computed with")
    q_create.add_argument("--max-lag", type=int, help="versions a result may be behind the ref")
    q_create.add_argument("--stale", choices=["requeue", "drop"], default="requeue")
    q_create.add_argument("--duplicate-after", type=float, help="seconds before a slow task may run twice")
    q_create.add_argument("--replicate", type=float, help="fraction of tasks run on two machines and compared (checked pools)")
    q_create.set_defaults(func=cmd_queue_create)
    q_ls = queue.add_parser("ls")
    q_ls.set_defaults(func=cmd_queue_ls)
    q_stats = queue.add_parser("stats")
    q_stats.add_argument("name")
    q_stats.set_defaults(func=cmd_queue_stats)
    q_add = queue.add_parser("add", help="tasks as JSON lines: {inputs, key?} or bare inputs")
    q_add.add_argument("name")
    q_add.add_argument("file", nargs="?", default="-")
    q_add.set_defaults(func=cmd_queue_add)
    q_results = queue.add_parser("results", help="accepted results as JSON lines")
    q_results.add_argument("name")
    q_results.add_argument("--after", type=int, default=0)
    q_results.add_argument("-f", "--follow", action="store_true")
    q_results.set_defaults(func=cmd_queue_results)
    for command in (q_create, q_ls, q_stats, q_add, q_results):
        command.add_argument("--pool")

    actor = sub.add_parser("actor", help="long-lived replicas, kept at a count or scaled by a queue (M3)").add_subparsers(
        dest="actor_command", required=True)
    a_create = actor.add_parser("create", help="create an actor from a job spec, or update it")
    a_create.add_argument("name")
    a_create.add_argument("spec", help="a job spec (YAML or JSON), as for pool submit")
    a_create.add_argument("--min", type=int, default=1)
    a_create.add_argument("--max", type=int)
    a_create.add_argument("--queue", help="scale with this queue's backlog")
    a_create.add_argument("--per-replica", type=int, default=1, help="queued tasks per replica")
    a_create.set_defaults(func=cmd_actor_create)
    a_ls = actor.add_parser("ls")
    a_ls.set_defaults(func=cmd_actor_ls)
    a_status = actor.add_parser("status")
    a_status.add_argument("name")
    a_status.set_defaults(func=cmd_actor_status)
    a_scale = actor.add_parser("scale")
    a_scale.add_argument("name")
    a_scale.add_argument("--min", type=int)
    a_scale.add_argument("--max", type=int)
    a_scale.set_defaults(func=cmd_actor_scale)
    a_stop = actor.add_parser("stop")
    a_stop.add_argument("name")
    a_stop.set_defaults(func=cmd_actor_stop)
    for command in (a_create, a_ls, a_status, a_scale, a_stop):
        command.add_argument("--pool")

    rnd = sub.add_parser("round", help="synchronisation rounds on a ref (M5)").add_subparsers(dest="round_command", required=True)
    r_open = rnd.add_parser("open", help="open a round on the ref's current version")
    r_open.add_argument("ref")
    r_open.add_argument("--quorum", type=int, required=True)
    r_open.add_argument("--min", type=int, help="contributions needed at the deadline (default: the quorum)")
    r_open.add_argument("--deadline", type=float, default=600, help="seconds")
    r_open.add_argument("--grace", type=float, default=5, help="seconds after the quorum for stragglers")
    r_open.add_argument("--late", choices=["drop", "next"], default="drop")
    r_open.add_argument("--decay", type=float, default=0.5, help="weight factor for late work carried to the next round")
    r_open.add_argument("--repeat", action="store_true", help="open the next round when one is reduced")
    r_open.add_argument("--mode", choices=["reduce", "gossip"], default="reduce",
                        help="gossip: no reducer; each worker averages with sampled peers (M7)")
    r_open.add_argument("--fanout", type=int, default=2, help="gossip: peers each worker samples")
    r_open.add_argument("--bias", type=float, default=0.5, help="gossip: preference for fast links (0 = uniform)")
    r_open.add_argument("--reducer", choices=["artifacts", "torchft"], default="artifacts",
                        help="--mode reduce only; torchft: the contributors reduce in-band with torchft (needs --lighthouse)")
    r_open.add_argument("--lighthouse", help="torchft lighthouse address, http://host:port")
    r_open.add_argument("--reduce", choices=["flat", "tree"], default="flat",
                        help="tree: each island reduces its members on one of its machines, then one partial per "
                             "island goes to the final reducer (--reducer artifacts only)")
    r_open.add_argument("--handoff", type=float, default=30,
                        help="tree: seconds a planned reducer has before other machines of its island may take its stage")
    r_open.set_defaults(func=cmd_round_open)
    r_status = rnd.add_parser("status", help="the round in progress on a ref")
    r_status.add_argument("ref")
    r_status.set_defaults(func=cmd_round_status)
    for command in (r_open, r_status):
        command.add_argument("--pool")

    from pool import agent

    agent.add_commands(sub)

    run = sub.add_parser("run", help="upload this folder, run a command, stream logs, fetch outputs")
    run.add_argument("--pool")
    run.add_argument("--image", default=DEFAULT_IMAGE)
    run.add_argument("--gpus", type=int, default=0)
    run.add_argument("--gpu-mem", type=float, help="GB of GPU memory each GPU must have")
    run.add_argument("--cpus", type=float, default=2)
    run.add_argument("--mem", type=float, default=8, help="GB of memory")
    run.add_argument("--shm-gb", dest="shm_gb", type=float, help="size of /dev/shm (default 4; Ray and vLLM like more)")
    run.add_argument("--tmp-gb", dest="tmp_gb", type=float, help="size of /tmp (default 8)")
    run.add_argument("--pids", type=int, help="process and thread limit (default 4096)")
    run.add_argument("--code", default=".", help="folder mounted at /work (default: this one)")
    run.add_argument("--no-code", action="store_true")
    run.add_argument("--data", action="append", help="[NAME=]DIR, mounted read-only at /inputs/NAME")
    run.add_argument("--model", action="append", help="NAME=model@tag, a pool model at /inputs/NAME")
    run.add_argument("--fit", help="NAME[:full|offload|stage:k/n[:kv_gb]]: only machines that hold this model")
    run.add_argument("--flops", type=float, help="work in the job, for the network check")
    run.add_argument("--dtype", default="bf16")
    run.add_argument("--bytes-out", type=float, help="bytes the job writes to /outputs")
    run.add_argument("--link-bytes", type=float, help="a gang: bytes each rank sends to the next (for placement: maxflow)")
    run.add_argument("--bulk", action="store_true", help="skip the network check (for example pre-staging)")
    run.add_argument("--priority", type=int, default=0, help="among your own jobs; higher first")
    run.add_argument("--env", action="append", help="K=V")
    run.add_argument("--label", action="append", help="K=V: only machines with this label")
    run.add_argument("--timeout", type=float, default=60, help="minutes")
    run.add_argument("--retries", type=int, default=1, help="re-runs if the machine disappears")
    run.add_argument("--gang", type=int, help="run N ranks on N machines, started together (RANK, WORLD_SIZE, MASTER_ADDR)")
    run.add_argument("--network", choices=["none", "island", "peers"],
                     help="island: a gang's ranks reach only each other; peers: the hosts' network (machines must allow it)")
    run.add_argument("--image-for", action="append", help="LABEL=VALUE=IMAGE: another image on matching machines, "
                                                          "for example gpu=tegra=dustynv/l4t-pytorch:r36.4.0")
    run.add_argument("--name")
    run.add_argument("-d", "--detach", action="store_true", help="print the job id and return")
    run.add_argument("--to", help="where the outputs go (default pool-outputs/<job>)")
    run.add_argument("--no-get", action="store_true", help="do not download the outputs")
    run.add_argument("command", nargs=argparse.REMAINDER)
    run.set_defaults(func=cmd_run)

    submit = sub.add_parser("submit", help="submit a job spec (YAML or JSON)")
    submit.add_argument("spec")
    submit.add_argument("--pool")
    submit.add_argument("-f", "--follow", action="store_true", help="stream the logs and fetch the outputs")
    submit.add_argument("--to")
    submit.add_argument("--no-get", action="store_true")
    submit.set_defaults(func=cmd_submit)

    ps = sub.add_parser("ps", help="your recent jobs")
    ps.add_argument("--pool")
    ps.add_argument("--all", action="store_true", help="(pool admins) everyone's in the pool")
    ps.add_argument("-n", type=int, default=20)
    ps.set_defaults(func=cmd_ps)
    show = sub.add_parser("show", help="a job's placement, or why it waits")
    show.add_argument("job")
    show.set_defaults(func=cmd_show)
    logs = sub.add_parser("logs", help="a job's output")
    logs.add_argument("job")
    logs.add_argument("-f", "--follow", action="store_true")
    logs.set_defaults(func=cmd_logs)
    get = sub.add_parser("get", help="download a job's outputs")
    get.add_argument("job")
    get.add_argument("--to")
    get.set_defaults(func=cmd_get)
    cancel = sub.add_parser("cancel")
    cancel.add_argument("job")
    cancel.set_defaults(func=cmd_cancel)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["server"]:  # its own options; needs FastAPI, which the agent and CLI do not
        from pool.server import main as server_main

        return server_main(argv[1:])
    args = build_parser().parse_args(argv)
    try:
        return args.func(args) or 0
    except PoolError as exc:
        print(f"pool: {exc}", file=sys.stderr)
        return 1
    except ArtifactError as exc:
        print(f"pool: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
