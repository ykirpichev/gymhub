"""What the Compute page shows of the compute pool (#162): its machines, how they connect, and
which gyms each can take.

Two ways the board learns the pool. With GYMHUB_POOL_SERVER and GYMHUB_POOL_TOKEN it asks the pool
server itself (gymhub/board/poolmachines.py). A board that cannot reach the pool server (the server is
on a LAN, the board on Cloud Run) reads what the dispatcher saw instead: every heartbeat of a pool
bridge (gymhub/runner/agent.py) carries `pool_report`, built here from the pool server's
GET /v1/machines and GET /v1/links. Both paths give the page the same shape (`machine_view`).

No imports from the board (FastAPI) or the runner: both use this module.
"""

import time

MAX_MACHINES = 50
REPORT_KEYS = ("name", "online", "drain", "running", "last_seen", "gpus", "arch", "platform", "cpus", "memory_gb", "labels", "own", "owner")


def _gpus(profile: dict) -> list[dict]:
    out = []
    for gpu in profile.get("gpus") or []:
        out.append({"model": str(gpu.get("model") or gpu.get("name") or "unknown GPU")[:80], "memory_gb": gpu.get("memory_gb")})
    return out


def _platform(profile: dict) -> str | None:
    system, arch = profile.get("os"), profile.get("arch")
    return " ".join(part for part in (str(system).capitalize() if system else None, arch and str(arch)) if part) or None


def machine_view(machine: dict) -> dict:
    """A pool machine (pool/server.py GET /v1/machines) as the Compute page shows it."""
    profile = machine.get("profile") or {}
    return {
        "id": machine.get("id"), "name": machine.get("name"), "owner": machine.get("owner"),
        "own": machine.get("mine") is True,
        "online": bool(machine.get("online")), "last_seen": machine.get("last_seen"), "drain": bool(machine.get("drain")),
        "labels": machine.get("labels") or {}, "running": len(machine.get("running") or []),
        "gpus": _gpus(profile), "arch": profile.get("arch"), "platform": _platform(profile), "cpus": profile.get("cpus"),
        "memory_gb": profile.get("memory_gb"),
    }


def report(client, pool_name: str | None = None, machines: list[dict] | None = None) -> dict | None:
    """What a pool bridge's heartbeat tells the board of its pool (`pool_report`): every machine,
    online or not, and the LAN islands they form (pool/peers.py). None when the pool server does not answer."""
    suffix = f"?pool={pool_name}" if pool_name else ""
    pool = pool_name
    if machines is None:
        try:
            reply = client.call("GET", "/v1/machines" + suffix, timeout=15)
        except Exception:
            return None
        machines, pool = reply.get("machines") or [], reply.get("pool") or pool_name
    try:
        links = client.call("GET", "/v1/links" + suffix, timeout=15)
        islands = links.get("islands") or []
        pool = pool or links.get("pool")
    except Exception:  # an older pool server, or links not measured yet: no topology, machines still listed
        islands = []
    views = []
    for machine in machines[:MAX_MACHINES]:
        view = machine_view(machine)
        views.append({key: view[key] for key in REPORT_KEYS})
    # Firestore holds no list of lists: an island is {"machines": [names]}.
    return {"pool": pool, "at": time.time(), "machines": views,
            "islands": [{"machines": list(group)} for group in islands[:MAX_MACHINES] if isinstance(group, list)]}


def _num(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _text(value, limit: int = 80) -> str | None:
    return str(value)[:limit] if value is not None and value != "" else None


def clean_report(found) -> dict | None:
    """A `pool_report` from a heartbeat, keeping only the fields and sizes the page uses (a worker
    sends it; the board never trusts its shape)."""
    if not isinstance(found, dict) or not isinstance(found.get("machines"), list):
        return None
    machines = []
    for machine in found["machines"][:MAX_MACHINES]:
        if not isinstance(machine, dict) or not machine.get("name"):
            continue
        gpus = [{"model": _text(gpu.get("model")) or "unknown GPU", "memory_gb": _num(gpu.get("memory_gb"))}
                for gpu in (machine.get("gpus") or [])[:8] if isinstance(gpu, dict)]
        labels = machine.get("labels") if isinstance(machine.get("labels"), dict) else {}
        machines.append({
            "name": _text(machine["name"], 60), "online": bool(machine.get("online")), "drain": bool(machine.get("drain")),
            "running": int(_num(machine.get("running")) or 0), "last_seen": _num(machine.get("last_seen")), "gpus": gpus,
            "arch": _text(machine.get("arch"), 30), "platform": _text(machine.get("platform")), "cpus": _num(machine.get("cpus")),
            "memory_gb": _num(machine.get("memory_gb")), "own": machine.get("own") is True, "owner": _text(machine.get("owner")),
            "labels": {_text(k, 40): _text(v, 80) for k, v in list(labels.items())[:10]},
        })
    names = {machine["name"] for machine in machines}
    islands = []
    for group in (found.get("islands") or [])[:MAX_MACHINES]:
        members = group.get("machines") if isinstance(group, dict) else None
        kept = [str(name) for name in members if str(name) in names] if isinstance(members, list) else []
        if kept:
            islands.append({"machines": kept})
    return {"pool": _text(found.get("pool"), 60), "at": _num(found.get("at")), "machines": machines, "islands": islands}


def from_routing(bridge, at: float | None = None) -> dict | None:
    """A pool report made from a bridge's routing report (`capabilities["pool"]["list"]`, gymhub/runner/pooled.py
    `advertised`), sent by runners from before `pool_report`: only the online machines, with their GPU names."""
    listed = (bridge or {}).get("list") if isinstance(bridge, dict) else None
    if not isinstance(listed, list) or not listed:
        return None
    machines = [{"name": m.get("name"), "online": True, "own": m.get("own") is True,
                 "gpus": [{"model": g.get("name")} for g in m.get("gpus") or [] if isinstance(g, dict)]}
                for m in listed if isinstance(m, dict)]
    found = clean_report({"at": at, "machines": machines})
    if found is not None:
        found["partial"] = True  # online machines only, no platform or state: an older dispatcher
    return found


def gyms_for(machine: dict, gyms: list[tuple[str, bool, bool]]) -> list[str]:
    """The gyms a pool machine can take, of (id, needs a GPU, public): a GPU gym only on a machine
    with a GPU, a gym that is not public only on the bridge member's own machines
    (gymhub/runner/pooled.py `own_machines_only`)."""
    return [gym for gym, gpu, public in gyms if (machine.get("gpus") or not gpu) and (public or machine.get("own"))]
