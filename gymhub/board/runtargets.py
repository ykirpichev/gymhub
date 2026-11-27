"""Run on: where a learner may send a run, test or bench (the exercise toolbar's "Run on" choice).

Auto (no target) is the default: any machine that may take the job. A choice is a target on the job:

    {"runner": <worker id>}         only that board runner leases it
    {"pool_machine": <name>}        only a pool bridge that lists that machine leases it, and pins the
                                    pool job there (resources.machine, gymhub/runner/pooled.py)

The choices are the machines that could lease the job anyway, by the same rules as leasing
(gymhub/jobqueue.py, gymhub/board/workers.py `_lease_rules`, gymhub/routing.py): whose jobs a
machine takes, which gyms may reach it, the sandbox, capabilities, and, for a gym that is not public,
only the bridge's own pool machines. A target only narrows who leases a job, never widens it: the
lease applies every rule as before, plus the target (`routing.misses_target`).
"""

import time

from gymhub import gymaccess, routing

ONLINE_SECONDS = 90
MAX_RUNNERS = 50


def _worker(store, runner_id: str, runner: dict):
    """The worker behind a runner's heartbeat; "direct" for a runner on the store itself (no worker
    token: the owner's machine, leasing without worker rules); None when revoked or disabled."""
    from gymhub.board.workers import worker_from_doc

    if not runner.get("worker_id"):
        return "direct"
    doc = store.get("workers", runner_id)
    if not doc or doc.get("revoked"):
        return None
    return worker_from_doc(runner_id, doc)


def allowed(store, runner_id: str, runner: dict, job: dict) -> bool:
    """May this runner take the user's job at all (whose jobs, which gyms, the sandbox)? Says nothing
    of whether it is online or able."""
    from gymhub.board.workers import _lease_rules

    worker = _worker(store, runner_id, runner)
    if worker is None:
        return False
    if worker == "direct":
        return True
    return not _lease_rules(store, worker, runner.get("capabilities"), head_start=False)(job)


def able(runner_id: str, runner: dict, job: dict) -> bool:
    """Would this runner lease the job by what its last heartbeat said (gyms, target, capabilities)?"""
    capabilities = runner.get("capabilities")
    if not routing.serves(runner, job.get("gym", ""), bool(job.get("bundle"))):
        return False
    if routing.misses_target(runner_id, capabilities, job) or routing.refuses(capabilities, job):
        return False
    return routing.unmet(routing.for_job(capabilities, job), job.get("requires")) is None


def online(runner: dict) -> bool:
    return time.time() - runner.get("heartbeat_at", 0) < ONLINE_SECONDS


def _runners(store) -> list[tuple[str, dict]]:
    return list(store.recent("runners", "heartbeat_at", MAX_RUNNERS))


def _gpu(gpus: list[dict]) -> str:
    return (gpus[0].get("name") or "GPU") if gpus else "CPU"


def choices(store, job: dict) -> list[dict]:
    """Where this job could run now: online, able and allowed runners, then the pool machines of
    online bridges that would grade it there. Each: {target, name, kind (runner|pool), hardware, via}."""
    job = {**job, "gym_public": gymaccess.public(store.get("gyms", job.get("gym", "")))}
    found, machines = [], {}
    for runner_id, runner in _runners(store):
        if not online(runner):
            continue
        name = runner.get("name") or runner_id
        capabilities = runner.get("capabilities")
        pinned = {**job, "target": {"runner": runner_id}}
        if able(runner_id, runner, pinned) and allowed(store, runner_id, runner, pinned):
            on_pool = routing.on_pool(capabilities, pinned)
            found.append({"target": {"runner": runner_id}, "name": name, "kind": "runner",
                          "hardware": "the pool" if on_pool else routing.hardware(capabilities)})
        for machine in routing.pool_machines((capabilities or {}).get("pool")):
            pinned = {**job, "target": {"pool_machine": machine["name"]}}
            if machine["name"] in machines or not able(runner_id, runner, pinned) or not allowed(store, runner_id, runner, pinned):
                continue
            machines[machine["name"]] = {"target": pinned["target"], "name": machine["name"], "kind": "pool",
                                         "hardware": _gpu(machine.get("gpus") or []), "via": name}
    return found + sorted(machines.values(), key=lambda item: item["name"])


def check(store, job: dict) -> str | None:
    """Validate a job's target before it is queued. None when it may be sent there (even if that
    machine is offline now: it waits); else why not. Never lets a target reach a machine the user
    could not be routed to: the runner (or a bridge listing the pool machine) must allow the job."""
    target = job.get("target")
    if not isinstance(target, dict) or len(target) != 1:
        return "Run on: pick a machine or Auto."
    (kind, name), = target.items()
    if kind not in ("runner", "pool_machine") or not isinstance(name, str) or not 0 < len(name) <= 80:
        return "Run on: pick a machine or Auto."
    job = {**job, "gym_public": gymaccess.public(store.get("gyms", job.get("gym", "")))}
    for runner_id, runner in _runners(store):
        if kind == "runner" and runner_id != name:
            continue
        if kind == "pool_machine":
            bridge = (runner.get("capabilities") or {}).get("pool")
            if routing.pool_machine(bridge, job, name) is None:
                continue
        if allowed(store, runner_id, runner, job):
            return None
    return f"You cannot run this on {name}. Choose another machine or Auto."


def waiting(store, job: dict) -> str | None:
    """The queue warning for a job with a target: None when that machine can take it now."""
    target = job["target"]
    if any(choice["target"] == target for choice in choices(store, job)):
        return None
    name = next(iter(target.values()))
    return f"{name} is offline or cannot run this now. The job waits for it; switch Run on to Auto to run it elsewhere."
