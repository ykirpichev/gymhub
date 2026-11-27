"""Refs and queues from inside a pool job (M3): the job's side of the agent's unix socket.

Pool jobs run with `--network none`, so they never talk to the server themselves. The agent opens a
unix socket in the job's folder and forwards each request as the job's owner, in the job's pool:

    POOL_SOCKET   the socket (/run/pool/pool.sock in a container)
    POOL_LIB      the folder that holds this file as pool_job.py (/run/pool): no install needed
    POOL_REFS     where fetched ref versions appear (/refs)

    import os, sys; sys.path.insert(0, os.environ["POOL_LIB"]); import pool_job

    path, version = pool_job.fetch_ref("grpo/policy")            # newest version, as a folder
    for task in pool_job.lease("rollouts", n=4, wait=10):
        ...
        pool_job.result("rollouts", task["id"], {"reward": 0.8}, computed_with=version)
    pool_job.publish_ref("grpo/policy", "policy-v2", parent=version)  # a folder under /outputs

Only the standard library, so it works in any image with Python 3.10+.
"""

import http.client
import json
import os
import socket
import time


class PoolJobError(Exception):
    def __init__(self, status: int, detail):
        super().__init__(f"{status}: {detail}")
        self.status, self.detail = status, detail


class _UnixConnection(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: float):
        super().__init__("localhost", timeout=timeout)
        self._path = path

    def connect(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self._path)
        self.sock = sock


def call(op: str, timeout: float = 120, **args) -> dict:
    """One operation through the agent. Raises PoolJobError with the server's message."""
    path = os.environ.get("POOL_SOCKET")
    if not path:
        raise PoolJobError(0, "POOL_SOCKET is not set: this is not running as a pool job")
    conn = _UnixConnection(path, timeout)
    try:
        conn.request("POST", "/api", json.dumps({"op": op, "args": args}), {"Content-Type": "application/json"})
        reply = conn.getresponse()
        data = json.loads(reply.read() or b"{}")
    finally:
        conn.close()
    if reply.status >= 400:
        raise PoolJobError(reply.status, data.get("detail", data))
    return data


# ---- refs -----------------------------------------------------------------------------------------

def ref(name: str, version: int | None = None) -> dict:
    """The current version of a ref (or one version): {name, version, artifact, parent, meta}."""
    return call("ref.get", name=name, version=version)["ref"]


def wait_ref(name: str, after: int, timeout: float = 30) -> dict | None:
    """A version newer than `after`, or None if none appears within `timeout` seconds."""
    deadline = time.monotonic() + timeout
    while True:
        left = deadline - time.monotonic()
        try:
            found = call("ref.get", timeout=60, name=name, after=after, wait=max(0.0, min(left, 30)))
        except PoolJobError as exc:
            if exc.status != 404:
                raise
            found = {"newer": False}
        if found.get("newer"):
            return found["ref"]
        if left <= 0:
            return None


def fetch_ref(name: str, version: int | None = None) -> tuple[str, int]:
    """Make a version of a ref available as a read-only folder; returns (path, version). The agent
    keeps it in its cache, so the next job on this machine gets it without a download."""
    reply = call("ref.fetch", timeout=3600, name=name, version=version)
    return reply["path"], reply["version"]


def publish_ref(name: str, folder: str, parent: int | None, force: bool = False, meta: dict | None = None) -> dict:
    """Publish a folder of this job's outputs (a path under POOL_OUTPUTS, or relative to it) as the
    next version of a ref. Fails with status 409 unless `parent` is the current version (or `force`)."""
    outputs = os.environ.get("POOL_OUTPUTS", "/outputs")
    relative = os.path.relpath(folder, outputs) if os.path.isabs(folder) else folder
    return call("ref.publish", timeout=3600, name=name, path=relative, parent=parent, force=force, meta=meta)["ref"]


# ---- queues ---------------------------------------------------------------------------------------

def create_queue(name: str, ref: str | None = None, max_lag: int | None = None, stale: str = "requeue",
                 duplicate_after: float | None = None) -> dict:
    return call("queue.create", name=name, ref=ref, max_lag=max_lag, stale=stale, duplicate_after=duplicate_after)["queue"]


def add_tasks(queue: str, tasks: list[dict]) -> list[str]:
    """tasks: [{inputs, key?, requires_version?}] (inline JSON, at most 64 KB each)."""
    return call("tasks.add", name=queue, tasks=tasks)["ids"]


def lease(queue: str, n: int = 1, wait: float = 0) -> list[dict]:
    """Up to n tasks: [{id, key, inputs, requires_version, tries, duplicate}]. Waits up to `wait` s for one."""
    return call("tasks.lease", timeout=max(60, wait + 30), name=queue, n=n, wait=wait)["tasks"]


def result(queue: str, task: str, value, computed_with: int | None = None) -> dict:
    """Report a task's result. {"accepted": False, ...} when a duplicate won, or when the result is stale."""
    return call("task.result", name=queue, task=task, result=value, computed_with=computed_with)


def fail(queue: str, task: str, reason: str) -> dict:
    """Give a task back to the queue."""
    return call("task.result", name=queue, task=task, result=None, failed=reason or "failed")


def results(queue: str, after: int = 0, wait: float = 0, limit: int = 500) -> dict:
    """Accepted results in order: {results: [{seq, task, key, result, computed_with, machine}], next}."""
    return call("results", timeout=max(60, wait + 30), name=queue, after=after, wait=wait, limit=limit)


def stats(queue: str) -> dict:
    return call("queue.stats", name=queue)["queue"]


# ---- checkpoints ------------------------------------------------------------------------------------

def save_checkpoint() -> dict:
    """Save POOL_CHECKPOINT (/checkpoint). If this job runs again (a lost machine, a preemption), it
    starts with that folder restored and POOL_RESUMED=1. The agent also saves it on preemption."""
    return call("checkpoint.save", timeout=3600)


def resumed() -> bool:
    return os.environ.get("POOL_RESUMED") == "1"


# ---- rounds (M5) ----------------------------------------------------------------------------------

def open_round(ref: str, quorum: int, min: int | None = None, deadline_s: float = 600, grace_s: float = 5,
               late: str = "drop", decay: float = 0.5, repeat: bool = False, mode: str = "reduce", fanout: int = 2,
               bias: float = 0.5, reducer: str = "artifacts", lighthouse: str | None = None, reduce: str = "flat",
               handoff_s: float = 30) -> dict:
    """Open a round on the ref's current version (or return the one already in progress).
    mode="gossip": no reducer; each worker mixes with `fanout` sampled peers (mix_round).
    reducer="torchft" (mode "reduce" only) with a lighthouse address: see pool_reducers.TorchftRounds.
    reduce="tree": island first (pool/rounds.py); run pool_reducers.reduce_round on machines of each island."""
    return call("round.open", ref=ref, quorum=quorum, min=min, deadline_s=deadline_s, grace_s=grace_s, late=late,
                decay=decay, repeat=repeat, mode=mode, fanout=fanout, bias=bias, reducer=reducer,
                lighthouse=lighthouse, reduce=reduce, handoff_s=handoff_s)["round"]


def current_round(ref: str, after: str | None = None, wait: float = 0) -> dict | None:
    """The round in progress on a ref; with `after`, wait up to `wait` s for an open round other than that one."""
    return call("round.current", timeout=max(60, wait + 30), ref=ref, after=after, wait=wait)["round"]


def contribute(ref: str, folder: str, weight: float, base_version: int, round: str | None = None,
               held: bool | None = None) -> dict:
    """Contribute a folder of this job's outputs to the round on `base_version` (or to `round`, for
    gossip rounds, which follow each other on one version). {"accepted", "late"?, "round"}.
    held: keep the chunks on this machine for peers instead of uploading them (default: when the
    round reduces as a tree, so that the update stays in its island)."""
    outputs = os.environ.get("POOL_OUTPUTS", "/outputs")
    relative = os.path.relpath(folder, outputs) if os.path.isabs(folder) else folder
    return call("round.contribute", timeout=3600, ref=ref, path=relative, weight=weight, base_version=base_version,
                round=round, held=held)


def mix_round(round_id: str, wait: float = 0) -> dict | None:
    """For a worker of a gossip round: once it closed, {"items": [{"artifact", "weight", "self", ...}]},
    the contributions to average this job's own with (weights add up to 1). None while still open.

        mix = pool_job.mix_round(rid, wait=30)
        folders = [(mine if i["self"] else pool_job.fetch_artifact(i["artifact"]), i["weight"], False) for i in mix["items"]]
        pool_reducers.weighted_mean(folders, out)
    """
    return call("round.mix", timeout=max(60, wait + 30), round=round_id, wait=wait)["mix"]


def claim_round(ref: str, wait: float = 0) -> dict | None:
    """For a reducer: a ready round with its contributions ("items": artifact, weight, late, machine).
    In a tree round, "stage" is "island" (reduce the items, hand in with submit_partial) or "final"
    (the items are the islands' partials, each with its "members"; finish_round as usual)."""
    return call("round.claim", timeout=max(60, wait + 30), ref=ref, wait=wait)["round"]


def fetch_artifact(artifact: str) -> str:
    """A readable artifact (a contribution) as a local folder."""
    return call("artifact.fetch", timeout=3600, artifact=artifact)["path"]


def finish_round(round_id: str, folder: str) -> dict:
    """Publish a folder of this job's outputs as the round's result: the ref's next version."""
    outputs = os.environ.get("POOL_OUTPUTS", "/outputs")
    relative = os.path.relpath(folder, outputs) if os.path.isabs(folder) else folder
    return call("round.finish", timeout=3600, round=round_id, path=relative)


def submit_partial(round_id: str, folder: str, dropped: list[str] | None = None) -> dict:
    """For an island reducer of a tree round: hand in a folder of this job's outputs (the weighted mean
    of the items it was given, held on this machine) as the island's partial; `dropped` names the
    contributions left out (outliers). {"accepted", "weight", "included", ...}."""
    outputs = os.environ.get("POOL_OUTPUTS", "/outputs")
    relative = os.path.relpath(folder, outputs) if os.path.isabs(folder) else folder
    return call("round.partial", timeout=3600, round=round_id, path=relative, dropped=list(dropped or []))


# ---- checks (M6) ------------------------------------------------------------------------------------

def report(machine: str, ok: bool, kind: str = "other", detail: str = "") -> dict:
    """Report a check on a machine's work (a task result's or contribution's "machine"): kind is
    logprobs, toploc, reward, update, duplicate or other. In a checked pool, enough failures exclude it."""
    return call("verify.report", machine=machine, ok=ok, kind=kind, detail=detail)["machine"]
