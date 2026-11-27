"""Built-in reducers for rounds (M5), importable in a job as pool_reducers (it sits next to pool_job).

    import pool_job, pool_reducers
    claimed = pool_job.claim_round("diloco/weights", wait=30)
    folders = [(pool_job.fetch_artifact(item["artifact"]), item["weight"], item["late"]) for item in claimed["items"]]
    pool_reducers.weighted_mean(folders, os.environ["POOL_OUTPUTS"] + "/merged", decay=claimed["settings"]["decay"])
    pool_job.finish_round(claimed["id"], "merged")

Every contribution folder holds the same files. `.npy` files are averaged with numpy; `.safetensors`
files too, tensor by tensor, when the safetensors package is installed (it is in most training
images). A late contribution's weight is multiplied by `decay`. Other files are copied from the
first contribution.

A round opened with `reduce: tree` (pool/rounds.py) is reduced island first: `reduce_round` claims
and does whichever stage this machine may take. An island stage merges the island's members into a
partial (checking each member first with `robust=True`) and hands it in with pool_job.submit_partial;
the final stage merges the partials, which carry their islands' total weights, so the result equals
the flat weighted mean, and can recompute a share of them from their members (`verify`).

With `reducer: torchft` a round is reduced in-band instead: the contributors average among
themselves with torchft (DiLoCo, streaming DiLoCo) over their own links, and the round records who
took part. TorchftRounds below is that path.
"""

import json
import os
import queue
import shutil
import socket
import subprocess
import threading
import time


def _weights(folders, decay: float) -> list[float]:
    raw = [float(weight) * (decay if late else 1.0) for _path, weight, late in folders]
    total = sum(raw)
    if total <= 0:
        raise ValueError("the contributions' weights add up to nothing")
    return [w / total for w in raw]


def weighted_mean(folders: list[tuple[str, float, bool]], out: str, decay: float = 0.5) -> str:
    """Average (path, weight, late) contribution folders file by file into `out`; returns `out`."""
    import numpy as np

    if not folders:
        raise ValueError("no contributions")
    weights = _weights(folders, decay)
    os.makedirs(out, exist_ok=True)
    first = folders[0][0]
    for root, _dirs, files in os.walk(first):
        relative = os.path.relpath(root, first)
        os.makedirs(os.path.join(out, relative), exist_ok=True)
        for name in files:
            target = os.path.join(out, relative, name)
            sources = [os.path.join(path, relative, name) for path, _w, _l in folders]
            if name.endswith(".npy"):
                total = sum(w * np.load(src).astype(np.float64) for w, src in zip(weights, sources))
                np.save(target, total.astype(np.load(sources[0]).dtype))
            elif name.endswith(".safetensors"):
                from safetensors.numpy import load_file, save_file

                tensors = [load_file(src) for src in sources]
                merged = {key: sum(w * t[key].astype(np.float64) for w, t in zip(weights, tensors)).astype(tensors[0][key].dtype)
                          for key in tensors[0]}
                save_file(merged, target)
            else:
                shutil.copy2(sources[0], target)
    return out


def robust_mean(folders: list[tuple[str, float, bool]], out: str, base: str | None = None, decay: float = 0.5,
                norm_factor: float = 3.0, min_cosine: float = 0.0) -> tuple[str, list[int]]:
    """Update sanity (M6): drop contributions whose change from `base` is far larger than the median
    change of the others (more than `norm_factor` times its norm) or points away from it (cosine
    below `min_cosine`; see `outliers`), then take the weighted mean of the rest. Returns (out, indices dropped).
    Without `base`, the contributions themselves are compared (use that for deltas and gradients).
    Only .npy files are compared; report the dropped contributions' machines with pool_job.report."""
    import numpy as np

    def vector(path: str) -> np.ndarray:
        parts = []
        for root, dirs, files in os.walk(path):
            dirs.sort()  # the same order in every contribution
            for name in sorted(files):
                if name.endswith(".npy"):
                    parts.append(np.load(os.path.join(root, name)).astype(np.float64).ravel())
        return np.concatenate(parts) if parts else np.zeros(0)

    origin = vector(base) if base else None
    changes = [vector(path) - origin if origin is not None else vector(path) for path, _w, _l in folders]
    dropped = outliers(changes, norm_factor, min_cosine)
    kept = [item for index, item in enumerate(folders) if index not in dropped]
    return weighted_mean(kept, out, decay), dropped


def outliers(changes, norm_factor: float = 3.0, min_cosine: float = 0.0) -> list[int]:
    """Update sanity on in-memory updates (flat arrays, one per contributor), the test robust_mean uses:
    the indices whose norm is more than `norm_factor` times the reference's, or whose cosine with it is
    below `min_cosine` (see `compare`). With fewer than three updates nothing is dropped (two cannot
    outvote each other). The same inputs give the same answer on every machine, so ranks that
    all-gathered their updates can each apply it and stay in step."""
    if len(changes) < 3:
        return []
    return [index for index, (ratio, cosine) in enumerate(compare(changes))
            if ratio > norm_factor or cosine < min_cosine]


def compare(changes) -> list[tuple[float, float]]:
    """(norm ratio, cosine) of each update against a reference: the element-wise median of the other
    updates, or of all of them when there are only three (the median of two is their mean, which one
    bad update can drag anywhere)."""
    import numpy as np

    stacked = np.stack([np.asarray(change, dtype=np.float64).ravel() for change in changes])
    everyone = np.median(stacked, axis=0) if len(stacked) <= 3 else None
    found = []
    for index, change in enumerate(stacked):
        median = everyone if everyone is not None else np.median(np.delete(stacked, index, axis=0), axis=0)
        median_norm = float(np.linalg.norm(median)) or 1e-12
        norm = float(np.linalg.norm(change))
        found.append((norm / median_norm, float(change @ median) / ((norm or 1e-12) * median_norm)))
    return found


# ---- reduce: tree (#156) -------------------------------------------------------------------------------

def merge_stage(claimed: dict, out: str, fetch, robust: bool = False, base: str | None = None, report=None,
                verify: float = 0.0, tolerance: float = 1e-5, seed=None, **robust_args) -> dict:
    """Merge one claimed stage of a round into `out`. `fetch(artifact)` gives a folder; `report(machine,
    ok, kind, detail)` records a check (pool_job.report).

    flat or island stage: the weighted mean of the items (late ones times decay); with `robust`, each
    update is checked first (robust_mean, against `base` when the updates are weights) and outliers
    are left out and reported, so a poisoned update never reaches the island's partial.
    final stage: the items are the islands' partials, whose weights are their islands' total weights,
    so their weighted mean equals the flat reduction. With `verify`, that share of the partials
    (chosen at random, seeded by the round) is recomputed from its members and compared; a partial
    that does not match is left out and its reducer reported. Verifying fetches the members, so it
    costs the slow-link traffic the tree saves: keep the share small.

    An island stage leaves out a member it cannot fetch (its machine went away) instead of failing.

    Returns {"out", "dropped": [contribution ids left out: outliers and unreachable], "unreachable",
    "bad_partials": [island numbers]}."""
    import random

    decay = float(claimed.get("settings", {}).get("decay", 0.5))
    items = list(claimed["items"])
    bad = []
    if claimed.get("stage") == "final" and verify:
        rng = random.Random(str(seed if seed is not None else claimed.get("id")))
        for item in items:
            if item.get("forwarded") or not item.get("members") or rng.random() >= verify:
                continue
            if not verify_partial(item, fetch, decay, tolerance):
                bad.append(item)
                if report:
                    report(item["machine"], False, "update", f"the partial of island {item['island']} does not match its inputs")
        items = [item for item in items if item not in bad]
    unreachable = []
    if claimed.get("stage") == "island":  # a member whose machine went away is left out, not the whole island
        fetched = []
        for item in items:
            try:
                fetched.append((item, fetch(item["artifact"])))
            except Exception as exc:  # noqa: BLE001  (pool_job raises PoolJobError, a test's fetch anything)
                print(f"[pool_reducers] contribution {item.get('contribution')} from {item.get('machine')}: {exc}", flush=True)
                unreachable.append(item)
        items = [item for item, _path in fetched]
        paths = [path for _item, path in fetched]
    else:
        paths = None
    if not items:
        raise ValueError("nothing left to merge")
    if paths is None:
        paths = [fetch(item["artifact"]) for item in items]
    folders = [(path, item["weight"], item["late"]) for path, item in zip(paths, items)]
    dropped = []
    if robust and claimed.get("stage") != "final":
        _, indices = robust_mean(folders, out, base=base, decay=decay, **robust_args)
        dropped = [items[index] for index in indices]
        for item in dropped:
            if report:
                report(item["machine"], False, "update", "an outlying update, left out of the reduction")
    else:
        weighted_mean(folders, out, decay)
    return {"out": out, "dropped": [item.get("contribution") for item in dropped + unreachable],
            "unreachable": [item.get("contribution") for item in unreachable], "bad_partials": [i["island"] for i in bad]}


def verify_partial(item: dict, fetch, decay: float = 0.5, tolerance: float = 1e-5) -> bool:
    """Whether an island's partial (a final-stage item) is the weighted mean of its members: its
    .npy files are recomputed from the members' and compared within `tolerance` (relative)."""
    import tempfile

    import numpy as np

    with tempfile.TemporaryDirectory() as scratch:
        expected = weighted_mean([(fetch(m["artifact"]), m["weight"], m["late"]) for m in item["members"]],
                                 os.path.join(scratch, "expected"), decay)
        got = fetch(item["artifact"])
        for root, _dirs, files in os.walk(expected):
            for name in files:
                if not name.endswith(".npy"):
                    continue
                relative = os.path.relpath(os.path.join(root, name), expected)
                theirs = os.path.join(got, relative)
                if not os.path.isfile(theirs):
                    return False
                a = np.load(os.path.join(expected, relative)).astype(np.float64)
                b = np.load(theirs).astype(np.float64)
                if a.shape != b.shape or not np.allclose(a, b, rtol=tolerance, atol=tolerance * (float(np.abs(a).max()) if a.size else 0)):
                    return False
    return True


def reduce_round(ref: str, wait: float = 30, out_root: str | None = None, **merge_args) -> dict | None:
    """Claim and do one stage of a round on `ref` from a job: a flat round's reduction, a tree round's
    island stage (handed in with submit_partial) or its final stage (finish_round). None when there
    is nothing this machine may claim yet. Workers of a tree round can call it in a loop after
    contributing: the planned machine of each island picks up its island's stage.

        while pool_job.ref("w")["version"] == version:
            pool_reducers.reduce_round("w", wait=5, robust=True)
    """
    job = _pool_job()
    claimed = job.claim_round(ref, wait=wait)
    if not claimed:
        return None
    stage = claimed.get("stage", "flat")
    name = f"{claimed['id']}-{stage}" + (f"-{claimed['island']}" if stage == "island" else "")
    out = os.path.join(out_root or os.environ.get("POOL_OUTPUTS", "/outputs"), "rounds", name)
    shutil.rmtree(out, ignore_errors=True)
    merged = merge_stage(claimed, out, job.fetch_artifact, report=job.report, **merge_args)
    if stage == "island":
        reply = job.submit_partial(claimed["id"], out, merged["dropped"])
    else:
        reply = job.finish_round(claimed["id"], out)
    return {"round": claimed["id"], "stage": stage, "island": claimed.get("island"), "reply": reply, **merged}


# ---- reducer: torchft (M5) -----------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("", 0))
        return probe.getsockname()[1]


def _pool_job():
    try:
        import pool_job  # inside a job: next to this file in POOL_LIB
    except ImportError:
        from pool import job as pool_job
    return pool_job


class TorchftRounds:
    """Rounds of a ref reduced in-band by torchft (#45 sections 8 and 15: adopt torchft, don't rebuild it).

    torchft does the reduction: the replica groups (islands, one pool job or gang each) all-reduce
    their pseudo-gradients directly, and its Manager handles quorum, membership changes and
    recovery. The pool round adds what torchft keeps no record of: the quorum and the deadline, the
    lighthouse's address (the rendezvous), and a ledger of who contributed to each outer step with
    what weight, published as the ref's next version (for credit, checks and reports, M6).

        rounds = pool_reducers.TorchftRounds("c1/olmoe", quorum=2, deadline_s=1800)
        manager = rounds.manager(pg, state_dict=..., load_state_dict=...)   # a torchft.Manager
        with torchft.local_sgd.DiLoCo(manager, [model], inner, outer, sync_every=H) as diloco:
            for step ...:
                ...; inner.step()
                if rounds.synced(manager):                   # an outer step was committed in this step
                    rounds.record(manager, weight=tokens, meta={"loss": loss})
        rounds.close()

    The first job to open the round hosts torchft's lighthouse (`torchft_lighthouse`, from the job's
    image) on `host:port` and writes its address into the round; the others read it there. The
    round's min is torchft's min_replica_size, and its deadline torchft's quorum timeout. Recording
    runs in a background thread and never blocks training. The jobs need a network to each other
    (network: peers); rounds whose contributors have no direct link use the artifact path above.
    """

    def __init__(self, ref: str, quorum: int, min: int | None = None, deadline_s: float = 1800, grace_s: float = 5,
                 host: str | None = None, port: int = 29510, replica_id: str | None = None, start_lighthouse: bool = True):
        self.job = _pool_job()
        self.ref, self.quorum, self.min = ref, int(quorum), int(min or quorum)
        self.deadline_s, self.grace_s = float(deadline_s), float(grace_s)
        self.host = host or os.environ.get("POOL_NODE_IP") or os.environ.get("MASTER_ADDR") or socket.gethostname()
        self.replica_id = replica_id or os.environ.get("POOL_JOB_ID") or f"{socket.gethostname()}-{os.getpid()}"
        mine = f"http://{self.host}:{int(port)}"
        try:
            self.job.ref(ref)
        except self.job.PoolJobError as exc:
            if exc.status != 404:
                raise
            self._publish_start()
        opened = self.job.open_round(ref, self.quorum, min=self.min, deadline_s=self.deadline_s, grace_s=self.grace_s,
                                     repeat=True, reducer="torchft", lighthouse=mine)
        if opened["settings"].get("reducer") != "torchft":
            raise RuntimeError(f"the round in progress on {ref} is not reduced by torchft: {opened['settings']}")
        self.lighthouse = opened["settings"]["lighthouse"]
        self.version = opened["base_version"]  # the ref version this job's next outer step builds on
        self.hosting = self.lighthouse == mine
        self.process = None
        if self.hosting and start_lighthouse:
            self.process = subprocess.Popen(["torchft_lighthouse", "--bind", f"{self.host}:{int(port)}",
                                             "--min_replicas", str(self.min), "--join_timeout_ms", str(int(self.grace_s * 1000)),
                                             "--quorum_tick_ms", "100"])
        os.environ["TORCHFT_LIGHTHOUSE"] = self.lighthouse
        self._step = 0
        self.errors: list[str] = []
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _publish_start(self) -> None:
        folder = os.path.join(os.environ.get("POOL_OUTPUTS", "/outputs"), "torchft-rounds", "start")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "round.json"), "w") as handle:
            json.dump({"start": True}, handle)
        try:
            self.job.publish_ref(self.ref, folder, parent=0)
        except self.job.PoolJobError as exc:
            if exc.status != 409:  # another island published it first
                raise

    def manager(self, pg=None, **kwargs):
        """A torchft.Manager for this job's replica group, pointed at the round's lighthouse."""
        from datetime import timedelta

        from torchft import Manager, ProcessGroupGloo

        kwargs.setdefault("min_replica_size", self.min)
        kwargs.setdefault("replica_id", self.replica_id)
        kwargs.setdefault("lighthouse_addr", self.lighthouse)
        kwargs.setdefault("use_async_quorum", False)
        kwargs.setdefault("quorum_timeout", timedelta(seconds=self.deadline_s))
        kwargs.setdefault("timeout", timedelta(seconds=self.deadline_s))
        kwargs.setdefault("hostname", self.host)
        if (os.environ.get("RANK", "0") == "0" and not os.environ.get("TORCHELASTIC_RUN_ID")
                and "store_addr" not in kwargs and getattr(self, "_store", None) is None):
            # torchft expects the replica group's store at MASTER_ADDR:MASTER_PORT, which torchrun
            # hosts; a plain gang rank 0 (or a lone job) hosts it here
            from torch.distributed import TCPStore

            addr = os.environ.setdefault("MASTER_ADDR", self.host)
            port = int(os.environ.setdefault("MASTER_PORT", str(_free_port())))
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            self._store = TCPStore(addr, port, is_master=True, wait_for_workers=False)
        return Manager(pg=pg if pg is not None else ProcessGroupGloo(timeout=timedelta(seconds=self.deadline_s)), **kwargs)

    def synced(self, manager) -> bool:
        """True once per outer step: torchft's step counter moves when an outer step is committed."""
        step = manager.current_step()
        moved, self._step = step != self._step, step
        return moved

    def record(self, manager, weight: float, meta: dict | None = None, publish: str | None = None) -> None:
        """Note this replica's part in the outer step just taken (weight: its tokens, say). The replica
        of participating rank 0 also closes the pool round: `publish` is a folder under the outputs to
        publish as the ref's version (a checkpoint), else a small JSON record of the round."""
        item = {"version": self.version, "weight": float(weight), "leader": manager.participating_rank() == 0,
                "publish": publish, "meta": {**(meta or {}), "replica": self.replica_id, "step": manager.current_step(),
                                             "participants": manager.num_participants()}}
        self.version += 1
        self._queue.put(item)

    def _drain(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                self._record(item)
            except Exception as exc:  # the ledger must never stop the training
                self.errors.append(f"v{item['version']}: {exc}")
                print(f"[pool_reducers] torchft round on v{item['version']}: {exc}", flush=True)

    def _record(self, item: dict) -> None:
        version = item["version"]
        root = os.path.join(os.environ.get("POOL_OUTPUTS", "/outputs"), "torchft-rounds")
        folder = os.path.join(root, f"v{version}-{self.replica_id}")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "contribution.json"), "w") as handle:
            json.dump(item["meta"], handle)
        give_up = time.monotonic() + self.deadline_s
        while True:  # the round on this version opens once the previous one is finished
            try:
                reply = self.job.contribute(self.ref, folder, item["weight"], base_version=version)
                break
            except self.job.PoolJobError as exc:
                if exc.status != 404 or time.monotonic() > give_up:
                    raise
                time.sleep(0.5)
        if not reply.get("accepted"):
            raise RuntimeError(f"contribution refused: {reply}")
        if not item["leader"]:
            return
        claimed = self.job.claim_round(self.ref, wait=min(self.deadline_s, 600))
        if not claimed or claimed["base_version"] != version:
            raise RuntimeError(f"no ready round of {self.ref} on v{version}: {claimed and claimed['base_version']}")
        out = item["publish"]
        if not out:
            out = os.path.join(root, f"result-v{version}")
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, "round.json"), "w") as handle:
                json.dump({"base_version": version, "step": item["meta"]["step"], "participants": item["meta"]["participants"],
                           "contributions": [{"machine": i["machine"], "job": i["job"], "weight": i["weight"], "late": i["late"]}
                                             for i in claimed["items"]]}, handle)
        self.job.finish_round(claimed["id"], out)

    def close(self, timeout: float = 120) -> None:
        """Let the ledger catch up, then stop the lighthouse if this job hosts it."""
        self._queue.put(None)
        self._thread.join(timeout)
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(10)
            except subprocess.TimeoutExpired:
                self.process.kill()
