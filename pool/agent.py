"""The agent: one per machine (#45 sections 3, 5 and 7). Standard library only.

    python -m pool.cli agent register --server URL --code CODE [--name NAME]   once (the installer does this)
    python -m pool.cli agent run [--gpus 0,1|none] [--max-jobs N] [--label k=v] [--allow-image PATTERN]
                                 [--cache-gb 200] [--bench-image IMAGE] [--network-mbps N] [--limit-mbps N]
    python -m pool.cli agent bench --image IMAGE     measure TFLOP/s per dtype and host-to-GPU bandwidth
    pool-agent status | pause | resume | uninstall    the owner's commands (pool agent ... with --config)

Settings: $POOL_AGENT_CONFIG or --config (the system install: /etc/pool/agent.json), else
~/.config/pool/agent.json. `pause` leaves agent.paused next to it: no new leases until `resume`.

It only makes outbound HTTPS calls: heartbeats every ~10 s (which bring cancellations and the use
policy, and carry what the cache holds), a long-poll for work, log chunks every second, results.

Inputs are cached by content. An artifact's chunks are fetched once into the chunk cache, its
files are assembled once into a read-only tree, and every later job that uses it mounts that tree:
placing a 35 GB model a second time costs no download. The cache is evicted least recently used
first, above --cache-gb.

Each job runs in its own container: no network, read-only root filesystem, every capability
dropped, no privilege escalation, this user's uid, limits on memory, CPUs, processes and /tmp,
only the GPUs it was given. /work: its code; /inputs/<name>: its inputs (read-only); /outputs:
what comes back; /checkpoint: scratch for the job's lifetime.

Use policy (pool machine policy): `hours` (only lease in that local window) and `idle_only`
(only while no one else uses the GPUs). When either stops allowing work, running jobs get
SIGTERM, 30 s to stop, and go back to the queue without using a retry.

Peers (pool/peers.py): the agent also runs a small peer server (--peer-port, default 8766) and
reports its LAN address. Through the server it finds the pool's other agents, measures its link to
each, fetches chunks from the peers that hold them before asking the server (several peers at once,
peers.Swarm; the server is the tracker, and each chunk is announced as soon as it lands), and meets
the other ranks of a gang. With --iroh, chunks move between machines through the iroh-blobs sidecar
(peers.IrohSidecar), with the HTTP peer server as the fallback. Jobs with network: peers run with
the host's network, and only on machines whose owner allows it (--allow-peer-network); every other
job still has no network at all.

The machine's owner can see everything a job brings (the pool does not hide job data from them).
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

from pool import island, models, peers, profile as profiles, update
from pool.spec import image_for
from pool import artifacts as artifact_format
from pool.artifacts import ArtifactError, ChunkStore, build_manifest, chunks_of, materialize, read_chunk
from pool.client import Client, PoolError, RateLimiter, config_dir, pairs, write_config
from pool.jobsocket import JobSocket

LOG_FLUSH_SECONDS = 1.0
LOG_CHUNK = 64 * 1024
PREEMPT_GRACE = 30
SUPERSEDED = "superseded by a newer attempt of the same job"
MEASURE_EVERY = 6 * 3600
SERVER_STREAMS = 4  # chunks fetched from the server at once (the swarm adds PER_PEER per peer on top)
PEER_CHECK_SECONDS = 60
GANG_WAIT_SECONDS = 600
EX_CONFIG = 78  # main_run's exit status when the machine is not (or no longer) registered: the unit does not restart it
# The system install (pool/installer.py); `pool-agent uninstall` removes these.
SYSTEM = {"unit": Path("/etc/systemd/system/pool-agent.service"), "wrapper": Path("/usr/local/bin/pool-agent"),
          "prefix": Path("/opt/pool"), "etc": Path("/etc/pool"), "state": Path("/var/lib/pool")}

BENCH = r"""
import json, time, torch
out = {"tflops": {}, "h2d_gbps": None, "device": None}
def rate(make, n, reps):
    a, b = make(n), make(n)
    for _ in range(3): a @ b
    sync(); start = time.perf_counter()
    for _ in range(reps): a @ b
    sync(); return 2 * n ** 3 * reps / (time.perf_counter() - start) / 1e12
if torch.cuda.is_available():
    sync = torch.cuda.synchronize
    out["device"] = torch.cuda.get_device_name(0)
    for name, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16), ("fp32", torch.float32)):
        torch.backends.cuda.matmul.allow_tf32 = False
        out["tflops"][name] = round(rate(lambda n: torch.randn(n, n, device="cuda", dtype=dtype), 8192, 10), 1)
    torch.backends.cuda.matmul.allow_tf32 = True
    out["tflops"]["tf32"] = round(rate(lambda n: torch.randn(n, n, device="cuda"), 8192, 10), 1)
    host = torch.empty(1 << 28, dtype=torch.float32).pin_memory()
    sync(); start = time.perf_counter()
    for _ in range(5): host.to("cuda", non_blocking=True)
    sync(); out["h2d_gbps"] = round(5 * host.numel() * 4 / (time.perf_counter() - start) / 1e9, 1)
else:
    sync = lambda: None
    out["tflops"]["fp32"] = round(rate(lambda n: torch.randn(n, n), 2048, 5), 2)
print("POOL_BENCH " + json.dumps(out), flush=True)
"""


def size(n: float) -> str:
    return f"{n / 1024**3:.2f} GB" if n >= 1024**3 else f"{n / 1024**2:.1f} MB"


def config_path() -> Path:
    """This machine's agent.json: $POOL_AGENT_CONFIG (the system install: /etc/pool/agent.json, or
    --config), else agent.json in the member settings folder (~/.config/pool)."""
    value = os.environ.get("POOL_AGENT_CONFIG")
    return Path(value) if value else config_dir() / "agent.json"


def pause_path() -> Path:
    """While this file exists (pool-agent pause) the agent takes no new jobs; running ones finish."""
    return config_path().with_suffix(".paused")


def load_agent_config() -> dict:
    path = config_path()
    return json.loads(path.read_text()) if path.is_file() else {}


def save_agent_config(config: dict) -> Path:
    return write_config(config_path(), config)


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def register(server: str, code: str, name: str, labels: dict, gpus: list[int] | None, allow_images: list[str]) -> dict:
    client = Client(server)
    reply = client.call("POST", "/v1/agent/register", {
        "code": code, "name": name, "labels": {**profiles.platform_labels(), **labels},
        "profile": {**profiles.collect(allow_images, gpus), "agent_digest": update.package_digest(Path(__file__).resolve().parent)}})
    config = {"server": server.rstrip("/"), "token": reply["token"], "machine_id": reply["machine_id"],
              "name": reply["name"], "pool": reply.get("pool"), "labels": labels, "gpus": gpus, "allow_images": allow_images}
    path = save_agent_config(config)
    log(f"registered as {reply['name']} ({reply['machine_id']}) in pool {reply.get('pool')}; settings in {path}")
    return config


def run_bench(image: str, gpus: list[int] | None = None) -> dict:
    """TFLOP/s per dtype and host-to-GPU GB/s, measured in a container of `image` (needs PyTorch)."""
    args = ["docker", "run", "--rm", "--network", "none", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", f"{os.getuid()}:{os.getgid()}", "--env", "HOME=/tmp", "--shm-size", "1g"]
    if gpus is None or gpus:
        args += ["--gpus", "all" if gpus is None else '"device=' + ",".join(map(str, gpus)) + '"']
    result = subprocess.run(args + [image, "python", "-c", BENCH], capture_output=True, text=True, timeout=1800)
    for line in result.stdout.splitlines():
        if line.startswith("POOL_BENCH "):
            return {**json.loads(line[len("POOL_BENCH "):]), "image": image, "measured_at": time.time()}
    raise RuntimeError(f"benchmark failed: {(result.stderr or result.stdout).strip()[-400:]}")


def in_hours(window: str | None, at: datetime | None = None) -> bool:
    """Whether local time `at` falls in "HH:MM-HH:MM" (which may wrap past midnight)."""
    if not window:
        return True
    at = at or datetime.now()
    start, end = window.split("-")
    minutes = at.hour * 60 + at.minute
    s = int(start[:2]) * 60 + int(start[3:])
    e = int(end[:2]) * 60 + int(end[3:])
    return s <= minutes < e if s < e else minutes >= s or minutes < e


# ---- the content cache -----------------------------------------------------------------------------


def tree_bytes(manifest: dict) -> int:
    """Disk a materialized tree takes: once built, it is the only copy of its chunks."""
    return sum(f["size"] for f in manifest["files"])


class Cache:
    """Chunks (cache/chunks) and assembled trees (cache/trees/<id>), least recently used out first."""

    def __init__(self, root: Path, quota_bytes: int):
        self.root = Path(root)
        self.chunks = ChunkStore(self.root / "chunks")
        self.trees = self.root / "trees"
        self.trees.mkdir(parents=True, exist_ok=True)
        self.quota = quota_bytes
        self.lock = threading.RLock()
        self.pins: dict[str, int] = {}
        self.building: dict[str, threading.Lock] = {}
        self.added: set[str] = set()
        self.removed: set[str] = set()
        self.index = {"trees": {}, "chunks": {}}  # id -> {"size", "used"}
        self.sources: dict[str, tuple[Path, int, int, str]] = {}  # chunk -> (file in a tree, offset, length, tree)
        self.announce = None  # announce(sha): tell the tracker a chunk landed here (set by the agent)
        self.peer_fetch = None  # peer_fetch(peer, sha, ticket, dest) -> bytes; default: HTTP (peers.fetch_chunk)
        self.last_fetch: dict = {}
        for path in self.chunks.root.glob("*/*"):
            if path.is_file() and not path.name.endswith(".part"):
                self.index["chunks"][path.name] = {"size": path.stat().st_size, "used": path.stat().st_mtime}
        for meta in self.trees.glob("*.json"):
            tree = self.trees / meta.stem
            if tree.is_dir():
                manifest = json.loads(meta.read_text())
                self.index["trees"][meta.stem] = {"size": tree_bytes(manifest), "used": meta.stat().st_mtime,
                                                  "chunks": list(chunks_of(manifest))}
                self._add_sources(meta.stem, manifest)
            else:
                meta.unlink()
        for sha in [sha for sha in self.index["chunks"] if sha in self.sources]:  # a tree already holds these bytes
            self.chunks.delete(sha)
            self.index["chunks"].pop(sha)
        for leftover in self.trees.iterdir():  # half-built trees from a crash
            if leftover.is_dir() and leftover.name not in self.index["trees"]:
                shutil.rmtree(leftover, ignore_errors=True)

    def _add_sources(self, key: str, manifest: dict) -> None:
        """Where each chunk of a built tree can be read back: a new artifact that shares chunks
        with a cached one (a checkpoint with unchanged layers) copies them instead of downloading."""
        for file in manifest["files"]:
            for index, sha in enumerate(file.get("chunks", [])):
                chunk = artifact_format.CHUNK
                length = min(chunk, file["size"] - index * chunk)
                self.sources.setdefault(sha, (self.trees / key / file["path"], index * chunk, length, key))

    def _drop_sources(self, key: str) -> None:
        for sha in [sha for sha, source in self.sources.items() if source[3] == key]:
            self.sources.pop(sha)
        for other, tree in self.index["trees"].items():
            if other != key:
                manifest_path = self.trees / f"{other}.json"
                if manifest_path.is_file() and any(sha not in self.sources for sha in tree["chunks"]):
                    self._add_sources(other, json.loads(manifest_path.read_text()))

    def peer_chunk(self, sha: str) -> bytes | None:
        """A chunk's bytes for a peer: from the chunk store, or read back from a built tree."""
        if not artifact_format.is_hex(sha):
            return None
        with self.lock:
            source = self.sources.get(sha)
        try:
            if self.chunks.has(sha):
                return self.chunks.path(sha).read_bytes()
            if source is not None:
                return read_chunk(source[:3])  # verifies the hash
        except (OSError, ArtifactError):
            return None
        return None

    def peer_has(self, sha: str) -> bool:
        if not artifact_format.is_hex(sha):
            return False
        with self.lock:
            return self.chunks.has(sha) or sha in self.sources

    def held(self) -> list[str]:
        with self.lock:
            out = set(self.index["chunks"])
            for tree in self.index["trees"].values():
                out.update(tree["chunks"])
            return sorted(out)

    def report(self) -> dict:
        with self.lock:
            add, remove = sorted(self.added), sorted(self.removed - self.added)
            self.added.clear()
            self.removed.clear()
        return {"add": add, "remove": remove}

    def report_pending(self) -> bool:
        with self.lock:
            return bool(self.added or self.removed)

    def used_bytes(self) -> int:
        with self.lock:
            return sum(c["size"] for c in self.index["chunks"].values()) + sum(t["size"] for t in self.index["trees"].values())

    def pin(self, artifact: str) -> None:
        with self.lock:
            self.pins[artifact] = self.pins.get(artifact, 0) + 1

    def unpin(self, artifact: str) -> None:
        with self.lock:
            self.pins[artifact] = self.pins.get(artifact, 1) - 1
            if self.pins[artifact] <= 0:
                self.pins.pop(artifact)

    def ensure(self, client: Client, artifact: str, limiter: RateLimiter | None = None, say=None, threads: int = 8,
               peer_sources=None) -> tuple[Path, int]:
        """The artifact's tree, fetching what is missing. Returns (path, bytes downloaded).
        `peer_sources(artifact, chunks)` -> {sha: {"ticket", "peers"}}: peers to try before the server."""
        key = artifact.removeprefix("sha256:")
        with self.lock:
            gate = self.building.setdefault(key, threading.Lock())
        with gate:  # two jobs needing the same model download it once
            tree = self.trees / key
            with self.lock:
                if key in self.index["trees"] and tree.is_dir():
                    self.index["trees"][key]["used"] = time.time()
                    os.utime(self.trees / f"{key}.json")
                    if say:
                        say(f"[pool] {artifact[7:19]} already on this machine ({size(self.index['trees'][key]['size'] or 0)} on disk)\n")
                    return tree, 0
            manifest = client.manifest(artifact)
            needed = chunks_of(manifest)
            missing = []
            reused = 0
            for sha in needed:
                if self.chunks.has(sha):
                    continue
                with self.lock:
                    source = self.sources.get(sha)
                try:
                    if source is None:
                        raise ArtifactError("not cached")
                    self.chunks.put(sha, read_chunk(source[:3]))  # verifies the hash
                    with self.lock:
                        self.index["chunks"][sha] = {"size": needed[sha], "used": time.time()}
                    reused += needed[sha]
                except (OSError, ArtifactError):
                    missing.append(sha)
            if reused and say:
                say(f"[pool] reused {size(reused)} from other artifacts cached here\n")
            fetched = 0
            if missing:
                total = sum(needed[sha] for sha in missing)
                if say:
                    say(f"[pool] fetching {len(missing)} of {len(needed)} chunks, {size(total)}\n")
                started = time.monotonic()
                offers: dict = {}
                if peer_sources is not None:
                    try:
                        offers = peer_sources(artifact, missing) or {}
                    except Exception as exc:  # peers are an optimisation: the server always works
                        log(f"asking for peer sources failed: {exc}")

                def from_peer(peer: dict, sha: str, ticket: str) -> int:
                    try:
                        return (self.peer_fetch or peers.fetch_chunk)(peer, sha, ticket, self.chunks.path(sha))
                    except (OSError, ValueError) as exc:
                        log(f"chunk {sha[:12]} from peer {peer['name']} failed ({exc}); trying the next source")
                        raise

                def refresh(shas: list[str]) -> dict:
                    return peer_sources(artifact, shas) or {}

                swarm = peers.Swarm({sha: needed[sha] for sha in missing}, offers, from_peer,
                                    lambda sha: client.fetch_chunk(sha, artifact, self.chunks.path(sha), limiter),
                                    threads=threads, refresh=refresh if peer_sources is not None else None,
                                    announce=self.announce, server_slots=SERVER_STREAMS)
                stats = swarm.run()
                fetched = stats["bytes"]
                from_peers = stats["peers"]
                with self.lock:
                    for sha in missing:
                        self.index["chunks"][sha] = {"size": needed[sha], "used": time.time()}
                        self.added.add(sha)
                self.last_fetch = stats
                if say:
                    seconds = time.monotonic() - started
                    via = "".join(f", {size(n)} from peer {name}" for name, n in sorted(from_peers.items()))
                    say(f"[pool] fetched {size(fetched)} in {seconds:.1f} s ({fetched * 8 / 1e6 / max(seconds, 1e-3):.0f} Mbit/s)"
                        f"{via}{', the rest from the server' if via and stats['server'] else ''}\n")
            elif say and needed:
                say(f"[pool] all {len(needed)} chunks already on this machine\n")
            building = self.trees / f"{key}.building"
            shutil.rmtree(building, ignore_errors=True)
            materialize(manifest, self.chunks, building)
            building.rename(tree)
            (self.trees / f"{key}.json").write_text(json.dumps(manifest))
            with self.lock:
                self.index["trees"][key] = {"size": tree_bytes(manifest), "used": time.time(), "chunks": list(needed)}
                for sha in needed:  # the tree now holds these bytes (hard links or copies): one copy on disk
                    self.index["chunks"].pop(sha, None)
                    self.chunks.delete(sha)
                self._add_sources(key, manifest)
            self.evict()
            return tree, fetched

    def adopt(self, manifest: dict, folder: Path) -> Path:
        """Keep a folder this machine made as an artifact's tree, so that the peer server serves its
        chunks: a held artifact (the contributions and partials of tree rounds, pool/rounds.py), which
        peers fetch from here and the server never stores. Nothing is downloaded or uploaded."""
        key = artifact_format.manifest_id(manifest).removeprefix("sha256:")
        with self.lock:
            gate = self.building.setdefault(key, threading.Lock())
        with gate:
            tree = self.trees / key
            with self.lock:
                if key in self.index["trees"] and tree.is_dir():
                    self.index["trees"][key]["used"] = time.time()
                    return tree
            building = self.trees / f"{key}.building"
            shutil.rmtree(building, ignore_errors=True)
            building.mkdir(parents=True)
            for directory in manifest.get("dirs", []):
                building.joinpath(*Path(directory).parts).mkdir(parents=True, exist_ok=True)
            for file in manifest["files"]:
                target = building.joinpath(*Path(file["path"]).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(Path(folder).joinpath(*Path(file["path"]).parts), target)  # the job may change its outputs
                os.chmod(target, file["mode"])
            building.rename(tree)
            (self.trees / f"{key}.json").write_text(json.dumps(manifest))
            with self.lock:
                self.index["trees"][key] = {"size": tree_bytes(manifest), "used": time.time(), "chunks": list(chunks_of(manifest))}
                self._add_sources(key, manifest)
            self.evict()
            return tree

    def evict(self) -> None:
        with self.lock:
            while self.used_bytes() > self.quota:
                pinned = {p.removeprefix("sha256:") for p in self.pins}
                loose = sorted(self.index["chunks"].items(), key=lambda item: item[1]["used"])
                trees = sorted(((k, v) for k, v in self.index["trees"].items() if k not in pinned), key=lambda item: item[1]["used"])
                if loose:
                    sha, _ = loose[0]
                    self.chunks.delete(sha)
                    self.index["chunks"].pop(sha)
                    if not any(sha in t["chunks"] for t in self.index["trees"].values()):
                        self.removed.add(sha)
                elif trees:
                    key, tree = trees[0]
                    shutil.rmtree(self.trees / key, ignore_errors=True)
                    (self.trees / f"{key}.json").unlink(missing_ok=True)
                    self.index["trees"].pop(key)
                    self._drop_sources(key)
                    still = set(self.held())
                    self.removed.update(sha for sha in tree["chunks"] if sha not in still)
                else:
                    break  # everything left is in use


# ---- runtimes ----------------------------------------------------------------------------------------


class GangTimeout(Exception):
    pass


class JobStopped(Exception):
    """Cancelled or preempted before the job's process started."""


class ImageUnavailable(RuntimeError):
    """The job's image is not on this machine and cannot be pulled (a private registry without a
    login, say). The job's error then starts with "ImageUnavailable:", so a submitter can tell a
    machine that could not start the job from a job that ran and failed."""


def gpu_note(job: dict) -> str:
    """How a job's GPUs read in its log: "GPUs [0]", or why it has none."""
    if job.get("gpus"):
        return f"GPUs {job['gpus']}"
    if not (job["spec"].get("resources") or {}).get("gpus"):
        return "no GPU requested"
    return "GPUs none"


class Running:
    def __init__(self, job: dict):
        self.job = job
        self.proc: subprocess.Popen | None = None
        self.rendezvous: dict | None = None  # the gang rendezvous reply (Agent.gang_env)
        self.stop_reason: str | None = None
        self.preempted: str | None = None
        self.stopped_at: float | None = None  # when stop or preemption was requested


class DockerRuntime:
    """`docker run` with gymhub's sandbox hardening."""

    name = "docker"

    def __init__(self, machine_id: str = "", pids: int = 4096, tmp_gb: int = 8, shm_gb: int = 4):
        self.machine_id = machine_id
        self.pids, self.tmp_gb, self.shm_gb = pids, tmp_gb, shm_gb

    def cleanup(self) -> int:
        """Kill containers left by an earlier run of this agent (a crash, a reboot of the service):
        their jobs were requeued by the server when the leases ran out."""
        result = subprocess.run(["docker", "ps", "-q", "--filter", f"label=pool.machine={self.machine_id}"],
                                capture_output=True, text=True)
        stale = result.stdout.split()
        if stale:
            subprocess.run(["docker", "kill", *stale], capture_output=True)
        return len(stale)

    def container_ids(self) -> set[str]:
        result = subprocess.run(["docker", "ps", "-q", "--no-trunc", "--filter", f"label=pool.machine={self.machine_id}"],
                                capture_output=True, text=True)
        return set(result.stdout.split())

    def prepare(self, image: str, say) -> None:
        if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0:
            return
        say(f"[pool] pulling {image} (only the first time on this machine)\n")
        result = subprocess.run(["docker", "pull", "-q", image], capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            raise ImageUnavailable(f"image {image} is not on this machine and could not be pulled: "
                                   f"{result.stderr.strip()[-400:]}")

    def name_of(self, job: dict) -> str:
        return f"pool-{job['id']}-{job['tries']}"

    def daemon_cpus(self) -> int | None:
        """CPUs the Docker daemon has: on Docker Desktop its VM's, which can be fewer than the host's."""
        if not hasattr(self, "_daemon_cpus"):
            try:
                out = subprocess.run(["docker", "info", "--format", "{{.NCPU}}"], capture_output=True, text=True, timeout=30)
                self._daemon_cpus = int(out.stdout.strip()) if out.returncode == 0 else None
            except (OSError, ValueError, subprocess.TimeoutExpired):
                self._daemon_cpus = None
        return self._daemon_cpus

    def cpus_arg(self, cpus: float) -> str:
        """`docker run --cpus` refuses more CPUs than the daemon has, so cap the job's request there."""
        available = self.daemon_cpus()
        return f"{min(float(cpus), available) if available else float(cpus):g}"

    def command(self, job: dict, dirs: dict, env: dict, memory_gb: float) -> list[str]:
        spec, res = job["spec"], job["spec"]["resources"]
        network = "host" if spec.get("network") == "peers" else "none"
        if job.get("island_holder"):  # network: island (pool/island.py): the holder's namespace
            network = f"container:{job['island_holder']}"
        args = [
            "docker", "run", "--rm", "--init", "--name", self.name_of(job),
            "--label", f"pool.machine={self.machine_id}", "--label", f"pool.job={job['id']}",
            "--network", network, "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--user", f"{os.getuid()}:{os.getgid()}",
            "--memory", f"{int(memory_gb * 1024)}m", "--memory-swap", f"{int(memory_gb * 1024)}m",
            "--cpus", self.cpus_arg(res["cpus"]), "--pids-limit", str(res.get("pids") or self.pids),
            "--tmpfs", f"/tmp:rw,exec,nosuid,size={int((res.get('tmp_gb') or self.tmp_gb) * 1024)}m",
            "--shm-size", f"{int((res.get('shm_gb') or self.shm_gb) * 1024)}m",
            "--workdir", "/work" if dirs.get("code") else "/outputs",
        ]
        if job["gpus"]:
            args += ["--gpus", '"device=' + ",".join(str(index) for index in job["gpus"]) + '"']
        if dirs.get("code"):
            args += ["--mount", f"type=bind,src={dirs['code']},dst=/work,readonly"]
        for path, tree in dirs["inputs"].items():
            args += ["--mount", f"type=bind,src={tree},dst=/inputs/{path},readonly"]
        args += ["--mount", f"type=bind,src={dirs['outputs']},dst=/outputs"]
        args += ["--mount", f"type=bind,src={dirs['checkpoint']},dst=/checkpoint"]
        if dirs.get("socket"):  # refs and queues (pool/jobsocket.py)
            args += ["--mount", f"type=bind,src={dirs['socket']},dst=/run/pool",
                     "--mount", f"type=bind,src={dirs['refs']},dst=/refs,readonly"]
        container_env = {"HOME": "/tmp", "PYTHONUNBUFFERED": "1", **spec["env"],
                         "POOL_WORK": "/work", "POOL_INPUTS": "/inputs", "POOL_OUTPUTS": "/outputs",
                         "POOL_CHECKPOINT": "/checkpoint", **env}
        for key, value in container_env.items():
            args += ["--env", f"{key}={value}"]
        return args + [job.get("image") or spec["image"], *spec["command"]]

    def kill(self, running: Running) -> None:
        subprocess.run(["docker", "kill", self.name_of(running.job)], capture_output=True, timeout=60)

    def terminate(self, running: Running, grace: int) -> None:
        subprocess.run(["docker", "stop", "-t", str(grace), self.name_of(running.job)], capture_output=True, timeout=grace + 60)


class ProcessRuntime:
    """No container: the command runs as a plain process on this machine. For tests and for a
    machine whose owner is the only member; never for other people's code."""

    name = "process"

    def prepare(self, image: str, say) -> None:
        pass

    def cleanup(self) -> int:
        return 0

    def container_ids(self) -> set[str]:
        return set()

    def command(self, job: dict, dirs: dict, env: dict, memory_gb: float) -> list[str]:
        return list(job["spec"]["command"])

    def cwd(self, dirs: dict) -> Path:
        return dirs.get("code") or dirs["outputs"]

    def kill(self, running: Running) -> None:
        if running.proc and running.proc.poll() is None:
            try:
                os.killpg(running.proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                running.proc.kill()

    def terminate(self, running: Running, grace: int) -> None:
        if running.proc and running.proc.poll() is None:
            try:
                os.killpg(running.proc.pid, signal.SIGTERM)
                running.proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                self.kill(running)
            except (ProcessLookupError, PermissionError):
                pass


RAY_SCRIPT = r"""
set -e
if [ "$RANK" = 0 ]; then
  ray start --head --port="$MASTER_PORT" --num-gpus="$POOL_GPUS" --disable-usage-stats $POOL_RAY_ARGS
  python3 - <<'PY'
import os, time
import ray
ray.init(address="auto")
deadline = time.time() + 600
while len([n for n in ray.nodes() if n.get("Alive")]) < int(os.environ["WORLD_SIZE"]):
    if time.time() > deadline:
        raise SystemExit("not every rank joined the Ray cluster in 600 s")
    time.sleep(1)
PY
  status=0
  "$@" || status=$?
  ray stop --force > /dev/null 2>&1 || true
  exit $status
else
  # joins the head and stays until the head stops (rank 0's command ended)
  ray start --address="$MASTER_ADDR:$MASTER_PORT" --num-gpus="$POOL_GPUS" --block --disable-usage-stats $POOL_RAY_ARGS || true
fi
"""


def runtime_command(spec: dict, env: dict, gpus: list) -> list[str]:
    """The command a gang rank runs (spec runtime, M5): as given, under torchrun, or in a Ray cluster.

    torchrun: one node per rank, one process per GPU, rendezvous at rank 0 (MASTER_ADDR/PORT).
    ray: rank 0 starts a head, waits for every rank, runs the command as the driver, then stops
    Ray; the other ranks join and stay until the head goes away. Ray never spans islands: a gang
    runs in one."""
    command = list(spec["command"])
    runtime = spec.get("runtime", "plain")
    if runtime == "torchrun":
        head = ["torchrun", f"--nnodes={env['WORLD_SIZE']}", f"--node-rank={env['RANK']}",
                f"--nproc-per-node={max(1, len(gpus))}", f"--master-addr={env['MASTER_ADDR']}",
                f"--master-port={env['MASTER_PORT']}"]
        if os.path.basename(command[0]).startswith("python"):
            return head + command[1:]  # torchrun runs the script with Python itself
        return head + ["--no-python"] + command
    if runtime == "ray":
        return ["sh", "-c", RAY_SCRIPT, "pool-ray", *command]
    return command


def name_of(ref: dict) -> str:
    return f"{ref.get('name')}@{ref.get('version')}"


def foreign_gpu_use(ours: set[str]) -> list[int]:
    """PIDs using a GPU that do not belong to this agent's containers: the owner's own work."""
    out = profiles._run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"])
    foreign = []
    for line in (out or "").splitlines():
        line = line.strip()
        if not line.isdigit():
            continue
        try:
            cgroup = Path(f"/proc/{line}/cgroup").read_text()
        except OSError:
            continue
        if not any(container in cgroup for container in ours):
            foreign.append(int(line))
    return foreign


class Agent:
    def __init__(self, config: dict, work: Path, runtime=None, max_jobs: int = 4, cpus: float | None = None,
                 memory_gb: float | None = None, max_output_mb: int = 16 * 1024, cache_gb: float | None = None,
                 network_mbps: float | None = None, limit_mbps: float | None = None, foreign=None,
                 peer_port: int | None = peers.DEFAULT_PORT, allow_peer_network: bool = False, peer_addr: str | None = None,
                 iroh: dict | None = None, peer_limit_mbps: float | None = None):
        self.config = config
        self.client = Client(config["server"], config["token"])
        self.work = Path(work)
        self.runtime = runtime or DockerRuntime(config.get("machine_id", ""))
        self.max_jobs = max_jobs
        self.profile = profiles.collect(config.get("allow_images"), config.get("gpus"))
        if config.get("bench"):
            self.profile["bench"] = config["bench"]
        self.cpus = cpus or float(self.profile["cpus"])
        self.memory_gb = memory_gb or round(self.profile["memory_gb"] * 0.9, 1)
        self.max_output = max_output_mb * 1024**2
        self.warming: set[str] = set()  # ref artifacts fetched ahead of time (Agent.warm)
        free_disk = shutil.disk_usage(self.work if self.work.exists() else Path.home()).free
        quota = int(cache_gb * 1024**3) if cache_gb else max(10 * 1024**3, int(free_disk * 0.3))
        self.cache = Cache(self.work / "cache", quota)
        self.network_mbps = network_mbps
        self.limiter = RateLimiter(limit_mbps) if limit_mbps else None
        self.foreign = foreign or (lambda: foreign_gpu_use(self.runtime.container_ids()))
        self.running: dict[str, Running] = {}
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.drain = False
        self.policy: dict = {}
        self.paused: str | None = None
        self.pause_file = pause_path()
        self.owner_paused = False
        self.removed = False
        self.lease_wait = 25.0  # seconds one lease request waits for work: a pause takes effect within this
        self.poll_seconds = 10.0
        self.measured_at = 0.0
        self.peer_port = peer_port
        self.allow_peer_network = allow_peer_network
        self.peer_addr = peer_addr
        self.peer_key: str | None = None
        self.peer_server: peers.PeerServer | None = None
        self.peer_bind = "0.0.0.0"
        self.peer_links: dict[str, dict] = {}  # machine id -> {"mbps", "rtt_ms", "at"}
        self.peers_checked_at = 0.0
        self.peer_limit_mbps = peer_limit_mbps
        self.iroh = iroh  # {"binary", "image", "port", "lan", "bind"}: start the iroh-blobs sidecar (peers.IrohSidecar)
        self.sidecar: peers.IrohSidecar | None = None
        self.peer_bytes = {"http": 0, "iroh": 0}  # chunk bytes received from peers, by transport
        self.peer_bytes_lock = threading.Lock()
        self.cache.announce = self.announce_chunk
        self.cache.peer_fetch = self.fetch_from_peer
        self.package_root = Path(__file__).resolve().parent  # pool/update.py: keeping up with the server
        self.own_digest = update.package_digest(self.package_root)
        self.server_digest: str | None = None
        self.auto_update = bool(config.get("auto_update"))
        self.restart = False
        self.update_noted: str | None = None
        # The agent's version in its profile, so the server can tell an outdated agent (update.agent_version).
        self.profile.update({"agent_digest": self.own_digest, "auto_update": self.auto_update})
        self.profile_changed = False  # sent with the next heartbeat rather than at the next measurement

    # ---- profile -----------------------------------------------------------------------------------
    def measure_network(self) -> dict:
        rtt = []
        for _ in range(3):
            start = time.monotonic()
            self.client.call("GET", "/healthz", timeout=10)
            rtt.append((time.monotonic() - start) * 1000)
        size = 16 << 20
        start = time.monotonic()
        with self.client._request("GET", f"/v1/agent/speedtest?bytes={size}", timeout=300) as reply:
            got = 0
            while True:
                block = reply.read(1 << 20)
                if not block:
                    break
                if self.limiter:
                    self.limiter.take(len(block))
                got += len(block)
        down = got * 8 / 1e6 / max(time.monotonic() - start, 1e-3)
        payload = os.urandom(8 << 20)
        start = time.monotonic()
        with self.client._request("POST", "/v1/agent/speedtest", payload, {"Content-Type": "application/octet-stream"}, timeout=300) as reply:
            reply.read()
        up = len(payload) * 8 / 1e6 / max(time.monotonic() - start, 1e-3)
        network = {"rtt_ms": round(min(rtt), 1), "down_mbps": round(down, 1), "up_mbps": round(up, 1), "measured_at": time.time()}
        if self.network_mbps:  # declared by the owner: wins over a measurement against a nearby server
            network.update({"down_mbps": self.network_mbps, "up_mbps": self.network_mbps, "declared": True})
        return network

    def refresh_profile(self) -> None:
        try:
            self.profile["network"] = self.measure_network()
        except PoolError as exc:
            log(f"network measurement failed: {exc}")
        self.profile["cache_gb"] = round(self.cache.quota / 1024**3, 1)
        self.measured_at = time.time()

    # ---- peers ---------------------------------------------------------------------------------------
    def start_peer_server(self) -> None:
        """Serve chunks and speed tests to the pool's other agents, and advertise where."""
        if self.peer_port is None:
            return
        addr = self.peer_addr or peers.local_address(self.client.server)
        if addr and not peers.private_address(addr):
            log(f"warning: peer address {addr} is not a private (LAN) address; peers outside the LAN are refused")
        self.peer_server = None
        for host in dict.fromkeys([peers.bind_address(addr), "0.0.0.0"]):
            try:
                self.peer_server = peers.PeerServer(self.peer_port, lambda: self.peer_key, self.cache.peer_chunk, host=host,
                                                    has_chunk=self.cache.peer_has, limit_mbps=self.peer_limit_mbps,
                                                    log=log).start()
                self.peer_bind = host
                break
            except OSError as exc:
                error = exc
        if self.peer_server is None:
            log(f"peer server not started (port {self.peer_port}: {error}); this machine only talks to the server")
            return
        self.profile["peer"] = {"addr": addr, "port": self.peer_server.port, "network": self.allow_peer_network,
                                "ifname": peers.interface_of(addr)}
        can, why = self.island_support(addr)
        self.profile["peer"].update({"island": can, "island_why": why})
        log(f"peer server on {addr}:{self.peer_server.port} (listening on {self.peer_bind}, LAN sources only)" + (" (peer networking allowed for gangs)" if self.allow_peer_network else ""))
        if self.iroh is not None:
            self.start_sidecar()

    def start_sidecar(self) -> None:
        """The optional iroh-blobs sidecar: chunks between machines over QUIC (hole punching, relays).
        Without it, or when it fails, peers use the plain HTTP peer server."""
        options = self.iroh or {}
        try:
            self.sidecar = peers.IrohSidecar.launch(self.peer_server.port, port=options.get("port") or peers.SIDECAR_PORT,
                                                    binary=options.get("binary"), image=options.get("image"),
                                                    lan=bool(options.get("lan")), bind=options.get("bind"),
                                                    agent_host=self.peer_bind)
            self.profile["peer"]["iroh"] = self.sidecar.status()
        except OSError as exc:
            log(f"iroh sidecar not started ({exc}); peers use HTTP")
            self.sidecar = None
            return
        log(f"iroh sidecar {self.profile['peer']['iroh']['id'][:12]} on control port {self.sidecar.port}")

    def stop_sidecar(self) -> None:
        if self.sidecar is not None:
            self.sidecar.stop()
            self.sidecar = None

    def fetch_from_peer(self, peer: dict, sha: str, ticket: str, dest: Path) -> int:
        """One chunk from a peer: through the iroh sidecars when both machines run one, else HTTP."""
        if self.sidecar is not None and peer.get("iroh"):
            try:
                got = self.sidecar.fetch(peer, sha, ticket, dest)
                with self.peer_bytes_lock:
                    self.peer_bytes["iroh"] += got
                return got
            except (OSError, ValueError) as exc:
                log(f"chunk {sha[:12]} from {peer.get('name')} over iroh failed ({exc}); trying HTTP")
        got = peers.fetch_chunk(peer, sha, ticket, dest)
        with self.peer_bytes_lock:
            self.peer_bytes["http"] += got
        return got

    def announce_chunk(self, sha: str) -> None:
        """Tell the tracker at once that a chunk landed here, so machines fetching the same artifact
        right now can take it from this one (the heartbeat would report it only later)."""
        if self.peer_server is not None:
            self.client.call("POST", "/v1/agent/chunks", {"add": [sha]}, timeout=10)

    def island_support(self, addr: str) -> tuple[bool, str]:
        """Whether gang jobs with network: island can run here (pool/island.py)."""
        if self.runtime.name != "docker":
            return False, "jobs here run without containers"
        if self.config.get("island") is False:
            return False, "turned off by the machine's owner"
        try:
            island.ipv4(addr)
        except ValueError:
            return False, f"peer address {addr} is not IPv4"
        if not peers.private_address(addr):
            return False, f"peer address {addr} is not a private (LAN) address"
        return island.available()

    def check_peers(self, force: bool = False) -> None:
        """Measure the link to peers not measured yet (or all, every MEASURE_EVERY) and report them."""
        if self.peer_server is None or self.peer_key is None:
            return
        self.peers_checked_at = time.time()
        found = self.client.call("GET", "/v1/agent/peers", timeout=30).get("peers", [])
        report = []
        for peer in found:
            known = self.peer_links.get(peer["id"])
            if known and not force and time.time() - known["at"] < MEASURE_EVERY:
                continue
            try:
                link = peers.measure(peer, self.peer_key, self.config["machine_id"])
            except OSError as exc:
                log(f"peer {peer['name']} at {peer['addr']}:{peer['port']} unreachable: {exc}")
                continue
            self.peer_links[peer["id"]] = {**link, "at": time.time()}
            report.append({"peer": peer["id"], **link})
            log(f"link to {peer['name']}: {link['mbps']:g} Mbit/s, {link['rtt_ms']:g} ms")
        if report:
            self.client.call("POST", "/v1/agent/links", {"links": report}, timeout=30)

    def peer_sources(self, artifact: str, chunks: list[str]) -> dict:
        if self.peer_server is None:
            return {}
        return self.client.call("POST", "/v1/agent/peer-sources", {"artifact": artifact, "chunks": chunks}, timeout=30).get("sources", {})

    def gang_env(self, job: dict, running: "Running", say, flush) -> dict:
        """Wait until every rank of the gang has a machine; return the rendezvous environment."""
        deadline = time.monotonic() + GANG_WAIT_SECONDS
        announced = None
        while True:
            if running.stop_reason:
                raise JobStopped(running.stop_reason)
            reply = self.client.call("POST", f"/v1/agent/jobs/{job['id']}/rendezvous", {}, timeout=30)
            if reply.get("stop"):
                raise JobStopped("the gang was stopped")
            if reply.get("ready"):
                break
            if reply.get("waiting") != announced:
                announced = reply.get("waiting")
                say(f"[pool] gang {job['spec']['gang']['id']}: waiting for rank(s) {announced} to get a machine\n")
                flush()
            if time.monotonic() > deadline:
                raise GangTimeout(f"the gang did not assemble in {GANG_WAIT_SECONDS} s")
            time.sleep(2)
        say(f"[pool] gang {reply['gang']}: rank {reply['rank']} of {reply['world_size']}; "
            + ", ".join(f"r{p['rank']}={p['machine']}({p['addr']})" for p in reply["peers"]) + "\n")
        env = {"RANK": str(reply["rank"]), "WORLD_SIZE": str(reply["world_size"]), "LOCAL_RANK": "0",
               "LOCAL_WORLD_SIZE": "1", "MASTER_ADDR": reply["master_addr"], "MASTER_PORT": str(reply["master_port"]),
               "POOL_GANG_ID": reply["gang"], "POOL_PEERS": json.dumps(reply["peers"])}
        running.rendezvous = reply
        if job["spec"].get("network") == "island":
            lo, hi = reply["ports"]
            addr = (self.profile.get("peer") or {}).get("addr")
            env.update(island.job_env(addr, lo, hi))
            if job["spec"].get("runtime") == "ray":
                env["POOL_RAY_ARGS"] = island.ray_args(lo, addr)
            return env
        ifname = (self.profile.get("peer") or {}).get("ifname")
        if ifname:
            env.update({"GLOO_SOCKET_IFNAME": ifname, "NCCL_SOCKET_IFNAME": ifname})
        return env

    # ---- capacity and policy -----------------------------------------------------------------------
    def free(self) -> dict:
        with self.lock:
            jobs = [r.job for r in self.running.values()]
        used_gpus = {index for job in jobs for index in job["gpus"]}
        exclusive = {index for job in jobs if not job["spec"]["resources"].get("share_gpu") for index in job["gpus"]}
        reserved: dict[int, float] = {}
        for job in jobs:
            if job["spec"]["resources"].get("share_gpu"):
                for index in job["gpus"]:
                    reserved[index] = reserved.get(index, 0.0) + job["spec"]["resources"]["gpu_memory_gb"]
        sizes = {gpu["index"]: float(gpu.get("memory_gb") or 0) for gpu in self.profile["gpus"]}
        return {
            "gpus": [gpu["index"] for gpu in self.profile["gpus"] if gpu["index"] not in used_gpus],
            # GPUs that hold only shared jobs, and the GPU memory still free on each (resources.share_gpu)
            "shared_gpus": {str(index): round(sizes.get(index, 0.0) - used, 2) for index, used in reserved.items()
                            if index not in exclusive},
            "cpus": round(self.cpus - sum(job["spec"]["resources"]["cpus"] for job in jobs), 2),
            "memory_gb": round(self.memory_gb - sum(self.job_memory(job) for job in jobs), 2),
        }

    def job_memory(self, job: dict) -> float:
        """Memory a job's container may use: what it asked for, or what its model fit needs."""
        res = job["spec"]["resources"]
        memory = res["memory_gb"]
        fit = res.get("fits")
        if fit and fit.get("meta"):
            need = models.need(fit["meta"], fit["mode"], fit["kv_gb"], res["gpus"])
            memory = max(memory, need["ram_gb"])
            if any(gpu.get("unified_memory") for gpu in self.profile["gpus"]):
                memory = max(memory, need["ram_gb"] + need["gpu_gb"] * max(res["gpus"], 1))
        return memory

    def check_policy(self) -> None:
        """Pause leasing, and preempt running jobs, when the use policy stops allowing work."""
        reason = None
        if not in_hours(self.policy.get("hours")):
            reason = f"outside the machine's hours ({self.policy['hours']})"
        elif self.policy.get("idle_only"):
            foreign = self.foreign()
            if foreign:
                reason = f"the owner is using the GPU (pid {foreign[0]})"
        if reason != self.paused:
            log(f"paused: {reason}" if reason else "resumed: the use policy allows jobs again")
        self.paused = reason
        if reason:
            with self.lock:
                running = list(self.running.values())
            for item in running:
                if item.stop_reason is None:
                    item.preempted = reason
                    item.stop_reason = f"preempted: {reason}"
                    item.stopped_at = time.monotonic()
                    log(f"{item.job['id']}: preempting ({reason})")
                    threading.Thread(target=self.runtime.terminate, args=(item, PREEMPT_GRACE), daemon=True).start()

    def check_owner_pause(self) -> bool:
        """True while the owner paused this machine (pool-agent pause): no new leases; running jobs finish."""
        paused = self.pause_file.exists()
        if paused != self.owner_paused:
            log("paused by the owner (pool-agent resume to take jobs again); running jobs finish" if paused
                else "resumed by the owner")
        self.owner_paused = paused
        return paused

    def labels(self) -> dict:
        """The owner's labels over the automatic arch and gpu ones (which the owner may override)."""
        return {**profiles.platform_labels(self.profile["gpus"]), **(self.config.get("labels") or {})}

    # ---- loops -------------------------------------------------------------------------------------
    def heartbeat_once(self, send_profile: bool = False, full_cache: bool = False) -> None:
        with self.lock:
            running = list(self.running)
        body = {"running": running, "labels": self.labels(), "free": self.free()}
        if send_profile:
            body["profile"] = dict(self.profile)
        body["cache"] = {"full": True, "add": self.cache.held(), "remove": []} if full_cache else self.cache.report()
        if full_cache:
            self.cache.report()  # the full list supersedes pending changes
        reply = self.client.call("POST", "/v1/agent/heartbeat", body, timeout=60)
        if send_profile and body["profile"] == self.profile:
            self.profile_changed = False
        self.poll_seconds = max(1.0, reply.get("poll_ms", 10000) / 1000)
        self.drain = bool(reply.get("drain"))
        self.policy = reply.get("policy") or {}
        if reply.get("peer_key"):
            self.peer_key = reply["peer_key"]
        self.server_digest = reply.get("agent") or self.server_digest
        for job_id in reply.get("cancel", []):
            self.stop(job_id, "cancelled")
        self.warm(reply.get("warm") or [])
        self.check_policy()

    def warm(self, refs: list[dict]) -> None:
        """Fetch new versions of refs that running jobs subscribe to, before they ask (fan-out from the origin)."""
        for ref in refs:
            artifact = ref.get("artifact")
            with self.lock:
                if not artifact or artifact in self.warming:
                    continue
                self.warming.add(artifact)

            def fetch(artifact=artifact, name=ref.get("name"), version=ref.get("version")):
                try:
                    self.cache.ensure(self.client, artifact, self.limiter, peer_sources=self.peer_sources)
                    log(f"warmed {name} v{version}")
                except Exception as exc:  # the job fetches it itself when it asks
                    log(f"warming {name} v{version} failed: {exc}")
                    with self.lock:
                        self.warming.discard(artifact)

            threading.Thread(target=fetch, daemon=True, name=f"warm-{name_of(ref)}").start()

    def heartbeat_loop(self) -> None:
        while not self.stopping.is_set():
            self.stopping.wait(self.poll_seconds)
            if self.stopping.is_set():
                break
            try:
                due = time.time() - self.measured_at > MEASURE_EVERY
                if due:
                    self.refresh_profile()
                self.heartbeat_once(send_profile=due or self.profile_changed)
                if time.time() - self.peers_checked_at > PEER_CHECK_SECONDS:
                    threading.Thread(target=self._check_peers_quietly, args=(due,), daemon=True, name="peers").start()
            except PoolError as exc:
                if exc.status == 401:
                    log("this machine was removed from the pool; stopping")
                    self.removed = True
                    self.stopping.set()
                    return
                log(f"heartbeat failed: {exc}")

    def _check_peers_quietly(self, force: bool) -> None:
        try:
            self.check_peers(force)
        except PoolError as exc:
            log(f"peer check failed: {exc}")

    def lease_once(self, wait: float | None = None) -> dict | None:
        wait = self.lease_wait if wait is None else wait
        with self.lock:
            busy = len(self.running) >= self.max_jobs
        if busy or self.drain or self.paused or self.check_owner_pause():
            return None
        jobs = self.client.call("POST", "/v1/agent/lease", {"free": self.free(), "wait": wait}, timeout=wait + 30).get("jobs", [])
        if not jobs:
            return None
        job = jobs[0]
        running = Running(job)
        with self.lock:
            stale = self.running.get(job["id"])
            self.running[job["id"]] = running
        if stale is not None:
            # The server gave this job out again (its lease lapsed, say after a network blip) and we got it:
            # the older attempt must stop, and must not report a result for the new one.
            stale.stop_reason = SUPERSEDED
            stale.stopped_at = time.monotonic()
            log(f"{job['id']}: leased again (attempt {job['tries']}); stopping the older attempt")
            try:
                self.runtime.kill(stale)
            except Exception as exc:
                log(f"{job['id']}: kill failed: {exc}")
        threading.Thread(target=self.execute, args=(running,), daemon=True, name=f"job-{job['id']}").start()
        return job

    def run_forever(self) -> None:
        stale = self.runtime.cleanup()
        if stale:
            log(f"stopped {stale} container(s) left from an earlier run")
        self.start_peer_server()
        self.refresh_profile()
        network = self.profile.get("network") or {}
        log(f"agent {self.config.get('name')} ({self.runtime.name}): {len(self.profile['gpus'])} GPU(s), "
            f"{self.cpus:g} CPUs, {self.memory_gb:g} GB for jobs; cache {self.cache.used_bytes() / 1024**3:.1f} of "
            f"{self.cache.quota / 1024**3:.0f} GB; link {network.get('down_mbps', '?')} Mbit/s down; server {self.client.server}")
        try:
            self.heartbeat_once(send_profile=True, full_cache=True)
        except PoolError as exc:
            if exc.status != 401:
                raise
            log("this machine was removed from the pool; stopping")
            self.removed = True
            self.stopping.set()
            self.stop_sidecar()
            return
        threading.Thread(target=self.heartbeat_loop, daemon=True, name="heartbeat").start()
        backoff = 1.0
        while not self.stopping.is_set():
            try:
                with self.lock:
                    full = len(self.running) >= self.max_jobs
                if self.maybe_update():
                    break
                if full or self.drain or self.paused or self.check_owner_pause():
                    self.stopping.wait(1)
                    continue
                self.lease_once()
                backoff = 1.0
            except PoolError as exc:
                if exc.status == 401:
                    log("this machine was removed from the pool; stopping")
                    self.removed = True
                    self.stopping.set()
                    break
                log(f"lease failed: {exc}; retrying in {backoff:.0f}s")
                self.stopping.wait(backoff)
                backoff = min(backoff * 2, 60)
        for job_id in list(self.running):
            self.stop(job_id, "agent stopped")
        self.stop_sidecar()

    def note_update(self, key: str, message: str) -> None:
        if self.update_noted != key:
            self.update_noted = key
            log(message)

    def report_update_error(self, message: str) -> None:
        """Put a failed self-update in the profile, for the next heartbeat. A success needs no report:
        the agent restarts, and its first heartbeat carries the new digest (and no error)."""
        if (self.profile.get("agent_update_error") or {}).get("message") != message:
            self.profile["agent_update_error"] = {"message": message, "at": round(time.time())}
            self.profile_changed = True

    def maybe_update(self) -> bool:
        """Install the server's agent package when it differs from ours (pool/update.py). Only with
        auto-update on, only between jobs. True when the agent should restart into the new files."""
        wanted = self.server_digest
        if not wanted or wanted == self.own_digest:
            return False
        if not self.auto_update:
            self.note_update(wanted, "the server has a newer agent: run `pool agent update` (or `--auto`), then restart")
            return False
        if in_container():
            self.note_update(wanted, "the server has a newer agent: run the installer again (or docker compose pull "
                                     "&& docker compose up -d) to restart the agent's container with it")
            return False
        if update.source_checkout(self.package_root):
            self.note_update(wanted, f"the server has a newer agent, but {self.package_root} is a source checkout: update it with git")
            return False
        with self.lock:
            busy = len(self.running)
        if busy:
            self.note_update(wanted, f"a newer agent is waiting until this machine's {busy} job(s) finish")
            return False
        try:
            data = self.client._request("GET", "/v1/agent/package.tar.gz", timeout=120).read()
            update.install(data, self.package_root, expect=wanted)
        except (PoolError, OSError, ValueError, RuntimeError) as exc:
            self.note_update(wanted, f"agent update failed: {exc}")
            self.report_update_error(update.short_error(exc, self.package_root))
            return False
        log(f"installed agent {wanted[:12]} (was {self.own_digest[:12]}); restarting")
        self.restart = True
        self.stopping.set()
        return True

    # ---- one job -----------------------------------------------------------------------------------
    def stop(self, job_id: str, reason: str) -> None:
        with self.lock:
            running = self.running.get(job_id)
        if running and running.stop_reason is None:
            running.stop_reason = reason
            running.stopped_at = time.monotonic()
            log(f"{job_id}: stopping ({reason})")
            try:
                self.runtime.kill(running)
            except Exception as exc:  # the process may already be gone
                log(f"{job_id}: kill failed: {exc}")

    def execute(self, running: Running) -> None:
        job = running.job
        attempt = f"{job['tries']}"
        root = self.work / "jobs" / f"{job['id']}-{attempt}"
        seq = [0]
        pending: list[str] = []
        pending_lock = threading.Lock()
        pinned: list[str] = []

        def say(text: str) -> None:
            with pending_lock:
                pending.append(text)

        def flush() -> None:
            with pending_lock:
                text = "".join(pending)
                pending.clear()
            while text:
                piece, text = text[:LOG_CHUNK], text[LOG_CHUNK:]
                seq[0] += 1
                try:
                    reply = self.client.call("POST", "/v1/agent/progress",
                                             {"job": job["id"], "key": f"{attempt}:{seq[0]}", "text": piece}, timeout=30)
                    if reply.get("stop") and running.stop_reason is None:
                        self.stop(job["id"], "cancelled")
                except PoolError as exc:
                    log(f"{job['id']}: log upload failed: {exc}")

        exit_code, error, outputs = None, None, None
        job_socket = None
        island_job = None  # the job whose island holder runs (network: island)
        try:
            log(f"{job['id']}: {job['spec']['name']}, {gpu_note(job)} (attempt {attempt})")
            dirs: dict = {"inputs": {}, "outputs": root / "outputs", "checkpoint": root / "checkpoint"}
            for path in (dirs["outputs"], dirs["checkpoint"]):
                path.mkdir(parents=True, exist_ok=True)
            placement = job.get("placement") or {}
            say(f"[pool] on {self.config.get('name')}, {gpu_note(job)}, attempt {attempt}"
                + (f"; inputs {size(placement['input_bytes'])}, {size(placement['missing_bytes'])} to fetch"
                   if placement.get("input_bytes") else "") + "\n")
            wanted = ([("code", job["spec"]["code"])] if job["spec"].get("code") else []) + \
                     [(item["path"], item["artifact"]) for item in job["spec"]["inputs"]]
            for slot, artifact in wanted:
                self.cache.pin(artifact)
                pinned.append(artifact)
                tree, _ = self.cache.ensure(self.client, artifact, self.limiter, say, peer_sources=self.peer_sources)
                if slot == "code" and job["spec"].get("code") == artifact and "code" not in dirs:
                    dirs["code"] = tree
                else:
                    dirs["inputs"][slot] = tree
                flush()
            if self.cache.report_pending():
                self.heartbeat_once()  # tell the server what is cached now, before the job runs for hours
            job["image"] = image_for(job["spec"], self.labels())
            self.runtime.prepare(job["image"], say)
            flush()
            env = {"POOL_JOB_ID": job["id"], "POOL_JOB_ATTEMPT": attempt, "POOL_MACHINE": self.config.get("machine_id", "")}
            if (job["spec"].get("gang") or {}).get("id"):
                env.update(self.gang_env(job, running, say, flush))
                if job["spec"].get("runtime", "plain") != "plain":
                    env["POOL_GPUS"] = str(len(job["gpus"] or []))
                    job = {**job, "spec": {**job["spec"], "command": runtime_command(job["spec"], env, job["gpus"] or [])}}
                    running.job = job
                    say(f"[pool] runtime {job['spec']['runtime']}: {' '.join(job['spec']['command'][:8])}\n")
            saved = job.get("checkpoint")
            if saved:  # a later attempt starts from the newest checkpoint (pool/actors.py)
                tree, _ = self.cache.ensure(self.client, saved["artifact"], self.limiter, say, peer_sources=self.peer_sources)
                shutil.rmtree(dirs["checkpoint"], ignore_errors=True)
                shutil.copytree(tree, dirs["checkpoint"])  # a writable copy
                env["POOL_RESUMED"] = "1"
                say(f"[pool] resuming from the checkpoint saved by attempt {saved.get('attempt')}\n")
                flush()
            # Refs and queues reach the server through a unix socket (pool/jobsocket.py): the job has no network.
            job_socket = JobSocket(self, job, root, container_paths=self.runtime.name != "process")
            env.update(job_socket.env)
            dirs["socket"], dirs["refs"] = job_socket.dir, job_socket.refs
            if job["spec"].get("network") == "island" and running.rendezvous:
                lo, hi = running.rendezvous["ports"]
                addr = (self.profile.get("peer") or {}).get("addr")
                others = [p["addr"] for p in running.rendezvous["peers"] if p["job"] != job["id"] and p.get("addr")]
                holder, iface = island.setup(job, self.config.get("machine_id", ""), addr, others, lo, hi,
                                             self.config.get("island_image") or island.IMAGE)
                island_job = job
                job = {**job, "island_holder": holder}
                running.job = job
                env.update(island.job_env(addr, lo, hi, iface))
                say(f"[pool] island: ports {lo}-{hi} on {addr}, reachable only by {', '.join(others) or 'no one'}\n")
            command = self.runtime.command(job, dirs, env, self.job_memory(job))
            if running.stop_reason:  # cancelled or preempted while its inputs were fetched: do not start it
                raise JobStopped(running.stop_reason)
            if self.runtime.name == "process":
                proc_env = {**os.environ, **job["spec"]["env"], **env, "PYTHONUNBUFFERED": "1",
                            "POOL_WORK": str(dirs.get("code", "")), "POOL_OUTPUTS": str(dirs["outputs"]),
                            "POOL_CHECKPOINT": str(dirs["checkpoint"]), "POOL_INPUTS": str(root / "inputs")}
                (root / "inputs").mkdir(exist_ok=True)
                for path, tree in dirs["inputs"].items():  # the process runtime sees inputs through links
                    (root / "inputs" / path).symlink_to(tree)
                running.proc = subprocess.Popen(command, cwd=self.runtime.cwd(dirs), env=proc_env, stdout=subprocess.PIPE,
                                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            else:
                running.proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)

            def read() -> None:
                for block in iter(lambda: running.proc.stdout.read1(LOG_CHUNK), b""):
                    say(block.decode(errors="replace"))

            reader = threading.Thread(target=read, daemon=True)
            reader.start()
            deadline = time.monotonic() + job["spec"]["timeout_minutes"] * 60
            last_kill = 0.0
            while running.proc.poll() is None:
                if time.monotonic() > deadline and running.stop_reason is None:
                    self.stop(job["id"], f"timed out after {job['spec']['timeout_minutes']:g} minutes")
                if running.stopped_at is not None:
                    # A stop can land while the process or container is still starting, when killing
                    # it does nothing: keep killing until it is gone (preemption first gets its grace).
                    grace = PREEMPT_GRACE + 10 if running.preempted else 0
                    if time.monotonic() - running.stopped_at >= grace and time.monotonic() - last_kill > 3:
                        last_kill = time.monotonic()
                        self.runtime.kill(running)
                try:  # a tick per LOG_FLUSH_SECONDS, but no waiting once the process has exited
                    running.proc.wait(timeout=LOG_FLUSH_SECONDS)
                except subprocess.TimeoutExpired:
                    pass
                flush()
            reader.join(timeout=10)
            exit_code = running.proc.returncode
            if running.preempted and job_socket is not None and any(dirs["checkpoint"].iterdir()):
                try:
                    job_socket.save_checkpoint()
                    say("\n[pool] saved /checkpoint; the next attempt resumes from it\n")
                except Exception as exc:
                    say(f"\n[pool] could not save /checkpoint: {exc}\n")
            if running.stop_reason:
                error = running.stop_reason
                say(f"\n[pool] stopped: {running.stop_reason}\n")
            else:
                say(f"\n[pool] exit code {exit_code}\n")
            flush()
            if running.stop_reason not in ("cancelled", SUPERSEDED) and not running.preempted:
                try:
                    manifest, sources = build_manifest(dirs["outputs"], follow_links=False, max_bytes=self.max_output)
                    if manifest["files"] or manifest.get("dirs"):
                        outputs = self.client.upload_manifest(manifest, sources, job=job["id"])["artifact"]
                except ArtifactError as exc:
                    error = error or f"outputs not kept: {exc}"
                    outputs = None
        except GangTimeout as exc:
            running.preempted = str(exc)  # handed back without using a retry, with its siblings
            say(f"\n[pool] {exc}; handing the gang back to the queue\n")
            try:
                flush()
            except Exception:
                pass
        except JobStopped as exc:
            error = str(exc)
            say(f"\n[pool] stopped before it started: {exc}\n")
            try:
                flush()
            except Exception:
                pass
        except Exception as exc:  # anything before or around the run: report it on the job
            error = f"{type(exc).__name__}: {exc}"
            log(f"{job['id']}: {error}\n{traceback.format_exc()}")
            say(f"\n[pool] {error}\n")
            try:
                flush()
            except Exception:
                pass
        finally:
            if island_job is not None:
                island.teardown(island_job)
            if job_socket is not None:
                job_socket.close()
            for artifact in pinned:
                self.cache.unpin(artifact)
            if running.stop_reason not in ("cancelled", SUPERSEDED):
                body = {"preempted": running.preempted} if running.preempted else {"exit_code": exit_code, "outputs": outputs, "error": error}
                for attempt_no in range(5):
                    try:
                        self.client.call("POST", f"/v1/agent/jobs/{job['id']}/complete", body, timeout=60)
                        break
                    except PoolError as exc:
                        log(f"{job['id']}: reporting the result failed ({exc}); retrying")
                        time.sleep(2 ** attempt_no)
            with self.lock:
                if self.running.get(job["id"]) is running:  # a newer attempt of the same job keeps its entry
                    self.running.pop(job["id"], None)
            shutil.rmtree(root, ignore_errors=True)
            log(f"{job['id']}: finished ({'exit ' + str(exit_code) if exit_code is not None else error})")
            try:  # report the freed room now: the open lease request picks it up within a second
                self.heartbeat_once()
            except PoolError:
                pass


def _gpus(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [] if value == "none" else [int(i) for i in value.split(",") if i != ""]


def main_run(args) -> int:
    config = load_agent_config()
    if not config:
        print(f"This machine is not registered ({config_path()}): pool agent register --server URL --code CODE", file=sys.stderr)
        return EX_CONFIG
    if args.gpus is not None:
        config["gpus"] = _gpus(args.gpus)
    if args.label:
        config["labels"] = pairs(args.label, "--label")
    if args.allow_image:
        config["allow_images"] = args.allow_image
    if args.bench_image and not config.get("bench"):
        log(f"measuring this machine with {args.bench_image} (once; pool agent bench to redo)")
        try:
            config["bench"] = run_bench(args.bench_image, config.get("gpus"))
            save_agent_config(config)
            log(f"measured: {config['bench']['tflops']} TFLOP/s, host to GPU {config['bench'].get('h2d_gbps')} GB/s")
        except Exception as exc:
            log(f"benchmark failed, using estimates: {exc}")
    runtime = ProcessRuntime() if args.unsafe_no_container else DockerRuntime(config["machine_id"])
    if args.unsafe_no_container:
        log("WARNING: --unsafe-no-container: jobs run as plain processes with this user's full access")
    work = Path(args.work or Path.home() / ".cache" / "pool-agent" / config["machine_id"])
    if args.allow_peer_network:
        config["allow_peer_network"] = True
        save_agent_config(config)
    agent = Agent(config, work, runtime=runtime, max_jobs=args.max_jobs, cpus=args.cpus, memory_gb=args.memory_gb,
                  cache_gb=args.cache_gb, network_mbps=args.network_mbps, limit_mbps=args.limit_mbps,
                  peer_port=None if args.peer_port == 0 else args.peer_port,
                  allow_peer_network=bool(config.get("allow_peer_network")), peer_addr=args.peer_addr,
                  iroh={"binary": args.iroh_binary, "image": args.iroh_image, "port": args.iroh_port, "lan": args.iroh_lan,
                        "bind": args.iroh_bind} if args.iroh else None, peer_limit_mbps=args.peer_limit_mbps)
    signal.signal(signal.SIGTERM, lambda *_: agent.stopping.set())
    try:
        agent.run_forever()
    except KeyboardInterrupt:
        agent.stopping.set()
    if agent.restart:  # new files are in place (Agent.maybe_update): become the new agent, same arguments
        os.execv(sys.executable, [sys.executable, "-m", "pool.cli", *sys.argv[1:]])
    return EX_CONFIG if agent.removed else 0


def main_update(args) -> int:
    """Install the server's agent package here; optionally turn auto-update on or off."""
    config = load_agent_config()
    if not config:
        print("This machine is not registered: pool agent register --server URL --code CODE", file=sys.stderr)
        return 1
    if args.auto is not None:
        config["auto_update"] = args.auto
        save_agent_config(config)
        print(f"auto-update {'on' if args.auto else 'off'} (the running agent picks this up when restarted)")
    if in_container():
        print("this agent runs in a container: run the installer again on the host (it fetches the server's agent), "
              "or with docker compose: docker compose pull && docker compose up -d", file=sys.stderr)
        return 1
    client = Client(config["server"], config["token"])
    root = Path(__file__).resolve().parent
    wanted = client.call("GET", "/healthz").get("agent")
    if not wanted:
        print("the server does not name an agent version (it is older than this agent)", file=sys.stderr)
        return 1
    if wanted == update.package_digest(root):
        print(f"up to date ({wanted[:12]})")
        return 0
    data = client._request("GET", "/v1/agent/package.tar.gz", timeout=120).read()
    update.install(data, root, expect=wanted)
    restart = {"system": "sudo systemctl restart pool-agent", "user": "systemctl --user restart pool-agent"}
    print(f"installed agent {wanted[:12]} in {root}; restart the agent: {restart.get(install_kind(), 'restart pool agent run')}")
    return 0


# ---- owner commands: pool-agent status | pause | resume | uninstall --------------------------------
def user_layout() -> dict:
    """The --no-sudo install (pool/installer.py): a user unit and the package in the home folder."""
    home = Path.home()
    return {"unit": home / ".config" / "systemd" / "user" / "pool-agent.service", "wrapper": home / ".local" / "bin" / "pool-agent",
            "prefix": home / ".local" / "share" / "pool-agent", "state": home / ".cache" / "pool-agent"}


def in_container() -> bool:
    """The agent runs in its own container (deploy/pool/Dockerfile, the installer's default): it
    drives the host's Docker through the mounted socket, and the host restarts it, not systemd."""
    return os.environ.get("POOL_IN_CONTAINER") == "1"


def install_kind() -> str | None:
    """'docker' (the agent's own container: sudo install.sh, or docker compose), 'system' (sudo install.sh
    --host-agent), 'user' (install.sh --no-sudo), or None (run by hand, say from a checkout)."""
    if in_container():
        return "docker"
    if SYSTEM["unit"].exists() or config_path().parent == SYSTEM["etc"]:
        return "system"
    if user_layout()["unit"].exists():
        return "user"
    return None


def _systemctl(kind: str | None, *args: str) -> subprocess.CompletedProcess | None:
    if kind in (None, "docker") or not shutil.which("systemctl"):
        return None
    command = ["systemctl", *(["--user"] if kind == "user" else []), *args]
    return subprocess.run(command, capture_output=True, text=True, timeout=120)


def main_status(args) -> int:
    config, kind = load_agent_config(), install_kind()
    if not config:
        print(f"not registered ({config_path()} is missing): run the installer from `pool machine add`")
        return 1
    print(f"machine   {config.get('name')} ({config.get('machine_id')}) in pool {config.get('pool')} at {config.get('server')}")
    print(f"settings  {config_path()}")
    state = _systemctl(kind, "is-active", "pool-agent.service")
    if kind is None:
        print("service   none installed (started by hand: pool agent run)")
    elif kind == "docker":
        print(f"service   running in a container (image {os.environ.get('POOL_IMAGE') or 'unknown'}; "
              "logs on the host: docker logs -f pool-agent)")
    else:
        where = "system unit" if kind == "system" else "user unit"
        print(f"service   {(state.stdout.strip() if state else '') or 'unknown'} (pool-agent.service, {where})")
    paused = pause_path()
    if paused.exists():
        since = datetime.fromtimestamp(paused.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"paused    yes, since {since}: no new jobs (pool-agent resume)")
    else:
        print("paused    no")
    try:
        me = Client(config["server"], config["token"]).call("GET", "/v1/agent/me", timeout=10)
        jobs = ", ".join(me["running"]) or "none"
        print(f"server    reachable; {'draining (pool machine drain), ' if me['drain'] else ''}running jobs: {jobs}")
    except PoolError as exc:
        if exc.status == 401:
            print("server    this machine was removed from the pool: uninstall it, or run the installer again with "
                  "a new code and --reregister")
            return 1
        print(f"server    unreachable: {exc}")
    return 0


def main_pause(args) -> int:
    config = load_agent_config()
    if not config:
        print(f"not registered ({config_path()} is missing)", file=sys.stderr)
        return 1
    pause_path().touch()
    print(f"{config.get('name')} is paused: it takes no new jobs (within half a minute); running jobs finish. "
          "Take jobs again with: pool-agent resume")
    return 0


def main_resume(args) -> int:
    pause_path().unlink(missing_ok=True)
    print(f"{load_agent_config().get('name', 'this machine')} takes jobs again")
    return 0


def _remove(path: Path) -> bool:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return True
    if path.is_dir():
        shutil.rmtree(path)
        return True
    return False


def main_uninstall(args) -> int:
    """Stop and remove the service, unregister the machine, delete the agent's files. Docker stays."""
    config, kind = load_agent_config(), install_kind()
    if kind == "system" and hasattr(os, "geteuid") and os.geteuid() != 0 and not in_container():
        print("the agent is installed for the whole system: sudo pool-agent uninstall", file=sys.stderr)
        return 1
    name = config.get("name") or "this machine"
    if not args.yes and sys.stdin.isatty():
        answer = input(f"Remove {name} from the pool at {config.get('server', '?')} and delete the agent? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("nothing changed")
            return 1
    if kind and kind != "docker":
        _systemctl(kind, "disable", "--now", "pool-agent.service")
        print("stopped and disabled pool-agent.service")
    if config.get("token"):
        try:
            Client(config["server"], config["token"]).call("DELETE", "/v1/agent/me", timeout=30)
            print(f"unregistered {name} from {config['server']}")
        except PoolError as exc:
            if exc.status == 401:
                print(f"{name} was already removed from the pool")
            else:
                print(f"warning: could not unregister ({exc}); remove it from a member's computer: pool machine rm {name}",
                      file=sys.stderr)
    if config.get("machine_id") and shutil.which("docker"):
        found = subprocess.run(["docker", "ps", "-q", "--filter", f"label=pool.machine={config['machine_id']}"],
                               capture_output=True, text=True)
        if found.returncode == 0 and found.stdout.split():
            subprocess.run(["docker", "kill", *found.stdout.split()], capture_output=True)
    paths = [config_path(), pause_path(), config_path().with_suffix(".tmp")]
    if kind == "docker":  # the host's `pool-agent uninstall` (pool/installer.py) removes the container and folders
        pass
    elif kind == "system":
        paths += [SYSTEM["unit"], SYSTEM["wrapper"], SYSTEM["etc"], SYSTEM["state"], SYSTEM["prefix"]]
    else:
        layout = user_layout()
        paths += [layout["unit"], layout["wrapper"]]
        if config.get("machine_id"):
            paths.append(layout["state"] / config["machine_id"])
        if not update.source_checkout(layout["prefix"] / "pool"):
            paths.append(layout["prefix"])
    removed = [str(path) for path in paths if _remove(path)]
    if kind == "user" and layout["state"].is_dir() and not any(layout["state"].iterdir()):
        layout["state"].rmdir()
    if kind and kind != "docker":
        _systemctl(kind, "daemon-reload")
    print("removed " + (", ".join(removed) or "nothing else") + "; Docker stays installed")
    return 0


def main_register(args) -> int:
    register(args.server, args.code, args.name, pairs(args.label, "--label"), _gpus(args.gpus), args.allow_image or ["*"])
    return 0


def main_bench(args) -> int:
    config = load_agent_config()
    result = run_bench(args.image, _gpus(args.gpus) if args.gpus is not None else config.get("gpus"))
    print(json.dumps(result, indent=2))
    if config:
        config["bench"] = result
        save_agent_config(config)
        print("saved; the agent reports it with its next profile (restart it to send it now)")
    return 0


def _with_config(func):
    """Run `func` with --config, when given, as the agent's settings file (config_path)."""
    if func is None:
        return None

    def run(args) -> int:
        if getattr(args, "config", None):
            os.environ["POOL_AGENT_CONFIG"] = str(Path(args.config).expanduser().resolve())
        return func(args)

    run.__name__ = func.__name__
    return run


def add_commands(subparsers) -> None:
    import argparse

    agent = subparsers.add_parser("agent", help="run this machine's agent").add_subparsers(dest="agent_command", required=True)
    reg = agent.add_parser("register", help="join a pool with a machine code (pool machine add)")
    reg.add_argument("--server", required=True)
    reg.add_argument("--code", required=True)
    reg.add_argument("--name", default=os.uname().nodename.split(".")[0])
    run = agent.add_parser("run", help="lease and run jobs until stopped")
    run.add_argument("--max-jobs", type=int, default=4)
    run.add_argument("--cpus", type=float, help="CPUs jobs may use in total (default: all)")
    run.add_argument("--memory-gb", type=float, help="memory jobs may use in total (default: 90%%)")
    run.add_argument("--work", help="scratch and cache folder (default ~/.cache/pool-agent/<id>)")
    run.add_argument("--cache-gb", type=float, help="size of the input cache (default: 30%% of free disk, at least 10 GB)")
    run.add_argument("--bench-image", help="measure TFLOP/s with this image (needs PyTorch) on first start")
    run.add_argument("--network-mbps", type=float, help="declare this machine's link speed instead of measuring it")
    run.add_argument("--limit-mbps", type=float, help="cap downloads at this rate (also simulates a slow link)")
    run.add_argument("--peer-port", type=int, default=peers.DEFAULT_PORT,
                     help=f"port of the peer server other agents fetch chunks from (default {peers.DEFAULT_PORT}; 0: none)")
    run.add_argument("--peer-addr", help="the address peers reach this machine at (default: the one it reaches the server from)")
    run.add_argument("--allow-peer-network", action="store_true",
                     help="let gang jobs with network: peers use this machine's network (remembered)")
    run.add_argument("--peer-limit-mbps", type=float, help="cap what peers may take from this machine (also simulates a slow uplink)")
    run.add_argument("--iroh", action="store_true",
                     help="move chunks between machines through the iroh-blobs sidecar (QUIC, hole punching, relays); HTTP stays the fallback")
    run.add_argument("--iroh-binary", help="the pool-iroh binary (default: $POOL_IROH_BIN or pool-iroh on PATH, else the image)")
    run.add_argument("--iroh-image", default=peers.SIDECAR_IMAGE,
                     help=f"the sidecar's container image when there is no binary (default {peers.SIDECAR_IMAGE}: docker build -t pool-iroh pool/sidecar/iroh)")
    run.add_argument("--iroh-port", type=int, default=peers.SIDECAR_PORT, help=f"the sidecar's loopback control port (default {peers.SIDECAR_PORT})")
    run.add_argument("--iroh-bind", help="the sidecar's QUIC address, for example 0.0.0.0:7842 (default: any port)")
    run.add_argument("--iroh-lan", action="store_true", help="the sidecar uses direct addresses only: no relays, no address lookup")
    run.add_argument("--unsafe-no-container", action="store_true", help=argparse.SUPPRESS)
    upd = agent.add_parser("update", help="install the server's version of the agent here")
    upd.add_argument("--auto", action=argparse.BooleanOptionalAction, default=None,
                     help="also let the agent update itself between jobs from now on (--no-auto: stop)")
    upd.set_defaults(func=main_update)
    bench = agent.add_parser("bench", help="measure TFLOP/s per dtype and host-to-GPU bandwidth")
    bench.add_argument("--image", required=True, help="an image with PyTorch, for example nvcr.io/nvidia/pytorch:25.01-py3")
    reg.set_defaults(func=main_register)
    run.set_defaults(func=main_run)
    bench.set_defaults(func=main_bench)
    status = agent.add_parser("status", help="this machine: service, registration, paused or not, running jobs")
    pause = agent.add_parser("pause", help="take no new jobs (running ones finish) until resume")
    resume = agent.add_parser("resume", help="take jobs again after pause")
    uninstall = agent.add_parser("uninstall", help="stop the service, unregister, delete the agent (Docker stays)")
    uninstall.add_argument("--yes", "-y", action="store_true", help="do not ask")
    for parser, func in ((status, main_status), (pause, main_pause), (resume, main_resume), (uninstall, main_uninstall)):
        parser.set_defaults(func=func)
    for parser in (reg, run, upd, bench, status, pause, resume, uninstall):
        parser.add_argument("--config", help="this machine's settings file (default: $POOL_AGENT_CONFIG, else ~/.config/pool/agent.json)")
        parser.set_defaults(func=_with_config(parser.get_default("func")))
    for parser in (reg, run, bench):
        parser.add_argument("--gpus", help="GPU indices to offer, for example 0,1, or none")
    for parser in (reg, run):
        parser.add_argument("--label", action="append", help="k=v, matched against a job's resources.labels")
        parser.add_argument("--allow-image", action="append", help="image pattern jobs may use (default: any)")
