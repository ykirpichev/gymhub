"""The compute pool, M1 and M2, and the LAN prototype: peers and gangs (issue #45).

End to end: a real server on a free port, agents with the process runtime (no Docker or GPU
needed), and the CLI. Units: manifests, the agent cache, models, placement, quotas, GC.

POOL_TEST_POSTGRES=postgresql://user:pw@host:port/postgres runs the server and database tests
against Postgres (each test gets a fresh database); otherwise SQLite.
"""

import base64
import io
import json
import os
import socket
import struct
import tarfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import pytest
import uvicorn

from pool import artifacts, cli, models, peers, scheduler, spec as specs
from pool.agent import Agent, Cache, ProcessRuntime, in_hours, register
from pool.artifacts import ArtifactError, ChunkStore, build_manifest, manifest_id
from pool.client import Client, PoolError, load_config
from pool.db import load_job, now, open_db
from pool.server import collect_garbage, create_app

PG = os.environ.get("POOL_TEST_POSTGRES")


@pytest.fixture()
def db_url(tmp_path):
    """SQLite in tmp_path, or a fresh Postgres database when POOL_TEST_POSTGRES is set."""
    if not PG:
        yield str(tmp_path / "pool.db")
        return
    import psycopg

    name = "pool_test_" + uuid.uuid4().hex[:10]
    with psycopg.connect(PG, autocommit=True) as conn:
        conn.execute(f"create database {name}")
    yield PG.rsplit("/", 1)[0] + "/" + name
    with psycopg.connect(PG, autocommit=True) as conn:
        conn.execute(f"drop database if exists {name} with (force)")


@pytest.fixture()
def small_chunks(monkeypatch):
    """1 KiB chunks and 100-byte inline files, so tests exercise chunking with small files."""
    monkeypatch.setattr(artifacts, "CHUNK", 1024)
    monkeypatch.setattr(artifacts, "INLINE_MAX", 100)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def pool_env(tmp_path, monkeypatch, db_url):
    monkeypatch.setenv("POOL_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("POOL_SERVER", raising=False)
    monkeypatch.delenv("POOL_TOKEN", raising=False)
    app = create_app(tmp_path / "server", db_url=db_url, gc_seconds=None, poll_ms=500)
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", timeout_graceful_shutdown=1))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(200):
        try:
            Client(url).call("GET", "/healthz")
            break
        except PoolError:
            time.sleep(0.05)
    assert cli.main(["login", url, "--code", app.state.bootstrap_code, "--name", "ann"]) == 0
    env = {"url": url, "app": app, "db": app.state.db, "tmp": tmp_path, "agents": []}
    yield env
    for agent in env["agents"]:
        agent.stopping.set()
    server.should_exit = True
    thread.join(timeout=5)


def start_agent(env, name="box1", pool=None, member=None, **kwargs) -> Agent:
    member = member or Client.member()
    code = member.call("POST", "/v1/machine-codes", {"name": name, "pool": pool})["code"]
    config = register(env["url"], code, name, kwargs.pop("labels", {}), [], ["*"])
    kwargs.setdefault("peer_port", 0)  # an ephemeral port each, on loopback
    kwargs.setdefault("peer_addr", "127.0.0.1")
    agent = Agent(config, env["tmp"] / f"work-{name}", runtime=ProcessRuntime(), cpus=8, memory_gb=16,
                  cache_gb=kwargs.pop("cache_gb", 1), **kwargs)
    threading.Thread(target=agent.run_forever, daemon=True).start()
    env["agents"].append(agent)
    for _ in range(200):
        if member.call("GET", f"/v1/machines?pool={pool or ''}")["machines"]:
            break
        time.sleep(0.05)
    return agent


def invite(env, role="member", pool=None, name="bob") -> Client:
    code = Client.member().call("POST", "/v1/invites", {"pool": pool, "role": role})["code"]
    token = Client(env["url"]).call("POST", "/v1/login", {"code": code, "name": name})["token"]
    return Client(env["url"], token)


def write(folder: Path, files: dict[str, str | bytes]) -> Path:
    for name, content in files.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode())
    return folder


def wait_for(predicate, timeout=20.0, step=0.1):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(step)
    raise AssertionError("timed out waiting")


def job_state(job_id: str) -> dict:
    return Client.member().call("GET", f"/v1/jobs/{job_id}")["job"]


TRAIN = """
import os, pathlib
data = pathlib.Path(os.environ["POOL_INPUTS"]) / "data"
total = sum(int(line) for line in (data / "numbers.txt").read_text().split())
print("epoch 1 loss 0.5", flush=True)
print("sum", total, flush=True)
out = pathlib.Path(os.environ["POOL_OUTPUTS"])
(out / "model").mkdir()
(out / "model" / "weights.txt").write_text(str(total))
(out / "metrics.json").write_text('{"sum": %d}' % total)
"""


# ---- end to end ---------------------------------------------------------------------------------


def test_run_streams_logs_and_returns_outputs(pool_env, capsys, monkeypatch):
    start_agent(pool_env)
    code = write(pool_env["tmp"] / "project", {"train.py": TRAIN, ".git/HEAD": "x", "__pycache__/x.pyc": "x"})
    data = write(pool_env["tmp"] / "data", {"numbers.txt": "1 2 3 4"})
    monkeypatch.chdir(pool_env["tmp"])
    rc = cli.main(["run", "--code", str(code), "--data", str(data), "--to", str(pool_env["tmp"] / "got"), "--", "python3", "train.py"])
    out, err = capsys.readouterr()
    assert rc == 0, err
    assert "epoch 1 loss 0.5" in out and "sum 10" in out and "[pool] exit code 0" in out
    assert "succeeded on box1" in err
    assert (pool_env["tmp"] / "got" / "model" / "weights.txt").read_text() == "10"
    assert json.loads((pool_env["tmp"] / "got" / "metrics.json").read_text()) == {"sum": 10}

    assert cli.main(["ps"]) == 0
    listing = capsys.readouterr().out
    assert "train" in listing and "succeeded" in listing and "default" in listing
    job_id = listing.splitlines()[1].split()[0]
    assert cli.main(["logs", job_id]) == 0
    assert "sum 10" in capsys.readouterr().out
    assert cli.main(["get", job_id, "--to", str(pool_env["tmp"] / "again")]) == 0
    assert (pool_env["tmp"] / "again" / "metrics.json").is_file()
    assert cli.main(["show", job_id]) == 0
    assert "placed on box1" in capsys.readouterr().out


def test_failed_job_reports_its_exit_code(pool_env, capsys):
    start_agent(pool_env)
    code = write(pool_env["tmp"] / "p", {"fail.py": "import sys; print('boom'); sys.exit(3)"})
    rc = cli.main(["run", "--code", str(code), "--no-get", "--", "python3", "fail.py"])
    out, err = capsys.readouterr()
    assert rc == 3
    assert "boom" in out and "failed" in err and "exit code 3" in err


def test_cancel_stops_a_running_job(pool_env, capsys):
    agent = start_agent(pool_env)
    code = write(pool_env["tmp"] / "p", {"slow.py": "import time\nprint('started', flush=True)\ntime.sleep(120)"})
    assert cli.main(["run", "-d", "--code", str(code), "--", "python3", "slow.py"]) == 0
    job_id = capsys.readouterr().out.strip().splitlines()[-1]
    wait_for(lambda: job_state(job_id)["status"] == "running" and agent.running)
    assert cli.main(["cancel", job_id]) == 0
    wait_for(lambda: not agent.running)
    assert job_state(job_id)["status"] == "cancelled"


def test_cancel_while_starting_never_runs_the_job(pool_env, capsys):
    """A cancel that lands while the job is still being prepared (inputs, image) used to find no
    process to kill, and the job then started anyway."""
    agent = start_agent(pool_env)
    gate = threading.Event()
    preparing = threading.Event()

    def slow_prepare(image, say):
        preparing.set()
        gate.wait(20)

    agent.runtime.prepare = slow_prepare
    code = write(pool_env["tmp"] / "p", {"slow.py": "import time\nprint('started', flush=True)\ntime.sleep(120)"})
    assert cli.main(["run", "-d", "--code", str(code), "--", "python3", "slow.py"]) == 0
    job_id = capsys.readouterr().out.strip().splitlines()[-1]
    assert preparing.wait(20)
    assert cli.main(["cancel", job_id]) == 0
    wait_for(lambda: next(iter(agent.running.values())).stop_reason if agent.running else True)
    started = list(agent.running.values())
    gate.set()
    wait_for(lambda: not agent.running)
    assert all(item.proc is None for item in started), "a cancelled job must not start"
    assert job_state(job_id)["status"] == "cancelled"


def test_spec_file_with_local_folders(pool_env, capsys):
    start_agent(pool_env)
    project = write(pool_env["tmp"] / "proj", {"train.py": TRAIN})
    write(project / "data", {"numbers.txt": "5 5"})
    (project / "job.json").write_text(json.dumps({
        "name": "from-spec", "image": "python:3.12-slim", "command": ["python3", "train.py"],
        "code": ".", "inputs": [{"path": "data", "dir": "data"}]}))
    rc = cli.main(["submit", str(project / "job.json"), "-f", "--to", str(pool_env["tmp"] / "o")])
    assert rc == 0, capsys.readouterr().err
    assert (pool_env["tmp"] / "o" / "model" / "weights.txt").read_text() == "10"


def test_second_job_with_the_same_input_fetches_nothing(pool_env, capsys, small_chunks):
    agent = start_agent(pool_env)
    import random

    digits = " ".join(str(random.randint(0, 9)) for _ in range(3000))  # 6 KB of varied text: 6 distinct chunks
    data = write(pool_env["tmp"] / "data", {"numbers.txt": digits})
    code = write(pool_env["tmp"] / "code", {"count.py": "import os; print(len(open(os.environ['POOL_INPUTS'] + '/data/numbers.txt').read().split()))"})
    member = Client.member()
    logs = []
    for _ in range(2):
        assert cli.main(["run", "--code", str(code), "--data", str(data), "--no-get", "--", "python3", "count.py"]) == 0
        capsys.readouterr()
        job_id = member.call("GET", "/v1/jobs?limit=1")["jobs"][0]["id"]
        logs.append("".join(c["text"] for c in member.call("GET", f"/v1/jobs/{job_id}/logs")["chunks"]))
    assert "3000" in logs[0] and "3000" in logs[1]
    assert "fetching" in logs[0] and "of 6 chunks" in logs[0]
    assert "fetching" not in logs[1] and "already on this machine" in logs[1]
    machines = Client.member().call("GET", "/v1/machines")["machines"]
    assert pool_env["db"].one("select count(*) as n from chunk_locations")["n"] >= 6, "the server knows what box1 holds"
    assert agent.cache.used_bytes() >= 6000


def test_a_new_artifact_reuses_chunks_cached_in_another(pool_env, capsys, small_chunks):
    """A checkpoint that changed one file: its unchanged chunks are copied from the cached tree of
    the old one, not downloaded, and each file's bytes are on disk once."""
    import random

    agent = start_agent(pool_env)
    random.seed(7)
    weights = "".join(random.choice("0123456789abcdef") for _ in range(8000))  # 8 distinct 1 KiB chunks
    code = write(pool_env["tmp"] / "code", {"size.py": "import os; print(os.path.getsize(os.environ['POOL_INPUTS'] + '/ckpt/weights.bin'))"})
    logs = []
    for step in (1, 2):
        ckpt = write(pool_env["tmp"] / f"ckpt{step}", {"weights.bin": weights, "step.txt": f"step {step} " * 30})
        assert cli.main(["run", "--code", str(code), "--data", f"ckpt={ckpt}", "--no-get", "--", "python3", "size.py"]) == 0
        capsys.readouterr()
        member = Client.member()
        job_id = member.call("GET", "/v1/jobs?limit=1")["jobs"][0]["id"]
        logs.append("".join(c["text"] for c in member.call("GET", f"/v1/jobs/{job_id}/logs")["chunks"]))
    assert "8000" in logs[1]
    assert "reused" in logs[1], logs[1]
    assert "fetching 1 of 9 chunks" in logs[1], logs[1]  # only step.txt's chunk travels
    assert not any(agent.cache.chunks.root.glob("*/*")), "built trees are the only copy of their chunks"


def test_preempted_job_requeues_without_using_a_retry(pool_env, capsys):
    owner_busy = []
    agent = start_agent(pool_env, foreign=lambda: list(owner_busy))
    Client.member().call("PUT", "/v1/machines/box1/policy", {"idle_only": True})
    code = write(pool_env["tmp"] / "p", {"job.py": "import os, time\nprint('attempt', os.environ['POOL_JOB_ATTEMPT'], flush=True)\n"
                                                 "time.sleep(float(os.environ.get('NAP', '0')))\n"})
    assert cli.main(["run", "-d", "--retries", "0", "--env", "NAP=60", "--code", str(code), "--", "python3", "job.py"]) == 0
    job_id = capsys.readouterr().out.strip().splitlines()[-1]
    wait_for(lambda: job_state(job_id)["status"] == "running" and agent.running)
    owner_busy.append(4242)  # the owner starts using the GPU
    wait_for(lambda: job_state(job_id)["status"] == "queued" and not agent.running)
    state = job_state(job_id)
    assert state["preemptions"] == 1 and state["tries"] == 1
    assert agent.paused and "owner is using the GPU" in agent.paused
    owner_busy.clear()
    wait_for(lambda: job_state(job_id)["status"] == "running")
    Client.member().call("POST", f"/v1/jobs/{job_id}/cancel")
    assert cli.main(["logs", job_id]) == 0
    assert "preempted" in capsys.readouterr().out


# ---- artifacts ------------------------------------------------------------------------------------


def test_chunked_upload_sends_each_chunk_once(pool_env, small_chunks):
    ann = Client.member()
    folder = write(pool_env["tmp"] / "m", {"weights.bin": os.urandom(5000), "config.json": "{}"})
    first = ann.upload_folder(folder)
    assert first["sent_bytes"] == 5000
    assert ann.upload_folder(folder)["sent_bytes"] == 0
    with open(folder / "weights.bin", "r+b") as handle:  # a new checkpoint: only the last chunk changed
        handle.seek(4990)
        handle.write(b"x" * 10)
    changed = ann.upload_folder(folder)
    assert changed["artifact"] != first["artifact"] and 0 < changed["sent_bytes"] <= 1024


def test_dedupe_never_reveals_other_members_uploads(pool_env, small_chunks):
    ann = Client.member()
    folder = write(pool_env["tmp"] / "secret", {"s.bin": os.urandom(3000)})
    mine = ann.upload_folder(folder)
    bob = invite(pool_env)
    with pytest.raises(PoolError) as exc:
        bob.manifest(mine["artifact"])
    assert exc.value.status == 404
    theirs = bob.upload_folder(folder)
    assert theirs["artifact"] == mine["artifact"] and theirs["sent_bytes"] == 3000, "bob must send the bytes himself"
    assert bob.manifest(mine["artifact"])["files"][0]["path"] == "s.bin"


def test_bad_chunk_is_refused(pool_env, small_chunks):
    ann = Client.member()
    manifest = {"v": 1, "files": [{"path": "x.bin", "size": 200, "mode": 0o644, "chunks": ["0" * 64]}]}
    reply = ann.upload_manifest  # declare it, then send bytes that do not hash to the id
    with pytest.raises(PoolError) as exc:
        reply(manifest, {"0" * 64: (write(pool_env["tmp"] / "b", {"x.bin": os.urandom(200)}) / "x.bin", 0, 200)})
    assert exc.value.status == 400


def test_machines_read_only_artifacts_of_their_jobs(pool_env, small_chunks):
    member = Client.member()
    artifact = member.upload_folder(write(pool_env["tmp"] / "d", {"x.bin": os.urandom(2000)}))["artifact"]
    code = member.call("POST", "/v1/machine-codes", {})["code"]
    config = register(pool_env["url"], code, "m1", {}, [], ["*"])
    machine = Client(pool_env["url"], config["token"])
    with pytest.raises(PoolError):
        machine.manifest(artifact)
    member.call("POST", "/v1/jobs", {"spec": {"image": "x", "command": ["true"], "code": artifact}})
    leased = machine.call("POST", "/v1/agent/lease", {"free": {"gpus": [], "cpus": 4, "memory_gb": 8}, "wait": 0})["jobs"]
    assert leased and machine.manifest(artifact)
    result, sources = build_manifest(write(pool_env["tmp"] / "r", {"y.bin": os.urandom(500)}))
    with pytest.raises(PoolError) as exc:  # machines upload results only for their own running jobs
        machine.upload_manifest(result, sources, job="j-nope")
    assert exc.value.status == 403
    out = machine.upload_manifest(result, sources, job=leased[0]["id"])
    assert member.manifest(out["artifact"]), "a result belongs to the job's owner"


def test_storage_quota(pool_env, small_chunks, capsys):
    assert cli.main(["pool", "set", "default", "quota.storage_gb=0.000001"]) == 0  # about 1 KB
    with pytest.raises(PoolError) as exc:
        Client.member().upload_folder(write(pool_env["tmp"] / "big", {"a.bin": os.urandom(5000)}))
    assert exc.value.status == 413
    assert cli.main(["pool", "set", "default", "quota.storage_gb=none"]) == 0
    Client.member().upload_folder(pool_env["tmp"] / "big")


def test_manifest_validation():
    good = {"v": 1, "files": [{"path": "a/b.txt", "size": 3, "mode": 0o644, "data": base64.b64encode(b"abc").decode()}]}
    artifacts.validate(good)
    for path in ("../x", "/abs", "a//b", "a/./b", ""):
        with pytest.raises(ArtifactError):
            artifacts.validate({"v": 1, "files": [{**good["files"][0], "path": path}]})
    with pytest.raises(ArtifactError):  # 100 MB needs two 64 MiB chunks
        artifacts.validate({"v": 1, "files": [{"path": "w", "size": 100 * 1024**2, "mode": 0o644, "chunks": ["a" * 64]}]})
    with pytest.raises(ArtifactError):
        artifacts.validate({"v": 1, "files": [{"path": "w", "size": 4, "mode": 0o644, "data": base64.b64encode(b"abc").decode()}]})
    assert manifest_id(good) == manifest_id(json.loads(json.dumps(good)))


def test_build_manifest_and_links(tmp_path):
    outside = write(tmp_path / "host", {"passwd": "secret"})
    folder = write(tmp_path / "out", {"a.txt": "1", "sub/b.txt": "2"})
    (folder / "leak").symlink_to(outside / "passwd")
    first, _ = build_manifest(folder)
    assert sorted(f["path"] for f in first["files"]) == ["a.txt", "sub/b.txt"], "links are skipped, never followed"
    followed, _ = build_manifest(folder, follow_links=True)
    assert "leak" in [f["path"] for f in followed["files"]]
    (folder / "a.txt").touch()  # a new mtime does not change the id
    assert manifest_id(build_manifest(folder)[0]) == manifest_id(first)


class FakeClient:
    """Serves artifacts from memory to a Cache."""

    def __init__(self):
        self.manifests, self.chunks, self.fetched = {}, {}, 0

    def add(self, files: dict[str, bytes]) -> str:
        manifest, chunks = artifacts.manifest_of_bytes(files)
        artifact = manifest_id(manifest)
        self.manifests[artifact] = manifest
        self.chunks.update(chunks)
        return artifact

    def manifest(self, artifact):
        return self.manifests[artifact]

    def fetch_chunk(self, sha, artifact, dest, limiter=None):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.chunks[sha])
        self.fetched += len(self.chunks[sha])
        return len(self.chunks[sha])


def test_cache_reuses_and_evicts(tmp_path, small_chunks):
    fake = FakeClient()
    first = fake.add({"w.bin": os.urandom(3000)})
    second = fake.add({"w.bin": os.urandom(3000)})
    cache = Cache(tmp_path / "cache", quota_bytes=4000)
    tree, fetched = cache.ensure(fake, first)
    assert fetched == 3000 and (tree / "w.bin").stat().st_size == 3000
    assert cache.ensure(fake, first)[1] == 0, "a cached tree costs no download"
    assert len(cache.report()["add"]) == 3
    cache.pin(second)
    cache.ensure(fake, second)
    assert not (cache.trees / first.removeprefix("sha256:")).exists(), "the least recently used tree goes first"
    assert (cache.trees / second.removeprefix("sha256:")).is_dir()
    report = cache.report()
    assert len(report["remove"]) == 3 and len(report["add"]) == 3
    assert set(cache.held()) == set(artifacts.chunks_of(fake.manifests[second]))
    again = Cache(tmp_path / "cache", quota_bytes=4000)  # a restarted agent finds its cache
    assert again.ensure(fake, second)[1] == 0


# ---- pools, roles, quotas -------------------------------------------------------------------------


def test_pools_and_roles(pool_env, capsys):
    ann = Client.member()
    assert cli.main(["pool", "create", "lab"]) == 0
    start_agent(pool_env, "lab-box", pool="lab")
    bob = invite(pool_env)  # member of default
    with pytest.raises(PoolError) as exc:
        bob.call("POST", "/v1/pools", {"name": "mine"})
    assert exc.value.status == 403
    code = ann.call("POST", "/v1/invites", {"pool": "lab", "role": "viewer"})["code"]
    assert bob.call("POST", "/v1/join", {"code": code})["pool"] == "lab"
    with pytest.raises(PoolError) as exc:
        bob.call("POST", "/v1/jobs", {"pool": "lab", "spec": {"image": "x", "command": ["true"]}})
    assert exc.value.status == 403, "viewers do not submit"
    assert [m["name"] for m in bob.call("GET", "/v1/machines?pool=lab")["machines"]] == ["lab-box"]
    assert bob.call("GET", "/v1/machines?pool=default")["machines"] == []
    carol = invite(pool_env, pool="lab", name="carol")
    with pytest.raises(PoolError) as exc:
        carol.call("GET", "/v1/machines?pool=default")
    assert exc.value.status == 404
    job = ann.call("POST", "/v1/jobs", {"pool": "default", "spec": {"image": "x", "command": ["true"]}})["job"]
    with pytest.raises(PoolError):
        carol.call("GET", f"/v1/jobs/{job['id']}")
    with pytest.raises(PoolError) as exc:
        carol.call("POST", "/v1/invites", {"pool": "lab"})
    assert exc.value.status == 403
    assert cli.main(["pools"]) == 0
    assert "lab" in capsys.readouterr().out


def test_max_running_jobs(pool_env, capsys):
    start_agent(pool_env)
    assert cli.main(["pool", "set", "default", "quota.max_running_jobs=1"]) == 0
    code = write(pool_env["tmp"] / "p", {"nap.py": "import time; time.sleep(30)"})
    ids = []
    for _ in range(2):
        assert cli.main(["run", "-d", "--code", str(code), "--", "python3", "nap.py"]) == 0
        ids.append(capsys.readouterr().out.strip().splitlines()[-1])
    wait_for(lambda: job_state(ids[0])["status"] == "running")
    time.sleep(1.5)
    assert job_state(ids[1])["status"] == "queued"
    why = Client.member().call("GET", f"/v1/jobs/{ids[1]}")["why"]
    assert any("at most 1 running jobs" in line for line in why)
    for job_id in ids:
        Client.member().call("POST", f"/v1/jobs/{job_id}/cancel")


def test_codes_work_once(pool_env):
    member = Client.member()
    code = member.call("POST", "/v1/machine-codes", {})["code"]
    register(pool_env["url"], code, "m1", {}, [], ["*"])
    with pytest.raises(PoolError) as exc:
        register(pool_env["url"], code, "m2", {}, [], ["*"])
    assert exc.value.status == 403
    assert load_config("agent.json")["name"] == "m1"


def test_installer_names_the_server(pool_env):
    import urllib.request

    script = urllib.request.urlopen(pool_env["url"] + "/install.sh").read().decode()
    assert f"SERVER='{pool_env['url']}'" in script and "pool.cli agent register" in script
    package = urllib.request.urlopen(pool_env["url"] + "/v1/agent/package.tar.gz").read()
    names = tarfile.open(fileobj=io.BytesIO(package)).getnames()
    assert {"pool/agent.py", "pool/cli.py", "pool/models.py"} <= set(names)


# ---- placement (database level) ------------------------------------------------------------------


@pytest.fixture()
def db(db_url):
    database = open_db(db_url)
    database.run("insert into pools (id, name, settings, created_at) values ('p1', 'default', '{}', ?)", now())
    for member in ("ann", "bob"):
        database.run("insert into members (id, name, created_at) values (?, ?, ?)", member, member, now())
        database.run("insert into pool_members (pool, member, role, created_at) values ('p1', ?, 'member', ?)", member, now())
    return database


def add_machine(db, machine_id, owner="ann", gpus=(), free=None, policy=None, network=None, labels=None, memory=64, online=True):
    profile = {"gpus": [{"index": i, "memory_gb": g, **({"unified_memory": True} if u else {})} for i, (g, u) in enumerate(gpus)],
               "cpus": 16, "memory_gb": memory, "allow_images": ["*"], **({"network": network} if network else {})}
    free = free or {"gpus": [i for i in range(len(gpus))], "cpus": 16, "memory_gb": memory}
    db.run("insert into machines (id, pool, name, owner, labels, profile, policy, free, free_at, created_at, last_seen) "
           "values (?, 'p1', ?, ?, ?, ?, ?, ?, ?, ?, ?)", machine_id, machine_id, owner, json.dumps(labels or {}),
           json.dumps(profile), json.dumps(policy or {}), json.dumps(free), now() if online else None, now(), now() if online else 0)
    return {"id": machine_id, "pool": "p1", "name": machine_id, "owner": owner, "labels": labels or {}, "profile": profile,
            "policy": policy or {}, "free": free, "drain": 0}


def add_job(db, owner="ann", created=None, spec=None, status="queued", **extra):
    job_id = "j" + uuid.uuid4().hex[:8]
    spec = specs.normalize({"image": "x", "command": ["true"], **(spec or {})})
    db.run("insert into jobs (id, pool, owner, name, priority, spec, status, created_at, started_at, finished_at) "
           "values (?, 'p1', ?, 'x', ?, ?, ?, ?, ?, ?)", job_id, owner, spec["priority"], json.dumps(spec), status,
           created if created is not None else now(), extra.get("started_at"), extra.get("finished_at"))
    return job_id


def add_artifact(db, size, owner="ann", created=None, pieces=4):
    artifact = "sha256:" + uuid.uuid4().hex * 2
    shas = [uuid.uuid4().hex * 2 for _ in range(pieces)]
    db.run("insert into artifacts (id, kind, size, files, manifest, created_at) values (?, 'data', ?, 1, '{}', ?)", artifact, size, now())
    for sha in shas:
        db.run("insert into chunks (sha, size, created_at) values (?, ?, ?)", sha, size // pieces, created or now())
        db.run("insert into artifact_chunks (artifact, chunk) values (?, ?)", artifact, sha)
    db.run("insert into artifact_owners (artifact, member, pool, created_at) values (?, ?, 'p1', ?)", artifact, owner, created or now())
    return artifact, shas


FREE = {"gpus": [0, 1], "cpus": 16, "memory_gb": 64}


def job_spec(**resources) -> dict:
    return specs.normalize({"image": "nvcr.io/nvidia/pytorch:25.01-py3", "command": ["python", "x.py"], "resources": resources})


def test_fits_resources_labels_images_policy():
    machine = {"id": "a1", "owner": "ann", "labels": {"zone": "home"}, "policy": {},
               "profile": {"gpus": [{"index": 0, "memory_gb": 24}, {"index": 1, "memory_gb": 80}], "allow_images": ["nvcr.io/*"]}}
    assert scheduler.fits(job_spec(), machine, FREE) == []
    assert scheduler.fits(job_spec(gpus=1), machine, FREE) == [0]
    assert scheduler.fits(job_spec(gpus=1, gpu_memory_gb=40), machine, FREE) == [1]
    assert scheduler.fits(job_spec(gpus=2, gpu_memory_gb=40), machine, FREE) is None
    assert scheduler.fits(job_spec(memory_gb=100), machine, FREE) is None
    assert scheduler.fits(job_spec(labels={"zone": "lab"}), machine, FREE) is None
    assert scheduler.fits(specs.normalize({"image": "docker.io/evil/miner", "command": ["x"]}), machine, FREE) is None
    owner_only = {**machine, "policy": {"who": "owner"}}
    assert scheduler.fits(job_spec(), owner_only, FREE, "ann") == []
    assert scheduler.fits(job_spec(), owner_only, FREE, "bob") is None
    listed = {**machine, "policy": {"who": ["bob"]}}
    assert scheduler.fits(job_spec(), listed, FREE, "bob", "bob") == []


def test_a_job_for_own_machines_runs_only_on_its_submitters():
    machine = {"id": "a1", "owner": "ann", "labels": {}, "policy": {}, "profile": {"gpus": [], "allow_images": ["*"]}}
    own = job_spec(machines="own")
    assert own["resources"]["machines"] == "own" and "machines" not in job_spec()["resources"]  # older specs unchanged
    assert job_spec(machines="any")["resources"] == job_spec()["resources"]
    assert scheduler.fits(own, machine, FREE, "ann") == []
    why: list[str] = []
    assert scheduler.fits(own, machine, FREE, "bob", "ann", why) is None and "not your machine" in why[0]  # a name is not enough
    assert scheduler.fits(own, machine, FREE) is None  # no submitter known: no machine (fail closed)
    with pytest.raises(specs.SpecError, match="resources.machines"):
        job_spec(machines="mine")


def test_own_machines_keep_a_job_off_other_members_machines(pool_env, tmp_path):
    """Placement end to end: another member's machine is online and idle, and passes the job over
    (`explain` says why); the submitter's own machine runs it as soon as it is online."""
    ann = Client.member()
    bob = invite(pool_env, name="bob")
    start_agent(pool_env, "theirs", member=bob)
    code = ann.upload_folder(write(tmp_path / "src", {"x.txt": "secret"}), kind="code")["artifact"]
    spec = {"image": "x", "command": ["sh", "-c", "cat x.txt"], "code": code, "resources": {"machines": "own"}}
    job = ann.call("POST", "/v1/jobs", {"spec": spec})["job"]["id"]
    time.sleep(2)  # several polls of the other member's agent
    found = ann.call("GET", f"/v1/jobs/{job}")
    assert found["job"]["status"] == "queued" and any("theirs" in line and "not your machine" in line for line in found["why"])
    anyone = ann.call("POST", "/v1/jobs", {"spec": {**spec, "resources": {}}})["job"]["id"]
    assert wait_for(lambda: ann.call("GET", f"/v1/jobs/{anyone}")["job"]["status"] == "succeeded")
    assert ann.call("GET", f"/v1/jobs/{anyone}")["job"]["machine"] == "theirs"  # without the field, anyone's machine
    start_agent(pool_env, "mine")
    assert wait_for(lambda: ann.call("GET", f"/v1/jobs/{job}")["job"]["status"] == "succeeded")
    assert ann.call("GET", f"/v1/jobs/{job}")["job"]["machine"] == "mine"
    mine = {m["name"]: m["mine"] for m in ann.call("GET", "/v1/machines")["machines"]}
    assert mine == {"theirs": False, "mine": True}


def fake_model(folder: Path, experts=4, top_k=1) -> Path:
    tensors = {"model.embed.weight": ([8, 4], 64), "model.layers.0.attn.q.weight": ([4, 4], 32)}
    for e in range(experts):
        tensors[f"model.layers.0.mlp.experts.{e}.up.weight"] = ([4, 8], 64)
    header, offset = {}, 0
    for name, (shape, size) in tensors.items():
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode()
    folder.mkdir(parents=True)
    (folder / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * offset)
    (folder / "config.json").write_text(json.dumps({"architectures": ["FakeMoe"], "num_hidden_layers": 1, "hidden_size": 4,
                                                    "num_experts": experts, "num_experts_per_tok": top_k}))
    return folder


def test_describe_counts_active_parameters(tmp_path):
    meta = models.describe(fake_model(tmp_path / "m"))
    assert meta["params"] == 32 + 16 + 4 * 32
    assert meta["active_params"] == 32 + 16 + 32, "one of four experts is active"
    assert meta["experts"] == 4 and meta["top_k"] == 1 and meta["architecture"] == "FakeMoe"
    assert meta["bytes_by_dtype"] == {"BF16": 96 + 4 * 64}


def test_fits_with_a_model():
    meta = {"params": 36e9, "active_params": 4e9, "weight_bytes": int(37 * 1024**3)}
    spec = specs.normalize({"image": "x", "command": ["x"], "inputs": [{"path": "base", "artifact": "sha256:" + "a" * 64, "model": "m@t"}],
                            "resources": {"gpus": 1, "fits": {"model": "base", "mode": "full", "kv_gb": 4}}})
    spec["resources"]["fits"]["meta"] = meta
    rtx = {"id": "r", "owner": "ann", "labels": {}, "profile": {"gpus": [{"index": 0, "memory_gb": 24}], "allow_images": ["*"]}}
    assert scheduler.fits(spec, rtx, {"gpus": [0], "cpus": 8, "memory_gb": 64}) is None, "37 GB does not fit in 24 GB"
    spec["resources"]["fits"]["mode"] = "offload"
    assert scheduler.fits(spec, rtx, {"gpus": [0], "cpus": 8, "memory_gb": 64}) == [0], "active experts fit, the rest in RAM"
    assert scheduler.fits(spec, rtx, {"gpus": [0], "cpus": 8, "memory_gb": 30}) is None, "but not without the RAM"
    spec["resources"]["fits"]["mode"] = "full"
    spark = {"id": "s", "owner": "ann", "labels": {}, "profile": {"gpus": [{"index": 0, "memory_gb": 120, "unified_memory": True}], "allow_images": ["*"]}}
    assert scheduler.fits(spec, spark, {"gpus": [0], "cpus": 8, "memory_gb": 100}) == [0]
    assert scheduler.fits(spec, spark, {"gpus": [0], "cpus": 8, "memory_gb": 30}) is None, "unified memory is counted as RAM"
    need = models.need(meta, "stage:1/2", 0, 1)
    assert 18 < need["gpu_gb"] < 22


def test_fair_share_uses_recent_gpu_time(db):
    two_hours = 7200
    add_job(db, "ann", created=now() - 3 * 3600, spec={"resources": {"gpus": 1}}, status="succeeded",
            started_at=now() - 3 * 3600, finished_at=now() - 3600)
    old_ann = add_job(db, "ann", created=now() - 600)
    bob = add_job(db, "bob", created=now() - 10)
    usage = scheduler.usage(db, "p1")
    assert abs(usage["ann"] - two_hours) < 5
    order = [job["id"] for job in scheduler.fair_order(db, "p1", owner_first=False)]
    assert order.index(bob) < order.index(old_ann), "bob used no GPU time today"
    high = add_job(db, "ann", created=now(), spec={"priority": 5})
    order = [job["id"] for job in scheduler.fair_order(db, "p1", owner_first=False)]
    assert order.index(high) < order.index(old_ann), "priority orders a member's own jobs"
    assert scheduler.fair_order(db, "p1", machine_owner="ann")[0]["owner"] == "ann", "owner first on their machine"


def test_network_check_refuses_slow_links(db):
    artifact, _ = add_artifact(db, 1024**3)  # 1 GB of inputs
    slow = add_machine(db, "slow", network={"down_mbps": 10})
    quick = {"image": "x", "command": ["x"], "code": artifact, "cost": {"flops": 1e12, "dtype": "bf16"}}
    add_job(db, spec=quick, created=now() - 3600)
    verdict = scheduler.evaluate(db, load_job(db.one("select * from jobs")), slow, 0.25)
    assert not verdict["ok"] and verdict["t_net"] > 800
    assert scheduler.claim(db, slow, slow["free"]) is None
    lines = scheduler.explain(db, load_job(db.one("select * from jobs")))
    assert any("link too slow" in line for line in lines)
    db.run("delete from jobs")
    add_job(db, spec={**quick, "cost": {**quick["cost"], "bulk": True}}, created=now() - 3600)
    assert scheduler.claim(db, slow, slow["free"]) is not None, "bulk jobs skip the check"


def test_cache_affinity_waits_for_the_machine_with_the_inputs(db):
    artifact, shas = add_artifact(db, 4 * 1024**3)
    holder = add_machine(db, "holder")
    other = add_machine(db, "other")
    for sha in shas:
        db.run("insert into chunk_locations (chunk, machine, seen_at) values (?, 'holder', ?)", sha, now())
    job_id = add_job(db, spec={"code": artifact})
    assert scheduler.claim(db, other, other["free"]) is None, "holder would do better"
    assert scheduler.claim(db, holder, holder["free"])["id"] == job_id
    late = add_job(db, spec={"code": artifact}, created=now() - 120)  # waited past affinity_wait_s
    assert scheduler.claim(db, other, other["free"])["id"] == late


def test_explain_lists_each_machine(db):
    add_machine(db, "away", online=False)
    add_machine(db, "lab", labels={"zone": "lab"})
    add_machine(db, "home", labels={"zone": "home"})
    job = load_job(db.one("select * from jobs where id=?", add_job(db, spec={"resources": {"labels": {"zone": "home"}}})))
    lines = scheduler.explain(db, job)
    assert "away: offline" in lines
    assert any(line.startswith("lab: never fits (labels)") for line in lines)
    assert any(line.startswith("home: can run it") for line in lines)


def test_lost_machine_requeues_then_fails(db):
    job_id = add_job(db, spec={"retries": 1})
    machine = add_machine(db, "m")
    for expected in ("queued", "failed"):
        assert scheduler.claim(db, machine, machine["free"])["id"] == job_id
        db.run("update jobs set lease_expires=? where id=?", now() - 1, job_id)
        scheduler.expire_leases(db)
        assert load_job(db.one("select * from jobs where id=?", job_id))["status"] == expected
    assert "stopped reporting" in load_job(db.one("select * from jobs where id=?", job_id))["error"]


def test_a_job_is_claimed_once(db):
    add_job(db)
    machines = [add_machine(db, f"m{i}") for i in range(6)]
    results = []
    threads = [threading.Thread(target=lambda m=m: results.append(scheduler.claim(db, m, m["free"]))) for m in machines]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(1 for r in results if r) == 1


def test_gc_expires_unused_artifacts(db, tmp_path):
    store = ChunkStore(tmp_path / "chunks")
    old = now() - 40 * 86400
    expired, expired_chunks = add_artifact(db, 4000, created=old)
    kept, _ = add_artifact(db, 4000, created=old)
    add_job(db, spec={"code": kept})  # a queued job still needs it
    for sha in expired_chunks:  # stand-ins for the chunk files (their names are not real hashes of b"x")
        store.path(sha).parent.mkdir(parents=True, exist_ok=True)
        store.path(sha).write_bytes(b"x")
    result = collect_garbage(db, store)
    assert result["artifacts_deleted"] == 1 and result["chunks_deleted"] == 4
    assert not db.one("select 1 as x from artifacts where id=?", expired)
    assert db.one("select 1 as x from artifacts where id=?", kept)
    assert not any(store.path(sha).exists() for sha in expired_chunks)


# ---- small pieces -------------------------------------------------------------------------------


def test_hours_window_wraps_midnight():
    at = lambda h, m=0: datetime(2026, 1, 1, h, m)
    assert in_hours("22:00-08:00", at(23)) and in_hours("22:00-08:00", at(7, 59))
    assert not in_hours("22:00-08:00", at(8)) and not in_hours("22:00-08:00", at(12))
    assert in_hours("09:00-17:00", at(12)) and not in_hours("09:00-17:00", at(18))
    assert in_hours(None)


@pytest.mark.parametrize("bad, message", [
    ({"command": ["x"]}, "image"),
    ({"image": "x"}, "command"),
    ({"image": "x", "command": ["x"], "network": "egress"}, "network"),
    ({"image": "x", "command": ["x"], "network": "peers"}, "needs a gang"),
    ({"image": "x", "command": ["x"], "gang": {"size": 0}}, "gang.size"),
    ({"image": "x", "command": ["x"], "images": {"tegra": "y"}}, "label=value"),
    ({"image": "x", "command": ["x"], "env": {"POOL_JOB_ID": "x"}}, "POOL_"),
    ({"image": "x", "command": ["x"], "inputs": [{"path": "../etc", "artifact": "sha256:" + "a" * 64}]}, "input path"),
    ({"image": "x", "command": ["x"], "resources": {"gpus": -1}}, "gpus"),
    ({"image": "x", "command": ["x"], "resources": {"fits": {"model": "base"}}}, "must name a model input"),
    ({"image": "x", "command": ["x"], "cost": {"dtype": "fp7"}}, "dtype"),
])
def test_bad_specs(bad, message):
    with pytest.raises(specs.SpecError, match=message):
        specs.normalize(bad)


def test_cli_fit_and_settings_parsing():
    args = cli.build_parser().parse_args(["run", "--gpus", "1", "--fit", "base:stage:1/2:4", "--model", "base=qwen@fp8", "--", "x"])
    assert cli._resources(args)["fits"] == {"model": "base", "mode": "stage:1/2", "kv_gb": 4.0}
    assert cli._settings(["alpha=0.5", "quota.gpu_hours_per_day=10", "quota.storage_gb=none"]) == \
        {"alpha": 0.5, "quota": {"gpu_hours_per_day": 10, "storage_gb": None}}


def test_model_ref_end_to_end(pool_env, capsys):
    start_agent(pool_env)
    folder = fake_model(pool_env["tmp"] / "fake-moe")
    assert cli.main(["model", "add", "fake-moe@v1", str(folder)]) == 0
    assert "fake-moe@v1" in capsys.readouterr().out
    assert cli.main(["model", "ls"]) == 0
    assert "FakeMoe" in capsys.readouterr().out
    code = write(pool_env["tmp"] / "c", {"read.py": "import json, os; print(json.load(open(os.environ['POOL_INPUTS'] + '/base/config.json'))['architectures'])"})
    assert cli.main(["run", "--code", str(code), "--model", "base=fake-moe", "--no-get", "--", "python3", "read.py"]) == 0
    assert "['FakeMoe']" in capsys.readouterr().out
    bob = invite(pool_env)  # models are shared with the pool
    assert bob.call("GET", "/v1/models")["models"][0]["model"] == "fake-moe@v1"
    reply = Client.member().call("POST", "/v1/jobs", {"spec": {  # a fit needs GPUs, and this machine has none
        "image": "x", "command": ["true"], "inputs": [{"path": "base", "model": "fake-moe@v1"}],
        "resources": {"gpus": 1, "fits": {"model": "base"}}}})
    assert "no machine in the pool can run this job" in reply["warning"]
    Client.member().call("POST", f"/v1/jobs/{reply['job']['id']}/cancel")


# ---- sandbox sizes (spike S1) -----------------------------------------------------------------------


def _docker_args(resources: dict) -> list[str]:
    from pool.agent import DockerRuntime

    spec = specs.normalize({"image": "img", "command": ["python", "x.py"], "resources": resources})
    job = {"id": "j1", "tries": 1, "gpus": [0], "spec": spec}
    dirs = {"inputs": {}, "outputs": Path("/tmp/o"), "checkpoint": Path("/tmp/c")}
    return DockerRuntime("a1").command(job, dirs, {}, spec["resources"]["memory_gb"])


def _flag(args: list[str], name: str) -> str:
    return args[args.index(name) + 1]


def test_sandbox_defaults_stay_locked_down():
    args = _docker_args({"memory_gb": 16})
    assert _flag(args, "--network") == "none" and _flag(args, "--cap-drop") == "ALL"
    assert "--read-only" in args and _flag(args, "--security-opt") == "no-new-privileges"
    assert _flag(args, "--pids-limit") == "4096" and _flag(args, "--shm-size") == "4096m"
    assert _flag(args, "--tmpfs") == "/tmp:rw,exec,nosuid,size=8192m"
    assert "--ipc" not in args and "--cap-add" not in args and "--privileged" not in args


def test_jobs_may_ask_for_bigger_shm_tmp_and_pids():
    args = _docker_args({"memory_gb": 64, "shm_gb": 16, "tmp_gb": 32, "pids": 16384})
    assert _flag(args, "--shm-size") == "16384m" and _flag(args, "--pids-limit") == "16384"
    assert _flag(args, "--tmpfs") == "/tmp:rw,exec,nosuid,size=32768m"
    assert _flag(args, "--network") == "none" and _flag(args, "--cap-drop") == "ALL"


@pytest.mark.parametrize("resources", [{"memory_gb": 8, "shm_gb": 16}, {"pids": 10**6}, {"tmp_gb": 0}, {"shm_gb": "big"}])
def test_sandbox_sizes_are_bounded(resources):
    with pytest.raises(specs.SpecError):
        specs.normalize({"image": "x", "command": ["y"], "resources": resources})


# ---- the LAN prototype: peers, islands, gangs -----------------------------------------------------------


def test_peer_tickets_and_headers():
    key = "ab" * 32
    good = peers.ticket(key, "m1", "c" * 64)
    assert peers.check_ticket(key, good, "c" * 64)
    assert not peers.check_ticket(key, good, "d" * 64)  # a ticket is for one chunk
    assert not peers.check_ticket("cd" * 32, good, "c" * 64)  # and one pool
    assert not peers.check_ticket(key, peers.ticket(key, "m1", "c" * 64, expires=time.time() - 1), "c" * 64)
    assert peers.check_peer_header(key, peers.peer_header(key, "m1")) == "m1"
    assert peers.check_peer_header("cd" * 32, peers.peer_header(key, "m1")) is None
    assert peers.check_peer_header(key, "garbage") is None


def test_islands_group_fast_links_only():
    links = [{"src": "a", "dst": "b", "mbps": 25000, "rtt_ms": 0.2},
             {"src": "b", "dst": "c", "mbps": 300, "rtt_ms": 3.0}]
    assert peers.islands(["a", "b", "c", "d"], links) == [["a", "b"], ["c"], ["d"]]


def test_gang_and_image_specs():
    spec = specs.normalize({"image": "x86img", "command": ["x"], "gang": {"size": 3}, "network": "peers",
                            "images": {"gpu=tegra": "l4t-img"}})
    assert spec["gang"] == {"size": 3} and spec["network"] == "peers"
    assert specs.image_for(spec, {"gpu": "tegra"}) == "l4t-img"
    assert specs.image_for(spec, {"gpu": "discrete"}) == "x86img"


def test_gangs_need_peer_capable_machines():
    plain = {"id": "a", "owner": "ann", "labels": {}, "policy": {}, "profile": {"gpus": [], "allow_images": ["*"]}}
    addressed = {**plain, "profile": {**plain["profile"], "peer": {"addr": "10.0.0.2", "port": 8766}}}
    networked = {**plain, "profile": {**plain["profile"], "peer": {"addr": "10.0.0.2", "port": 8766, "network": True}}}
    gang = specs.normalize({"image": "x", "command": ["x"], "gang": {"size": 2}})
    peered = specs.normalize({"image": "x", "command": ["x"], "gang": {"size": 2}, "network": "peers"})
    why: list = []
    assert scheduler.fits(gang, plain, FREE, why=why) is None and "peer address" in why[0]
    assert scheduler.fits(gang, addressed, FREE) == []
    assert scheduler.fits(peered, addressed, FREE) is None
    assert scheduler.fits(peered, networked, FREE) == []
    tegra_only = {**networked, "profile": {**networked["profile"], "allow_images": ["l4t-*"]}, "labels": {"gpu": "tegra"}}
    both = specs.normalize({"image": "x", "command": ["x"], "images": {"gpu=tegra": "l4t-img"}})
    assert scheduler.fits(both, tegra_only, FREE) == []  # the allowlist sees the image this machine would run


def test_peer_network_runs_with_the_host_network():
    from pool.agent import DockerRuntime

    spec = specs.normalize({"image": "img", "command": ["x"], "gang": {"size": 2}, "network": "peers"})
    job = {"id": "j1", "tries": 1, "gpus": [], "spec": spec, "image": "l4t-img"}
    args = DockerRuntime("a1").command(job, {"inputs": {}, "outputs": Path("/tmp/o"), "checkpoint": Path("/tmp/c")}, {}, 4)
    assert _flag(args, "--network") == "host" and _flag(args, "--cap-drop") == "ALL" and "--read-only" in args
    assert args[-2:] == ["l4t-img", "x"]


def _gang(db, size=2, retries=1):
    ids = []
    for rank in range(size):
        job_id = add_job(db, spec={"retries": retries})
        spec = json.loads(db.one("select spec from jobs where id=?", job_id)["spec"])
        spec["gang"] = {"id": "gabc123", "size": size, "rank": rank}
        db.run("update jobs set spec=? where id=?", json.dumps(spec), job_id)
        db.run("insert into gang_members (gang, rank, job) values ('gabc123', ?, ?)", rank, job_id)
        ids.append(job_id)
    return ids


def _peer_machine(db, name):
    machine = add_machine(db, name)
    machine["profile"]["peer"] = {"addr": f"10.0.0.{len(name)}", "port": 8766}
    db.run("update machines set profile=? where id=?", json.dumps(machine["profile"]), name)
    return machine


def _status(db, job_id):
    return load_job(db.one("select * from jobs where id=?", job_id))


def test_gang_ranks_go_to_different_machines_and_meet(db):
    r0, r1 = _gang(db)
    a, b = _peer_machine(db, "a"), _peer_machine(db, "bb")
    assert scheduler.claim(db, a, a["free"])["id"] == r0
    assert scheduler.claim(db, a, a["free"]) is None  # one rank per machine
    assert scheduler.rendezvous(db, _status(db, r0)) == {"ready": False, "waiting": [1]}
    assert scheduler.claim(db, b, b["free"])["id"] == r1
    meet = scheduler.rendezvous(db, _status(db, r1))
    assert meet["ready"] and meet["rank"] == 1 and meet["world_size"] == 2 and meet["master_addr"] == "10.0.0.1"
    assert [p["machine"] for p in meet["peers"]] == ["a", "bb"]


def test_a_lost_rank_sends_the_whole_gang_back(db):
    r0, r1 = _gang(db)
    a, b = _peer_machine(db, "a"), _peer_machine(db, "bb")
    scheduler.claim(db, a, a["free"])
    scheduler.claim(db, b, b["free"])
    db.run("update jobs set lease_expires=? where id=?", now() - 1, r1)
    scheduler.expire_leases(db)
    assert _status(db, r0)["status"] == "queued" and _status(db, r0)["preemptions"] == 1  # no retry used
    assert _status(db, r1)["status"] == "queued"


def test_a_rank_out_of_retries_fails_the_gang(db):
    r0, r1 = _gang(db, retries=0)
    a, b = _peer_machine(db, "a"), _peer_machine(db, "bb")
    scheduler.claim(db, a, a["free"])
    scheduler.claim(db, b, b["free"])
    db.run("update jobs set lease_expires=? where id=?", now() - 1, r0)
    scheduler.expire_leases(db)
    assert _status(db, r0)["status"] == "failed"
    assert _status(db, r1)["status"] == "failed" and "rank 0 failed" in _status(db, r1)["error"]


def test_agents_find_each_other_and_measure_links(pool_env):
    start_agent(pool_env, "box1")
    start_agent(pool_env, "box2")
    member = Client.member()
    reply = wait_for(lambda: (r := member.call("GET", "/v1/links")) and len(r["links"]) == 2 and r)
    assert {(l["src"], l["dst"]) for l in reply["links"]} == {("box1", "box2"), ("box2", "box1")}
    assert all(l["mbps"] > 0 for l in reply["links"])
    assert cli.main(["machine", "links"]) == 0


def test_a_machine_fetches_chunks_from_a_peer(pool_env, capsys, small_chunks):
    one = start_agent(pool_env, "box1")
    data = write(pool_env["tmp"] / "data", {"numbers.txt": "\n".join(str(i) for i in range(2000))})
    code = write(pool_env["tmp"] / "code", {"train.py": TRAIN})
    two = start_agent(pool_env, "box2")
    two.drain = True
    assert cli.main(["machine", "drain", "box2"]) == 0
    assert cli.main(["run", "--code", str(code), "--data", str(data), "--no-get", "--", "python3", "train.py"]) == 0
    assert "on box1" in capsys.readouterr().out
    assert cli.main(["machine", "drain", "box1"]) == 0
    assert cli.main(["machine", "drain", "box2", "--off"]) == 0
    wait_for(lambda: Client.member().call("GET", "/v1/machines")["machines"][0]["cached_gb"] is not None and
             pool_env["db"].one("select count(*) as n from chunk_locations l join machines m on m.id=l.machine where m.name='box1'")["n"] > 0)
    assert cli.main(["run", "--code", str(code), "--data", str(data), "--no-get", "--", "python3", "train.py"]) == 0
    out = capsys.readouterr().out
    assert "on box2" in out and "from peer box1" in out and "sum 1999000" in out
    assert one.peer_server.served_bytes > 0


def test_peer_refuses_chunks_without_a_ticket(pool_env, small_chunks):
    import urllib.error
    import urllib.request

    agent = start_agent(pool_env, "box1")
    wait_for(lambda: agent.peer_key)
    base = f"http://127.0.0.1:{agent.peer_server.port}"
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(f"{base}/peer/chunks/{'a' * 64}", timeout=5)
    assert caught.value.code == 403
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(f"{base}/peer/speedtest?bytes=10", timeout=5)
    assert caught.value.code == 403


GANG = """
import json, os
print("rank", os.environ["RANK"], "of", os.environ["WORLD_SIZE"], "master", os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"],
      "peers", len(json.loads(os.environ["POOL_PEERS"])), flush=True)
"""


def test_gang_runs_one_rank_per_machine(pool_env, capsys):
    start_agent(pool_env, "box1")
    start_agent(pool_env, "box2")
    code = write(pool_env["tmp"] / "gang", {"gang.py": GANG})
    assert cli.main(["run", "--gang", "2", "--code", str(code), "--no-get", "--", "python3", "gang.py"]) == 0
    out = capsys.readouterr().out
    assert "rank 0 of 2 master 127.0.0.1" in out and "peers 2" in out
    jobs = Client.member().call("GET", "/v1/jobs")["jobs"]
    ranks = [j for j in jobs if j["name"].startswith("gang.r")]
    assert len(ranks) == 2 and {j["status"] for j in ranks} == {"succeeded"}
    assert len({j["machine"] for j in ranks}) == 2


def test_a_failing_rank_fails_the_gang(pool_env, capsys):
    start_agent(pool_env, "box1")
    start_agent(pool_env, "box2")
    code = write(pool_env["tmp"] / "gang", {"gang.py": "import os, sys, time\nif os.environ['RANK'] == '1': sys.exit(3)\ntime.sleep(60)\n"})
    assert cli.main(["run", "--gang", "2", "--code", str(code), "--no-get", "-d", "--", "python3", "gang.py"]) == 0
    first = capsys.readouterr().out.strip().splitlines()[-1]
    final = wait_for(lambda: (j := job_state(first))["status"] in ("failed", "succeeded") and j, timeout=30)
    assert final["status"] == "failed" and "rank 1 failed" in final["error"]


def test_a_jobs_gpus_in_its_log():
    """#163: "GPUs none" read like a missing GPU; a CPU job says it asked for none."""
    from pool.agent import gpu_note

    assert gpu_note({"gpus": [0], "spec": {"resources": {"gpus": 1}}}) == "GPUs [0]"
    assert gpu_note({"gpus": [], "spec": {"resources": {"gpus": 0}}}) == "no GPU requested"
    assert gpu_note({"gpus": None, "spec": {"resources": {"gpus": 1}}}) == "GPUs none"


def test_an_image_that_cannot_be_pulled_is_image_unavailable(monkeypatch):
    from types import SimpleNamespace

    from pool import agent as agent_module
    from pool.agent import DockerRuntime, ImageUnavailable

    calls = []

    def fake_run(args, **kwargs):
        calls.append(args[:3])
        return SimpleNamespace(returncode=1, stdout="", stderr="Error response from daemon: unauthorized")

    monkeypatch.setattr(agent_module.subprocess, "run", fake_run)
    with pytest.raises(ImageUnavailable, match="image ghcr.io/x/grader:l4t-r36 is not on this machine.*unauthorized"):
        DockerRuntime("a1").prepare("ghcr.io/x/grader:l4t-r36", lambda text: None)
    assert calls == [["docker", "image", "inspect"], ["docker", "pull", "-q"]]
