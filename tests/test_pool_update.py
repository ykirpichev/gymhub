"""Agents keeping up with the server's pool package (pool/update.py)."""

import io
import json
import shutil
import tarfile
import threading
import time
from pathlib import Path

import pytest

import pool
from pool import update
from pool.agent import Agent, ProcessRuntime, register
from pool.client import Client
from pool.server import package_tarball

from tests.test_pool import db_url, pool_env, wait_for  # noqa: F401  (fixtures)

PACKAGE = Path(pool.__file__).resolve().parent


def tarball(files: dict[str, bytes], extra=None) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        if extra:
            extra(tar)
    return buffer.getvalue()


def installed_copy(tmp_path: Path) -> Path:
    """A copy of the package as install.sh lays it out (no pyproject.toml next to it)."""
    root = tmp_path / "share" / "pool"
    shutil.copytree(PACKAGE, root, ignore=shutil.ignore_patterns("__pycache__"))
    return root


def test_served_package_matches_the_folder_digest():
    files = update.read_package(package_tarball())
    assert update.files_digest(files) == update.package_digest(PACKAGE)


def test_package_with_unexpected_entries_is_refused():
    good = {"pool/__init__.py": b"", "pool/agent.py": b""}
    for bad in ({**good, "pool/../evil.py": b"x"}, {**good, "pool/sub/x.py": b"x"}, {**good, "other.py": b"x"}):
        with pytest.raises(ValueError):
            update.read_package(tarball(bad))

    def link(tar):
        info = tarfile.TarInfo("pool/link.py")
        info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
        tar.addfile(info)

    with pytest.raises(ValueError):
        update.read_package(tarball(good, extra=link))
    with pytest.raises(ValueError):
        update.read_package(tarball({"pool/__init__.py": b""}))  # no agent.py


def test_install_swaps_files_and_checks_the_digest(tmp_path):
    root = installed_copy(tmp_path)
    (root / "stale.py").write_text("old = True\n")
    files = {f"pool/{p.name}": p.read_bytes() for p in PACKAGE.glob("*.py")}
    files["pool/agent.py"] += b"\n# newer\n"
    data = tarball(files)
    with pytest.raises(ValueError):
        update.install(data, root, expect="0" * 64)
    assert not (root / "agent.py").read_bytes().endswith(b"# newer\n")  # untouched after a refusal
    digest = update.install(data, root, expect=update.files_digest({k.split("/", 1)[1]: v for k, v in files.items()}))
    assert (root / "agent.py").read_bytes().endswith(b"# newer\n")
    assert not (root / "stale.py").exists()
    assert update.package_digest(root) == digest
    assert not (root.parent / ".pool-new").exists() and not (root.parent / ".pool-old").exists()


def test_a_source_checkout_is_never_overwritten():
    assert update.source_checkout(PACKAGE)  # this repo has pyproject.toml next to pool/
    with pytest.raises(RuntimeError):
        update.install(package_tarball(), PACKAGE)


def make_agent(env, tmp_path, auto: bool) -> tuple[Agent, Path]:
    member = Client.member()
    code = member.call("POST", "/v1/machine-codes", {"name": "box1"})["code"]
    config = register(env["url"], code, "box1", {}, [], ["*"])
    config["auto_update"] = auto
    agent = Agent(config, tmp_path / "work", runtime=ProcessRuntime(), cpus=4, memory_gb=8, cache_gb=1, peer_port=0,
                  peer_addr="127.0.0.1")
    root = installed_copy(tmp_path)
    (root / "agent.py").write_bytes((root / "agent.py").read_bytes() + b"\n# older\n")
    agent.package_root, agent.own_digest = root, update.package_digest(root)
    agent.profile["agent_digest"] = agent.own_digest
    env["agents"].append(agent)
    return agent, root


def test_agent_with_auto_update_installs_the_servers_package_and_restarts(pool_env, tmp_path):
    agent, root = make_agent(pool_env, tmp_path, auto=True)
    thread = threading.Thread(target=agent.run_forever, daemon=True)
    thread.start()
    thread.join(timeout=20)
    assert not thread.is_alive() and agent.restart
    assert update.package_digest(root) == update.package_digest(PACKAGE)


def test_agent_without_auto_update_only_says_so(pool_env, tmp_path):
    agent, root = make_agent(pool_env, tmp_path, auto=False)
    before = update.package_digest(root)
    threading.Thread(target=agent.run_forever, daemon=True).start()
    wait_for(lambda: agent.server_digest)
    assert agent.maybe_update() is False
    assert not agent.restart and update.package_digest(root) == before
    assert agent.update_noted == agent.server_digest


def test_agent_waits_for_its_jobs_before_updating(pool_env, tmp_path):
    agent, root = make_agent(pool_env, tmp_path, auto=True)
    agent.server_digest = update.package_digest(PACKAGE)
    agent.running["j1"] = object()
    assert agent.maybe_update() is False and not agent.restart
    agent.running.clear()
    assert agent.maybe_update() is True and agent.restart


# ---- agents report their version (profile agent_digest); the server tells current from outdated -----------

def machines() -> dict:
    return {m["name"]: m for m in Client.member().call("GET", "/v1/machines")["machines"]}


def test_agent_reports_its_digest_and_the_listing_tells_current_outdated_and_unknown(pool_env, tmp_path, capsys):
    from pool import cli
    from tests.test_pool import start_agent

    current = update.package_digest(PACKAGE)
    start_agent(pool_env, "box-new")  # runs this repo's files: the server's own package
    old, _ = make_agent(pool_env, tmp_path, auto=False)  # box1: one file differs
    threading.Thread(target=old.run_forever, daemon=True).start()
    # an agent from before agents reported their version: no agent_digest at registration or in heartbeats
    code = Client.member().call("POST", "/v1/machine-codes", {"name": "box-old"})["code"]
    bare = {"cpus": 2, "memory_gb": 4, "gpus": []}
    token = Client(pool_env["url"]).call("POST", "/v1/agent/register",
                                         {"code": code, "name": "box-old", "labels": {}, "profile": bare})["token"]
    Client(pool_env["url"], token).call("POST", "/v1/agent/heartbeat", {"running": [], "profile": bare})

    wait_for(lambda: machines()["box1"]["profile"].get("auto_update") is False)
    found = machines()
    assert found["box-new"]["profile"]["agent_digest"] == current
    assert found["box-new"]["agent"] == {"digest": current, "short": current[:12], "state": "current", "auto_update": False}
    assert found["box1"]["agent"]["state"] == "outdated" and found["box1"]["agent"]["digest"] == old.own_digest != current
    assert found["box-old"]["agent"] == {"digest": None, "short": None, "state": "unknown"}
    assert Client.member().call("GET", "/v1/machines")["agent"] == current

    capsys.readouterr()
    assert cli.main(["machine", "ls"]) == 0
    rows = {line.split()[0]: line.rstrip() for line in capsys.readouterr().out.splitlines()[2:]}
    assert rows["box-new"].endswith(f"{current[:12]} current")
    assert rows["box1"].endswith(f"{old.own_digest[:12]} outdated")
    assert rows["box-old"].endswith("-            unknown")

    # pool show: a queued job lists each machine's agent next to why it waits
    job = Client.member().call("POST", "/v1/jobs", {"spec": {"image": "x", "command": ["true"],
                                                             "resources": {"labels": {"zone": "nowhere"}}}})["job"]
    assert cli.main(["show", job["id"]]) == 0
    out = capsys.readouterr().out
    assert f"agent on box1: {old.own_digest[:12]} outdated" in out and f"agent on box-new: {current[:12]} current" in out
    assert "agent on box-old: -            unknown" in out


def test_a_failed_self_update_appears_in_the_profile(pool_env, tmp_path, monkeypatch, capsys):
    from pool import cli

    agent, root = make_agent(pool_env, tmp_path, auto=True)

    def denied(data, package_root, expect=None):  # what a root-owned /opt/pool did to the Spark's agent
        raise PermissionError(13, "Permission denied", str(Path(package_root).with_name(".pool-new")))

    monkeypatch.setattr(update, "install", denied)
    threading.Thread(target=agent.run_forever, daemon=True).start()
    wait_for(lambda: machines()["box1"]["agent"].get("update_error"), timeout=30)
    seen = machines()["box1"]["agent"]
    assert seen["state"] == "outdated" and seen["auto_update"] is True and not agent.restart
    message = seen["update_error"]["message"]
    assert message == "PermissionError: [Errno 13] Permission denied: '../.pool-new'", message
    assert str(tmp_path) not in message and abs(seen["update_error"]["at"] - time.time()) < 60
    capsys.readouterr()
    assert cli.main(["machine", "ls"]) == 0
    assert "agent self-update failed 0 min ago: PermissionError" in capsys.readouterr().out


def test_update_errors_are_short_and_keep_paths_inside_the_package(tmp_path):
    root = tmp_path / "opt" / "pool"
    error = update.short_error(OSError(f"cannot write {root / 'agent.py'}\nwhile reading /etc/shadow"), root)
    assert error == "OSError: cannot write ../pool/agent.py while reading <path>"
    assert len(update.short_error(RuntimeError("x" * 1000), root)) == 200


def test_metrics_flag_machines_with_an_outdated_agent(db_url):
    from pool import metrics
    from pool.db import now, open_db
    from tests.test_pool_migrations import samples

    db = open_db(db_url)
    db.run("insert into pools (id, name, settings, created_at) values ('p1', 'lab', '{}', ?)", now())
    for machine, digest in (("a", "abc"), ("b", "def"), ("c", None)):
        profile = json.dumps({"agent_digest": digest} if digest else {})
        db.run("insert into machines (id, pool, name, owner, profile, created_at, last_seen) values (?, 'p1', ?, 'm', ?, ?, ?)",
               machine, machine, profile, now(), now())
    found = samples(metrics.render(db, "1", None, "abc"))
    assert found['pool_machine_agent_outdated{pool="lab",machine="a"}'] == 0
    assert found['pool_machine_agent_outdated{pool="lab",machine="b"}'] == 1
    assert found['pool_machine_agent_outdated{pool="lab",machine="c"}'] == 1, "no digest reported: an agent from before"
