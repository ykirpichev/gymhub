"""The one-command installer (pool/installer.py) and the owner's commands (#133, worker-v2 section 3).

The script itself runs on real machines only; here: what the server serves (the package digest, the
refusal of plain http on a public address), the self-unregister endpoint, the agent's settings path,
the owner's pause, and uninstall of a --no-sudo install.
"""

import hashlib
import shutil
import subprocess
import time

import pytest
from fastapi.testclient import TestClient

from pool import agent as agent_mod, cli
from pool.agent import pause_path, register
from pool.client import Client, PoolError, load_config
from pool.server import create_app, lan_host, package_tarball

from tests.test_pool import db_url, job_state, pool_env, start_agent, wait_for, write  # noqa: F401  (fixtures)


def member_headers(app) -> dict:
    client = TestClient(app)
    token = client.post("/v1/login", json={"code": app.state.bootstrap_code, "name": "ann"}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_package_is_byte_for_byte_stable():
    assert package_tarball() == package_tarball()


def test_install_script_carries_the_package_sha256(tmp_path):
    app = create_app(tmp_path / "s", public_url="https://pool.example.com", gc_seconds=None)
    client = TestClient(app)
    script = client.get("/install.sh").text
    package = client.get("/v1/agent/package.tar.gz").content
    digest = hashlib.sha256(package).hexdigest()
    assert f"PACKAGE_SHA256='{digest}'" in script and "SERVER='https://pool.example.com'" in script
    assert "__SERVER__" not in script and "__SHA256__" not in script
    assert client.get("/healthz").json()["package_sha256"] == digest

    reply = client.post("/v1/machine-codes", json={"name": "box1"}, headers=member_headers(app)).json()
    assert reply["sha256"] == digest
    assert reply["install"] == f"curl -fsSL https://pool.example.com/install.sh | sudo sh -s -- --code {reply['code']} --name box1"
    assert "| sh -s -- --no-sudo --code" in reply["install_no_sudo"]


def test_install_script_is_valid_posix_sh(tmp_path):
    app = create_app(tmp_path / "s", public_url="http://192.168.1.10:8765", gc_seconds=None)
    path = tmp_path / "install.sh"
    path.write_text(TestClient(app).get("/install.sh").text)
    for shell in ("sh", "dash"):
        if shutil.which(shell):
            result = subprocess.run([shell, "-n", str(path)], capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
    if shutil.which("shellcheck"):
        result = subprocess.run(["shellcheck", "-s", "sh", str(path)], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout


def test_install_script_refuses_a_package_it_did_not_expect(tmp_path):
    app = create_app(tmp_path / "s", public_url="http://127.0.0.1:1", gc_seconds=None)
    path = tmp_path / "install.sh"
    path.write_text(TestClient(app).get("/install.sh").text)
    result = subprocess.run(["sh", str(path), "--code", "x", "--sha256", "0" * 64], capture_output=True, text=True)
    assert result.returncode != 0 and "not the " + "0" * 64 in result.stderr


@pytest.mark.parametrize("url", ["http://pool.example.com", "http://8.8.8.8:8765"])
def test_plain_http_on_a_public_address_is_refused(tmp_path, url):
    app = create_app(tmp_path / "s", public_url=url, gc_seconds=None)
    client = TestClient(app)
    assert client.get("/install.sh").status_code == 400
    reply = client.post("/v1/machine-codes", json={}, headers=member_headers(app))
    assert reply.status_code == 400 and "https" in reply.json()["detail"]


@pytest.mark.parametrize("url", ["https://pool.example.com", "http://192.168.1.10:8765", "http://spark:8765",
                                 "http://spark-10dc.local:8765", "http://127.0.0.1:8765", "http://100.101.1.2:8765",
                                 "http://[fd00::1]:8765"])
def test_https_or_the_lan_is_allowed(tmp_path, url):
    app = create_app(tmp_path / "s", public_url=url, gc_seconds=None)
    client = TestClient(app)
    assert client.get("/install.sh").status_code == 200
    assert client.post("/v1/machine-codes", json={}, headers=member_headers(app)).status_code == 200


def test_lan_host():
    assert lan_host("10.1.2.3") and lan_host("localhost") and lan_host("jetson") and lan_host("box.lan")
    assert not lan_host("1.1.1.1") and not lan_host("pool.example.com") and not lan_host(None)


def test_machine_unregisters_itself(pool_env):
    code = Client.member().call("POST", "/v1/machine-codes", {})["code"]
    config = register(pool_env["url"], code, "m1", {}, [], ["*"])
    machine = Client(config["server"], config["token"])
    me = machine.call("GET", "/v1/agent/me")
    assert me["name"] == "m1" and me["running"] == [] and me["drain"] is False
    assert machine.call("DELETE", "/v1/agent/me")["name"] == "m1"
    with pytest.raises(PoolError) as exc:
        machine.call("GET", "/v1/agent/me")
    assert exc.value.status == 401
    assert Client.member().call("GET", "/v1/machines")["machines"] == []


def test_config_path_from_flag_or_environment(pool_env, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("POOL_AGENT_CONFIG", "")  # empty means unset; restored after the test (cli sets it for --config)
    assert agent_mod.config_path() == tmp_path / "config" / "agent.json"
    etc = tmp_path / "etc" / "agent.json"
    code = Client.member().call("POST", "/v1/machine-codes", {})["code"]
    assert cli.main(["agent", "register", "--config", str(etc), "--server", pool_env["url"], "--code", code, "--name", "sys1"]) == 0
    assert etc.is_file() and (etc.stat().st_mode & 0o777) == 0o600
    assert not load_config("agent.json")  # not in the member settings folder
    assert agent_mod.load_agent_config()["name"] == "sys1"
    monkeypatch.setenv("POOL_AGENT_CONFIG", str(etc))
    assert agent_mod.pause_path() == tmp_path / "etc" / "agent.paused"
    capsys.readouterr()
    assert cli.main(["agent", "pause"]) == 0
    assert (tmp_path / "etc" / "agent.paused").exists()
    assert cli.main(["agent", "status"]) == 0
    out = capsys.readouterr().out
    assert "sys1" in out and "paused    yes" in out and "running jobs: none" in out
    assert cli.main(["agent", "resume"]) == 0
    assert not (tmp_path / "etc" / "agent.paused").exists()


def test_paused_agent_takes_no_new_jobs_and_resumes(pool_env, capsys):
    pause_path().parent.mkdir(parents=True, exist_ok=True)
    pause_path().touch()
    agent = start_agent(pool_env)
    code = write(pool_env["tmp"] / "p", {"hi.py": "print('hi')"})
    assert cli.main(["run", "-d", "--code", str(code), "--", "python3", "hi.py"]) == 0
    job_id = capsys.readouterr().out.strip().splitlines()[-1]
    time.sleep(2)
    assert agent.owner_paused and job_state(job_id)["status"] == "queued"
    pause_path().unlink()
    wait_for(lambda: job_state(job_id)["status"] == "succeeded")
    assert not agent.owner_paused


def test_pause_lets_the_running_job_finish(pool_env, capsys):
    agent = start_agent(pool_env)
    agent.lease_wait = 0.5  # a pause takes effect once the lease request in flight returns
    code = write(pool_env["tmp"] / "p", {"slow.py": "import time\ntime.sleep(3)\nprint('done')"})
    assert cli.main(["run", "-d", "--code", str(code), "--", "python3", "slow.py"]) == 0
    first = capsys.readouterr().out.strip().splitlines()[-1]
    wait_for(lambda: agent.running)
    pause_path().touch()
    time.sleep(1)
    assert cli.main(["run", "-d", "--code", str(code), "--", "python3", "slow.py"]) == 0
    second = capsys.readouterr().out.strip().splitlines()[-1]
    wait_for(lambda: job_state(first)["status"] == "succeeded")
    time.sleep(1.5)
    assert job_state(second)["status"] == "queued"


def test_removed_machine_stops_the_agent_for_good(pool_env):
    agent = start_agent(pool_env)
    Client(agent.config["server"], agent.config["token"]).call("DELETE", "/v1/agent/me")
    wait_for(lambda: agent.stopping.is_set(), timeout=40)
    assert agent.removed


def test_uninstall_a_user_install(pool_env, tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(agent_mod, "SYSTEM", {key: tmp_path / "absent" / key for key in agent_mod.SYSTEM})
    calls = []
    monkeypatch.setattr(agent_mod, "_systemctl", lambda kind, *args: calls.append((kind, *args)))
    layout = agent_mod.user_layout()
    for path in (layout["unit"], layout["wrapper"], layout["prefix"] / "pool" / "agent.py"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    code = Client.member().call("POST", "/v1/machine-codes", {})["code"]
    config = register(pool_env["url"], code, "gone1", {}, [], ["*"])
    (layout["state"] / config["machine_id"]).mkdir(parents=True)
    pause_path().touch()
    assert agent_mod.install_kind() == "user"
    assert cli.main(["agent", "uninstall", "--yes"]) == 0
    assert "unregistered gone1" in capsys.readouterr().out
    assert ("user", "disable", "--now", "pool-agent.service") in calls
    assert Client.member().call("GET", "/v1/machines")["machines"] == []
    for path in (layout["unit"], layout["wrapper"], layout["prefix"], layout["state"], agent_mod.config_path(), pause_path()):
        assert not path.exists(), path


@pytest.mark.skipif(not hasattr(agent_mod.os, "geteuid") or agent_mod.os.geteuid() == 0, reason="needs a non-root user")
def test_uninstall_of_a_system_install_needs_root(tmp_path, monkeypatch, capsys):
    unit = tmp_path / "pool-agent.service"
    unit.write_text("x")
    monkeypatch.setitem(agent_mod.SYSTEM, "unit", unit)
    assert cli.main(["agent", "uninstall", "--yes"]) == 1
    assert "sudo pool-agent uninstall" in capsys.readouterr().err
    assert unit.exists()


def fake_process(proc, pid, cmdline, environ=(), cwd=None):
    folder = proc / str(pid)
    folder.mkdir(parents=True)
    (folder / "cmdline").write_bytes(b"\0".join(part.encode() for part in cmdline) + b"\0")
    (folder / "environ").write_bytes(b"\0".join(item.encode() for item in environ) + b"\0")
    if cwd:
        (folder / "cwd").symlink_to(cwd)


def test_installer_finds_only_the_older_no_sudo_agents(tmp_path):
    """The system install stops the user's older agent (the Spark's user unit, the Jetson's setsid nohup
    process) and never one started from elsewhere, such as a checkout's .venv."""
    from pool.installer import OLD_AGENTS_SH

    proc, old = tmp_path / "proc", "/home/ann/.local/share/pool-agent"
    fake_process(proc, 101, ["python3", "-m", "pool.cli", "agent", "run"], ["HOME=/home/ann", f"PYTHONPATH={old}"])  # nohup
    fake_process(proc, 102, ["/usr/bin/python3", "-m", "pool.cli", "agent", "run"], [f"PYTHONPATH={old}/"])  # user unit
    fake_process(proc, 103, ["/home/ann/projects/human-gym/.venv/bin/python", "-m", "pool.cli", "agent", "run"],
                 ["PYTHONPATH="], cwd=str(tmp_path))  # a checkout's agent: not ours
    fake_process(proc, 104, ["python3", "-m", "http.server"], [f"PYTHONPATH={old}"])  # not an agent
    fake_process(proc, 105, ["python3", "-m", "pool.cli", "agent", "run"], ["PYTHONPATH=/srv/other:/opt/x"], cwd=old)  # cwd
    script = OLD_AGENTS_SH + '\npgrep() { printf "101\\n102\\n103\\n104\\n105\\n"; }\nold_agent_pids ann "$1"\n'
    result = subprocess.run(["sh", "-c", script, "sh", old], capture_output=True, text=True,
                            env={"PATH": "/usr/bin:/bin", "POOL_INSTALL_PROC": str(proc)})
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["101", "102", "105"]


def test_system_install_takes_over_the_older_agent():
    from pool.installer import render

    script = render("http://spark-10dc.local:8765", "a" * 64)
    assert "__OLD_AGENTS__" not in script and 'stop_old_agents "$RUN_AS" "$OLD_DIR"' in script
    assert 'systemctl --user -M "$RUN_AS@" disable --now pool-agent.service' in script
    assert 'XDG_RUNTIME_DIR="/run/user/$uid" systemctl --user disable --now' in script
    assert 'mv "$OLD_CONF" "$OLD_CONF.moved-to-etc-pool"' in script  # keeps the machine's registration

