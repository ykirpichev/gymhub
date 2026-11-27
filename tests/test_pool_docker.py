"""The pool in Docker (docs/pool.md, "Docker"): what the agent does differently in its own container,
the installer's container mode, and the deploy/pool files. The images themselves are built and run
by .github/workflows/pool-images.yml (deploy/pool/smoke.sh)."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from pool import agent as agent_mod, island, jobsocket, profile
from pool.client import write_config
from pool.installer import AGENT_IMAGE, render

REPO = Path(__file__).resolve().parents[1]
DEPLOY = REPO / "deploy" / "pool"


# ---- the agent in its container -----------------------------------------------------------------
def test_the_agent_knows_it_runs_in_a_container(monkeypatch):
    monkeypatch.delenv("POOL_IN_CONTAINER", raising=False)
    assert not agent_mod.in_container()
    monkeypatch.setenv("POOL_IN_CONTAINER", "1")
    assert agent_mod.in_container() and agent_mod.install_kind() == "docker"


def test_status_in_a_container_names_the_container_not_a_unit(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("POOL_IN_CONTAINER", "1")
    monkeypatch.setenv("POOL_IMAGE", "ghcr.io/x/pool:abc1234")
    monkeypatch.setenv("POOL_AGENT_CONFIG", str(tmp_path / "agent.json"))
    write_config(tmp_path / "agent.json", {"name": "box", "machine_id": "m1", "server": "http://127.0.0.1:1", "token": "t"})
    agent_mod.main_status(None)
    assert "running in a container (image ghcr.io/x/pool:abc1234" in capsys.readouterr().out


def test_uninstall_in_a_container_unregisters_and_leaves_the_folders_to_the_host(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("POOL_IN_CONTAINER", "1")
    monkeypatch.setenv("POOL_AGENT_CONFIG", str(tmp_path / "etc" / "agent.json"))
    monkeypatch.setattr(agent_mod, "SYSTEM", {key: tmp_path / key for key in agent_mod.SYSTEM})
    for key in ("prefix", "state"):
        (tmp_path / key).mkdir()
    write_config(tmp_path / "etc" / "agent.json", {"name": "box", "server": "http://127.0.0.1:1"})
    monkeypatch.setattr(agent_mod, "_systemctl", lambda *a: pytest.fail("no systemctl in a container"))
    assert agent_mod.main_uninstall(type("Args", (), {"yes": True})()) == 0
    assert not (tmp_path / "etc" / "agent.json").exists()
    assert (tmp_path / "prefix").is_dir() and (tmp_path / "state").is_dir()  # mounts: the host removes them


def test_a_container_agent_does_not_update_itself(monkeypatch, capsys):
    monkeypatch.setenv("POOL_IN_CONTAINER", "1")
    notes = []
    fake = type("Agent", (), {"server_digest": "new", "own_digest": "old", "auto_update": True,
                              "note_update": lambda self, key, message: notes.append(message)})()
    assert agent_mod.Agent.maybe_update(fake) is False
    assert "run the installer again" in notes[0]


def test_pool_agent_update_in_a_container_points_at_the_installer(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("POOL_IN_CONTAINER", "1")
    monkeypatch.setenv("POOL_AGENT_CONFIG", str(tmp_path / "agent.json"))
    write_config(tmp_path / "agent.json", {"server": "http://127.0.0.1:1", "token": "t"})
    assert agent_mod.main_update(type("Args", (), {"auto": None})()) == 1
    assert "installer again" in capsys.readouterr().err


def test_job_sockets_go_where_the_host_can_mount_them(tmp_path, monkeypatch):
    sockets = tmp_path / "run"
    sockets.mkdir()
    monkeypatch.setenv("POOL_SOCKET_DIR", str(sockets))
    job_socket = jobsocket.JobSocket(None, {"id": "j1"}, tmp_path / "job", container_paths=True)
    try:
        assert job_socket.dir.parent == sockets and job_socket.path.exists()
    finally:
        job_socket.close()


def test_islands_in_a_container_need_only_docker(monkeypatch):
    monkeypatch.setenv("POOL_IN_CONTAINER", "1")
    monkeypatch.setattr(island.shutil, "which", lambda tool, path=None: "/usr/local/bin/docker" if tool == "docker" else None)
    assert island.available() == (True, "")
    monkeypatch.delenv("POOL_IN_CONTAINER")
    assert island.available() == (False, "no iptables on this machine")


# ---- a Jetson's GPU without nvidia-smi ----------------------------------------------------------
def test_a_jetson_gpu_is_found_from_its_device_nodes(tmp_path, monkeypatch):
    release, model, node = tmp_path / "nv_tegra_release", tmp_path / "model", tmp_path / "nvmap"
    monkeypatch.setattr(profile, "TEGRA_RELEASE", release)
    monkeypatch.setattr(profile, "TEGRA_MODEL", model)
    monkeypatch.setattr(profile, "TEGRA_DEVICES", (str(node),))
    monkeypatch.setattr(profile, "_run", lambda *a, **k: None)  # no nvidia-smi
    monkeypatch.setattr(profile, "memory_gb", lambda: 61.4)
    assert profile.gpus() == []  # not a Jetson
    release.write_text("# R36 (release), REVISION: 4.3\n")
    assert profile.gpus() == []  # a Jetson whose container has no GPU
    node.write_text("")
    model.write_bytes(b"NVIDIA Jetson AGX Orin Developer Kit\x00")
    assert profile.gpus() == [{"index": 0, "model": "NVIDIA Jetson AGX Orin Developer Kit", "uuid": "tegra-0",
                               "memory_gb": 61.4, "unified_memory": True, "compute_capability": "8.7"}]


def test_nvidia_smi_wins_over_the_jetson_fallback(monkeypatch):
    monkeypatch.setattr(profile, "_run", lambda *a, **k: "0, Orin, [N/A], 8.7, GPU-abc\n")
    monkeypatch.setattr(profile, "tegra_gpus", lambda: pytest.fail("not needed"))
    assert profile.gpus()[0]["uuid"] == "GPU-abc"


# ---- the installer ------------------------------------------------------------------------------
def test_the_installer_runs_the_agent_in_a_container_by_default(monkeypatch):
    monkeypatch.delenv("POOL_AGENT_IMAGE", raising=False)
    script = render("http://192.168.1.10:8765", "a" * 64)
    assert f"AGENT_IMAGE='{AGENT_IMAGE}'" in script and "__AGENT_IMAGE__" not in script
    assert "HOST_AGENT=''" in script and "--host-agent) HOST_AGENT=1" in script
    assert '--restart unless-stopped' in script and "-v /var/run/docker.sock:/var/run/docker.sock" in script
    assert '-v "$STATE:$STATE"' in script  # job folders at the same path for the host's Docker
    assert 'mv "$UNIT" "$UNIT.replaced-by-container"' in script  # takes over a host agent


def test_a_server_can_name_the_agent_image(monkeypatch):
    monkeypatch.setenv("POOL_AGENT_IMAGE", "registry.lan:5000/pool:v2")
    assert "AGENT_IMAGE='registry.lan:5000/pool:v2'" in render("http://192.168.1.10:8765", "a" * 64)
    with pytest.raises(ValueError):
        render("http://192.168.1.10:8765", "a" * 64, agent_image="pool'; rm -rf /")


def test_the_installer_reads_agent_json_without_python(tmp_path):
    path = write_config(tmp_path / "agent.json", {"server": "http://spark:8765", "token": "t", "name": "jetson-orin", "labels": {}})
    script = render("http://192.168.1.10:8765", "a" * 64)
    function = re.search(r"^config_field\(\) \{.*?^\}", script, re.S | re.M).group(0)
    out = subprocess.run(["sh", "-c", function + '\nconfig_field "$1" name; config_field "$1" server', "sh", str(path)],
                         capture_output=True, text=True, check=True).stdout
    assert out.split() == ["jetson-orin", "http://spark:8765"]


def test_the_installers_wrapper_runs_owner_commands_in_the_container(tmp_path):
    script = render("http://192.168.1.10:8765", "a" * 64)
    assert "exec docker exec $CONTAINER pool-agent" in script
    assert "docker exec $CONTAINER pool-agent uninstall --yes" in script


# ---- deploy/pool --------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["entrypoint.sh", "dispatcher-entrypoint.sh", "smoke.sh"])
def test_scripts_are_posix_sh(name):
    path = DEPLOY / name
    assert subprocess.run(["sh", "-n", str(path)], capture_output=True).returncode == 0
    if shutil.which("shellcheck"):
        result = subprocess.run(["shellcheck", "-s", "sh", str(path)], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout


@pytest.mark.parametrize("dockerfile", ["Dockerfile", "Dockerfile.dispatcher"])
def test_dockerfiles_copy_only_what_their_dockerignore_lets_in(dockerfile):
    text = (DEPLOY / dockerfile).read_text()
    ignore = (DEPLOY / f"{dockerfile}.dockerignore").read_text().split()
    allowed = {line[1:] for line in ignore if line.startswith("!")}
    for source in re.findall(r"^COPY (?:--\S+ )?(\S+) ", text, re.M):
        assert source in allowed, f"{dockerfile} copies {source}, which {dockerfile}.dockerignore leaves out"
        assert (REPO / source).exists(), source


def test_the_agent_image_entrypoint_drops_root():
    text = (DEPLOY / "entrypoint.sh").read_text()
    assert "setpriv --reuid=\"$POOL_UID\"" in text and "stat -c %g \"$SOCK\"" in text
    assert 'POOL_SOCKET_DIR="$STATE/run"' in text


def test_compose_files_mount_the_state_at_the_same_path():
    agent = (DEPLOY / "compose.agent.yml").read_text()
    assert "- /var/lib/pool:/var/lib/pool" in agent and "- /var/run/docker.sock:/var/run/docker.sock" in agent
    assert "network_mode: host" in agent and "container_name: pool-agent" in agent
    assert "driver: nvidia" in (DEPLOY / "compose.agent.gpu.yml").read_text()
    jetson = (DEPLOY / "compose.agent.jetson.yml").read_text()
    assert "runtime: nvidia" in jetson and "/etc/nv_tegra_release:/etc/nv_tegra_release:ro" in jetson
    controller = (DEPLOY / "compose.controller.yml").read_text()
    assert "POOL_PUBLIC_URL:?" in controller and "pool-data:/data" in controller


def test_compose_files_are_valid(tmp_path):
    if not shutil.which("docker") or subprocess.run(["docker", "compose", "version"], capture_output=True).returncode != 0:
        pytest.skip("needs docker compose")
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(tmp_path), "POOL_PUBLIC_URL": "http://10.0.0.1:8765"}
    for files in (["compose.controller.yml"], ["compose.agent.yml"], ["compose.agent.yml", "compose.agent.gpu.yml"],
                  ["compose.agent.yml", "compose.agent.jetson.yml"]):
        args = [arg for name in files for arg in ("-f", str(DEPLOY / name))]
        result = subprocess.run(["docker", "compose", *args, "config", "--format", "json"], capture_output=True, text=True, env=env)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["services"]
