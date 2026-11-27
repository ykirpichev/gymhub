"""Gang runtimes (M5): torchrun and Ray, started by the agent around a rank's command."""

import os
import stat
import sys

import pytest

from pool import spec as specs
from pool.agent import runtime_command
from pool.client import Client
from tests.test_pool import db_url, job_state, pool_env, start_agent, wait_for, write  # noqa: F401  (fixtures)

ENV = {"RANK": "1", "WORLD_SIZE": "2", "MASTER_ADDR": "10.0.0.5", "MASTER_PORT": "29511"}


def test_torchrun_and_ray_commands():
    torch_spec = specs.normalize({"image": "x", "command": ["python3", "train.py", "--lr", "1"], "gang": {"size": 2}, "runtime": "torchrun"})
    assert runtime_command(torch_spec, ENV, [0, 1]) == [
        "torchrun", "--nnodes=2", "--node-rank=1", "--nproc-per-node=2", "--master-addr=10.0.0.5", "--master-port=29511",
        "train.py", "--lr", "1"]
    binary = {**torch_spec, "command": ["./trainer"]}
    assert runtime_command(binary, ENV, [])[-2:] == ["--no-python", "./trainer"]
    ray_spec = specs.normalize({"image": "x", "command": ["python3", "driver.py"], "gang": {"size": 2}, "runtime": "ray"})
    command = runtime_command(ray_spec, ENV, [0])
    assert command[:2] == ["sh", "-c"] and "ray start --head" in command[2] and command[-2:] == ["python3", "driver.py"]
    plain = specs.normalize({"image": "x", "command": ["true"]})
    assert runtime_command(plain, ENV, []) == ["true"]
    with pytest.raises(specs.SpecError):
        specs.normalize({"image": "x", "command": ["true"], "runtime": "ray"})  # a runtime needs a gang
    with pytest.raises(specs.SpecError):
        specs.normalize({"image": "x", "command": ["true"], "gang": {"size": 2}, "runtime": "mpi"})


FAKE_TORCHRUN = f"""#!{sys.executable}
import os, sys
args = sys.argv[1:]
options = [a for a in args if a.startswith("--")]
rest = [a for a in args if not a.startswith("--")]
node_rank = next(a.split("=")[1] for a in options if a.startswith("--node-rank="))
print("torchrun", " ".join(options), flush=True)
os.environ["RANK"] = node_rank  # one process per node here
os.execv(sys.executable, [sys.executable, *rest])
"""

FAKE_RAY_CLI = """#!/bin/sh
echo "ray $*"
exit 0
"""

FAKE_RAY_MODULE = """import os
def init(address=None):
    pass
def nodes():
    return [{"Alive": True}] * int(os.environ["WORLD_SIZE"])
"""


@pytest.fixture()
def fake_tools(pool_env, monkeypatch):
    bin_dir = pool_env["tmp"] / "fakebin"
    write(bin_dir, {"torchrun": FAKE_TORCHRUN, "ray": FAKE_RAY_CLI})
    for name in ("torchrun", "ray"):
        (bin_dir / name).chmod((bin_dir / name).stat().st_mode | stat.S_IEXEC)
    lib = write(pool_env["tmp"] / "fakelib", {"ray.py": FAKE_RAY_MODULE})
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("PYTHONPATH", f"{lib}:{os.environ.get('PYTHONPATH', '')}")
    os.symlink(sys.executable, bin_dir / "python3")  # the Ray wrapper's python3: the interpreter with this repo's deps


def run_gang(env, runtime: str, script: str) -> list[str]:
    start_agent(env, "box1")
    start_agent(env, "box2")
    member = Client.member()
    code = member.upload_folder(write(env["tmp"] / f"code-{runtime}", {"main.py": script}))["artifact"]
    job = member.call("POST", "/v1/jobs", {"spec": {"name": runtime, "image": "x", "command": ["python3", "main.py"],
                                                    "code": code, "gang": {"size": 2}, "runtime": runtime}})["job"]
    ranks = job["gang"]["jobs"]
    logs = []
    for rank_job in ranks:
        final = wait_for(lambda: (s := job_state(rank_job))["status"] in ("succeeded", "failed") and s, timeout=60)
        text = "".join(c["text"] for c in member.call("GET", f"/v1/jobs/{rank_job}/logs")["chunks"])
        assert final["status"] == "succeeded", text
        logs.append(text)
    return logs


def test_a_torchrun_gang(pool_env, fake_tools):
    logs = run_gang(pool_env, "torchrun", "import os\nprint('worker rank', os.environ['RANK'], 'of', os.environ['WORLD_SIZE'])\n")
    for rank, text in enumerate(logs):
        assert f"--nnodes=2 --node-rank={rank} --nproc-per-node=1 --master-addr=127.0.0.1" in text
        assert f"worker rank {rank} of 2" in text


def test_a_ray_gang_runs_the_driver_on_rank_0(pool_env, fake_tools):
    logs = run_gang(pool_env, "ray", "print('driver ran')\n")
    assert "ray start --head" in logs[0] and "driver ran" in logs[0] and "ray stop" in logs[0]
    assert "ray start --address=127.0.0.1:" in logs[1] and "--block" in logs[1] and "driver ran" not in logs[1]
