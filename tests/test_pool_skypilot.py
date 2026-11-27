"""M7 SkyPilot machine source (pool/skypilot.py, pool machine cloud). No cloud is ever called: `sky`
is a fake module, and dry runs must work without it."""

import json
import os
import stat
import subprocess
import sys
import types

import pytest

from pool import cli, skypilot
from pool.agent import register
from pool.client import PoolError
from tests.test_pool import db_url, pool_env  # noqa: F401  (fixtures)


class FakeSky(types.ModuleType):
    """Records what pool asks SkyPilot to do; the 0.8+ API (request ids, stream_and_get)."""

    def __init__(self, detach_ok=True):
        super().__init__("sky")
        self.calls = []
        self.detach_ok = detach_ok
        outer = self

        class Task:
            @staticmethod
            def from_yaml_config(config):
                outer.calls.append(("task", json.loads(json.dumps(config))))
                return ("task", config["name"])
        self.Task = Task

    def launch(self, task, **kwargs):
        if "detach_run" in kwargs and not self.detach_ok:
            raise TypeError("launch() got an unexpected keyword argument 'detach_run'")
        self.calls.append(("launch", task, kwargs))
        return "request-1"

    def down(self, name):
        self.calls.append(("down", name))
        return "request-2"

    def stream_and_get(self, request):
        self.calls.append(("wait", request))
        return {"request": request}


@pytest.fixture()
def no_sky(monkeypatch):
    monkeypatch.setitem(sys.modules, "sky", None)  # `import sky` raises ImportError


@pytest.fixture()
def fake_sky(monkeypatch):
    fake = FakeSky()
    monkeypatch.setitem(sys.modules, "sky", fake)
    return fake


def test_the_task_installs_registers_and_runs_the_agent():
    config = skypilot.task("https://pool.example:8765/", "CODE123", "gpu-a100", "A100:1", "gcp", use_spot=True,
                           disk_gb=200, labels={"zone": "cloud"})
    assert config["resources"] == {"accelerators": "A100:1", "cloud": "gcp", "disk_size": 200, "use_spot": True}
    assert config["envs"] == {"POOL_SERVER": "https://pool.example:8765", "POOL_CODE": "CODE123", "POOL_MACHINE": "gpu-a100"}
    assert '$POOL_SERVER/v1/agent/package.tar.gz' in config["setup"]
    assert "agent register" in config["setup"] and '--code "$POOL_CODE"' in config["setup"]
    assert "--label cloud=gcp --label source=skypilot --label zone=cloud" in config["setup"]
    assert "test -f ~/.config/pool/agent.json ||" in config["setup"]  # a re-run of setup does not reuse the spent code
    assert config["run"].endswith("python3 -m pool.cli agent run")
    assert "CODE123" not in config["setup"] + config["run"]  # the code travels as an env var only
    for bad in ({"name": "GPU_box"}, {"accelerators": "A100;rm -rf"}, {"server": "pool.example"}):
        args = {"server": "https://x", "code": "c", "name": "ok", **bad}
        with pytest.raises(skypilot.SkyError):
            skypilot.task(**args)


def test_a_dry_run_needs_no_skypilot_and_hides_the_code(no_sky):
    config = skypilot.task("https://x", "SECRET", "box", "L4:1")
    reply = skypilot.launch(config, idle_minutes=30)
    assert reply["dry_run"] and reply["task"]["envs"]["POOL_CODE"] == skypilot.CODE_PLACEHOLDER
    assert reply["command"] == ["sky", "launch", "--yes", "--cluster", "box", "--detach-run",
                                "--idle-minutes-to-autostop", "30", "--down", "box.sky.yaml"]
    assert skypilot.down("box")["command"] == ["sky", "down", "--yes", "box"]
    assert config["envs"]["POOL_CODE"] == "SECRET"  # the caller's config is untouched
    with pytest.raises(skypilot.SkyError, match="not installed"):
        skypilot.launch(config, dry_run=False)


def test_launch_and_down_go_through_the_sky_api(fake_sky):
    config = skypilot.task("https://x", "SECRET", "box", "L4:1")
    reply = skypilot.launch(config, idle_minutes=20, dry_run=False)
    assert reply["launched"] and reply["task"]["envs"]["POOL_CODE"] == "<hidden>"
    kinds = [c[0] for c in fake_sky.calls]
    assert kinds == ["task", "launch", "wait"]
    assert fake_sky.calls[0][1]["envs"]["POOL_CODE"] == "SECRET"
    assert fake_sky.calls[1][2] == {"cluster_name": "box", "detach_run": True, "idle_minutes_to_autostop": 20, "down": True}
    skypilot.down("box", dry_run=False)
    assert fake_sky.calls[-2:] == [("down", "box"), ("wait", "request-2")]


def test_launch_retries_without_detach_run_on_older_skypilot(monkeypatch):
    fake = FakeSky(detach_ok=False)
    old = types.ModuleType("sky")  # an older API: no stream_and_get, results come back directly
    old.Task, old.launch = fake.Task, fake.launch
    monkeypatch.setitem(sys.modules, "sky", old)
    reply = skypilot.launch(skypilot.task("https://x", "c", "box"), dry_run=False)
    assert fake.calls[-1] == ("launch", ("task", "box"), {"cluster_name": "box"}) and reply["result"] == "'request-1'"


def test_the_cli_dry_run_creates_nothing(pool_env, capsys, no_sky):
    codes = pool_env["db"].one("select count(*) as n from codes")["n"]
    assert cli.main(["machine", "cloud", "gpu-box", "--gpus", "A100:1", "--cloud", "gcp", "--spot"]) == 0
    out = capsys.readouterr().out
    assert "Dry run: nothing was created" in out and skypilot.CODE_PLACEHOLDER in out and '"use_spot": true' in out
    assert pool_env["db"].one("select count(*) as n from codes")["n"] == codes
    assert cli.main(["machine", "cloud", "gpu-box", "--down"]) == 0 and "Would run: sky down" in capsys.readouterr().out
    assert cli.main(["machine", "cloud", "Bad_Name"]) == 2


def test_the_cli_writes_a_task_whose_code_registers_a_machine(pool_env, capsys, no_sky):
    path = pool_env["tmp"] / "box.sky.yaml"
    assert cli.main(["machine", "cloud", "cloud-box", "--gpus", "L4:1", "--write", str(path)]) == 0
    assert "sky launch --yes --cluster cloud-box" in capsys.readouterr().out
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    config = json.loads(path.read_text())  # JSON is YAML, which `sky launch` reads
    code = config["envs"]["POOL_CODE"]
    assert code != skypilot.CODE_PLACEHOLDER and config["envs"]["POOL_SERVER"] == pool_env["url"]
    # what the VM does with it: run the task's setup here, in a scratch home (no cloud involved)
    home = pool_env["tmp"] / "vm-home"
    home.mkdir()
    env = {"PATH": os.environ["PATH"], "HOME": str(home), **config["envs"]}
    done = subprocess.run(["bash", "-c", config["setup"]], env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    row = pool_env["db"].one("select name, labels from machines where name='cloud-box'")
    assert row and json.loads(row["labels"])["source"] == "skypilot"
    assert (home / ".config" / "pool" / "agent.json").exists()
    again = subprocess.run(["bash", "-c", config["setup"]], env=env, capture_output=True, text=True, timeout=120)
    assert again.returncode == 0, again.stderr  # setup runs again on `sky launch` of a stopped VM: no second register
    with pytest.raises(PoolError):
        register(pool_env["url"], code, "other", {}, [], ["*"])  # the code was used once


def test_the_cli_launch_uses_a_fresh_code(pool_env, capsys, fake_sky):
    assert cli.main(["machine", "cloud", "cloud-box", "--gpus", "L4:1", "--idle-minutes", "0", "--launch"]) == 0
    assert "Launched cloud-box" in capsys.readouterr().out
    task = fake_sky.calls[0][1]
    assert task["envs"]["POOL_CODE"] != skypilot.CODE_PLACEHOLDER
    assert "idle_minutes_to_autostop" not in fake_sky.calls[1][2]  # 0: never stop
