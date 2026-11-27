"""A SkyPilot machine source (M7, #45 section 15): rent a cloud VM that joins a pool as a machine.

    pool machine cloud gpu-a100 --gpus A100:1 --cloud gcp --spot          # dry run: prints the task
    pool machine cloud gpu-a100 --gpus A100:1 --cloud gcp --spot --launch # launches it (costs money)
    pool machine cloud gpu-a100 --down                                     # tears it down

SkyPilot (https://skypilot.readthedocs.io) picks a cloud, region and instance for the requested
resources, provisions the VM and runs a task on it. The task here installs the pool agent from the
pool server (the same package the one-line installer fetches), registers it with a one-time machine
code and runs it. From then on it is an ordinary machine of the pool: outbound only, jobs in
containers. With `idle_minutes` SkyPilot stops (or with `down`, deletes) the VM after it has been
idle that long, so a forgotten VM does not run up a bill; `pool machine drain` plus this stops it.

Nothing here needs SkyPilot until a real launch: the task is a plain dict (written as JSON, which is
valid YAML for `sky launch`), and the dry run (the default) only shows it and the command. `sky`
is imported lazily, only by launch(dry_run=False) and down(), so the CLI stays standard library.
"""

import json
import re
import shlex

AGENT_DIR = "~/.local/share/pool-agent"
NAME = re.compile(r"^[a-z]([-a-z0-9]{0,61}[a-z0-9])?$")  # SkyPilot cluster names (cloud VM names)
ACCELERATORS = re.compile(r"^[A-Za-z0-9-]+(:[0-9]+(\.[0-9]+)?)?$")
CODE_PLACEHOLDER = "<machine-code>"


class SkyError(Exception):
    pass


def task(server: str, code: str, name: str, accelerators: str | None = None, cloud: str | None = None,
         region: str | None = None, cpus: str | None = None, memory: str | None = None, use_spot: bool = False,
         disk_gb: int | None = None, labels: dict | None = None, image_id: str | None = None,
         agent_args: list[str] | None = None) -> dict:
    """The SkyPilot task (its YAML as a dict) for a VM that runs `pool agent` in a pool."""
    if not NAME.match(name):
        raise SkyError("the name is the VM's name: lowercase letters, digits and dashes, starting with a letter")
    if accelerators and not ACCELERATORS.match(accelerators):
        raise SkyError("accelerators look like A100:1, H100:8 or L4")
    if not server.startswith(("http://", "https://")):
        raise SkyError("the server is a URL the VM can reach (http(s)://host:port)")
    resources: dict = {}
    for key, value in (("accelerators", accelerators), ("cloud", cloud), ("region", region), ("cpus", cpus),
                       ("memory", memory), ("disk_size", disk_gb), ("image_id", image_id)):
        if value:
            resources[key] = value
    if use_spot:
        resources["use_spot"] = True
    marks = {"source": "skypilot", **({"cloud": cloud} if cloud else {}), **(labels or {})}
    label_args = " ".join(f"--label {shlex.quote(f'{k}={v}')}" for k, v in sorted(marks.items()))
    extra = " ".join(shlex.quote(a) for a in agent_args or [])
    setup = "\n".join([
        "set -eu",
        "python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' "
        "|| { echo 'pool needs Python 3.10 or newer on the image' >&2; exit 1; }",
        "command -v docker > /dev/null || echo 'warning: no Docker on this image; pool jobs run in containers' >&2",
        f"mkdir -p {AGENT_DIR}",
        f'curl -fsSL "$POOL_SERVER/v1/agent/package.tar.gz" | tar -xz -C {AGENT_DIR}',
        f"test -f ~/.config/pool/agent.json || PYTHONPATH={AGENT_DIR} python3 -m pool.cli agent register "
        f'--server "$POOL_SERVER" --code "$POOL_CODE" --name "$POOL_MACHINE" {label_args}',
    ])
    run = f"PYTHONPATH={AGENT_DIR} exec python3 -m pool.cli agent run {extra}".rstrip()
    return {"name": name, "resources": resources, "envs": {"POOL_SERVER": server.rstrip("/"), "POOL_CODE": code,
                                                          "POOL_MACHINE": name},
            "setup": setup, "run": run}


def command(name: str, path: str, idle_minutes: int | None = None, down: bool = True) -> list[str]:
    """The `sky launch` command for a task file written by write()."""
    args = ["sky", "launch", "--yes", "--cluster", name, "--detach-run"]
    if idle_minutes:
        args += ["--idle-minutes-to-autostop", str(int(idle_minutes))] + (["--down"] if down else [])
    return args + [path]


def write(config: dict, path: str) -> str:
    """The task as a file `sky launch` reads (JSON is valid YAML). Readable by its owner only: it holds the code."""
    import os

    with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    return path


def _sky():
    try:
        import sky  # noqa: PLC0415  (optional: pip install "skypilot[gcp]" or another cloud)
    except ImportError:
        raise SkyError("SkyPilot is not installed: pip install 'skypilot[<cloud>]' and run `sky check`") from None
    return sky


def _finish(sky, result):
    # SkyPilot 0.8+ returns a request id to wait on; older versions return the result directly
    if isinstance(result, str) and hasattr(sky, "stream_and_get"):
        return sky.stream_and_get(result)
    return result


def launch(config: dict, idle_minutes: int | None = None, down: bool = True, dry_run: bool = True) -> dict:
    """Launch the task's VM with SkyPilot's Python API, or (dry_run, the default) only describe it.
    A dry run never imports sky and never calls a cloud."""
    shown = json.loads(json.dumps(config))
    shown["envs"]["POOL_CODE"] = CODE_PLACEHOLDER if dry_run else "<hidden>"
    summary = {"name": config["name"], "resources": config["resources"], "idle_minutes": idle_minutes,
               "down_when_idle": bool(idle_minutes and down), "task": shown}
    if dry_run:
        return {**summary, "dry_run": True,
                "command": command(config["name"], f"{config['name']}.sky.yaml", idle_minutes, down)}
    sky = _sky()
    job = sky.Task.from_yaml_config(config)
    kwargs = {"cluster_name": config["name"], "detach_run": True}
    if idle_minutes:
        kwargs.update(idle_minutes_to_autostop=int(idle_minutes), down=bool(down))
    try:
        result = _finish(sky, sky.launch(job, **kwargs))
    except TypeError:  # SkyPilot versions without detach_run on the API
        kwargs.pop("detach_run")
        result = _finish(sky, sky.launch(job, **kwargs))
    return {**summary, "dry_run": False, "launched": True, "result": repr(result)[:500]}


def down(name: str, dry_run: bool = True) -> dict:
    """Tear the VM down (`sky down`). Remove the machine from the pool with `pool machine rm`."""
    if not NAME.match(name):
        raise SkyError("not a VM name")
    if dry_run:
        return {"name": name, "dry_run": True, "command": ["sky", "down", "--yes", name]}
    sky = _sky()
    _finish(sky, sky.down(name))
    return {"name": name, "dry_run": False, "down": True}
