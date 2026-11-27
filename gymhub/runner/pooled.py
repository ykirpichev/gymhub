"""Grade other users' jobs on the compute pool instead of this machine (issue #45, M4 in small).

The runner keeps its place between the board and the machines: it leases board jobs as before. In
pool mode (GYMHUB_EXECUTOR=pool, the default; `local` turns it off), an isolated job (another user's code, or any job of an uploaded gym) becomes a
pool job, so it can run on any machine of the pool that fits it: a Jetson, a DGX Spark, a box with a
consumer GPU. The board and its worker API do not change.

The graders come with the image (#132): CI builds self-contained grader images (deploy/grader/) with
gymkit/ and gpu_gym/ at /opt/gym, on the image's PYTHONPATH, so a pool machine needs no repo checkout
and no venv. What the pool job gets, as its code folder (/work, read-only):

    /work/gyms/<id>/                 the gym as a slice without answer keys and reference solutions
    /work/job/<solution file>        the submitted code; custom_cases.json for run mode
    /work/job/progress.json          the user's progress, which the grade updates
    /work/gymkit/, /work/gpu_gym/    only for an image that is not a grader image (`ships_graders`):
                                     the graders, which then come first on PYTHONPATH

gymkit is pointed at its gym by path (`python -m gymkit gyms/<id> ...`), gpu_gym by GPU_GYM_ROOT, so
neither looks for the gym next to its own code.

Paths in the command go through $POOL_WORK and $POOL_OUTPUTS, so it also runs under an agent
without containers (tests).

The grade writes the updated progress to /outputs/progress.json, which comes back with the job.
Compile caches live in the job's /tmp. Routing: a gym that needs a GPU asks the pool for one.

The image follows the machine's platform, by the labels every pool agent sets (pool/profile.py:
arch, and gpu = tegra | unified | discrete | none), through the spec's per-platform `images`
(`default_images`):

    a job without a GPU           grader:cpu       (amd64 + arm64)
    a GPU job                     grader:cuda      (amd64 + arm64: x86 cards, DGX Spark / GB10)
    a GPU job on a Jetson         grader:l4t-r36   (gpu=tegra, JetPack 6 / L4T r36)

GYMHUB_POOL_IMAGE and GYMHUB_POOL_IMAGES still override that, for example for a machine whose
driver is too old for the CUDA 13 image.

The grader packages on ghcr.io are private, and pool machines have no ghcr login. So for now each
machine builds the GPU image it needs, under the same tag, and the agent uses the local image
without pulling (#163): on a Jetson `sh deploy/grader/build-local.sh l4t-r36`, on a DGX Spark
`sh deploy/grader/build-local.sh cuda`. A machine that has neither the image nor a way to pull it
fails the job before it starts (the pool's "ImageUnavailable" error). `run` and `validate` raise
`ImageUnavailable` for that, so the board shows a machine problem with the fix, not a failed grade.

The bridge to the worker API: a runner in pool mode reports, with each heartbeat, what the pool can
grade (`capabilities["pool"]`, from `advertised`). The board then routes it isolated jobs by the
pool's GPUs, not its own, and lets it take other users' code, because the pool's containers isolate
it (gymhub/routing.py). The pool is the only place another user's code runs: a runner has no sandbox
of its own (the host-mount sandbox is gone, #132), so without a pool it grades only its owner's
jobs. Runners that send no "pool" are routed by their own capabilities, so the two kinds serve the
same queue side by side; whichever fits and asks first takes a job. A runner in pool mode also
speaks either backend (direct or --api with a token).

Who may see the gym (#117): /work carries the gym's graders and hidden tests, and a pool machine's
owner can read whatever runs on it. The board knows who may see a gym, the pool does not, and pool
members are not board users. So only a public gym's job goes to any machine of the pool. Every other
job (the board's `gym_public` on the leased job is missing or false: fail closed) is submitted with
resources.machines: own, which the pool scheduler honours: it runs only on machines owned by the pool
member this runner submits as, which are as trusted as this runner, which the board already let hold
the gym. `advertised` reports those machines apart (`own`), so the board routes such jobs by them,
and, when none is online, by this machine's own capabilities: they are then graded here if they
are the owner's own (`takes`), or left to another worker.

Run on (gymhub/routing.py): `advertised` lists the pool's online machines by name (`list`), so the
board can offer them in its "Run on" choice. A job the learner pinned to one (`target.pool_machine`)
is submitted with resources.machine: <name>, which the pool scheduler honours. Where a job ran goes
back to the board: the pool machine's name as soon as the pool places it (`on_machine`), and with
the result.

Network: every job is submitted with network: none (`check_network`). `island` and `peers` (the
host's network) are for pool members' own gangs, never for board-submitted gym code.

Dispatcher (#135, docs/design/worker-v2.md section 6): with GYMHUB_DISPATCH_ONLY=1 as well, the
runner executes nothing itself. Every job goes to the pool: the owner's own jobs too, every action
(run, test, bench) and the validation of uploaded gym versions (`stage_validation`). It tells the
board so (`dispatch_only` in `advertised`), and the board then leases it only jobs it can send to the
pool (gymhub/routing.py `refuses`). Saving, resetting and hints of the owner's jobs stay file edits
here; they run no code. So a dispatcher needs no GPU, no compiler and no Docker, and can run next to
the board or the pool server.

Settings (environment):
    GYMHUB_EXECUTOR=pool|local                 pool (default): grade isolated jobs on the pool when a pool
                                               member is configured; local: never use the pool
    GYMHUB_DISPATCH_ONLY=1                     with GYMHUB_EXECUTOR=pool: every job to the pool, nothing here
                                               (GYMHUB_POOL_ACTIONS is then ignored)
    GYMHUB_POOL_ACTIONS=run,test,bench         which actions go to the pool (default: all three). A bench
                                               result is recorded under the GPU model of the pool machine
                                               that measured it
    GYMHUB_GRADER_IMAGE=<repo>                 where the grader images live (default
                                               ghcr.io/ykirpichev/gymhub/grader)
    GYMHUB_GRADER_VERSION=<short sha>          pin the grader images (tags cpu-<sha>, ...; default: the
                                               floating tags cpu, cuda, l4t-r36; a machine keeps a floating
                                               tag it has pulled, so pin to roll a new build out)
    GYMHUB_POOL_IMAGE=<image>                  override: the image for most machines (needs Python with the
                                               graders' dependencies; nvcc for CUDA exercises)
    GYMHUB_POOL_IMAGES=gpu=tegra=<image>,...   override: per-platform images, by pool machine label
    GYMHUB_POOL_SHIP_GRADERS=1|0               ship gymkit/ and gpu_gym/ in /work (default: only when an
                                               image the job may use is not a grader image)
    GYMHUB_POOL=<pool>                         the pool to use (default: the member's default pool)
    GYMHUB_POOL_TOOLS=g++,nvcc                 the compilers the pool image has (default: both)
    POOL_SERVER, POOL_TOKEN                    the pool member to submit as (default: ~/.config/pool/member.json)
"""

import json
import os
import shlex
import shutil
import tempfile
import time
from pathlib import Path

from gymhub.runner import slices
from gymhub.runner.adapters import REPO, GymAdapter

GRADERS = ("gymkit", "gpu_gym")
GRADER_IMAGE = "ghcr.io/ykirpichev/gymhub/grader"
TIMEOUTS_MIN = {"run": 6, "test": 6, "bench": 21, "validate": 31}
ALL_ACTIONS = {"run", "test", "bench", "validate"}
# What /work needs to validate an uploaded version (gymhub/runner/validate.py and what it imports).
VALIDATOR = ("gymhub/__init__.py", "gymhub/kinds.py", "gymhub/packages.py", "gymhub/routing.py",
             "gymhub/runner/__init__.py", "gymhub/runner/validate.py")
POLL_SECONDS = 1.0


def enabled() -> bool:
    return os.environ.get("GYMHUB_EXECUTOR", "pool").strip().lower() != "local"


def dispatch_only() -> bool:
    """GYMHUB_EXECUTOR=pool with GYMHUB_DISPATCH_ONLY=1: every job goes to the pool, none runs here."""
    return enabled() and os.environ.get("GYMHUB_DISPATCH_ONLY", "").strip().lower() in ("1", "true", "yes")


def actions() -> set[str]:
    if dispatch_only():
        return set(ALL_ACTIONS)
    return {a.strip() for a in os.environ.get("GYMHUB_POOL_ACTIONS", "run,test,bench").split(",") if a.strip()}


def images() -> dict[str, str]:
    """GYMHUB_POOL_IMAGES: label=value=image, comma-separated."""
    out = {}
    for item in os.environ.get("GYMHUB_POOL_IMAGES", "").split(","):
        match, _, image = item.strip().rpartition("=")
        if match and "=" in match and image:
            out[match] = image
    return out


def grader_image(flavor: str) -> str:
    """The grader image of one flavor (cpu, cuda, l4t-r36), built by .github/workflows/grader-images.yml."""
    repo = os.environ.get("GYMHUB_GRADER_IMAGE") or GRADER_IMAGE
    version = os.environ.get("GYMHUB_GRADER_VERSION", "").strip()
    return f"{repo}:{flavor}" + (f"-{version}" if version else "")


def default_images(needs_gpu: bool) -> tuple[str, dict[str, str]]:
    """The image for a job, and its per-platform images by machine label (pool/spec.py `images`).

    A CPU job runs the multi-arch cpu image everywhere, a Jetson included. A GPU job runs the
    multi-arch cuda image (x86 cards, DGX Spark), except on a Jetson (gpu=tegra), whose L4T userland
    needs its own image."""
    if not needs_gpu:
        return grader_image("cpu"), {}
    return grader_image("cuda"), {"gpu=tegra": grader_image("l4t-r36")}


def choose_images(needs_gpu: bool) -> tuple[str, dict[str, str]]:
    """`default_images`, with GYMHUB_POOL_IMAGE and GYMHUB_POOL_IMAGES (label=value=image) on top."""
    image, per_platform = default_images(needs_gpu)
    return os.environ.get("GYMHUB_POOL_IMAGE") or image, {**per_platform, **images()}


def is_grader_image(image: str) -> bool:
    """Is this one of the self-contained grader images (the graders inside, at /opt/gym)?"""
    name = image.split("@")[0]  # without a digest
    if ":" in name.rpartition("/")[2]:
        name = name.rpartition(":")[0]  # without a tag (a registry's port has a "/" after it)
    return name == (os.environ.get("GYMHUB_GRADER_IMAGE") or GRADER_IMAGE)


def ships_graders(needs_gpu: bool) -> bool:
    """Must /work carry gymkit/ and gpu_gym/? Only when an image the job may run in is not a grader
    image (a legacy image, or the process runtime in tests). GYMHUB_POOL_SHIP_GRADERS=1|0 decides."""
    setting = os.environ.get("GYMHUB_POOL_SHIP_GRADERS", "").strip().lower()
    if setting in ("1", "true", "yes", "0", "false", "no"):
        return setting in ("1", "true", "yes")
    image, per_platform = choose_images(needs_gpu)
    return not all(is_grader_image(i) for i in [image, *per_platform.values()])


def _copy_package(name: str, dest: Path) -> None:
    shutil.copytree(REPO / name, dest / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def stage(adapter: GymAdapter, slug: str, lang: str, code: str, options: dict, progress: dict | None, folder: Path,
          graders: bool = True) -> list[str]:
    """Lay out the job's /work folder; return the command, with paths as the job sees them. `graders`:
    also ship gymkit/ and gpu_gym/ (an image without them, see `ships_graders`)."""
    for name in GRADERS if graders else ():
        if (REPO / name).is_dir():
            _copy_package(name, folder)
    gym_root = adapter.root or adapter.mounts[0]
    gym_id = adapter.id
    shutil.copytree(slices.gym_slice(gym_root), folder / "gyms" / gym_id)  # no answer keys, no reference solutions
    job = folder / "job"
    job.mkdir()
    name = adapter.solution_path(slug, lang).name
    (job / name).write_text(code)
    (job / "progress.json").write_text(json.dumps(progress or {"version": 1, "exercises": {}}))
    module = "gpu_gym" if adapter.command_prefix[-1:] == ["gpu_gym"] else "gymkit"
    # gymkit by path ("gyms/<id>", relative to /work): its own folder has no gyms/ in a grader image.
    args = ["python3", "-m", module] + ([] if module == "gpu_gym" else [f"gyms/{gym_id}"])
    args += [options.get("action", "test"), slug, "--lang", lang, "--file", f"job/{name}"]
    if options.get("cases"):
        (job / "custom_cases.json").write_text(json.dumps(options["cases"]))
        args += ["--cases", "job/custom_cases.json"]
        if options.get("only_custom"):
            args.append("--only-custom")
    runs = int(options.get("runs") or 0)
    if options.get("action") == "bench" and 1 <= runs <= 100:
        args += ["--runs", str(runs)]
    # Paths go through $POOL_WORK and $POOL_OUTPUTS (/work and /outputs in a container). The graders
    # write progress where GYMKIT_PROGRESS_FILE or GPU_GYM_PROGRESS_FILE point: into the outputs.
    variable = "GPU_GYM_PROGRESS_FILE" if module == "gpu_gym" else "GYMKIT_PROGRESS_FILE"
    # gpu_gym finds its gym through GPU_GYM_ROOT (default gyms/gpu); an uploaded GPU gym has its own id.
    root = f' GPU_GYM_ROOT="$POOL_WORK/gyms/{gym_id}"' if module == "gpu_gym" else ""
    # /work first (shipped graders win over the image's), then the image's own PYTHONPATH (/opt/gym).
    script = ('cd "$POOL_WORK" && export PYTHONPATH="$POOL_WORK${PYTHONPATH:+:$PYTHONPATH}"'
              f'{root} {variable}="$POOL_OUTPUTS/progress.json" && cp job/progress.json "$POOL_OUTPUTS/progress.json" && '
              f'exec {shlex.join(args)}')
    return ["sh", "-c", script]


class ImageUnavailable(RuntimeError):
    """The pool machine could not get the job's image: nothing was graded. `output` is the job's log
    with what to do about it."""

    def __init__(self, message: str, output: str):
        super().__init__(message)
        self.output = output


def check_image(final: dict, output: str) -> None:
    """Raise ImageUnavailable when the pool job failed because its machine could not get the image
    (pool/agent.py ImageUnavailable)."""
    error = final.get("error") or ""
    if final.get("status") != "failed" or not error.startswith("ImageUnavailable:"):
        return
    image = error.split("image ", 1)[1].split(" ", 1)[0] if "image " in error else "the grader image"
    tag = image.rpartition(":")[2]  # a flavor, maybe with -<GYMHUB_GRADER_VERSION>
    flavor = next((f for f in ("l4t-r36", "cuda", "cpu") if tag == f or tag.startswith(f + "-")), None) \
        if is_grader_image(image) else None
    machine = final.get("machine") or "the pool machine"
    message = f"Not graded: {machine} does not have the image {image} and could not pull it."
    fix = (f"On {machine}, from the repo root: sh deploy/grader/build-local.sh {flavor}" if flavor
           else f"Pull or build {image} on {machine}.")
    raise ImageUnavailable(message, f"{output}\n{message}\n{fix}\n")


def gpu_reason(adapter: GymAdapter, lang: str, needs_gpu: bool) -> str:
    """The job log's first line about GPUs, from the gym's side (the pool knows only the request)."""
    if needs_gpu:
        return f"[gymhub] {adapter.id} [{lang}] asks the pool for a GPU\n"
    return f"[gymhub] {adapter.id} [{lang}] runs on the CPU: no GPU requested\n"


def spec_for(adapter: GymAdapter, action: str, command: list[str], needs_gpu: bool, own_only: bool = True,
             machine: str | None = None) -> dict:
    """The pool job spec. `own_only` (the default, fail closed) keeps it on the machines of the pool
    member this runner submits as (resources.machines: own); only a public gym's job may go anywhere.
    `machine` pins it to that pool machine ("Run on")."""
    image, per_platform = choose_images(needs_gpu)
    resources = {"gpus": 1 if needs_gpu else 0, "cpus": 2, "memory_gb": 4, **({"machines": "own"} if own_only else {})}
    if machine:
        resources["machine"] = machine
    spec = {
        "name": f"gym-{adapter.id}-{action}"[:64], "image": image, "command": command,
        "resources": resources,
        "env": {"PYTHONDONTWRITEBYTECODE": "1",
                "GYM_SANDBOX": "1", "NO_COLOR": "1", "TERM": "dumb", "GPU_GYM_CACHE": "/tmp/cache/gpu",
                "GYMKIT_CACHE": "/tmp/cache/cpp", "TRITON_CACHE_DIR": "/tmp/cache/triton",
                "TORCH_EXTENSIONS_DIR": "/tmp/cache/torch_extensions"},
        "timeout_minutes": TIMEOUTS_MIN.get(action, 6), "retries": 1,
        "network": "none",
    }
    if per_platform:
        spec["images"] = per_platform
    return spec


def check_network(spec: dict) -> dict:
    """Board-submitted gym code never gets a network: `island` and `peers` (the host's network) are
    for pool members' own jobs (docs/design/worker-v2.md section 4). Raises before anything is sent."""
    if spec.get("network", "none") != "none":
        raise RuntimeError(f"a board job may not use network: {spec.get('network')} (only none)")
    return spec


def _needs_gpu(adapter: GymAdapter, job: dict, lang: str) -> bool:
    from gymhub import routing

    if job.get("requires") is None:
        return routing.requirements(adapter.id, None, lang)["gpu"]
    return bool((job.get("requires") or {}).get("gpu"))


def target_machine(job: dict) -> str | None:
    """The pool machine the learner chose for this job ("Run on"), if any."""
    from gymhub import routing

    return routing.target_machine(job)


def run(client, adapter: GymAdapter, job: dict, slug: str, lang: str, action: str, on_output, timeout: int, on_machine=None):
    """Grade one board job on the pool. `on_output(text)` gets the output so far (the last part) and
    returns True when the board cancelled the job; `on_machine(name)` hears which pool machine runs it.
    Returns (output, ok, exit_code, user_progress, hardware)."""
    options = {**(job.get("options") or {}), "action": action}
    needs_gpu = _needs_gpu(adapter, job, lang)
    with tempfile.TemporaryDirectory(prefix="gymhub-pool-") as tmp:
        folder = Path(tmp) / "work"
        folder.mkdir()
        command = stage(adapter, slug, lang, job["code"], options, job.get("user_progress") if action != "run" else None, folder,
                        graders=ships_graders(needs_gpu))
        spec = spec_for(adapter, action, command, needs_gpu, own_machines_only(job), target_machine(job))
        output, final, outputs = _submit(client, folder, spec, Path(tmp) / "outputs" if action != "run" else None, on_output, timeout,
                                         on_machine, note=gpu_reason(adapter, lang, needs_gpu))
        check_image(final, output)
        progress = None
        if outputs is not None and (outputs / "progress.json").is_file():
            progress = json.loads((outputs / "progress.json").read_text())
        ok = final["status"] == "succeeded"
        return output, ok, final.get("exit_code"), progress, machine_hardware(client, os.environ.get("GYMHUB_POOL"), final.get("machine"))


def stage_validation(package_root: Path, folder: Path, graders: bool = True) -> list[str]:
    """Lay out /work to validate an uploaded gym version: the whole package as the uploader sent it
    (validation runs its reference solutions), the validator, and the graders when the image lacks
    them. The result goes to /outputs/result.json."""
    for name in GRADERS if graders else ():
        if (REPO / name).is_dir():
            _copy_package(name, folder)
    for name in VALIDATOR:
        target = folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, target)
    shutil.copytree(package_root, folder / "package", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    script = ('cd "$POOL_WORK" && export PYTHONPATH="$POOL_WORK${PYTHONPATH:+:$PYTHONPATH}" && '
              'exec python3 -m gymhub.runner.validate package --out "$POOL_OUTPUTS/result.json"')
    return ["sh", "-c", script]


def validate(client, adapter: GymAdapter, job: dict, on_output, timeout: int, on_machine=None):
    """Validate an uploaded gym version on the pool (gymhub/runner/validate.py). Returns
    (output, ok, exit_code, result), result being the validator's result.json, or None."""
    needs_gpu = bool((job.get("requires") or {}).get("gpu"))
    with tempfile.TemporaryDirectory(prefix="gymhub-pool-") as tmp:
        folder = Path(tmp) / "work"
        folder.mkdir()
        command = stage_validation(adapter.mounts[0], folder, graders=ships_graders(needs_gpu))
        spec = spec_for(adapter, "validate", command, needs_gpu, own_machines_only(job))
        output, final, outputs = _submit(client, folder, spec, Path(tmp) / "outputs", on_output, timeout, on_machine)
        check_image(final, output)
        result = None
        if outputs is not None and (outputs / "result.json").is_file():
            result = json.loads((outputs / "result.json").read_text())
        return output, final["status"] == "succeeded", final.get("exit_code"), result


def _submit(client, folder: Path, spec: dict, outputs: Path | None, on_output, timeout: int, on_machine=None,
            note: str = ""):
    """Upload `folder` as the job's /work, submit `spec`, stream its logs to `on_output` until it ends
    (cancelling it when the board asks or past `timeout`), and fetch its outputs into `outputs`.
    `on_machine(name)` hears the pool machine running it, each time it changes (the pool's logs say
    so; an older pool server does not, and then only the final job does). `note` follows the header.
    Returns (output text, the final pool job, the outputs folder or None)."""
    machine = None

    def placed(name):
        nonlocal machine
        if name and name != machine:
            machine = name
            if on_machine is not None:
                on_machine(name)

    pool_name = os.environ.get("GYMHUB_POOL")
    code = client.upload_folder(folder, pool=pool_name, kind="code")["artifact"]
    spec = check_network({**spec, "code": code})
    submitted = client.call("POST", "/v1/jobs", {"pool": pool_name, "spec": spec})
    pool_job = submitted["job"]["id"]
    text, after, started, cancelled = "", 0, time.time(), False
    header = f"[pool job {pool_job}]\n{note}"
    while True:
        reply = client.call("GET", f"/v1/jobs/{pool_job}/logs?after={after}&wait=5", timeout=60)
        for chunk in reply["chunks"]:
            text += chunk["text"]
        after = reply["next"]
        placed(reply.get("machine"))
        if reply["status"] in ("succeeded", "failed", "cancelled") and not reply["chunks"]:
            break
        try:
            stop = not cancelled and on_output(header + text[-150_000:])
        except BaseException:  # the board revoked this runner's token: nothing may keep running for it
            try:
                client.call("POST", f"/v1/jobs/{pool_job}/cancel")
            except Exception:
                pass
            raise
        if not cancelled and (stop or time.time() - started > timeout):
            cancelled = True  # asked once; the loop then waits for the pool to report the end
            client.call("POST", f"/v1/jobs/{pool_job}/cancel")
        if not reply["chunks"]:
            time.sleep(POLL_SECONDS)
    final = client.call("GET", f"/v1/jobs/{pool_job}")["job"]
    placed(final.get("machine"))
    fetched = None
    if final.get("outputs") and outputs is not None:
        client.download_folder(final["outputs"], outputs)
        fetched = outputs
    where = f"\n[graded on {final.get('machine') or 'the pool'}]\n"
    if cancelled and final["status"] != "succeeded":
        where += "Cancelled from the board, or over the time limit.\n"
    return header + text + where, final, fetched


def advertised(client, pool_name: str | None = None, machines: list[dict] | None = None) -> dict | None:
    """What this runner tells the board it can grade on the pool (`capabilities["pool"]`, see
    gymhub/routing.py): the actions it sends there, and the GPUs of the pool's online machines.

    Tools (nvcc, g++) come from the job's image, not the machine: GYMHUB_POOL_TOOLS (default
    g++,nvcc). None when the pool cannot be reached or has no machine online, so the runner is routed
    like any other worker until it has (a job leased for a pool that is down would only fail or wait)."""
    if machines is None:
        machines = fetch_machines(client, pool_name)
    if machines is None:
        return None
    online = [m for m in machines if m.get("online") and not m.get("drain")]
    if not online:
        return None  # nothing would grade it soon; leave the job to a worker that can
    own = [m for m in online if m.get("mine") is True]  # an older pool server does not say: none
    tools = sorted({t.strip() for t in os.environ.get("GYMHUB_POOL_TOOLS", "g++,nvcc").split(",") if t.strip()})
    found = {"actions": sorted(actions()), "machines": len(online), "gpus": _gpus(online), "tools": tools,
             "own": {"machines": len(own), "gpus": _gpus(own)},
             "list": [{"name": m["name"], "gpus": _gpus([m]), "own": m.get("mine") is True} for m in online if m.get("name")][:50]}
    if dispatch_only():
        found["dispatch_only"] = True  # the owner's jobs too: the board routes every job by the pool
    return found


def fetch_machines(client, pool_name: str | None = None) -> list[dict] | None:
    """The pool's machines (pool/server.py GET /v1/machines), or None when the server does not answer."""
    try:
        return client.call("GET", "/v1/machines" + (f"?pool={pool_name}" if pool_name else ""), timeout=15)["machines"]
    except Exception:
        return None


def _gpus(machines: list[dict]) -> list[dict]:
    gpus, seen = [], set()
    for machine in machines:
        for gpu in (machine.get("profile") or {}).get("gpus") or []:
            name = gpu.get("model") or gpu.get("name") or "unknown GPU"
            if name not in seen:
                seen.add(name)
                gpus.append({"name": name})
    return gpus


def own_machines_only(job: dict) -> bool:
    """Must this job stay on the bridge member's own pool machines? Yes unless the board said its gym
    is public (gymhub/gymaccess.py): a missing or false `gym_public` keeps it there (fail closed)."""
    return job.get("gym_public") is not True


def takes(bridge: dict | None, job: dict) -> bool:
    """Does this runner grade the job on the pool, by what it last told the board (`advertised`)?
    Mirrors routing.on_pool for a gym that is not public: with no own pool machine online, the board
    routed it by this machine's own capabilities, so it is graded here (only the owner's job is). Unknown
    (no report yet): the pool, where resources.machines: own still keeps it on our machines."""
    if bridge is None or not own_machines_only(job):
        return True
    from gymhub import routing

    return bool(routing.own_pool(bridge)["machines"])


def machine_hardware(client, pool_name: str | None, machine: str | None) -> str | None:
    """The hardware class (gymhub/routing.py) of the pool machine that ran a job: its GPU model, or CPU and arch."""
    if not machine:
        return None
    try:
        machines = client.call("GET", "/v1/machines" + (f"?pool={pool_name}" if pool_name else ""))["machines"]
    except Exception:
        return None
    found = next((m for m in machines if m["name"] == machine), None)
    return profile_hardware(found.get("profile")) if found and found.get("profile") else None


def profile_hardware(profile: dict) -> str:
    """The board's hardware class for a pool machine's profile (pool/profile.py names a GPU's model
    "model"; a runner's capabilities call it "name"): the first GPU's model, else CPU and arch."""
    from gymhub import routing

    gpus = profile.get("gpus") or []
    if gpus:
        return gpus[0].get("model") or gpus[0].get("name") or "unknown GPU"
    return routing.hardware({"gpus": [], "arch": profile.get("arch", "?")})
