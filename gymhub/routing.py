"""Which machine may run which job.

A job's requirements come from its gym (does it need an NVIDIA GPU?) and its language (CUDA needs
nvcc, C++ needs g++). A machine reports its capabilities with each heartbeat
(gymhub/runner/capabilities.py). A machine only leases jobs it can run.

Benchmarks record the hardware they ran on, so best times are compared per GPU, never across GPUs.

The pool bridge: a runner with GYMHUB_EXECUTOR=pool (gymhub/runner/pooled.py) grades isolated jobs on
the compute pool, not on its own hardware. Its heartbeat then also carries `capabilities["pool"]`:
the actions it sends to the pool and what the pool's online machines have. For those jobs the board
routes by the pool's capabilities (`for_job`), and the pool's container isolation is what lets it take
another user's code (`sandbox`). Everything else, and every machine whose heartbeat has no "pool", is
routed by the machine's own capabilities. A runner has no sandbox of its own, so it reports
`sandbox: false` and gets only its owner's isolated jobs; only runners from before the host-mount
sandbox was removed (#132) still report true.

A gym that is not public (`job["gym_public"]`, which the board sets when it leases; missing means
not public) is graded only on the bridge's own pool machines, never on another pool member's
(gymhub/runner/pooled.py). The board routes its jobs by those machines (`capabilities["pool"]["own"]`);
when the bridge has none online, by the runner's own capabilities, as if it had no pool: graded on
it if it is the owner's own, or left to another machine.

A dispatcher (GYMHUB_DISPATCH_ONLY=1, #135) says `dispatch_only` in its pool report: it sends every
job to the pool, the owner's own too, and runs nothing itself. So every job of its actions is routed
by the pool, and a job it could not send there (a gym that is not public with none of its own pool
machines online, or an action it does not take) is never leased to it (`refuses`).

Run on (`job["target"]`): the learner may pick where a run, test or bench runs, from the machines
they could be routed to anyway (gymhub/board/runtargets.py). {"runner": id} is leased only by that
runner; {"pool_machine": name} only by a pool bridge that lists that machine online in its pool
report (`capabilities["pool"]["list"]`) and would grade the job there, and only by that machine's
capabilities. A target only narrows who leases the job (`misses_target`): every other rule still holds.
"""

LANG_TOOLS = {"cuda": ["nvcc"], "cpp": ["g++"]}
TOOL_NAMES = {"nvcc": "the CUDA toolkit (nvcc)", "g++": "a C++ compiler (g++)"}  # in messages
GPU_GYMS = {"gpu", "llm"}  # used when a gym document does not say (published before requirements existed)


def requirements(gym_id: str, gym_doc: dict | None, lang: str) -> dict:
    declared = (gym_doc or {}).get("requires")
    gpu = bool(declared.get("gpu")) if isinstance(declared, dict) else gym_id in GPU_GYMS
    return {"gpu": gpu, "tools": list(LANG_TOOLS.get(lang, []))}


def serves(runner: dict, gym_id: str, bundle: bool) -> bool:
    """Does this runner (its heartbeat doc) take jobs of this gym? Local gyms are listed in "gyms";
    uploaded gyms (bundle) go to any runner new enough to fetch bundles."""
    if bundle:
        return bool((runner.get("capabilities") or {}).get("bundles"))
    return not runner.get("gyms") or gym_id in runner["gyms"]


def on_pool(capabilities: dict | None, job: dict) -> bool:
    """Will this runner grade the job on the compute pool? Isolated jobs (and any job of an uploaded
    gym) of the actions its pool bridge takes; the owner's own jobs run in place, except on a
    dispatcher (`dispatch_only`), which sends them to the pool too."""
    bridge = (capabilities or {}).get("pool")
    if not isinstance(bridge, dict):
        return False
    if job.get("action") not in (bridge.get("actions") or []):
        return False
    # A job pinned to a pool machine goes to the pool, the owner's own too.
    if not (job.get("isolated") or job.get("bundle") or bridge.get("dispatch_only") or target_machine(job)):
        return False
    # A gym that is not public goes only to the bridge's own pool machines: none online, not to the pool.
    return bool(job.get("gym_public")) or bool(own_pool(bridge)["machines"])


def refuses(capabilities: dict | None, job: dict) -> bool:
    """A dispatcher (`dispatch_only`, in its pool report or, while the pool cannot be reached, in its
    capabilities) never takes a job it would not grade on the pool: it has nowhere else to run it.
    Every other runner takes what its capabilities allow."""
    return is_dispatcher(capabilities) and not on_pool(capabilities, job) and not file_edit(capabilities, job)


FILE_ACTIONS = {"save", "reset", "hint"}


def is_dispatcher(capabilities: dict | None) -> bool:
    capabilities = capabilities or {}
    bridge = capabilities.get("pool")
    return bool(capabilities.get("dispatch_only")) or (isinstance(bridge, dict) and bool(bridge.get("dispatch_only")))


def file_edit(capabilities: dict | None, job: dict) -> bool:
    """The owner's save, reset or hint on a dispatcher: an edit of the owner's files on its machine,
    which runs no code, so it needs nothing of the machine."""
    return (is_dispatcher(capabilities) and job.get("action") in FILE_ACTIONS
            and not job.get("isolated") and not job.get("bundle"))


def own_pool(bridge: dict) -> dict:
    """What a pool bridge reports of its own machines (the pool member it submits as owns them):
    {"machines": n, "gpus": [...]}; none when it does not say (an older runner)."""
    own = bridge.get("own")
    if not isinstance(own, dict):
        return {"machines": 0, "gpus": []}
    return {"machines": int(own.get("machines") or 0), "gpus": list(own.get("gpus") or [])}


def for_job(capabilities: dict | None, job: dict) -> dict | None:
    """The capabilities that decide whether this runner can run the job: the pool's when it grades
    the job there (with the pool's isolation as its sandbox; for a gym that is not public, only the
    bridge's own machines), else the machine's own."""
    if on_pool(capabilities, job):
        bridge = capabilities["pool"]
        if not job.get("gym_public"):
            bridge = {**bridge, **own_pool(bridge)}
        name = target_machine(job)
        if name:
            machine = pool_machine(bridge, job, name)
            bridge = {**bridge, "machines": 1 if machine else 0, "gpus": (machine or {}).get("gpus") or []}
        return {**bridge, "bundles": bool(capabilities.get("bundles")), "sandbox": True}
    if file_edit(capabilities, job):
        return None  # a dispatcher saves or resets the owner's file: no GPU or compiler involved
    return capabilities


def target_of(job: dict) -> dict | None:
    """The job's "Run on" choice: {"runner": id} or {"pool_machine": name}, or None (Auto)."""
    target = job.get("target")
    return target if isinstance(target, dict) and target else None


def target_machine(job: dict) -> str | None:
    """The pool machine the job is pinned to, if any."""
    name = (target_of(job) or {}).get("pool_machine")
    return name if isinstance(name, str) and name else None


def pool_machines(bridge: dict | None) -> list[dict]:
    """The online pool machines a bridge lists: [{"name", "gpus": [{"name"}], "own"}]; none from an
    older runner, which only counts them."""
    listed = (bridge or {}).get("list")
    if not isinstance(listed, list):
        return []
    return [m for m in listed if isinstance(m, dict) and isinstance(m.get("name"), str)]


def pool_machine(bridge: dict | None, job: dict, name: str) -> dict | None:
    """That machine of the bridge's pool, if the job may go there: listed online, and, for a gym
    that is not public, one of the bridge's own machines."""
    for machine in pool_machines(bridge):
        if machine["name"] == name:
            return machine if job.get("gym_public") or machine.get("own") is True else None
    return None


def misses_target(worker: str, capabilities: dict | None, job: dict) -> bool:
    """True when the job was sent somewhere else than this runner ("Run on"). A pool machine target
    is met only by a bridge that would grade the job on that machine."""
    target = target_of(job)
    if target is None:
        return False
    if "runner" in target:
        return target["runner"] != worker
    name = target_machine(job)
    if name is None:
        return True  # a target we do not understand: nobody takes it (fail closed)
    return not on_pool(capabilities, job) or pool_machine(capabilities["pool"], job, name) is None


def unmet(capabilities: dict | None, requires: dict | None) -> str | None:
    """None when the machine can run the job, else what is missing (for messages)."""
    if not requires:
        return None
    if capabilities is None:
        return None  # an older runner that does not report capabilities: trust its gym list
    if requires.get("gpu") and not capabilities.get("gpus"):
        return "an NVIDIA GPU"
    tools = set(capabilities.get("tools") or [])
    missing = [TOOL_NAMES.get(tool, tool) for tool in requires.get("tools", []) if tool not in tools]
    return ", ".join(missing) if missing else None


def hardware(capabilities: dict | None) -> str:
    """The hardware class benchmarks are grouped by: the GPU model, or the CPU architecture."""
    if not capabilities:
        return "unknown"
    gpus = capabilities.get("gpus") or []
    if gpus:
        return gpus[0]["name"]
    return f"CPU {capabilities.get('arch', '?')}"
