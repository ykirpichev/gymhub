"""Job specs (#45 section 7.1), the M2 subset.

    name: train-mnist
    image: nvcr.io/nvidia/pytorch:25.01-py3
    command: ["python", "train.py"]
    priority: 0                      # among your own jobs; higher first
    resources: {gpus: 1, gpu_memory_gb: 16, cpus: 4, memory_gb: 16, labels: {zone: home},
                fits: {model: base, mode: full, kv_gb: 8},   # the machine must hold this model
                shm_gb: 16, tmp_gb: 32, pids: 16384,         # sandbox sizes (defaults 4, 8, 4096)
                machines: own,               # any (default) | own: only machines you own, never another member's
                machine: spark}              # only this machine, by name (with machines: own, only if it is yours)
    code: sha256:...                 # mounted read-only at /work, the working directory
    inputs: [{path: data, artifact: sha256:...},           # each at /inputs/<path>, read-only
             {path: base, model: qwen3-30b-a3b@fp8}]       # a ModelRef, resolved when submitted
    cost: {flops: 2e15, dtype: bf16, bytes_out: 1e8, bulk: false,   # for the network check
           link_bytes: 4e9}            # a gang: bytes each rank sends to the next (placement: maxflow)
    env: {EPOCHS: "3"}
    timeout_minutes: 60
    retries: 1                       # re-run elsewhere if the machine disappears
    network: none                    # none | island | peers (island, peers: only with a gang)
    gang: {size: 2}                  # run as `size` ranks on different machines, started together
    images: {"gpu=tegra": "dustynv/l4t-pytorch:r36.4.0"}   # per-platform image, by machine label

Results are files: whatever the job writes to /outputs comes back as one artifact.
resources.machines: own keeps a job, its code and its inputs on the submitter's own machines: for
content that other members of the pool may not see. A machine's use policy can only narrow this.
fits.mode: full (all weights in GPU memory), offload (only the active parameters in GPU memory,
the rest in RAM) or stage:k/n (k of n pipeline stages).

A gang is `size` jobs (rank 0 .. size-1), each on its own machine. Each rank starts only when
every rank has a machine; it gets RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT and POOL_PEERS (the
ranks' addresses). With network: island each rank gets its own network namespace in which it can
reach only the other ranks, on the gang's block of ports (POOL_PORT_MIN..POOL_PORT_MAX), and its
own loopback: no internet, no DNS, nothing else on the LAN and none of the host's services
(pool/island.py). With network: peers the ranks share the machines' network (host networking on
machines that allow it, `pool agent run --allow-peer-network`). If one rank fails, the gang fails;
if one rank's machine disappears, the whole gang goes back to the queue. Not yet: refs, actors,
checkpoints.
"""

import re

SHA = re.compile(r"^sha256:[0-9a-f]{64}$")
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
INPUT_PATH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
IMAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:@-]{0,254}$")
KNOWN = {"name", "image", "command", "resources", "code", "inputs", "env", "timeout_minutes", "retries", "network",
         "priority", "cost", "gang", "images", "subscribe", "runtime"}
RUNTIMES = ("plain", "torchrun", "ray")
REF_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")
NETWORKS = {"none", "island", "peers"}
LABEL_MATCH = re.compile(r"^[A-Za-z0-9._-]{1,64}=[A-Za-z0-9._-]{1,64}$")
RESOURCES = {"gpus", "gpu_memory_gb", "cpus", "memory_gb", "labels", "fits", "shm_gb", "tmp_gb", "pids", "share_gpu", "machines",
             "machine"}
MACHINES = ("any", "own")  # resources.machines: whose machines may run the job
# Sandbox sizes a job may ask for (spike S1: Ray, vLLM and verl need more than the defaults).
# /dev/shm and /tmp are memory-backed, so they also count against memory_gb when used.
SANDBOX = {"shm_gb": (4, 0.0625, 256), "tmp_gb": (8, 1, 512), "pids": (4096, 64, 65536)}
MODEL_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}(@[A-Za-z0-9][A-Za-z0-9._-]{0,63})?$")
DTYPES = {"fp64", "fp32", "tf32", "bf16", "fp16", "fp8", "int8", "fp4"}
FIT_MODE = re.compile(r"^(full|offload|stage:[1-9][0-9]*/[1-9][0-9]*)$")
LIMITS = {"timeout_minutes": 7 * 24 * 60, "retries": 5, "gpus": 64, "env": 64, "inputs": 32, "command": 256, "gang": 64,
          "images": 8}


class SpecError(ValueError):
    pass


def _number(value, field: str, minimum: float = 0, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < minimum or (maximum is not None and value > maximum):
        raise SpecError(f"{field} must be a number from {minimum}" + (f" to {maximum}" if maximum is not None else ""))
    return value


def normalize(spec: dict) -> dict:
    """Check a submitted spec and fill in the defaults. Raises SpecError with a readable message."""
    if not isinstance(spec, dict):
        raise SpecError("the spec must be an object")
    unknown = set(spec) - KNOWN
    if unknown:
        raise SpecError(f"unknown fields: {', '.join(sorted(unknown))} (supported: {', '.join(sorted(KNOWN))})")
    out: dict = {}
    name = spec.get("name") or "job"
    if not isinstance(name, str) or not NAME.match(name):
        raise SpecError("name: letters, digits, '.', '_' and '-', at most 64")
    out["name"] = name
    image = spec.get("image")
    if not isinstance(image, str) or not IMAGE.match(image):
        raise SpecError("image is required, for example nvcr.io/nvidia/pytorch:25.01-py3")
    out["image"] = image
    images = spec.get("images") or {}
    if not isinstance(images, dict) or len(images) > LIMITS["images"]:
        raise SpecError(f"images maps at most {LIMITS['images']} 'label=value' keys to images")
    for match, other in images.items():
        if not isinstance(match, str) or not LABEL_MATCH.match(match) or not isinstance(other, str) or not IMAGE.match(other):
            raise SpecError("images: keys are 'label=value' (for example gpu=tegra), values images")
    if images:
        out["images"] = dict(images)
    command = spec.get("command")
    if isinstance(command, str):
        command = ["sh", "-c", command]
    if not isinstance(command, list) or not command or len(command) > LIMITS["command"] or not all(isinstance(a, str) for a in command):
        raise SpecError("command must be a non-empty list of strings (or one shell string)")
    out["command"] = command

    resources = spec.get("resources") or {}
    if not isinstance(resources, dict) or set(resources) - RESOURCES:
        raise SpecError(f"resources may hold {', '.join(sorted(RESOURCES))}")
    labels = resources.get("labels") or {}
    if not isinstance(labels, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in labels.items()):
        raise SpecError("resources.labels must map strings to strings")
    out["resources"] = {
        "gpus": int(_number(resources.get("gpus", 0), "resources.gpus", 0, LIMITS["gpus"])),
        "gpu_memory_gb": _number(resources.get("gpu_memory_gb", 0), "resources.gpu_memory_gb"),
        "cpus": _number(resources.get("cpus", 1), "resources.cpus", 0.1),
        "memory_gb": _number(resources.get("memory_gb", 4), "resources.memory_gb", 0.1),
        "labels": labels,
    }
    for key, (_default, low, high) in SANDBOX.items():
        if resources.get(key) is not None:  # only when asked for: older specs stay unchanged
            value = _number(resources[key], f"resources.{key}", low, high)
            out["resources"][key] = int(value) if key == "pids" else value
    machines = resources.get("machines", "any")
    if machines not in MACHINES:
        raise SpecError("resources.machines is any or own (only machines the submitter owns)")
    if machines != "any":  # only when asked for: older specs stay unchanged
        out["resources"]["machines"] = machines
    machine = resources.get("machine")
    if machine is not None:  # pinned to one machine, by name; only when asked for
        if not isinstance(machine, str) or not NAME.match(machine):
            raise SpecError("resources.machine is a machine's name")
        out["resources"]["machine"] = machine
    if resources.get("share_gpu"):
        # Several small jobs on one GPU (rollers next to a trainer): each reserves gpu_memory_gb of it.
        if out["resources"]["gpus"] != 1 or out["resources"]["gpu_memory_gb"] <= 0:
            raise SpecError("resources.share_gpu needs gpus: 1 and the GPU memory it uses (gpu_memory_gb)")
        out["resources"]["share_gpu"] = True
    if out["resources"].get("shm_gb", 0) > out["resources"]["memory_gb"]:
        raise SpecError("resources.shm_gb cannot exceed resources.memory_gb (shared memory is memory)")
    fit = resources.get("fits")
    if fit is not None:
        if not isinstance(fit, dict) or set(fit) - {"model", "mode", "kv_gb"} or not isinstance(fit.get("model"), str):
            raise SpecError("resources.fits is {model: <input path or model name>, mode: full|offload|stage:k/n, kv_gb: n}")
        mode = fit.get("mode", "full")
        if not isinstance(mode, str) or not FIT_MODE.match(mode):
            raise SpecError("resources.fits.mode is full, offload or stage:k/n")
        if mode.startswith("stage:"):
            k, n = (int(x) for x in mode[6:].split("/"))
            if k > n:
                raise SpecError("resources.fits.mode stage:k/n needs k <= n")
        out["resources"]["fits"] = {"model": fit["model"], "mode": mode, "kv_gb": _number(fit.get("kv_gb", 0), "resources.fits.kv_gb")}
    out["priority"] = int(_number(spec.get("priority", 0), "priority", -100, 100))
    cost = spec.get("cost")
    if cost is not None:
        if not isinstance(cost, dict) or set(cost) - {"flops", "dtype", "bytes_out", "bulk", "link_bytes"}:
            raise SpecError("cost is {flops, dtype, bytes_out, bulk, link_bytes}")
        dtype = cost.get("dtype", "bf16")
        if dtype not in DTYPES:
            raise SpecError(f"cost.dtype is one of {', '.join(sorted(DTYPES))}")
        out["cost"] = {"flops": _number(cost.get("flops", 0), "cost.flops"), "dtype": dtype,
                       "bytes_out": _number(cost.get("bytes_out", 0), "cost.bytes_out"), "bulk": bool(cost.get("bulk", False))}
        if cost.get("link_bytes"):
            out["cost"]["link_bytes"] = _number(cost["link_bytes"], "cost.link_bytes")

    code = spec.get("code")
    if code is not None and (not isinstance(code, str) or not SHA.match(code)):
        raise SpecError("code must be an artifact id (sha256:...)")
    out["code"] = code
    inputs = spec.get("inputs") or []
    if not isinstance(inputs, list) or len(inputs) > LIMITS["inputs"]:
        raise SpecError(f"inputs must be a list of at most {LIMITS['inputs']}")
    seen = set()
    out["inputs"] = []
    for item in inputs:
        if not isinstance(item, dict) or set(item) not in ({"path", "artifact"}, {"path", "model"}, {"path", "artifact", "model"}):
            raise SpecError("each input is {path, artifact} or {path, model}")
        if "model" in item:
            if not isinstance(item["model"], str) or not MODEL_REF.match(item["model"]):
                raise SpecError(f"input {item.get('path')}: model is name or name@tag")
            if "artifact" not in item:  # resolved by the server when submitted
                if not isinstance(item["path"], str) or not INPUT_PATH.match(item["path"]) or item["path"] in seen:
                    raise SpecError(f"input path {item['path']!r}: one folder name, letters, digits, '.', '_', '-', unique")
                seen.add(item["path"])
                out["inputs"].append({"path": item["path"], "model": item["model"]})
                continue
        if not isinstance(item["path"], str) or not INPUT_PATH.match(item["path"]) or item["path"] in seen:
            raise SpecError(f"input path {item['path']!r}: one folder name, letters, digits, '.', '_', '-', unique")
        if not isinstance(item["artifact"], str) or not SHA.match(item["artifact"]):
            raise SpecError(f"input {item['path']}: artifact must be sha256:...")
        seen.add(item["path"])
        out["inputs"].append({"path": item["path"], "artifact": item["artifact"], **({"model": item["model"]} if "model" in item else {})})

    env = spec.get("env") or {}
    if not isinstance(env, dict) or len(env) > LIMITS["env"]:
        raise SpecError("env must be an object")
    for key, value in env.items():
        if not ENV_KEY.match(str(key)) or key.startswith("POOL_") or not isinstance(value, str) or len(value) > 4096:
            raise SpecError(f"env {key}: names are identifiers not starting with POOL_, values strings")
    out["env"] = dict(env)
    out["timeout_minutes"] = _number(spec.get("timeout_minutes", 60), "timeout_minutes", 1, LIMITS["timeout_minutes"])
    out["retries"] = int(_number(spec.get("retries", 1), "retries", 0, LIMITS["retries"]))
    gang = spec.get("gang")
    if gang is not None:
        if not isinstance(gang, dict) or set(gang) - {"size", "id", "rank"}:
            raise SpecError("gang is {size: n}")
        out["gang"] = {"size": int(_number(gang.get("size", 1), "gang.size", 1, LIMITS["gang"]))}
        if out["gang"]["size"] > 1 and out["resources"].get("machine"):
            raise SpecError("resources.machine pins one machine; a gang's ranks each need their own")
        for key in ("id", "rank"):  # set by the server when it splits a gang into its ranks
            if key in gang:
                out["gang"][key] = gang[key]
    network = spec.get("network", "none")
    if network not in NETWORKS:
        raise SpecError("network: none, island or peers (egress comes later)")
    if network in ("island", "peers") and not gang:
        raise SpecError(f"network: {network} needs a gang (gang: {{size: n}})")
    out["network"] = network
    runtime = spec.get("runtime", "plain")
    if runtime not in RUNTIMES:
        raise SpecError(f"runtime is one of {', '.join(RUNTIMES)}")
    if runtime != "plain" and not gang:
        raise SpecError(f"runtime: {runtime} starts one node per rank of a gang (gang: {{size: n}})")
    if runtime != "plain":
        out["runtime"] = runtime
    subscribe = spec.get("subscribe") or []
    if not isinstance(subscribe, list) or len(subscribe) > 16 or not all(isinstance(n, str) and REF_NAME.match(n) for n in subscribe):
        raise SpecError("subscribe is a list of at most 16 ref names (their new versions are fetched to the machine ahead of time)")
    if subscribe:
        out["subscribe"] = sorted(set(subscribe))
    if out["resources"].get("fits"):
        target = out["resources"]["fits"]["model"]
        if not any(item["path"] == target or item.get("model") == target for item in out["inputs"]):
            raise SpecError(f"resources.fits.model {target!r} must name a model input (its path or model name)")
    return out


def image_for(spec: dict, labels: dict) -> str:
    """The image a job uses on a machine with these labels: the first matching `images` entry, else `image`."""
    for match, image in (spec.get("images") or {}).items():
        key, _, value = match.partition("=")
        if labels.get(key) == value:
            return image
    return spec["image"]


def artifacts_of(spec: dict) -> list[str]:
    return ([spec["code"]] if spec.get("code") else []) + [item["artifact"] for item in spec.get("inputs", []) if item.get("artifact")]
