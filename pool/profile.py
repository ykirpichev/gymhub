"""What a machine can do (#45 section 4, "Profile"), the M1 subset: GPUs, CPUs, memory, disk,
container runtime. Measured TFLOP/s and network links come in M2.
"""

import os
import platform
import shutil
import subprocess
from pathlib import Path


def _run(command: list[str], timeout: float = 20) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def memory_gb() -> float:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024**2, 1)
    except OSError:
        pass
    out = _run(["sysctl", "-n", "hw.memsize"])  # macOS
    return round(int(out) / 1024**3, 1) if out and out.strip().isdigit() else 0.0


def gpus() -> list[dict]:
    out = _run(["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap,uuid", "--format=csv,noheader,nounits"])
    found = []
    for line in (out or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5 or not parts[0].isdigit():
            continue
        unified = not parts[2].replace(".", "").isdigit()  # GB10 and other unified-memory parts report [N/A]
        found.append({
            "index": int(parts[0]), "model": parts[1], "uuid": parts[4],
            "memory_gb": memory_gb() if unified else round(float(parts[2]) / 1024, 1),
            "unified_memory": unified, "compute_capability": parts[3],
        })
    return found or tegra_gpus()


TEGRA_RELEASE = Path("/etc/nv_tegra_release")
TEGRA_DEVICES = ("/dev/nvmap", "/dev/nvgpu", "/dev/nvhost-gpu")  # the integrated GPU's device nodes
TEGRA_MODEL = Path("/proc/device-tree/model")


def tegra_gpus() -> list[dict]:
    """A Jetson's integrated GPU when nvidia-smi says nothing: inside a container (the agent's own,
    deploy/pool/) the NVIDIA runtime mounts the GPU's device nodes but not nvidia-smi (CSV mode), and
    some JetPack releases have no nvidia-smi at all. One GPU, index 0, sharing the system's memory."""
    if not TEGRA_RELEASE.exists() or not any(Path(node).exists() for node in TEGRA_DEVICES):
        return []
    try:
        model = TEGRA_MODEL.read_text().strip("\x00 \n") or "Jetson"
    except OSError:
        model = "Jetson"
    orin = "orin" in model.lower()
    return [{"index": 0, "model": model, "uuid": "tegra-0", "memory_gb": memory_gb(), "unified_memory": True,
             "compute_capability": "8.7" if orin else ""}]


def driver() -> str | None:
    out = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    return out.splitlines()[0].strip() if out else None


def container_runtime() -> dict:
    version = _run(["docker", "version", "--format", "{{.Server.Version}}"])
    return {"docker": version.strip() if version else None}


def platform_labels(found: list[dict] | None = None) -> dict[str, str]:
    """Labels every agent sets for itself, so a job can ask for a kind of machine: CUDA images are
    built per platform, and one made for a GB10 or an x86 card does not run on a Jetson.

    arch: x86_64 | aarch64 | arm64 ...   gpu: tegra (Jetson) | unified (GB10, Grace) | discrete | none
    """
    found = gpus() if found is None else found
    if Path("/etc/nv_tegra_release").exists():
        kind = "tegra"
    elif not found:
        kind = "none"
    else:
        kind = "unified" if all(gpu["unified_memory"] for gpu in found) else "discrete"
    return {"arch": platform.machine().lower(), "gpu": kind}


def collect(allow_images: list[str] | None = None, gpu_indices: list[int] | None = None) -> dict:
    found = gpus()
    if gpu_indices is not None:
        found = [gpu for gpu in found if gpu["index"] in gpu_indices]
    return {
        "os": platform.system().lower(), "arch": platform.machine(), "python": platform.python_version(),
        "cpus": os.cpu_count() or 1, "memory_gb": memory_gb(),
        "disk_free_gb": round(shutil.disk_usage(Path.home()).free / 1024**3, 1),
        "gpus": found, "driver": driver(), **container_runtime(),
        "allow_images": allow_images or ["*"],
    }
