"""What this machine can run: reported to the board with every heartbeat, used to route jobs.

    {"os": "Linux", "arch": "aarch64", "cpus": 20, "memory_gb": 121,
     "gpus": [{"name": "NVIDIA GB10", "compute_capability": "12.1", "memory_gb": null}],
     "driver": "580.178.04", "cuda": "13.0", "tools": ["docker", "g++", "nvcc", "nvidia-ctk"]}

Only cheap commands run here (nvidia-smi, nvcc --version); the result is cached for the runner's life.
"""

import os
import platform
import re
import shutil
import subprocess
from functools import lru_cache

TOOLS = ("nvcc", "g++", "docker", "nvidia-ctk")


def _run(*command: str) -> str:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _memory_gb() -> int | None:
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024 / 1024)
    except OSError:
        pass
    return None


def _gpus() -> tuple[list[dict], str | None]:
    if not shutil.which("nvidia-smi"):
        return [], None
    out = _run("nvidia-smi", "--query-gpu=name,compute_cap,memory.total,driver_version", "--format=csv,noheader,nounits")
    gpus, driver = [], None
    for line in out.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        name, capability, memory, driver = parts[:4]
        memory_gb = round(int(memory) / 1024) if memory.isdigit() else None  # unified-memory GPUs report N/A
        gpus.append({"name": name, "compute_capability": capability, "memory_gb": memory_gb})
    return gpus, driver


def _nvcc() -> str | None:
    nvcc = shutil.which("nvcc") or ("/usr/local/cuda/bin/nvcc" if os.path.exists("/usr/local/cuda/bin/nvcc") else None)
    if not nvcc:
        return None
    match = re.search(r"release (\d+\.\d+)", _run(nvcc, "--version"))
    return match.group(1) if match else None


@lru_cache(maxsize=1)
def probe() -> dict:
    gpus, driver = _gpus()
    cuda = _nvcc()
    tools = sorted(tool for tool in TOOLS if shutil.which(tool) or (tool == "nvcc" and cuda))
    return {
        "os": platform.system(),
        "arch": platform.machine(),
        "cpus": os.cpu_count(),
        "memory_gb": _memory_gb(),
        "gpus": gpus,
        "driver": driver,
        "cuda": cuda,
        "tools": tools,
    }
