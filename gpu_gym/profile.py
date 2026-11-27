"""Nsight Compute summary for `bench --profile`: what limits the kernels of one solution call.

The solution runs once more in a child process under ncu. Only kernels launched inside the NVTX
range "gym_solution" are profiled, so the reference implementation, input setup and warm-up are not.
For each of the slowest kernels it shows:

    duration, grid x block, registers per thread
    achieved occupancy      sm__warps_active (% of peak, while active)
    SM and memory busy      sm__throughput, gpu__compute_memory_throughput (% of peak)
    L2 bandwidth            lts__t_bytes per second
    main warp stall         the largest smsp__average_warps_issue_stalled_<reason> ratio, with advice

GB10 (DGX Spark) has unified memory and no dram__* counters, so memory throughput is the compute-memory
pipeline's share of its peak. Profiling needs access to GPU performance counters: by default the
NVIDIA driver allows that only for root (see `available()` for how to enable it).
"""

import csv
import io
import os
import shutil
import subprocess
import sys
from pathlib import Path

NVTX_RANGE = "gym_solution"
STALLS = {
    "long_scoreboard": "waiting for global/local memory: improve coalescing, reuse data from shared memory or registers",
    "short_scoreboard": "waiting for shared memory or special math: watch bank conflicts and MUFU-heavy code",
    "barrier": "waiting at __syncthreads(): balance work between warps or sync less often",
    "membar": "waiting on memory fences: avoid unnecessary __threadfence()/atomics ordering",
    "mio_throttle": "memory input/output queue full: too many shared-memory or special instructions at once",
    "lg_throttle": "local/global memory queue full: use wider loads (float4) and fewer, larger requests",
    "math_pipe_throttle": "a math pipe is saturated: you are compute-bound on that unit",
    "tex_throttle": "texture/L1 pipeline saturated",
    "not_selected": "warps were ready but another was picked: latency is well hidden",
    "selected": "issuing: the warp was running instructions",
    "wait": "fixed-latency dependencies: increase instruction-level parallelism",
    "dispatch_stall": "the dispatcher stalled",
    "no_instruction": "instruction cache misses or branch targets: large or divergent code",
    "drain": "waiting for memory stores to drain at exit",
    "sleeping": "warps sleeping (nanosleep, yield)",
    "branch_resolving": "waiting for branch targets: reduce divergent branches",
    "misc": "miscellaneous stalls",
}
METRICS = [
    "gpu__time_duration.sum",
    "launch__grid_size",
    "launch__block_size",
    "launch__registers_per_thread",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_bytes.sum.per_second",
] + [f"smsp__average_warps_issue_stalled_{name}_per_issue_active.ratio" for name in STALLS]
NCU_CANDIDATES = ["/usr/local/cuda/bin/ncu", "/opt/nvidia/nsight-compute/ncu"]


def ncu_path() -> str | None:
    found = shutil.which("ncu")
    if found:
        return found
    return next((path for path in NCU_CANDIDATES if os.path.exists(path)), None)


def available() -> str | None:
    """None when profiling can run here, else why not (shown to the learner)."""
    if ncu_path() is None:
        return "Nsight Compute (ncu) is not installed on this machine."
    params = Path("/proc/driver/nvidia/params")
    if os.geteuid() != 0 and params.is_file() and "RmProfilingAdminOnly: 1" in params.read_text():
        return (
            "This machine's NVIDIA driver lets only root read GPU performance counters. Its owner can allow it with\n"
            "  echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf\n"
            "  sudo update-initramfs -u && sudo reboot"
        )
    return None


def _number(text: str) -> float | None:
    try:
        return float(str(text).replace(",", ""))
    except ValueError:
        return None


def parse(csv_text: str) -> list[dict]:
    """ncu --csv --page raw: a header row, a units row, then one row per profiled kernel."""
    lines = [line for line in csv_text.splitlines() if line.startswith('"')]
    rows = list(csv.reader(io.StringIO("\n".join(lines))))
    if len(rows) < 3:
        return []
    header, units = rows[0], rows[1]
    kernels = []
    for row in rows[2:]:
        values = dict(zip(header, row))
        unit = dict(zip(header, units))
        duration = _number(values.get("gpu__time_duration.sum", ""))
        if duration is not None and unit.get("gpu__time_duration.sum") in {"nsecond", "ns"}:
            duration /= 1000.0
        elif duration is not None and unit.get("gpu__time_duration.sum") in {"msecond", "ms"}:
            duration *= 1000.0
        l2 = _number(values.get("lts__t_bytes.sum.per_second", ""))
        l2_unit = unit.get("lts__t_bytes.sum.per_second", "")
        scale = {"byte/second": 1e-9, "Kbyte/second": 1e-6, "Mbyte/second": 1e-3, "Gbyte/second": 1.0, "Tbyte/second": 1e3}
        stalls = {
            name: _number(values.get(f"smsp__average_warps_issue_stalled_{name}_per_issue_active.ratio", ""))
            for name in STALLS
        }
        stalls = {name: value for name, value in stalls.items() if value is not None and name not in {"selected", "not_selected"}}
        kernels.append({
            "name": values.get("Kernel Name", "?"),
            "duration_us": duration,
            "grid": values.get("launch__grid_size") or values.get("Grid Size"),
            "block": values.get("launch__block_size") or values.get("Block Size"),
            "registers": _number(values.get("launch__registers_per_thread", "")),
            "occupancy_pct": _number(values.get("sm__warps_active.avg.pct_of_peak_sustained_active", "")),
            "sm_pct": _number(values.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", "")),
            "memory_pct": _number(values.get("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", "")),
            "l2_gbps": l2 * scale[l2_unit] if l2 is not None and l2_unit in scale else None,
            "stall": max(stalls, key=stalls.get) if stalls else None,
            "stall_ratio": max(stalls.values()) if stalls else None,
        })
    return kernels


def _short(name: str, limit: int = 60) -> str:
    name = name.split("(")[0] if "(" in name and len(name) > limit else name
    return name if len(name) <= limit else name[: limit - 1] + "…"


def summary(kernels: list[dict], top: int = 4) -> list[str]:
    if not kernels:
        return ["  profile no kernels ran inside the solution call (is all the work on the CPU?)"]
    total = sum(k["duration_us"] or 0 for k in kernels)
    lines = [f"  profile {len(kernels)} kernel launch{'es' if len(kernels) != 1 else ''}, {total:.1f} us in total"]
    fmt = lambda value, suffix="", digits=0: "?" if value is None else f"{value:.{digits}f}{suffix}"  # noqa: E731
    for kernel in sorted(kernels, key=lambda k: k["duration_us"] or 0, reverse=True)[:top]:
        lines.append(f"  · {_short(kernel['name'])}  {fmt(kernel['duration_us'], ' us', 1)}  grid {kernel['grid']} x block {kernel['block']}"
                     f"  {fmt(kernel['registers'])} regs/thread")
        lines.append(f"      occupancy {fmt(kernel['occupancy_pct'], '%')}   SM busy {fmt(kernel['sm_pct'], '%')}   memory busy {fmt(kernel['memory_pct'], '%')}"
                     f"   L2 {fmt(kernel['l2_gbps'], ' GB/s')}")
        if kernel["stall"]:
            lines.append(f"      main stall: {kernel['stall']} ({kernel['stall_ratio']:.1f} cycles per issued instruction): {STALLS[kernel['stall']]}")
        memory, sm = kernel["memory_pct"] or 0, kernel["sm_pct"] or 0
        if memory >= 60 and memory > sm:
            lines.append("      verdict: memory-bound; fewer bytes per result or better reuse will help most")
        elif sm >= 60:
            lines.append("      verdict: compute-bound; look at instruction count and the busiest pipe")
        elif (kernel["occupancy_pct"] or 100) < 35:
            lines.append("      verdict: low occupancy and neither unit busy; more warps per SM or more work per thread")
        else:
            lines.append("      verdict: latency-bound; neither memory nor compute is near its peak")
    return lines


def run(slug: str, lang: str, file: Path | None) -> bool:
    """Profile one call of the solution in a child process and print the summary."""
    reason = available()
    if reason:
        print("  profile unavailable: " + reason.replace("\n", "\n  "))
        return False
    command = [
        ncu_path(), "--target-processes", "all", "--nvtx", "--nvtx-include", f"{NVTX_RANGE}/",
        "--csv", "--page", "raw", "--metrics", ",".join(METRICS),
        sys.executable, "-m", "gpu_gym", "profile-target", slug, "--lang", lang,
    ] + (["--file", str(file)] if file else [])
    result = subprocess.run(command, capture_output=True, text=True, timeout=900)
    if "ERR_NVGPUCTRPERM" in result.stdout + result.stderr:
        print("  profile unavailable: no permission to read GPU performance counters on this machine.")
        return False
    kernels = parse(result.stdout)
    if result.returncode != 0 and not kernels:
        tail = (result.stderr or result.stdout).strip().splitlines()[-5:]
        print("  profile failed: " + " / ".join(tail))
        return False
    for line in summary(kernels):
        print(line)
    return True
