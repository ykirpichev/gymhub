"""bench --profile (issue #19): parsing Nsight Compute output, the summary, and plumbing the option."""

from pathlib import Path

from gpu_gym import profile
from gymhub.runner.adapters import gpu_adapter, kit_adapter
from tests.test_board_auth import H, board, client, sign_in  # noqa: F401  (fixture)

STALL_COLUMNS = [f"smsp__average_warps_issue_stalled_{name}_per_issue_active.ratio" for name in profile.STALLS]
HEADER = ["ID", "Process ID", "Kernel Name", "Block Size", "Grid Size", "gpu__time_duration.sum", "launch__block_size", "launch__grid_size",
          "launch__registers_per_thread", "sm__warps_active.avg.pct_of_peak_sustained_active", "sm__throughput.avg.pct_of_peak_sustained_elapsed",
          "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", "lts__t_bytes.sum.per_second", *STALL_COLUMNS]
UNITS = ["", "", "", "", "", "usecond", "", "", "register/thread", "%", "%", "%", "Gbyte/second", *["" for _ in STALL_COLUMNS]]


def _row(name, us, regs, occ, sm, mem, l2, stalls):
    return ["0", "1", name, "(256, 1, 1)", "(4096, 1, 1)", str(us), "256", "4096", str(regs), str(occ), str(sm), str(mem), str(l2),
            *[str(stalls.get(n, 0.1)) for n in profile.STALLS]]


def _csv(rows):
    quote = lambda cells: ",".join(f'"{cell}"' for cell in cells)  # noqa: E731
    return "==PROF== Connected to process\n" + "\n".join(quote(r) for r in [HEADER, UNITS, *rows]) + "\n"


def test_parse_and_summarize():
    text = _csv([
        _row("matmul_kernel(const float *, const float *, float *, int, int, int)", 1312.5, 40, 31.2, 22.0, 81.5, 610.0,
             {"long_scoreboard": 9.4, "selected": 12.0}),
        _row("reduce_kernel", 88.0, 18, 92.0, 75.0, 30.0, "1,204.0", {"math_pipe_throttle": 3.0}),
    ])
    kernels = profile.parse(text)
    assert [k["name"][:13] for k in kernels] == ["matmul_kernel", "reduce_kernel"]
    matmul = kernels[0]
    assert matmul["duration_us"] == 1312.5 and matmul["registers"] == 40 and matmul["l2_gbps"] == 610.0
    assert matmul["stall"] == "long_scoreboard"  # "selected" means issuing, never the stall to fix
    assert kernels[1]["l2_gbps"] == 1204.0
    lines = "\n".join(profile.summary(kernels))
    assert "2 kernel launches" in lines and "occupancy 31%" in lines and "regs/thread" in lines
    assert "main stall: long_scoreboard (9.4 cycles" in lines and "coalescing" in lines
    assert "verdict: memory-bound" in lines and "verdict: compute-bound" in lines
    assert lines.index("matmul_kernel") < lines.index("reduce_kernel")  # slowest first


def test_nanoseconds_and_empty_output():
    units = list(UNITS)
    units[HEADER.index("gpu__time_duration.sum")] = "nsecond"
    rows = [HEADER, units, _row("k", 2500, 32, 50, 10, 10, 100, {})]
    text = "\n".join(",".join(f'"{c}"' for c in r) for r in rows)
    assert profile.parse(text)[0]["duration_us"] == 2.5
    assert profile.parse("==PROF== nothing profiled") == []
    assert "no kernels ran" in profile.summary([])[0]


def test_available_explains_the_driver_restriction(monkeypatch, tmp_path):
    params = tmp_path / "params"
    params.write_text("RmProfilingAdminOnly: 1\n")
    real_path = profile.Path
    monkeypatch.setattr(profile, "Path", lambda p: params if p == "/proc/driver/nvidia/params" else real_path(p))
    monkeypatch.setattr(profile, "ncu_path", lambda: "/usr/local/cuda/bin/ncu")
    monkeypatch.setattr(profile.os, "geteuid", lambda: 1000)
    assert "NVreg_RestrictProfilingToAdminUsers=0" in profile.available()
    params.write_text("RmProfilingAdminOnly: 0\n")
    assert profile.available() is None
    monkeypatch.setattr(profile, "ncu_path", lambda: None)
    assert "not installed" in profile.available()


def test_option_reaches_only_gpu_bench_commands(board):
    assert gpu_adapter().command("bench", "matmul", "cuda", {"profile": True})[-1] == "--profile"
    assert "--profile" not in gpu_adapter().command("test", "matmul", "cuda", {"profile": True})
    assert "--profile" not in kit_adapter("algo").command("bench", "dijkstra", "python", {"profile": True})
    owner = client(board)
    sign_in(owner, "owner")
    board.store.set("problems", "algo__p", {"benchable": True}, merge=True)
    job_id = owner.post("/api/jobs", json={"gym": "algo", "slug": "p", "lang": "python", "action": "bench", "code": "x",
                                           "options": {"profile": True, "evil": "x"}}, headers=H).json()["id"]
    assert board.store.get("jobs", job_id)["options"] == {"profile": True}
