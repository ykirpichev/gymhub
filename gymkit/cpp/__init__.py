"""Compile and run C++ solutions through a per-exercise harness (problems/<slug>/cpp/harness.cpp).

The harness `#include "solution.cpp"`, converts JSON arguments to C++ types, and times only the
user's call. Builds are cached by content hash under .cache/gymkit/cpp/.
"""

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CACHE = Path(os.environ["GYMKIT_CACHE"]) if os.environ.get("GYMKIT_CACHE") else REPO / ".cache" / "gymkit" / "cpp"
CXX = os.environ.get("GYMKIT_CXX", "g++")

# Tests: catch out-of-bounds access and undefined behaviour with readable messages.
TEST_FLAGS = ["-std=c++20", "-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-D_GLIBCXX_ASSERTIONS"]
BENCH_FLAGS = ["-std=c++20", "-O2", "-DNDEBUG"]


class CompileError(Exception):
    pass


class RunError(Exception):
    pass


def harness_path(problem_dir: Path) -> Path:
    return problem_dir / "cpp" / "harness.cpp"


def reference_path(problem_dir: Path) -> Path:
    return problem_dir / "cpp" / "reference.cpp"


def build(problem_dir: Path, solution: Path, mode: str) -> Path:
    """Compile harness + solution. Return the executable."""
    harness = harness_path(problem_dir)
    if not harness.is_file():
        raise CompileError(f"{problem_dir.name} has no C++ harness")
    flags = TEST_FLAGS if mode == "test" else BENCH_FLAGS
    digest = hashlib.sha256()
    for part in (harness.read_bytes(), solution.read_bytes(), (HERE / "gym_io.hpp").read_bytes(), " ".join([CXX, *flags]).encode()):
        digest.update(part)
        digest.update(b"\0")
    out_dir = CACHE / digest.hexdigest()[:20]
    exe = out_dir / "main"
    if exe.is_file():
        return exe
    out_dir.mkdir(parents=True, exist_ok=True)
    # Compile from a private copy so the harness's `#include "solution.cpp"` finds exactly this file.
    with tempfile.TemporaryDirectory(dir=out_dir) as work:
        shutil.copy2(solution, Path(work) / "solution.cpp")
        command = [CXX, *flags, "-fdiagnostics-color=never", "-I", str(HERE), "-iquote", work, "-o", str(exe) + ".tmp", str(harness)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            text = result.stderr.replace(str(Path(work) / "solution.cpp"), solution.name)
            raise CompileError(_trim_compile_errors(text))
    os.replace(str(exe) + ".tmp", exe)
    return exe


def _trim_compile_errors(text: str, limit: int = 40) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    # Errors in the user's file first; template noise from headers is rarely useful.
    mine = [line for line in lines if line.startswith("solution.cpp")]
    chosen = mine if mine else lines
    out = chosen[:limit]
    if len(chosen) > limit:
        out.append(f"... {len(chosen) - limit} more lines")
    return "\n".join(out)


def run(exe: Path, cases: list[list], timeout: float, capture_dir: Path | None = None) -> list[dict]:
    """Run the harness on argument lists. Raises RunError on crash or timeout, with the results so far.

    With `capture_dir`, case i's printed output is also kept in <capture_dir>/case-<i>.txt, so it
    can be shown even when that case crashes the program.
    """
    payload = json.dumps({"cases": cases})
    env = {**os.environ, "ASAN_OPTIONS": "detect_leaks=0:abort_on_error=0:handle_abort=1:symbolize=1", "UBSAN_OPTIONS": "print_stacktrace=1:halt_on_error=1"}
    if capture_dir is not None:
        env["GYMKIT_CAPTURE_DIR"] = str(capture_dir)
    process = subprocess.Popen(
        [str(exe)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(payload, timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        error = RunError(f"did not finish within {timeout:g} s")
        error.results = _parse(stdout)
        error.timed_out = True
        raise error
    results = _parse(stdout)
    if process.returncode != 0:
        code = process.returncode
        reason = f"killed by {signal.Signals(-code).name}" if code < 0 else f"exited with code {code}"
        error = RunError(f"the program crashed ({reason})\n" + _trim_runtime_errors(stderr))
        error.results = results
        error.timed_out = False
        raise error
    return results


def _parse(stdout: str) -> list[dict]:
    results = []
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return results


def _trim_runtime_errors(stderr: str, limit: int = 25) -> str:
    # Builds compile a private copy of the solution; show it as plain solution.cpp.
    stderr = re.sub(r"\S*/solution\.cpp", "solution.cpp", stderr)
    lines = [line for line in stderr.splitlines() if line.strip()]
    keep = []
    for line in lines:
        if "solution.cpp" in line or "ERROR" in line or "runtime error" in line or "SUMMARY" in line or "Assertion" in line or "Error" in line:
            keep.append(line.strip())
    keep = keep or lines
    return "\n".join(f"  {line}" for line in keep[:limit])
