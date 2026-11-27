"""Run functional tests and timed benchmarks for a Python solution."""

import contextlib
import copy
import io
import importlib.util
import signal
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from gymkit import cpp
from gymkit.compare import _compare_array, _numpy, _short, _torch, default_compare  # noqa: F401 (graders import default_compare from here)
from gymkit.gym import STARTER_MARK, Gym
from gymkit.progress import Progress
from gymkit.spec import Problem


def _sync() -> None:
    torch = _torch()
    if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


def load_module(path: Path):
    name = f"gymkit_user_{path.parent.name}_{path.stat().st_mtime_ns}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _args(problem: Problem, case: dict, fresh: bool = False) -> list:
    """The case's arguments, deep-copied so a solution that mutates them cannot change the case. A
    `fresh` case (just made by performance_case and used for one call) needs no copy: copying a large
    one can take longer than solving it."""
    missing = [name for name in problem.params if name not in case]
    if missing:
        raise KeyError(f"{problem.slug} case is missing {missing}")
    return [case[name] if fresh else copy.deepcopy(case[name]) for name in problem.params]


def _invoke(problem: Problem, module, args: list) -> Any:
    if problem.call is not None:
        return problem.call(module, args)
    if not hasattr(module, "solve"):
        raise AttributeError("the solution does not define solve()")
    return module.solve(*args)


def _check(problem: Problem, got: Any, want: Any, case: dict) -> None:
    if problem.compare is not None:
        problem.compare(got, want, case)
    else:
        default_compare(got, want, problem.atol, problem.rtol)


def _describe(problem: Problem, case: dict) -> str:
    return ", ".join(f"{name}={_short(case[name], 60)}" for name in problem.params)


def _user_frames(exc: BaseException, path: Path) -> str:
    frames = [frame for frame in traceback.extract_tb(exc.__traceback__) if frame.filename == str(path)]
    if not frames:
        return ""
    frame = frames[-1]
    return f"  at {path.name}:{frame.lineno}  {frame.line or ''}".rstrip()


def run_functional(gym: Gym, problem: Problem, lang: str, record: bool = True, path: Path | None = None) -> bool:
    path = path or gym.solution_path(problem.slug, lang)
    progress = Progress(gym.progress_file)
    tag = f"{problem.slug}  [{lang}]"
    if not path.is_file():
        print(f"{tag}  missing {path}")
        return False
    starter = gym.starter_path(problem.slug, lang)
    if starter.is_file() and path.read_text() == starter.read_text() and STARTER_MARK in path.read_text():
        print(f"{tag}  not started")
        print(f"  Edit {path.relative_to(gym.root.parent.parent) if path.is_relative_to(gym.root.parent.parent) else path.name}")
        print("  The GYM_STARTER line marks the blank file. It disappears once you change the file.")
        return False
    if gym.lang(lang).kind == "cpp":
        return _cpp_functional(gym, problem, lang, path, progress, tag, record)
    try:
        module = load_module(path)
    except Exception as exc:
        print(f"{tag}  failed to load")
        print(f"  {type(exc).__name__}: {exc}")
        line = _user_frames(exc, path)
        if line:
            print(line)
        if record:
            progress.record_test(problem.slug, lang, passed=False)
        return False

    cases = problem.functional_cases()
    passed = 0
    for index, case in enumerate(cases, start=1):
        name = str(case.get("_name", f"case {index}"))
        try:
            want = problem.reference(*_args(problem, case))
            started = time.perf_counter()
            got = _invoke(problem, module, _args(problem, case))
            _sync()
            elapsed = time.perf_counter() - started
            _check(problem, got, want, case)
        except Exception as exc:
            print(f"{tag}  FAIL  {passed}/{len(cases)} passed, failed on {name}")
            print(f"  inputs: {_describe(problem, case)}")
            if isinstance(exc, AssertionError):
                for text in str(exc).splitlines():
                    print(f"  {text}")
            else:
                print(f"  {type(exc).__name__}: {exc}")
                line = _user_frames(exc, path)
                if line:
                    print(line)
            if record:
                progress.record_test(problem.slug, lang, passed=False)
            return False
        if elapsed > 10:
            print(f"  note: {name} took {elapsed:.1f} s")
        passed += 1
    print(f"{tag}  pass  {passed}/{len(cases)}")
    if record and progress.record_test(problem.slug, lang, passed=True):
        print(f"  Solved {problem.slug} in {lang} for the first time.")
    return True


class _Deadline(Exception):
    pass


def _time_once(fn, deadline_ms: float | None = None) -> float:
    """Time one call. With a deadline, interrupt a call that runs past it and return infinity."""
    def on_alarm(_signum, _frame):
        raise _Deadline()

    previous = None
    if deadline_ms is not None:
        previous = signal.signal(signal.SIGALRM, on_alarm)
        signal.setitimer(signal.ITIMER_REAL, deadline_ms / 1e3)
    try:
        _sync()
        started = time.perf_counter()
        fn()
        _sync()
        return (time.perf_counter() - started) * 1e3
    except _Deadline:
        return float("inf")
    finally:
        if deadline_ms is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)


def run_bench(gym: Gym, problem: Problem, lang: str, runs: int = 7, compare: bool = True, path: Path | None = None) -> bool:
    if problem.performance_case is None:
        print(f"{problem.slug} has no benchmark input. Tests passing is the whole exercise.")
        return run_functional(gym, problem, lang, path=path)
    if not run_functional(gym, problem, lang, path=path):
        return False
    path = path or gym.solution_path(problem.slug, lang)
    if gym.lang(lang).kind == "cpp":
        return _cpp_bench(gym, problem, lang, runs, compare, path)
    module = load_module(path)
    probe = problem.performance_case(0)
    limit = problem.limit_for(lang)

    def too_slow(ms: float) -> bool:
        if limit is not None and ms > 3 * limit:
            took = "did not finish within" if ms == float("inf") else f"took {ms:.0f} ms, over"
            print(f"  limit   FAIL  one run {took} 3x the {limit:g} ms limit. Stopped early.")
            print("          Look for a better algorithm.")
            return True
        return False

    try:
        want = problem.reference(*_args(problem, probe))
        probe_args = _args(problem, probe)
        got_box: list = []
        deadline = 3 * limit + 1 if limit is not None else None
        probe_ms = _time_once(lambda: got_box.append(_invoke(problem, module, probe_args)), deadline)
        if not got_box:
            too_slow(probe_ms)
            return False
        got = got_box[0]
        _check(problem, got, want, probe)
    except Exception as exc:
        print("  performance case failed")
        print(f"  inputs: {_describe(problem, probe)}")
        for text in str(exc).splitlines() or [type(exc).__name__]:
            print(f"  {text}")
        return False
    if too_slow(probe_ms):
        return False

    samples = []
    for seed in range(runs):
        case = problem.performance_case(1000 + seed)
        args = _args(problem, case, fresh=True)
        samples.append(_time_once(lambda: _invoke(problem, module, args), deadline))
        if too_slow(samples[-1]):
            return False
    median = statistics.median(samples)
    print(f"  time    {median:.4f} ms median   min {min(samples):.4f}  max {max(samples):.4f}  {runs} runs")
    print(f"  input   {_describe(problem, probe)}")

    ref_ms = None
    if compare:
        ref_samples = []
        for seed in range(max(3, runs // 2)):
            case = problem.performance_case(1000 + seed)
            args = _args(problem, case, fresh=True)
            ref_samples.append(_time_once(lambda: problem.reference(*args)))
        ref_ms = statistics.median(ref_samples)
        ratio = ref_ms / median if median > 0 else float("inf")
        verdict = f"{ratio:.2f}x faster than" if ratio >= 1 else f"{1 / ratio:.2f}x slower than"
        print(f"  ref     {ref_ms:.4f} ms   you are {verdict} the reference")

    return _finish_bench(gym, problem, lang, median, ref_ms, limit)


def _finish_bench(gym: Gym, problem: Problem, lang: str, median: float, ref_ms: float | None, limit: float | None) -> bool:
    ok = True
    if limit is not None:
        if median > limit:
            print(f"  limit   FAIL  {median:.1f} ms is over the {limit:g} ms limit. Look for a better algorithm.")
            ok = False
        else:
            print(f"  limit   ok    under the {limit:g} ms limit")

    previous = Progress(gym.progress_file).record_bench(problem.slug, lang, median, ref_ms)
    if previous is None:
        print("  best    first benchmark recorded")
    elif median < previous:
        print(f"  best    new best! {previous:.4f} -> {median:.4f} ms")
    else:
        print(f"  best    {previous:.4f} ms (this run {median / previous:.2f}x of best)")
    return ok


# ---- C++ ---------------------------------------------------------------------------------------


def _plain_args(problem: Problem, case: dict) -> list:
    return [case[name] for name in problem.params]


def _cpp_build(problem: Problem, path: Path, mode: str, tag: str):
    assert problem.directory is not None
    try:
        return cpp.build(problem.directory, path, mode)
    except cpp.CompileError as exc:
        print(f"{tag}  compile error")
        for line in str(exc).splitlines():
            print(f"  {line}")
        return None


def _cpp_functional(gym: Gym, problem: Problem, lang: str, path: Path, progress: Progress, tag: str, record: bool) -> bool:
    def failed() -> bool:
        if record:
            progress.record_test(problem.slug, lang, passed=False)
        return False

    exe = _cpp_build(problem, path, "test", tag)
    if exe is None:
        return failed()
    cases = problem.functional_cases()
    crash = None
    try:
        results = cpp.run(exe, [_plain_args(problem, case) for case in cases], timeout=60)
    except cpp.RunError as exc:
        results, crash = exc.results, exc
    for index, case in enumerate(cases):
        name = str(case.get("_name", f"case {index + 1}"))
        header = f"{tag}  FAIL  {index}/{len(cases)} passed, failed on {name}"
        if index >= len(results):
            print(header)
            print(f"  inputs: {_describe(problem, case)}")
            for line in str(crash or "no output for this case").splitlines():
                print(f"  {line}")
            return failed()
        outcome = results[index]
        if "error" in outcome:
            print(header)
            print(f"  inputs: {_describe(problem, case)}")
            print(f"  exception: {outcome['error']}")
            _print_stdout(outcome.get("stdout", ""))
            return failed()
        try:
            want = problem.reference(*_args(problem, case))
            _check(problem, outcome["result"], want, case)
        except AssertionError as exc:
            print(header)
            print(f"  inputs: {_describe(problem, case)}")
            for text in str(exc).splitlines():
                print(f"  {text}")
            _print_stdout(outcome.get("stdout", ""))
            return failed()
    print(f"{tag}  pass  {len(cases)}/{len(cases)}")
    if record and progress.record_test(problem.slug, lang, passed=True):
        print(f"  Solved {problem.slug} in {lang} for the first time.")
    return True


def _cpp_time(exe: Path, problem: Problem, case: dict, deadline_ms: float | None) -> tuple[float, object]:
    """Run one case in its own process. Return (ms, result); ms is infinity past the deadline."""
    timeout = (deadline_ms / 1e3 + 2.0) if deadline_ms is not None else 120.0
    try:
        results = cpp.run(exe, [_plain_args(problem, case)], timeout=timeout)
    except cpp.RunError as exc:
        if getattr(exc, "timed_out", False):
            return float("inf"), None
        raise
    if not results:
        raise RuntimeError("the program printed no result")
    if "error" in results[0]:
        raise RuntimeError(f"exception: {results[0]['error']}")
    ms = float(results[0]["ms"])
    if deadline_ms is not None and ms > deadline_ms:
        return float("inf"), results[0]["result"]
    return ms, results[0]["result"]


def _cpp_bench(gym: Gym, problem: Problem, lang: str, runs: int, compare: bool, path: Path) -> bool:
    assert problem.performance_case is not None and problem.directory is not None
    tag = f"{problem.slug}  [{lang}]"
    exe = _cpp_build(problem, path, "bench", tag)
    if exe is None:
        return False
    limit = problem.limit_for(lang)
    deadline = 3 * limit if limit is not None else None

    def too_slow(ms: float) -> bool:
        if limit is not None and ms > 3 * limit:
            took = "did not finish within" if ms == float("inf") else f"took {ms:.0f} ms, over"
            print(f"  limit   FAIL  one run {took} 3x the {limit:g} ms limit. Stopped early.")
            print("          Look for a better algorithm.")
            return True
        return False

    probe = problem.performance_case(0)
    try:
        probe_ms, got = _cpp_time(exe, problem, probe, deadline)
        if too_slow(probe_ms):
            return False
        _check(problem, got, problem.reference(*_args(problem, probe)), probe)
    except Exception as exc:
        print("  performance case failed")
        print(f"  inputs: {_describe(problem, probe)}")
        for text in str(exc).splitlines() or [type(exc).__name__]:
            print(f"  {text}")
        return False

    samples = []
    for seed in range(runs):
        ms, _ = _cpp_time(exe, problem, problem.performance_case(1000 + seed), deadline)
        if too_slow(ms):
            return False
        samples.append(ms)
    median = statistics.median(samples)
    print(f"  time    {median:.4f} ms median   min {min(samples):.4f}  max {max(samples):.4f}  {runs} runs")
    print(f"  input   {_describe(problem, probe)}")

    ref_ms = None
    reference = cpp.reference_path(problem.directory)
    if compare and reference.is_file():
        try:
            ref_exe = cpp.build(problem.directory, reference, "bench")
            ref_samples = [
                _cpp_time(ref_exe, problem, problem.performance_case(1000 + seed), None)[0]
                for seed in range(max(3, runs // 2))
            ]
            ref_ms = statistics.median(ref_samples)
            ratio = ref_ms / median if median > 0 else float("inf")
            verdict = f"{ratio:.2f}x faster than" if ratio >= 1 else f"{1 / ratio:.2f}x slower than"
            print(f"  ref     {ref_ms:.4f} ms   you are {verdict} the C++ reference")
        except Exception as exc:
            print(f"  ref     unavailable: {exc}")
    return _finish_bench(gym, problem, lang, median, ref_ms, limit)


# ---- run mode: every case, outputs shown, nothing recorded -------------------------------------


def _display(value: Any, limit: int = 600) -> str:
    torch = _torch()
    if torch is not None and isinstance(value, torch.Tensor):
        flat = value.detach().flatten().float().cpu()
        head = f"tensor(shape={list(value.shape)}, dtype={str(value.dtype).replace('torch.', '')})"
        if flat.numel() <= 32:
            return f"{head} {[round(v, 5) for v in flat.tolist()]}"
        return f"{head} first 8 {[round(v, 5) for v in flat[:8].tolist()]}"
    return _short(value, limit)


def _verdict(problem: Problem, got: Any, want: Any, case: dict) -> str | None:
    """None when correct, else the mismatch message."""
    try:
        _check(problem, got, want, case)
        return None
    except AssertionError as exc:
        return str(exc)


def _print_stdout(text: str) -> None:
    text = text.rstrip("\n")
    if text:
        print("  stdout:")
        for line in text.splitlines():
            print(f"  | {line}")


def case_template(problem: Problem) -> dict:
    """The first built-in case as editable JSON: the starting point for a custom case."""
    from gymkit.cases import template

    return template(problem.functional_cases()[0], problem.params)


def run_debug(
    gym: Gym,
    problem: Problem,
    lang: str,
    path: Path | None = None,
    cases_file: Path | None = None,
    only_custom: bool = False,
) -> bool:
    path = path or gym.solution_path(problem.slug, lang)
    tag = f"{problem.slug}  [{lang}]"
    print(f"{problem.slug} [{lang}]  run mode: every case, outputs shown, nothing recorded")
    if not path.is_file():
        print(f"{tag}  missing {path}")
        return False
    cases = [] if only_custom else problem.functional_cases()
    if cases_file is not None:
        from gymkit.cases import CaseError, load_cases

        try:
            cases = cases + load_cases(cases_file, problem.params)
        except CaseError as exc:
            print(f"{tag}  bad custom case")
            print(f"  {exc}")
            return False
    if not cases:
        print("No cases to run: add a custom case, or untick 'only my cases'.")
        return False

    def expected(case: dict):
        """The reference answer, or None with a message when the reference rejects the input."""
        try:
            return True, problem.reference(*_args(problem, case))
        except Exception as exc:
            print(f"  INVALID INPUT  the reference rejects this case: {type(exc).__name__}: {exc}")
            return False, None

    passed = 0

    def show(index: int, case: dict, got: Any, want: Any, stdout: str = "") -> None:
        nonlocal passed
        _print_stdout(stdout)
        print(f"  output:   {_display(got)}")
        print(f"  expected: {_display(want)}")
        problem_text = _verdict(problem, got, want, case)
        if problem_text is None:
            passed += 1
            print("  PASS")
        else:
            print("  FAIL")
            for line in problem_text.splitlines():
                print(f"    {line}")

    def header(index: int, case: dict) -> None:
        name = str(case.get("_name", f"case {index + 1}"))
        print()
        print(f"── case {index + 1}/{len(cases)}: {name}")
        print(f"  inputs:   {_describe(problem, case)}", flush=True)

    if gym.lang(lang).kind == "cpp":
        exe = _cpp_build(problem, path, "test", tag)
        if exe is None:
            return False
        import tempfile

        with tempfile.TemporaryDirectory() as capture_dir:
            crash = None
            try:
                results = cpp.run(exe, [_plain_args(problem, case) for case in cases], timeout=60, capture_dir=Path(capture_dir))
            except cpp.RunError as exc:
                results, crash = exc.results, exc
            for index, case in enumerate(cases):
                header(index, case)
                if index >= len(results):
                    printed = Path(capture_dir, f"case-{index}.txt")
                    _print_stdout(printed.read_text(errors="replace") if printed.is_file() else "")
                    print("  CRASH")
                    for line in str(crash or "no output for this case").splitlines():
                        print(f"    {line}")
                    print("\n  The program stopped here; later cases did not run.")
                    break
                outcome = results[index]
                valid, want = expected(case)
                if not valid:
                    continue
                if "error" in outcome:
                    _print_stdout(outcome.get("stdout", ""))
                    print(f"  ERROR  exception: {outcome['error']}")
                    continue
                show(index, case, outcome["result"], want, outcome.get("stdout", ""))
    else:
        try:
            module = load_module(path)
        except Exception as exc:
            print(f"{tag}  failed to load")
            print(f"  {type(exc).__name__}: {exc}")
            line = _user_frames(exc, path)
            if line:
                print(line)
            return False
        for index, case in enumerate(cases):
            header(index, case)
            valid, want = expected(case)
            if not valid:
                continue
            printed = io.StringIO()
            try:
                with contextlib.redirect_stdout(printed):
                    got = _invoke(problem, module, _args(problem, case))
                    _sync()
            except Exception as exc:
                _print_stdout(printed.getvalue())
                print(f"  ERROR  {type(exc).__name__}: {exc}")
                line = _user_frames(exc, path)
                if line:
                    print(line)
                continue
            show(index, case, got, want, printed.getvalue()[-16000:])
    print()
    failed = len(cases) - passed
    print(f"{tag}  {passed}/{len(cases)} passed{', all passed' if not failed else f', {failed} failed'}. Not recorded. Submit to record it.")
    return failed == 0
