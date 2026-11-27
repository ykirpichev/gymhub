"""Run functional tests and median-of-runs benchmarks for one solution."""

import importlib.util
import json
import statistics
from pathlib import Path

import torch

from gpu_gym import progress
from gpu_gym.catalog import LANG_FILES, STARTER_MARK
from gpu_gym.cuda_backend import l2_cache_bytes, load_cuda
from gpu_gym.paths import CACHE, ROOT
from gpu_gym.spec import Problem
from gpu_gym.ui import bold, cyan, dim, green, red, yellow

_FLUSH: torch.Tensor | None = None


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("PyTorch cannot see a CUDA device. Run ./setup.sh and use this project's .venv.")


def solution_path(problem: Problem, lang: str) -> Path:
    from gpu_gym.paths import SOLUTIONS

    return SOLUTIONS / problem.slug / LANG_FILES[lang]


def starter_path(problem: Problem, lang: str) -> Path:
    assert problem.directory is not None
    return problem.directory / "starter" / LANG_FILES[lang]


def is_unsolved(problem: Problem, lang: str, path: Path | None = None) -> bool:
    path = path or solution_path(problem, lang)
    starter = starter_path(problem, lang)
    if not path.is_file() or not starter.is_file():
        return False
    text = path.read_bytes()
    return text == starter.read_bytes() and STARTER_MARK in text


def _load_python(path: Path):
    name = f"gpu_gym_user_{path.parent.name}_{path.stem}_{path.stat().st_mtime_ns}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "solve"):
        raise RuntimeError(f"{path} does not define solve()")
    return module.solve


def load_solution(problem: Problem, lang: str, debug: bool, path: Path | None = None):
    path = path or solution_path(problem, lang)
    if not path.is_file():
        raise SystemExit(f"Missing {path}. Run gpu-gym reset {problem.slug} --force")
    if lang == "cuda":
        return load_cuda(path, problem.params, debug=debug)
    function = _load_python(path)

    def solve(case: dict, _function=function, _params=problem.params) -> None:
        _function(*(case[param.name] for param in _params))

    return solve


def _check_case(problem: Problem, case: dict) -> None:
    for param in problem.params:
        if param.name not in case:
            raise KeyError(f"{problem.slug} case is missing {param.name}")
        value = case[param.name]
        if param.kind == "tensor":
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{problem.slug} {param.name} must be a tensor")
            if value.dtype != param.torch_dtype or value.device.type != "cuda" or not value.is_contiguous():
                raise TypeError(
                    f"{problem.slug} {param.name} must be a contiguous {param.dtype} CUDA tensor, "
                    f"got dtype={value.dtype} device={value.device} contiguous={value.is_contiguous()}"
                )
        elif param.kind == "int" and not isinstance(value, int):
            raise TypeError(f"{problem.slug} {param.name} must be an int")
        elif param.kind == "float" and not isinstance(value, float):
            raise TypeError(f"{problem.slug} {param.name} must be a float")


def _split(problem: Problem, case: dict) -> tuple[dict, dict]:
    user: dict = {}
    reference: dict = {}
    for param in problem.params:
        value = case[param.name]
        if param.kind != "tensor":
            user[param.name] = value
            reference[param.name] = value
            continue
        if param.output:
            _poison(value)
            user[param.name] = value
            reference[param.name] = torch.empty_like(value)
        else:
            user[param.name] = value
            reference[param.name] = value.clone()
    return user, reference


def _poison(tensor: torch.Tensor) -> None:
    """Fill an output before the solution runs, so an element it never writes is caught: NaN for floats,
    the largest value of the type for integers."""
    if tensor.is_floating_point():
        tensor.fill_(float("nan"))
    else:
        tensor.fill_(torch.iinfo(tensor.dtype).max)


def _matches(got: torch.Tensor, want: torch.Tensor, atol: float, rtol: float) -> bool:
    """Floats (of any width) within the tolerance, compared in float32; integers exactly."""
    if got.dtype != want.dtype:
        return False
    if not want.is_floating_point():
        return bool(torch.equal(got, want))
    return bool(torch.allclose(got.float(), want.float(), atol=atol, rtol=rtol))


def _call(solve, problem: Problem, case: dict) -> None:
    del problem
    solve(case)


def _describe_mismatch(got: torch.Tensor, want: torch.Tensor, atol: float, rtol: float) -> str:
    got_flat = got.detach().flatten().float().cpu()
    want_flat = want.detach().flatten().float().cpu()
    bad = ~torch.isclose(got_flat, want_flat, atol=atol, rtol=rtol, equal_nan=False)
    count = int(bad.sum())
    lines = [f"{count} of {got_flat.numel()} values are wrong"]
    if got_flat.numel() <= 32:
        lines.append(f"  got:      {[round(v, 4) for v in got_flat.tolist()]}")
        lines.append(f"  expected: {[round(v, 4) for v in want_flat.tolist()]}")
        return "\n".join(lines)
    shape = tuple(got.shape)
    for flat in bad.nonzero().flatten()[:5].tolist():
        index = flat if len(shape) <= 1 else tuple(int(i) for i in torch.unravel_index(torch.tensor(flat), shape))
        lines.append(f"  [{index}]  got {got_flat[flat].item():.6g}  expected {want_flat[flat].item():.6g}")
    if count > 5:
        lines.append(f"  ... and {count - 5} more")
    if bool(torch.isnan(got_flat).any()):
        lines.append("  NaN in the output usually means an element was never written.")
    return "\n".join(lines)


def _compare(problem: Problem, actual: dict, expected: dict) -> None:
    torch.cuda.synchronize()
    for param in problem.params:
        if not param.output:
            continue
        got = actual[param.name]
        want = expected[param.name]
        if got.shape != want.shape:
            raise AssertionError(f"{param.name} shape {tuple(got.shape)} != {tuple(want.shape)}")
        if not _matches(got, want, problem.atol, problem.rtol):
            raise AssertionError(
                f"{param.name} mismatch (atol {problem.atol:g}, rtol {problem.rtol:g}): "
                + _describe_mismatch(got, want, problem.atol, problem.rtol)
            )


def _case_summary(problem: Problem, case: dict) -> str:
    parts = []
    for param in problem.params:
        value = case[param.name]
        if param.kind == "tensor":
            dtype = "" if param.dtype == "float32" else f" {param.dtype}"
            parts.append(f"{param.name}{list(value.shape)}{dtype}")
        else:
            parts.append(f"{param.name}={value}")
    return ", ".join(parts)


def flush_cache() -> None:
    global _FLUSH
    count = max(l2_cache_bytes() * 2, 32 << 20) // 4
    if _FLUSH is None or _FLUSH.numel() < count:
        _FLUSH = torch.empty(count, device="cuda", dtype=torch.float32)
    _FLUSH.fill_(1.0)
    torch.cuda.synchronize()


def _time_ms(solve, problem: Problem, case: dict) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    flush_cache()
    start.record()
    _call(solve, problem, case)
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _reference_solver(problem: Problem):
    def solve(case: dict) -> None:
        problem.reference(*(case[param.name] for param in problem.params))
        torch.cuda.synchronize()

    return solve


def copy_peak_gbps() -> float:
    """Device-to-device copy bandwidth, measured once and cached. A practical ceiling for memory-bound kernels."""
    name = torch.cuda.get_device_name()
    cache_file = CACHE / "device.json"
    cached: dict = {}
    if cache_file.is_file():
        try:
            cached = json.loads(cache_file.read_text())
        except json.JSONDecodeError:
            cached = {}
    if name in cached:
        return float(cached[name])
    count = 64 << 20  # 256 MiB of float32, far larger than L2
    source = torch.ones(count, device="cuda")
    target = torch.empty_like(source)
    for _ in range(3):
        target.copy_(source)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    repeats = 10
    start.record()
    for _ in range(repeats):
        target.copy_(source)
    end.record()
    end.synchronize()
    seconds = start.elapsed_time(end) / 1e3
    gbps = 2 * count * 4 * repeats / seconds / 1e9
    del source, target
    cached[name] = round(gbps, 1)
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cached, indent=2) + "\n")
    return gbps


def _tag(problem: Problem, lang: str) -> str:
    return f"{bold(problem.slug)}  [{lang}]"


def run_functional(problem: Problem, lang: str, debug: bool = False, record: bool = True, path: Path | None = None) -> bool:
    """Return True when every functional case matches the reference."""
    require_cuda()
    path = path or solution_path(problem, lang)
    if is_unsolved(problem, lang, path):
        print(f"{_tag(problem, lang)}  {yellow('not started')}")
        print(f"  Edit {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path.name}")
        print(dim("  The GPU_GYM_STARTER line marks the blank file. It disappears once you change the file."))
        return False
    try:
        solve = load_solution(problem, lang, debug=debug, path=path)
    except Exception as exc:
        print(f"{_tag(problem, lang)}  {red('failed to load')}")
        print(f"  {exc}")
        if record:
            progress.record_test(problem.slug, lang, passed=False)
        return False

    cases = problem.functional_cases()
    if not cases:
        print(f"{_tag(problem, lang)}  has no functional cases")
        return False
    passed = 0
    for index, case in enumerate(cases, start=1):
        _check_case(problem, case)
        name = str(case.get("_name", f"case {index}"))
        user, reference = _split(problem, case)
        try:
            problem.reference(*(reference[param.name] for param in problem.params))
            _call(solve, problem, user)
            _compare(problem, user, reference)
        except Exception as exc:
            print(f"{_tag(problem, lang)}  {red('FAIL')}  {passed}/{len(cases)} passed, failed on {bold(name)}")
            print(dim(f"  inputs: {_case_summary(problem, case)}"))
            for line in str(exc).splitlines():
                print(f"  {line}")
            if record:
                progress.record_test(problem.slug, lang, passed=False)
            return False
        passed += 1
    print(f"{_tag(problem, lang)}  {green('pass')}  {passed}/{len(cases)}")
    if record:
        first = not progress.is_solved(progress.load(), problem.slug, lang)
        progress.record_test(problem.slug, lang, passed=True)
        if first:
            print(green(f"  Solved {problem.slug} in {lang} for the first time. Run gpu-gym bench to time it."))
    return True


def _measure(solve, problem: Problem, runs: int, warmup: int) -> list[float]:
    for seed in range(warmup):
        case = problem.performance_case(10_000 + seed)
        _check_case(problem, case)
        user, _ = _split(problem, case)
        _time_ms(solve, problem, user)
    samples = []
    for seed in range(runs):
        case = problem.performance_case(20_000 + seed)
        _check_case(problem, case)
        user, _ = _split(problem, case)
        samples.append(_time_ms(solve, problem, user))
    return samples


def run_bench(
    problem: Problem,
    lang: str,
    runs: int = 15,
    warmup: int = 2,
    debug: bool = False,
    compare: bool = True,
    path: Path | None = None,
) -> bool:
    require_cuda()
    if not run_functional(problem, lang, debug=debug, path=path):
        return False
    solve = load_solution(problem, lang, debug=debug, path=path)
    probe = problem.performance_case(0)
    _check_case(problem, probe)
    user, reference = _split(problem, probe)
    try:
        problem.reference(*(reference[param.name] for param in problem.params))
        _call(solve, problem, user)
        _compare(problem, user, reference)
    except Exception as exc:
        print(f"{_tag(problem, lang)}  {red('performance case failed')}")
        print(dim(f"  inputs: {_case_summary(problem, probe)}"))
        for line in str(exc).splitlines():
            print(f"  {line}")
        return False

    samples = _measure(solve, problem, runs, warmup)
    median = statistics.median(samples)
    print(
        f"  time    {bold(f'{median:.4f} ms')} median   "
        + dim(f"min {min(samples):.4f}  max {max(samples):.4f}  {runs} runs on {_case_summary(problem, probe)}")
    )

    gbps = None
    if problem.traffic_bytes is not None:
        nbytes = problem.traffic_bytes(probe)
        gbps = nbytes / (median / 1e3) / 1e9
        peak = copy_peak_gbps()
        share = gbps / peak * 100
        colour = green if share >= 70 else yellow if share >= 35 else red
        print(
            f"  memory  {gbps:.1f} GB/s = {colour(f'{share:.0f}%')} of this GPU's copy peak ({peak:.0f} GB/s)   "
            + dim(f"{nbytes / 2**20:.1f} MiB moved")
        )
    tflops = None
    if problem.flops is not None:
        tflops = problem.flops(probe) / (median / 1e3) / 1e12
        print(f"  compute {tflops:.3f} TFLOP/s")

    if compare:
        ref_samples = _measure(_reference_solver(problem), problem, max(5, runs // 2), 1)
        ref = statistics.median(ref_samples)
        ratio = ref / median
        if ratio >= 1:
            verdict = green(f"{ratio:.2f}x faster than PyTorch")
        else:
            verdict = yellow(f"{1 / ratio:.2f}x slower than PyTorch")
        print(f"  torch   {ref:.4f} ms   you are {verdict}")

    previous = progress.record_bench(problem.slug, lang, median, gbps, tflops)
    if previous is None:
        print(dim("  best    first benchmark recorded"))
    elif median < previous:
        print(f"  best    {cyan('new best!')} {previous:.4f} -> {median:.4f} ms")
    else:
        print(dim(f"  best    {previous:.4f} ms (this run {median / previous:.2f}x of best)"))
    return True


def _show_tensor(name: str, got: torch.Tensor, want: torch.Tensor, atol: float, rtol: float) -> None:
    got_flat = got.detach().flatten().float().cpu()
    want_flat = want.detach().flatten().float().cpu()
    shape = list(got.shape)
    if got_flat.numel() <= 64:
        print(f"  {name} got:      {[round(v, 5) for v in got_flat.tolist()]}")
        print(f"  {name} expected: {[round(v, 5) for v in want_flat.tolist()]}")
        return
    print(f"  {name} {shape}  first 8 got:      {[round(v, 5) for v in got_flat[:8].tolist()]}")
    print(f"  {name} {shape}  first 8 expected: {[round(v, 5) for v in want_flat[:8].tolist()]}")
    if got_flat.shape == want_flat.shape:
        diff = (got_flat - want_flat).abs()
        finite = diff[torch.isfinite(diff)]
        worst = float(finite.max()) if finite.numel() else float("nan")
        nans = int(torch.isnan(got_flat).sum())
        print(f"  {name} max |got - expected| = {worst:.3g}" + (f", {nans} NaN" if nans else ""))


def custom_cases(problem: Problem, cases_file: Path) -> list[dict]:
    """User cases from a JSON file (see gymkit/cases.py), with int/float arguments coerced."""
    from gymkit.cases import load_cases

    cases = load_cases(cases_file, [param.name for param in problem.params])
    for case in cases:
        for param in problem.params:
            value = case[param.name]
            if param.kind == "int" and isinstance(value, (int, float)) and not isinstance(value, bool):
                case[param.name] = int(value)
            elif param.kind == "float" and isinstance(value, (int, float)) and not isinstance(value, bool):
                case[param.name] = float(value)
            elif param.kind == "tensor" and isinstance(value, torch.Tensor):
                case[param.name] = value.to(device="cuda", dtype=param.torch_dtype).contiguous()
    return cases


def case_template(problem: Problem) -> dict:
    """The first built-in case as editable JSON: the starting point for a custom case."""
    from gymkit.cases import template

    case = problem.functional_cases()[0]
    outputs = {param.name for param in problem.params if param.output}
    return template(case, [param.name for param in problem.params], outputs)


def run_debug(
    problem: Problem,
    lang: str,
    path: Path | None = None,
    debug: bool = False,
    cases_file: Path | None = None,
    only_custom: bool = False,
) -> bool:
    """Run every functional case (and any custom ones) and show each output. Records nothing."""
    require_cuda()
    path = path or solution_path(problem, lang)
    tag = _tag(problem, lang)
    print(dim(f"{problem.slug} [{lang}]  run mode: every case, outputs shown, nothing recorded"))
    try:
        solve = load_solution(problem, lang, debug=debug, path=path)
    except Exception as exc:
        print(f"{tag}  {red('failed to load')}")
        print(f"  {exc}")
        return False
    cases = [] if only_custom else problem.functional_cases()
    custom: list[dict] = []
    if cases_file is not None:
        from gymkit.cases import CaseError

        try:
            custom = custom_cases(problem, cases_file)
        except CaseError as exc:
            print(f"{tag}  {red('bad custom case')}")
            print(f"  {exc}")
            return False
    cases = cases + custom
    if not cases:
        print("No cases to run: add a custom case, or untick 'only my cases'.")
        return False
    passed = 0
    for index, case in enumerate(cases, start=1):
        name = str(case.get("_name", f"case {index}"))
        print()
        print(bold(f"── case {index}/{len(cases)}: {name}"))
        try:
            _check_case(problem, case)
        except (TypeError, KeyError) as exc:
            print(f"  {red('INVALID INPUT')} {exc}")
            continue
        print(dim(f"  inputs: {_case_summary(problem, case)}"), flush=True)
        is_custom = any(case is item for item in custom)
        if is_custom and any(param.kind == "tensor" and case[param.name].numel() <= 64 for param in problem.params if not param.output):
            for param in problem.params:
                if param.kind == "tensor" and not param.output and case[param.name].numel() <= 64:
                    print(dim(f"  {param.name} = {[round(v, 5) for v in case[param.name].flatten().tolist()]}"))
        user, reference = _split(problem, case)
        try:
            problem.reference(*(reference[param.name] for param in problem.params))
        except Exception as exc:
            print(f"  {red('INVALID INPUT')} the reference rejects this case: {exc}")
            continue
        try:
            _call(solve, problem, user)
            torch.cuda.synchronize()
        except Exception as exc:
            print(f"  {red('ERROR')} {exc}")
            if "illegal" in str(exc).lower() or "misaligned" in str(exc).lower():
                print(dim("  The CUDA context is broken after this error; stopping here."))
                break
            continue
        ok = True
        for param in problem.params:
            if not param.output:
                continue
            got, want = user[param.name], reference[param.name]
            _show_tensor(param.name, got, want, problem.atol, problem.rtol)
            if got.shape != want.shape:
                ok = False
                print(f"  {param.name} shape {list(got.shape)} != expected {list(want.shape)}")
            elif not _matches(got, want, problem.atol, problem.rtol):
                ok = False
                if got.numel() > 64:
                    for line in _describe_mismatch(got, want, problem.atol, problem.rtol).splitlines():
                        print(f"  {line}")
        if ok:
            passed += 1
            print(f"  {green('PASS')}")
        else:
            print(f"  {red('FAIL')}  (atol {problem.atol:g}, rtol {problem.rtol:g})")
    print()
    verdict = green("all passed") if passed == len(cases) else red(f"{len(cases) - passed} failed")
    print(f"{tag}  {passed}/{len(cases)} passed, {verdict}. " + dim("Not recorded. Submit to record it."))
    return passed == len(cases)
