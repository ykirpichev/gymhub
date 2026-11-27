"""Command line for local GPU exercises."""

import argparse
import os
import re
import time

# Keep Triton's current FP-fusion behavior and skip its version warning.
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

from gpu_gym import __version__, progress
from gpu_gym.catalog import LANG_FILES, LANGS, WARMUP, all_problems, get_problem
from gpu_gym.paths import PROBLEMS, ROOT, SOLUTIONS
from gpu_gym.runner import is_unsolved, run_bench, run_debug, run_functional
from gpu_gym.scaffold import ensure_solutions, new_problem, reset_solution, write_compile_db
from gpu_gym.spec import Problem
from gpu_gym.ui import bar, bold, cyan, dim, green, markdown, red, yellow


def _languages(name: str) -> tuple[str, ...]:
    if name == "all":
        return LANGS
    return (name,)


def _selected(slugs: list[str], all_of_them: bool):
    if all_of_them:
        return all_problems()
    if not slugs:
        raise SystemExit("Name an exercise, or pass --all. Try: gpu-gym list")
    return [get_problem(slug) for slug in slugs]


def _status_mark(data: dict, problem: Problem, lang: str) -> str:
    entry = data["exercises"].get(problem.slug, {}).get("langs", {}).get(lang)
    if entry and "first_solved" in entry:
        return green("✓")
    if entry and entry.get("attempts"):
        return red("✗")
    return dim("·")


def _cmd_list(_args: argparse.Namespace) -> int:
    ensure_solutions()
    problems = all_problems()
    if not problems:
        print(f"No exercises in {PROBLEMS}")
        return 1
    data = progress.load()
    print(bold(f"{'#':>2}  {'exercise':<16} {'level':<7} {'cu py tr':<9} title"))
    for number, problem in enumerate(problems):
        marks = " ".join(_status_mark(data, problem, lang) + " " for lang in LANGS)
        level = {"easy": green, "medium": yellow, "hard": red}[problem.difficulty](f"{problem.difficulty:<7}")
        print(f"{number:>2}  {problem.slug:<16} {level} {marks} {problem.title}")
    print()
    print(dim("✓ solved   ✗ tried, not passing yet   · not started      columns: cuda pytorch triton"))
    print(dim("gpu-gym next            open the next unsolved exercise"))
    print(dim("gpu-gym show <name>     read a problem      gpu-gym progress   see your stats"))
    return 0


def _tests_section(problem: Problem) -> str:
    """Describe the actual test inputs from problem.py, so the statement never drifts from the checker."""
    from gpu_gym.runner import _case_summary, copy_peak_gbps, require_cuda

    require_cuda()
    lines = ["## Tests", ""]
    lines.append(f"Checked against the PyTorch reference with `atol={problem.atol:g}`, `rtol={problem.rtol:g}`.")
    lines.append("")
    for case in problem.functional_cases():
        lines.append(f"- **{case.get('_name', 'case')}**: `{_case_summary(problem, case)}`")
    perf = problem.performance_case(0)
    lines.append("")
    lines.append(f"Benchmark input: `{_case_summary(problem, perf)}`")
    lines.append("")
    if problem.traffic_bytes is not None:
        nbytes = problem.traffic_bytes(perf)
        peak = copy_peak_gbps()
        floor_ms = nbytes / (peak * 1e9) * 1e3
        moved = (
            f"It moves at least {nbytes / 2**20:.1f} MiB. At this GPU's copy peak ({peak:.0f} GB/s) "
            f"that takes {floor_ms:.4f} ms"
        )
        if problem.flops is None:
            lines.append(moved + ", so no kernel can be much faster than that.")
        else:
            lines.append(moved + ".")
    if problem.flops is not None:
        lines.append(
            f"It does {problem.flops(perf) / 1e9:.2f} GFLOP of arithmetic, so it is limited by compute, "
            "not memory. Judge your kernel by the TFLOP/s line of `gpu-gym bench` and the PyTorch comparison."
        )
    return "\n".join(lines)


def _files_section(problem: Problem) -> str:
    lines = ["## Your files", ""]
    labels = {"cuda": "CUDA", "pytorch": "PyTorch", "triton": "Triton"}
    data = progress.load()
    for lang in LANGS:
        path = (SOLUTIONS / problem.slug / LANG_FILES[lang]).relative_to(ROOT)
        state = "solved" if progress.is_solved(data, problem.slug, lang) else (
            "not started" if is_unsolved(problem, lang) else "in progress"
        )
        lines.append(f"- {labels[lang]}: `{path}` ({state})")
    lines.append("")
    lines.append(f"`gpu-gym test {problem.slug}`, `gpu-gym watch {problem.slug}`, `gpu-gym bench {problem.slug}`")
    total = len(_hints(problem))
    if total:
        seen = progress.hints_seen(problem.slug)
        lines.append("")
        lines.append(f"Stuck? `gpu-gym hint {problem.slug}` ({total} hints, {seen} seen)")
    return "\n".join(lines)


def _show(problem: Problem) -> None:
    readme = PROBLEMS / problem.slug / "PROBLEM.md"
    text = readme.read_text().rstrip() if readme.is_file() else f"# {problem.title}\n\n{problem.summary}"
    markdown(text + "\n\n" + _tests_section(problem) + "\n\n" + _files_section(problem))


def _cmd_show(args: argparse.Namespace) -> int:
    ensure_solutions()
    _show(get_problem(args.slug))
    return 0


def _hints(problem: Problem) -> list[str]:
    path = PROBLEMS / problem.slug / "HINTS.md"
    if not path.is_file():
        return []
    parts = re.split(r"^## .*$", path.read_text(), flags=re.MULTILINE)
    return [part.strip() for part in parts[1:] if part.strip()]


def _cmd_hint(args: argparse.Namespace) -> int:
    problem = get_problem(args.slug)
    hints = _hints(problem)
    if not hints:
        print(f"{problem.slug} has no hints yet. Add problems/{problem.slug}/HINTS.md with '## Hint 1' sections.")
        return 1
    seen = progress.hints_seen(problem.slug)
    if args.all:
        start, wanted = 1, len(hints)
    elif args.number is not None:
        if not 1 <= args.number <= len(hints):
            print(f"{problem.slug} has hints 1 to {len(hints)}.")
            return 1
        start = wanted = args.number
    else:
        # Repeat the hints already seen, then reveal one more.
        start, wanted = 1, min(seen + 1, len(hints))
    body = [f"## Hint {number} of {len(hints)}\n\n{hints[number - 1]}" for number in range(start, wanted + 1)]
    markdown("\n\n".join(body))
    progress.set_hints_seen(problem.slug, wanted)
    if wanted < len(hints):
        print(dim(f"\nRun gpu-gym hint {problem.slug} again for hint {wanted + 1}."))
    return 0


def _practice(problems: list[Problem]) -> list[Problem]:
    return [problem for problem in problems if problem.slug not in WARMUP]


def _cmd_progress(args: argparse.Namespace) -> int:
    if args.reset is not None:
        target = None if args.reset == "all" else get_problem(args.reset).slug
        if not args.force:
            print(f"This erases recorded progress for {target or 'every exercise'}. Re-run with --force.")
            return 1
        progress.reset(target)
        print(f"Cleared progress for {target or 'every exercise'}.")
        return 0

    problems = _practice(all_problems())
    data = progress.load()
    solved = [problem for problem in problems if progress.is_solved(data, problem.slug)]
    print(bold("Progress") + f"   {len(solved)}/{len(problems)} exercises solved in at least one language")
    print(f"  {bar(len(solved) / max(len(problems), 1))} {len(solved) / max(len(problems), 1) * 100:.0f}%")
    per_lang = []
    for lang in LANGS:
        count = sum(progress.is_solved(data, problem.slug, lang) for problem in problems)
        per_lang.append(f"{lang} {count}/{len(problems)}")
    by_level = []
    for level in ("easy", "medium", "hard"):
        group = [problem for problem in problems if problem.difficulty == level]
        if group:
            done = sum(progress.is_solved(data, problem.slug) for problem in group)
            by_level.append(f"{level} {done}/{len(group)}")
    print("  " + "   ".join(per_lang))
    print("  " + "   ".join(by_level))
    print()
    header = f"{'exercise':<16} {'level':<7} " + "".join(f"{lang:<15}" for lang in LANGS) + "hints"
    print(bold(header))
    for problem in problems:
        entry = data["exercises"].get(problem.slug, {})
        cells = []
        for lang in LANGS:
            item = entry.get("langs", {}).get(lang, {})
            if "first_solved" in item:
                best = item.get("best_ms")
                text = f"✓ {best:.4f}ms" if best is not None else "✓ (no bench)"
                cells.append(green(f"{text:<15}"))
            elif item.get("attempts"):
                cells.append(red(f"{'✗ ' + str(item['attempts']) + ' tries':<15}"))
            else:
                cells.append(dim(f"{'·':<15}"))
        hints = entry.get("hints_seen", 0)
        print(f"{problem.slug:<16} {problem.difficulty:<7} " + "".join(cells) + (str(hints) if hints else dim("-")))
    print()
    recent = []
    for slug, entry in data["exercises"].items():
        for lang, item in entry.get("langs", {}).items():
            if "last_run" in item:
                recent.append((item["last_run"], slug, lang, item.get("last_status", "")))
    if recent:
        recent.sort(reverse=True)
        print(bold("Recent"))
        for when, slug, lang, status in recent[:5]:
            colour = green if status == "pass" else red
            print(f"  {when.replace('T', ' ')}  {slug:<16} {lang:<8} {colour(status)}")
        print()
    print(dim(f"Stored in {progress.path().relative_to(ROOT)}. Times are the median of gpu-gym bench."))
    return 0


def _cmd_next(args: argparse.Namespace) -> int:
    ensure_solutions()
    data = progress.load()
    lang = None if args.lang == "any" else args.lang
    for problem in _practice(all_problems()):
        if not progress.is_solved(data, problem.slug, lang):
            _show(problem)
            return 0
    print(green("Every exercise is solved" + (f" in {lang}" if lang else "") + "."))
    print("Try another language with --lang, beat your best times with gpu-gym bench, or add one with gpu-gym new.")
    return 0


def _cmd_watch(args: argparse.Namespace) -> int:
    ensure_solutions()
    problem = get_problem(args.slug)
    langs = _languages(args.lang)
    files = [SOLUTIONS / problem.slug / LANG_FILES[lang] for lang in langs]
    files += [PROBLEMS / problem.slug / "problem.py"]

    def stamp() -> tuple:
        return tuple(path.stat().st_mtime_ns if path.exists() else 0 for path in files)

    def run() -> None:
        print("\033[2J\033[H", end="")
        print(dim(f"watching {', '.join(str(path.relative_to(ROOT)) for path in files[:-1])}   Ctrl+C to stop"))
        print()
        for lang in langs:
            if args.bench:
                run_bench(problem, lang, runs=args.runs, warmup=2, compare=False)
            else:
                run_functional(problem, lang)
        print()
        print(dim(time.strftime("last run %H:%M:%S. Save the file to run again.")), flush=True)

    last = stamp()
    run()
    try:
        while True:
            time.sleep(0.4)
            current = stamp()
            if current != last:
                time.sleep(0.15)  # let the editor finish writing
                last = stamp()
                try:
                    run()
                except SystemExit as exc:
                    print(exc)
    except KeyboardInterrupt:
        print()
    return 0


def _file_arg(args: argparse.Namespace):
    """--file runs one exercise in one language from another file (the board does this for its users)."""
    from pathlib import Path

    if not args.file:
        return None
    if len(args.slugs) != 1 or args.all or args.lang == "all":
        raise SystemExit("--file needs exactly one exercise and one --lang")
    return Path(args.file).resolve()


def _cmd_test(args: argparse.Namespace) -> int:
    ensure_solutions()
    problems = _selected(args.slugs, args.all)
    path = _file_arg(args)
    ok = True
    for problem in problems:
        for lang in _languages(args.lang):
            ok = run_functional(problem, lang, debug=args.debug, path=path) and ok
    return 0 if ok else 1


def _cmd_run(args: argparse.Namespace) -> int:
    from pathlib import Path

    ensure_solutions()
    problem = get_problem(args.slug)
    path = Path(args.file).resolve() if args.file else None
    cases = Path(args.cases).resolve() if args.cases else None
    return 0 if run_debug(problem, args.lang, path=path, debug=args.debug, cases_file=cases, only_custom=args.only_custom) else 1


def _cmd_bench(args: argparse.Namespace) -> int:
    ensure_solutions()
    problems = _selected(args.slugs, args.all)
    path = _file_arg(args)
    ok = True
    for problem in problems:
        for lang in _languages(args.lang):
            ok = (
                run_bench(
                    problem,
                    lang,
                    runs=args.runs,
                    warmup=args.warmup,
                    debug=args.debug,
                    compare=not args.no_compare,
                    path=path,
                )
                and ok
            )
            if ok and args.profile:
                from gpu_gym import profile

                profile.run(problem.slug, lang, path)  # informational: never fails the bench
    return 0 if ok else 1


def _cmd_profile_target(args: argparse.Namespace) -> int:
    """Run inside ncu by `bench --profile`: one solution call on the perf case, in the NVTX range ncu profiles."""
    import torch

    from gpu_gym.profile import NVTX_RANGE
    from gpu_gym.runner import _call, _split, load_solution

    problem = get_problem(args.slug)
    from pathlib import Path

    solve = load_solution(problem, args.lang, debug=False, path=Path(args.file).resolve() if args.file else None)
    user, _reference = _split(problem, problem.performance_case(0))
    for _ in range(2):  # compile, autotune and warm caches outside the profiled range
        _call(solve, problem, user)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(NVTX_RANGE)
    _call(solve, problem, user)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    return 0


def _cmd_reset(args: argparse.Namespace) -> int:
    lang = None if args.lang == "all" else args.lang
    return reset_solution(args.slug, lang, force=args.force)


def _cmd_new(args: argparse.Namespace) -> int:
    title = args.title or args.slug.replace("_", " ")
    return new_problem(args.slug, title)


def _cmd_compile_db(args: argparse.Namespace) -> int:
    write_compile_db(args.arch or "sm_90")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpu-gym",
        description="Run GPU kernel exercises locally, with no submission limit.",
    )
    parser.add_argument("--version", action="version", version=f"gpu-gym {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("list", help="Show the exercises").set_defaults(func=_cmd_list)

    show = commands.add_parser("show", help="Print one problem statement")
    show.add_argument("slug")
    show.set_defaults(func=_cmd_show)

    test = commands.add_parser("test", help="Check a solution against the reference")
    test.add_argument("slugs", nargs="*")
    test.add_argument("--all", action="store_true")
    test.add_argument("--lang", default="cuda", choices=[*LANGS, "all"])
    test.add_argument("--debug", action="store_true", help="Compile CUDA with device debug info")
    test.add_argument("--file", help="Test this file instead of solutions/<exercise>/...")
    test.set_defaults(func=_cmd_test)

    run = commands.add_parser("run", help="Debug run: every case with its output, nothing recorded")
    run.add_argument("slug")
    run.add_argument("--lang", default="cuda", choices=list(LANGS))
    run.add_argument("--file", help="Run this file instead of solutions/<exercise>/...")
    run.add_argument("--debug", action="store_true", help="Compile CUDA with device debug info")
    run.add_argument("--cases", help="JSON file with extra cases (format: gymkit/cases.py)")
    run.add_argument("--only-custom", action="store_true", help="Skip the built-in cases")
    run.set_defaults(func=_cmd_run)

    bench = commands.add_parser("bench", help="Correctness, then the median of timed runs")
    bench.add_argument("slugs", nargs="*")
    bench.add_argument("--all", action="store_true")
    bench.add_argument("--lang", default="cuda", choices=[*LANGS, "all"])
    bench.add_argument("--runs", type=int, default=15)
    bench.add_argument("--warmup", type=int, default=2)
    bench.add_argument("--debug", action="store_true")
    bench.add_argument("--no-compare", action="store_true", help="Skip timing the PyTorch reference")
    bench.add_argument("--file", help="Benchmark this file instead of solutions/<exercise>/...")
    bench.add_argument("--profile", action="store_true", help="Then profile one call with Nsight Compute (slow)")
    bench.set_defaults(func=_cmd_bench)
    target = commands.add_parser("profile-target")  # internal: what `bench --profile` runs under ncu
    target.add_argument("slug")
    target.add_argument("--lang", default="cuda", choices=LANGS)
    target.add_argument("--file")
    target.set_defaults(func=_cmd_profile_target)

    watch = commands.add_parser("watch", help="Re-run the tests every time you save the solution")
    watch.add_argument("slug")
    watch.add_argument("--lang", default="cuda", choices=[*LANGS, "all"])
    watch.add_argument("--bench", action="store_true", help="Benchmark after each passing save")
    watch.add_argument("--runs", type=int, default=10)
    watch.set_defaults(func=_cmd_watch)

    hint = commands.add_parser("hint", help="Reveal the next hint for an exercise")
    hint.add_argument("slug")
    hint.add_argument("number", nargs="?", type=int, help="Show this hint number")
    hint.add_argument("--all", action="store_true", help="Show every hint")
    hint.set_defaults(func=_cmd_hint)

    stats = commands.add_parser("progress", aliases=["stats"], help="What you have solved and your best times")
    stats.add_argument("--reset", metavar="EXERCISE|all", help="Erase recorded progress")
    stats.add_argument("--force", action="store_true")
    stats.set_defaults(func=_cmd_progress)

    upcoming = commands.add_parser("next", help="Show the first exercise you have not solved")
    upcoming.add_argument("--lang", default="any", choices=[*LANGS, "any"])
    upcoming.set_defaults(func=_cmd_next)

    reset = commands.add_parser("reset", help="Copy the starter back over a solution")
    reset.add_argument("slug")
    reset.add_argument("--lang", default="all", choices=[*LANGS, "all"])
    reset.add_argument("--force", action="store_true")
    reset.set_defaults(func=_cmd_reset)

    create = commands.add_parser("new", help="Scaffold an exercise you can edit and rerun")
    create.add_argument("slug")
    create.add_argument("--title")
    create.set_defaults(func=_cmd_new)

    database = commands.add_parser("compile-db", help="Refresh compile_commands.json for CUDA autocomplete")
    database.add_argument("--arch", help="Arch clangd uses to parse CUDA. Default sm_90")
    database.set_defaults(func=_cmd_compile_db)
    return parser


def main(argv: list[str] | None = None) -> int:
    if not PROBLEMS.is_dir():
        print(f"Cannot find exercises. Expected {PROBLEMS}")
        return 1
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
