"""Command line shared by the LLM and algo gyms: llm-gym and algo-gym."""

import argparse
import os
import shutil
import sys

# Same as gpu_gym: keep Triton's current FP-fusion behavior and skip its version warning.
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

from gymkit.gym import Gym, load_gym
from gymkit.progress import Progress
from gymkit.runner import run_bench, run_debug, run_functional


def _cmd_list(gym: Gym, _args) -> int:
    data = Progress(gym.progress_file).load()
    langs = [lang.id for lang in gym.langs]
    print(f"{'#':>2}  {'exercise':<22} {'level':<7} {'track':<20} title")
    for number, problem in enumerate(gym.problems(), start=1):
        entry = data["exercises"].get(problem.slug, {}).get("langs", {})
        marks = ""
        for lang in langs:
            item = entry.get(lang, {})
            marks += "✓" if "first_solved" in item else "✗" if item.get("attempts") else "·"
        print(f"{number:>2}  {problem.slug:<22} {problem.difficulty:<7} {problem.track[:20]:<20} {marks}  {problem.title}")
    return 0


def _cmd_show(gym: Gym, args) -> int:
    problem = gym.load(args.slug)
    print(gym.statement(problem.slug).rstrip())
    print()
    for lang in gym.langs:
        print(f"Your file: {gym.solution_path(problem.slug, lang.id).relative_to(gym.root.parent.parent)}")
    return 0


def _langs(gym: Gym, value: str, slug: str | None = None) -> list[str]:
    if value != "all":
        return [value]
    return gym.langs_for(slug) if slug else [lang.id for lang in gym.langs]


def _file_arg(args):
    """--file runs one exercise in one language from another file (the board does this for its users)."""
    from pathlib import Path

    if not getattr(args, "file", None):
        return None
    if len(args.slugs) != 1 or getattr(args, "all", False) or args.lang == "all":
        raise SystemExit("--file needs exactly one exercise and one --lang")
    return Path(args.file).resolve()


def _cmd_test(gym: Gym, args) -> int:
    ok = True
    problems = gym.problems() if args.all else [gym.load(slug) for slug in args.slugs]
    if not problems:
        raise SystemExit("Name an exercise, or pass --all.")
    path = _file_arg(args)
    for problem in problems:
        for lang in _langs(gym, args.lang, problem.slug):
            ok = run_functional(gym, problem, lang, path=path) and ok
    return 0 if ok else 1


def _cmd_run(gym: Gym, args) -> int:
    from pathlib import Path

    problem = gym.load(args.slug)
    lang = args.lang or gym.langs_for(problem.slug)[0]
    path = Path(args.file).resolve() if args.file else None
    cases = Path(args.cases).resolve() if args.cases else None
    return 0 if run_debug(gym, problem, lang, path, cases_file=cases, only_custom=args.only_custom) else 1


def _cmd_bench(gym: Gym, args) -> int:
    ok = True
    for slug in args.slugs:
        for lang in _langs(gym, args.lang, slug):
            ok = run_bench(gym, gym.load(slug), lang, runs=args.runs, compare=not args.no_compare, path=_file_arg(args)) and ok
    return 0 if ok else 1


def _cmd_hint(gym: Gym, args) -> int:
    hints = gym.hints(args.slug)
    if not hints:
        print(f"{args.slug} has no hints.")
        return 1
    progress = Progress(gym.progress_file)
    seen = progress.load()["exercises"].get(args.slug, {}).get("hints_seen", 0)
    wanted = len(hints) if args.all else min(seen + 1, len(hints))
    for number in range(1, wanted + 1):
        print(f"## Hint {number} of {len(hints)}\n\n{hints[number - 1]}\n")
    progress.set_hints_seen(args.slug, wanted)
    return 0


def _cmd_next(gym: Gym, _args) -> int:
    progress = Progress(gym.progress_file)
    data = progress.load()
    for problem in gym.problems():
        if not progress.is_solved(data, problem.slug):
            return _cmd_show(gym, argparse.Namespace(slug=problem.slug))
    print("Every exercise is solved.")
    return 0


def _cmd_progress(gym: Gym, _args) -> int:
    progress = Progress(gym.progress_file)
    data = progress.load()
    problems = gym.problems()
    solved = sum(progress.is_solved(data, problem.slug) for problem in problems)
    print(f"{gym.title}: {solved}/{len(problems)} solved")
    for problem in problems:
        for lang in gym.langs:
            item = data["exercises"].get(problem.slug, {}).get("langs", {}).get(lang.id, {})
            if not item:
                continue
            state = "solved" if "first_solved" in item else f"{item.get('attempts', 0)} tries"
            best = f"  best {item['best_ms']:.3f} ms" if "best_ms" in item else ""
            print(f"  {problem.slug:<22} {lang.id:<8} {state}{best}")
    return 0


def _cmd_reset(gym: Gym, args) -> int:
    for lang in _langs(gym, args.lang, args.slug):
        source = gym.starter_path(args.slug, lang)
        target = gym.solution_path(args.slug, lang)
        if target.exists() and target.read_text() != source.read_text() and not args.force:
            print(f"{target.name} differs from the starter. Re-run with --force.")
            return 1
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        print(f"Restored {target}")
    return 0


def main(gym_id: str | None = None, argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if gym_id is None:
        if not argv:
            raise SystemExit("usage: gymkit <gym> <command> ...")
        gym_id = argv.pop(0)
    gym = load_gym(gym_id)
    gym.ensure_solutions()
    default_lang = gym.langs[0].id
    lang_choices = [lang.id for lang in gym.langs] + ["all"]

    parser = argparse.ArgumentParser(prog=f"{gym.id}-gym", description=gym.description)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list").set_defaults(func=_cmd_list)
    show = commands.add_parser("show")
    show.add_argument("slug")
    show.set_defaults(func=_cmd_show)
    test = commands.add_parser("test")
    test.add_argument("slugs", nargs="*")
    test.add_argument("--all", action="store_true")
    test.add_argument("--lang", default=default_lang, choices=lang_choices)
    test.add_argument("--file", help="Test this file instead of the saved solution")
    test.set_defaults(func=_cmd_test)
    run = commands.add_parser("run", help="Debug run: every case with its output, nothing recorded")
    run.add_argument("slug")
    run.add_argument("--lang", choices=[lang.id for lang in gym.langs])
    run.add_argument("--file", help="Run this file instead of the saved solution")
    run.add_argument("--cases", help="JSON file with extra cases (format: gymkit/cases.py)")
    run.add_argument("--only-custom", action="store_true", help="Skip the built-in cases")
    run.set_defaults(func=_cmd_run)
    bench = commands.add_parser("bench")
    bench.add_argument("slugs", nargs="+")
    bench.add_argument("--lang", default=default_lang, choices=lang_choices)
    bench.add_argument("--runs", type=int, default=7)
    bench.add_argument("--no-compare", action="store_true")
    bench.add_argument("--file", help="Benchmark this file instead of the saved solution")
    bench.set_defaults(func=_cmd_bench)
    hint = commands.add_parser("hint")
    hint.add_argument("slug")
    hint.add_argument("--all", action="store_true")
    hint.set_defaults(func=_cmd_hint)
    commands.add_parser("next").set_defaults(func=_cmd_next)
    commands.add_parser("progress").set_defaults(func=_cmd_progress)
    reset = commands.add_parser("reset")
    reset.add_argument("slug")
    reset.add_argument("--lang", default="all", choices=lang_choices)
    reset.add_argument("--force", action="store_true")
    reset.set_defaults(func=_cmd_reset)

    args = parser.parse_args(argv)
    return args.func(gym, args)


def llm_main() -> int:
    return main("llm")


def algo_main() -> int:
    return main("algo")
