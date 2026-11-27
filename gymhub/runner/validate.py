"""Validate an unpacked gym bundle: `python -m gymhub.runner.validate <package dir> --out result.json`.

Runs on the compute pool when the bundle is someone else's (gymhub/runner/pooled.py `validate`),
because it imports the author's graders. For every exercise:

    grader loads        problem.py builds its Problem, and its slug matches the folder
    starter fails       each starter file fails its tests (otherwise the exercise is already solved)
    reference passes    each reference solution in solution/ passes its tests

An exercise without reference solutions is accepted with a warning: nothing proves it can be solved.
A gpu_gym package (CUDA, PyTorch, Triton) is checked the same way with gpu_gym's own grader, on a
machine with a GPU: every declared language's starter must fail and its reference must pass.
The result file holds the checks and what only the grader can compute (case templates, whether there
is a benchmark), which the board adds to the public catalog when it publishes the version.
"""

import argparse
import concurrent.futures
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from gymhub import packages

CHECK_TIMEOUT = int(os.environ.get("GYMHUB_VALIDATE_CHECK_TIMEOUT", "300"))


def _parallel_checks(runtime: str) -> int:
    """How many test runs go at once: a few for CPU graders, one at a time for GPU ones (one GPU)."""
    if os.environ.get("GYMHUB_VALIDATE_JOBS"):
        return max(1, int(os.environ["GYMHUB_VALIDATE_JOBS"]))
    if runtime == "gpu_gym":
        return 1
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:  # not on Linux
        cpus = os.cpu_count() or 1
    return max(1, min(4, cpus))


def _tail(text: str, lines: int = 6) -> str:
    kept = [line for line in text.strip().splitlines() if line.strip()][-lines:]
    return " / ".join(kept)[-500:]


def _test(root: Path, slug: str, lang: str, file: Path, scratch: Path, runtime: str = "gymkit") -> tuple[bool, str]:
    """Grade `file`; `scratch` is this run's own folder (runs go in parallel), for its progress file."""
    progress = scratch / "progress.json"
    env = {
        **os.environ,
        "GYMKIT_PROGRESS_FILE": str(progress),
        "GPU_GYM_PROGRESS_FILE": str(progress),
        "PYTHONDONTWRITEBYTECODE": "1",
        "NO_COLOR": "1",
        "TERM": "dumb",
    }
    if runtime == "gpu_gym":
        env["GPU_GYM_ROOT"] = str(root)
        command = [sys.executable, "-m", "gpu_gym", "test", slug, "--lang", lang, "--file", str(file)]
    else:
        command = [sys.executable, "-m", "gymkit", str(root), "test", slug, "--lang", lang, "--file", str(file)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=CHECK_TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {CHECK_TIMEOUT} s"
    return result.returncode == 0, _tail(result.stdout + result.stderr)


def _load_gpu_problem(directory: Path):
    from gpu_gym.catalog import load_problem

    return load_problem(directory)


def _gpu_case_template(problem):
    from gpu_gym import runner as gpu_runner

    return gpu_runner.case_template(problem)


def validate(root: Path) -> dict:
    from gymkit.gym import load_gym
    from gymkit import examples
    from gymkit.runner import case_template

    package = packages.load(root)
    checks: list[dict] = []
    warnings: list[str] = []
    enrich: dict[str, dict] = {}
    loaded: dict = {}  # slug -> gymkit Problem with a Python starter: it may get examples for the browser

    def check(slug: str, lang: str, what: str, ok: bool, detail: str = "") -> None:
        checks.append({"slug": slug, "lang": lang, "what": what, "ok": ok, "detail": detail})
        print(f"{'ok  ' if ok else 'FAIL'} {slug} [{lang or '-'}] {what}" + (f": {detail}" if detail and not ok else ""), flush=True)

    for error in packages.validate_upload(package):
        check("", "", "package", False, error)
    if checks:
        return {"ok": False, "checks": checks, "warnings": warnings, "enrich": enrich}

    runtime = package.runtime
    gym = load_gym(package.root) if runtime == "gymkit" else None
    # Each test run is a subprocess of its own: they run a few at a time (_parallel_checks), and the
    # checks are reported in the same order as one after another.
    pending: list = []  # (slug, lang, what, (ok, detail) or a future of _test's (passed, detail))

    def now_(slug: str, lang: str, what: str, ok: bool, detail: str = "") -> None:
        pending.append((slug, lang, what, (ok, detail)))

    def later(slug: str, lang: str, what: str, file: Path) -> None:
        run_dir = scratch / f"{what.split()[0]}-{slug}-{lang}"
        run_dir.mkdir(parents=True, exist_ok=True)
        pending.append((slug, lang, what, runner.submit(_test, package.root, slug, lang, file, run_dir, runtime)))

    with tempfile.TemporaryDirectory(prefix="gym-validate-") as tmp, \
            concurrent.futures.ThreadPoolExecutor(_parallel_checks(runtime)) as runner:
        scratch = Path(tmp)
        for slug in package.slugs():
            kind = package.meta(slug).get("kind", "code")
            if kind != "code":  # graded on the board from answer.json, already checked by validate_upload
                now_(slug, "", f"{kind} answer key", True)
                continue
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    problem = gym.load(slug) if gym is not None else _load_gpu_problem(package.problem_dir(slug))
            except BaseException as exc:  # noqa: BLE001 - a grader may raise anything, even SystemExit
                now_(slug, "", "grader loads", False, f"{type(exc).__name__}: {exc}"[:500])
                continue
            now_(slug, "", "grader loads", True)
            item: dict = {"benchable": problem.performance_case is not None}
            try:
                template = case_template(problem) if gym is not None else _gpu_case_template(problem)
                if len(json.dumps(template)) <= 20_000:
                    item["case_template"] = template
            except Exception:
                pass
            enrich[slug] = item
            if gym is not None and "python" in package.starters(slug):
                loaded[slug] = problem
            for lang in sorted(package.starters(slug)):
                name = next(l["file"] for l in package.langs if l["id"] == lang)
                # Grade the starter's code itself: gymkit skips a file that still is the marked starter
                # ("not started"), which would make this check pass without running a single test.
                text = (package.problem_dir(slug) / "starter" / name).read_text()
                file = scratch / f"starter-{slug}-{lang}" / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text("".join(line for line in text.splitlines(keepends=True) if "GYM_STARTER" not in line))
                later(slug, lang, "starter fails", file)
            references = package.references(slug)
            if not references:
                warnings.append(f"{slug}: no reference solution in solution/, so nothing shows it can be solved")
            for lang, file in sorted(references.items()):
                later(slug, lang, "reference passes", file)
        for slug, lang, what, result in pending:  # in order, each as soon as it is done
            if not isinstance(result, concurrent.futures.Future):
                check(slug, lang, what, *result)
            elif what == "starter fails":
                passed, _ = result.result()
                check(slug, lang, what, not passed, "the starter already passes every test" if passed else "")
            else:
                passed, detail = result.result()
                check(slug, lang, what, passed, "" if passed else detail)
    # Examples for the browser (gymkit/examples.py), built once every check passed: the reference
    # solution they are run with here has then finished its own tests in time.
    ok = all(item["ok"] for item in checks)
    for slug, problem in loaded.items() if ok else ():
        reference = package.references(slug).get("python")
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                enrich[slug]["examples"] = examples.build(problem, reference.read_text() if reference else None)
        except Exception:  # noqa: BLE001 - published without examples
            pass
    return {"ok": ok, "checks": checks, "warnings": warnings, "enrich": enrich}


def main() -> int:
    parser = argparse.ArgumentParser(prog="gymhub.runner.validate")
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.root.resolve())
    for warning in result["warnings"]:
        print(f"note {warning}", flush=True)
    passed = sum(item["ok"] for item in result["checks"])
    print(f"\n{passed}/{len(result['checks'])} checks passed: " + ("the version can be published." if result["ok"] else "this version will NOT be published."))
    args.out.write_text(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
