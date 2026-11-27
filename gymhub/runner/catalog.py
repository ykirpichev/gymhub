"""Export one gym's catalog as JSON: `python -m gymhub.runner.catalog <gym>`.

The public catalog comes from the gym package (gymhub/packages.py): manifest, problem.json,
PROBLEM.md, HINTS.md and starters. Each runtime then adds what only its grader can compute:
the GPU gym's generated "Tests" section, the custom-case template, whether there is a
benchmark, and the examples a browser can run (gymkit/examples.py: inputs and expected outputs).
Grader code is loaded here, never shipped: only these derived values leave.

Runs in a subprocess so the long-lived runner never imports torch or touches the GPU itself.

A machine without a grader's dependencies (PyTorch on a CPU-only laptop, say) still publishes the
gym: the exercises without the derived values, and a `machine_note` on the gym that says what to
install so this machine can grade it. The board shows the note on the gym's page.
"""

import contextlib
import io
import json
import sys
from pathlib import Path

from gymhub import packages


def _template(build) -> dict | None:
    """Editable JSON for the example case; None if it cannot be built (the board then shows no template)."""
    try:
        value = build()
        return value if len(json.dumps(value)) <= 20_000 else None
    except (Exception, SystemExit):  # gpu_gym exits when PyTorch sees no CUDA device
        return None


# Modules a grader may import that are not installed with this package, and how to get them.
INSTALL_HINTS = {
    "torch": "PyTorch (pip install torch, see https://pytorch.org/get-started/locally/)",
    "triton": "Triton (pip install triton, Linux with an NVIDIA GPU)",
    "numpy": "NumPy (pip install numpy)",
}


def _missing_module(exc: BaseException) -> str | None:
    """The top-level module an ImportError is about, if it is one."""
    if isinstance(exc, ModuleNotFoundError) and exc.name:
        return exc.name.split(".")[0]
    return None


def machine_note(missing: set[str], needs_gpu: bool, has_gpu: bool, *, torch_sees_gpu: bool = True,
                 needs_nvcc: bool = False) -> str:
    """What this machine lacks to grade the gym, in words for the gym's page; "" when nothing.

    torch_sees_gpu: False when nvidia-smi lists a GPU but PyTorch cannot use it (a CPU build, or a CUDA
    build newer than the driver), so every run would fail. needs_nvcc: the gym has CUDA exercises and
    this machine has a GPU but no CUDA toolkit, so only those wait (PyTorch and Triton still run)."""
    parts = []
    if missing:
        names = [INSTALL_HINTS.get(name, f"the Python module {name!r}") for name in sorted(missing)]
        parts.append(f"The machine that publishes this gym cannot grade it yet: it needs {' and '.join(names)}. "
                     "Install it there and restart the runner.")
    if needs_gpu and not has_gpu:
        parts.append("Its exercises need an NVIDIA GPU, and that machine has none: you can read them and write "
                     "solutions, and runs wait until a machine with a GPU connects.")
    if needs_gpu and has_gpu and not missing and not torch_sees_gpu:
        parts.append("That machine has an NVIDIA GPU, but its PyTorch cannot use it (a CPU-only build, or a CUDA "
                     "build newer than the driver), so runs there fail. Run ./setup.sh again on it (or install the "
                     "PyTorch build for its driver from https://pytorch.org/get-started/locally/) and restart the runner.")
    if needs_gpu and has_gpu and needs_nvcc:
        parts.append("CUDA solutions need the CUDA toolkit (nvcc), which that machine lacks: PyTorch and Triton "
                     "solutions run, CUDA ones wait in the queue. Install the toolkit "
                     "(https://developer.nvidia.com/cuda-downloads) and restart the runner.")
    return " ".join(parts)


def _torch_sees_gpu() -> bool:
    """Can PyTorch use a CUDA device here? True when PyTorch is not installed (that note is the missing module)."""
    try:
        import torch
    except ImportError:
        return True
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _enrich_gpu(problems: list[dict]) -> set[str]:
    from gpu_gym.scaffold import ensure_solutions

    ensure_solutions()  # needs no torch: the board shows each exercise's code from these files
    for item in problems:
        item["benchable"] = True
    try:
        from gpu_gym import cli
        from gpu_gym import runner as gpu_runner
        from gpu_gym.catalog import get_problem
    except ImportError as exc:  # no PyTorch here: the statements without their generated Tests section
        missing = _missing_module(exc)
        if missing is None:
            raise
        return {missing}
    for item in problems:
        problem = get_problem(item["slug"])
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                tests = cli._tests_section(problem)
            item["statement"] += "\n\n" + tests
        except SystemExit:  # PyTorch without a CUDA device (gpu_gym.runner.require_cuda): the gym's note says so
            pass
        except Exception as exc:  # the statement is still useful without the tests section
            item["statement"] += f"\n\n(Tests section unavailable: {exc})"
        item["case_template"] = _template(lambda: gpu_runner.case_template(problem))
    return set()


def _examples(problem, package: packages.Package) -> str | None:
    """Example cases with expected outputs for the browser; None if there are none (see gymkit/examples.py)."""
    from gymkit import examples

    reference = package.references(problem.slug).get("python")
    try:
        return examples.build(problem, reference.read_text() if reference else None)
    except Exception:  # noqa: BLE001 - the exercise is still published, without examples
        return None


def _enrich_gymkit(gym_id: str | Path, problems: list[dict], package: packages.Package) -> set[str]:
    from gymkit.gym import load_gym
    from gymkit.runner import case_template

    gym = load_gym(gym_id)
    gym.ensure_solutions()
    missing = set()
    for item in problems:
        if item.get("kind", "code") != "code":
            continue
        try:
            problem = gym.load(item["slug"])
        except ImportError as exc:  # its grader needs a module this machine lacks: listed, not graded here
            name = _missing_module(exc)
            if name is None:
                raise
            missing.add(name)
            continue
        item["case_template"] = _template(lambda: case_template(problem))
        item["benchable"] = problem.performance_case is not None
        if "python" in item.get("langs", []):
            item["examples"] = _examples(problem, package)
    return missing


def build(gym_id: str | Path) -> dict:
    """The catalog of a gym under gyms/ (by id) or of a package directory (by path)."""
    from gymhub.runner.capabilities import probe

    package = packages.load(gym_id)
    catalog = packages.public_catalog(package)
    if package.runtime == "gpu_gym":
        missing = _enrich_gpu(catalog["problems"])
    else:
        missing = _enrich_gymkit(gym_id, catalog["problems"], package)
    needs_gpu = bool((package.manifest.get("requires") or {}).get("gpu"))
    machine = probe()
    has_gpu = bool(needs_gpu and machine["gpus"])
    langs = {lang.get("id") for lang in package.manifest.get("langs") or [] if isinstance(lang, dict)}
    torch_ok = _torch_sees_gpu() if has_gpu and not missing else True
    # Always set, so a note from before the module was installed is cleared on the next publish.
    catalog["gym"]["machine_note"] = machine_note(
        missing, needs_gpu, has_gpu, torch_sees_gpu=torch_ok, needs_nvcc="cuda" in langs and not machine.get("cuda"),
    )
    # Can the publishing machine itself grade it? The board's "Start here" skips gyms it cannot (#3 of the
    # launch review); a GPU it lacks shows in its capabilities, the modules and a broken PyTorch only here.
    catalog["gym"]["machine_ready"] = not missing and torch_ok
    return catalog


def main() -> int:
    gym_id = sys.argv[1]
    with contextlib.redirect_stdout(sys.stderr):
        data = build(gym_id)
    json.dump(data, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
