import importlib.util
from pathlib import Path

from gpu_gym.paths import PROBLEMS
from gpu_gym.spec import Problem

LANGS = ("cuda", "pytorch", "triton")
LANG_FILES = {
    "cuda": "solution.cu",
    "pytorch": "solution.py",
    "triton": "triton_solution.py",
}
STARTER_MARK = b"GPU_GYM_STARTER"
# Already solved examples. Listed, but not counted in progress totals or offered by `next`.
WARMUP = {"playground"}


def iter_problem_dirs() -> list[Path]:
    if not PROBLEMS.is_dir():
        return []
    dirs = []
    for path in PROBLEMS.iterdir():
        if path.name.startswith("_") or not path.is_dir():
            continue
        if (path / "problem.py").is_file():
            dirs.append(path)
    return sorted(dirs, key=lambda path: path.name)


def load_problem(directory: Path) -> Problem:
    module_path = directory / "problem.py"
    spec = importlib.util.spec_from_file_location(f"gpu_gym_problem_{directory.name}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    problem = module.build()
    if not isinstance(problem, Problem):
        raise TypeError(f"{module_path} build() must return a Problem")
    if problem.slug != directory.name:
        raise ValueError(
            f"{module_path} slug is {problem.slug!r}, but the directory is {directory.name!r}"
        )
    problem.directory = directory
    return problem


# Suggested order. Exercises created with `gpu-gym new` are listed after these.
LEARNING_ORDER = (
    # Warm-up: one thread per element, indexing, launch config.
    "playground",
    "vector_add",
    "relu",
    "saxpy",
    "gelu",
    "reverse",
    # Memory patterns: coalescing, shared-memory tiles.
    "transpose",
    "conv1d",
    "conv2d",
    # Cooperation inside a block: reductions and scans.
    "reduction",
    "histogram",
    "matvec",
    "prefix_sum",
    # Row-wise kernels used in real models.
    "row_softmax",
    "layer_norm",
    # LLM building blocks: memory-bound kernels around a transformer layer.
    "embedding",
    "swiglu",
    "rms_norm",
    "rope",
    "cross_entropy",
    # Compute-bound: tiling for reuse.
    "matmul",
    "attention",
    "causal_attention",
    "moe_matmul",
    # Data-dependent work: irregular rows, selection, sorting.
    "spmv_csr",
    "top_k",
    "sort",
)


def all_problems() -> list[Problem]:
    problems = [load_problem(directory) for directory in iter_problem_dirs()]
    rank = {slug: index for index, slug in enumerate(LEARNING_ORDER)}

    def sort_key(problem: Problem) -> tuple:
        return (rank.get(problem.slug, len(LEARNING_ORDER)), problem.slug)

    return sorted(problems, key=sort_key)


def get_problem(slug: str) -> Problem:
    directory = PROBLEMS / slug
    if not (directory / "problem.py").is_file():
        known = ", ".join(problem.slug for problem in all_problems()) or "(none)"
        raise SystemExit(f"Unknown exercise {slug!r}. Known exercises: {known}")
    return load_problem(directory)
