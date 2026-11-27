import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# GPU_GYM_ROOT points at another gym folder with the same layout: an uploaded GPU gym (gymhub/runner).
GYM = Path(os.environ["GPU_GYM_ROOT"]) if os.environ.get("GPU_GYM_ROOT") else ROOT / "gyms" / "gpu"
PROBLEMS = GYM / "problems"
SOLUTIONS = GYM / "solutions"
# GPU_GYM_CACHE moves compiled kernels and device info elsewhere (the board's sandbox uses a per-user cache).
CACHE = Path(os.environ["GPU_GYM_CACHE"]) if os.environ.get("GPU_GYM_CACHE") else ROOT / ".cache"
COMPILE_DB = ROOT / "compile_commands.json"
NVCC = "/usr/local/cuda/bin/nvcc"
CUDART = "/usr/local/cuda/lib64/libcudart.so"
CUDA_INCLUDE = "/usr/local/cuda/include"
CUDA_PATH = "/usr/local/cuda"
