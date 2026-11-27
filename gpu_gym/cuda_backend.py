"""Compile a solution.cu and call its extern \"C\" solve."""

import ctypes
import hashlib
import shutil
import subprocess
import textwrap
from pathlib import Path

import torch

from gpu_gym.paths import CACHE, CUDA_PATH, CUDART, NVCC
from gpu_gym.spec import DTYPES, Param

_CUDART: ctypes.CDLL | None = None


def gpu_arch() -> str:
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"


def _cudart() -> ctypes.CDLL:
    global _CUDART
    if _CUDART is None:
        _CUDART = ctypes.CDLL(CUDART)
    return _CUDART


def cuda_error_string(code: int) -> str:
    library = _cudart()
    library.cudaGetErrorString.restype = ctypes.c_char_p
    library.cudaGetErrorString.argtypes = [ctypes.c_int]
    message = library.cudaGetErrorString(ctypes.c_int(code))
    if not message:
        return f"cuda error {code}"
    return message.decode()


def l2_cache_bytes() -> int:
    library = _cudart()
    library.cudaDeviceGetAttribute.restype = ctypes.c_int
    library.cudaDeviceGetAttribute.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
    ]
    value = ctypes.c_int()
    # cudaDevAttrL2CacheSize
    status = library.cudaDeviceGetAttribute(ctypes.byref(value), 38, 0)
    if status != 0 or value.value <= 0:
        return 8 << 20
    return int(value.value)


def _ctype(param: Param):
    if param.kind == "tensor":
        return ctypes.c_void_p
    if param.kind == "int":
        return ctypes.c_int
    return ctypes.c_float


def _cxx_type(param: Param) -> str:
    if param.kind == "tensor":
        element = DTYPES[param.dtype]
        return f"{element}*" if param.output else f"const {element}*"
    if param.kind == "int":
        return "int"
    return "float"


def _wrapper_source(params: list[Param]) -> str:
    signature = ", ".join(f"{_cxx_type(param)} {param.name}" for param in params)
    names = ", ".join(param.name for param in params)
    return textwrap.dedent(
        f"""\
        #include <cstdint>
        #include <cuda_runtime.h>
        #include <cuda_fp16.h>
        #include <cuda_bf16.h>

        extern "C" void solve({signature});

        extern "C" int gpu_gym_launch({signature}) {{
            solve({names});
            cudaError_t status = cudaDeviceSynchronize();
            if (status != cudaSuccess) {{
                return (int)status;
            }}
            return (int)cudaGetLastError();
        }}
        """
    )


def _argument(param: Param, value):
    if param.kind == "tensor":
        return ctypes.c_void_p(value.data_ptr())
    if param.kind == "int":
        return int(value)
    return float(value)


class CudaSolution:
    def __init__(self, library: ctypes.CDLL, params: list[Param]):
        launch = library.gpu_gym_launch
        launch.restype = ctypes.c_int
        launch.argtypes = [_ctype(param) for param in params]
        self._launch = launch
        self._params = params

    def __call__(self, case: dict) -> None:
        status = self._launch(*(_argument(param, case[param.name]) for param in self._params))
        if status != 0:
            raise RuntimeError(cuda_error_string(status))


def load_cuda(source: Path, params: list[Param], debug: bool) -> CudaSolution:
    code = source.read_text()
    wrapper = _wrapper_source(params)
    arch = gpu_arch()
    nvcc = shutil.which("nvcc") or NVCC
    version = subprocess.run([nvcc, "--version"], check=False, capture_output=True, text=True)
    digest = hashlib.sha256()
    digest.update(code.encode())
    digest.update(wrapper.encode())
    digest.update(arch.encode())
    digest.update(b"debug" if debug else b"release")
    digest.update(version.stdout.encode())
    key = digest.hexdigest()[:16]
    cache_dir = CACHE / "cuda"
    cache_dir.mkdir(parents=True, exist_ok=True)
    so_path = cache_dir / f"{source.parent.name}-{key}.so"
    if not so_path.exists():
        wrapper_path = cache_dir / f"{key}.cu"
        wrapper_path.write_text(wrapper)
        libdir = str(Path(CUDA_PATH) / "lib64")
        cmd = [
            nvcc,
            "-O0" if debug else "-O3",
            "-std=c++17",
            f"-arch={arch}",
            "-shared",
            "-Xcompiler",
            "-fPIC",
            "-cudart",
            "shared",
            "-Xlinker",
            "-rpath",
            "-Xlinker",
            libdir,
            str(source),
            str(wrapper_path),
            "-o",
            str(so_path),
        ]
        if debug:
            cmd[1:1] = ["-G", "-g"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            so_path.unlink(missing_ok=True)
            log = (result.stdout + "\n" + result.stderr).strip()
            raise RuntimeError(
                "nvcc failed.\n"
                + " ".join(cmd)
                + "\n"
                + log
                + '\n\nsolve must be extern "C" and match the starter signature.'
            )
    return CudaSolution(ctypes.CDLL(str(so_path)), params)
