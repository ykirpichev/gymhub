from dataclasses import dataclass
from pathlib import Path
from typing import Callable

Case = dict
CaseFn = Callable[[], list[Case]]
PerfFn = Callable[[int], Case]
BytesFn = Callable[[Case], int]
FlopsFn = Callable[[Case], int]
ReferenceFn = Callable[..., None]


# Element types a tensor argument may have, and the C type `solve` receives for each in CUDA.
DTYPES = {
    "float32": "float",
    "float16": "__half",
    "bfloat16": "__nv_bfloat16",
    "int8": "int8_t",
    "uint8": "uint8_t",
    "int32": "int",
}


@dataclass(frozen=True)
class Param:
    """One argument of `solve`, in order. A tensor is float32 unless `dtype` says otherwise."""

    name: str
    kind: str
    output: bool = False
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.kind not in {"tensor", "int", "float"}:
            raise ValueError(f"unknown param kind {self.kind!r} on {self.name}")
        if self.output and self.kind != "tensor":
            raise ValueError(f"{self.name} is an output but is not a tensor")
        if self.dtype not in DTYPES:
            raise ValueError(f"{self.name}: dtype must be one of {', '.join(DTYPES)}, not {self.dtype!r}")
        if self.dtype != "float32" and self.kind != "tensor":
            raise ValueError(f"{self.name}: only a tensor has a dtype")

    @property
    def torch_dtype(self):
        import torch

        return getattr(torch, self.dtype)


@dataclass
class Problem:
    slug: str
    title: str
    difficulty: str
    summary: str
    atol: float
    rtol: float
    params: list[Param]
    reference: ReferenceFn
    functional_cases: CaseFn
    performance_case: PerfFn
    traffic_bytes: BytesFn | None = None
    # Floating point operations for one call. Set it for compute-bound problems.
    flops: FlopsFn | None = None
    directory: Path | None = None

    def __post_init__(self) -> None:
        if self.difficulty not in {"easy", "medium", "hard"}:
            raise ValueError(f"{self.slug}: difficulty must be easy, medium, or hard")
        names = [param.name for param in self.params]
        if len(names) != len(set(names)):
            raise ValueError(f"{self.slug}: duplicate parameter names")
        if not any(param.output for param in self.params):
            raise ValueError(f"{self.slug}: at least one output tensor is required")
