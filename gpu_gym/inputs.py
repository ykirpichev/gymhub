"""Small helpers for building CUDA cases: float32 unless a dtype is given."""

import torch

# Where cases are built. Always "cuda" when grading; tests set "cpu" to check problems on a machine without a GPU.
DEVICE = "cuda"


def cuda_rand(*shape: int, seed: int, low: float = 0.0, high: float = 1.0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Uniform values in [low, high), drawn in float32 on the CPU (the same values on every machine), then cast."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    data = torch.empty(*shape, dtype=torch.float32)
    data.uniform_(low, high, generator=generator)
    return data.to(dtype).to(DEVICE).contiguous()


def cuda_randint(*shape: int, seed: int, low: int, high: int, dtype: torch.dtype = torch.int32) -> torch.Tensor:
    """Integers in [low, high), drawn on the CPU, then cast."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    data = torch.randint(low, high, shape, generator=generator, dtype=torch.int64)
    return data.to(dtype).to(DEVICE).contiguous()


def cuda_empty(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.empty(*shape, device=DEVICE, dtype=dtype)


def cuda_tensor(values: list, shape: tuple[int, ...] | None = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    tensor = torch.tensor(values, dtype=dtype, device=DEVICE)
    if shape is not None:
        tensor = tensor.reshape(shape)
    return tensor.contiguous()
