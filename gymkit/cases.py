"""Custom test cases as JSON: encode a built-in case into an editable template, decode user cases.

Plain values (numbers, strings, lists, dicts) are written as-is. Tensors use a small object form:

    {"tensor": [[1, 2], [3, 4]]}                 explicit values (dtype float32 unless "dtype" is given)
    {"randn": [64, 128], "seed": 1}              normal random values of this shape
    {"rand": [1024]}                             uniform [0, 1) values
    {"randint": [16], "low": 0, "high": 10}      integers in [low, high)
    {"zeros": [4, 4]}  {"ones": [8]}  {"empty": [32]}
    {"arange": 10}                               0, 1, ..., 9

Any of them can add "dtype": "float32" | "float16" | "bfloat16" | "int32" | "int64" | "bool".
Tensors are created on the GPU when one is available.

Used by gymkit (LLM and algo gyms) and gpu_gym, so every gym accepts the same format.
"""

import json
from pathlib import Path
from typing import Any

TENSOR_KEYS = ("tensor", "randn", "rand", "randint", "zeros", "ones", "empty", "arange")
MAX_CASES = 20
MAX_EXPLICIT = 64  # tensors up to this many elements are written out in templates


class CaseError(ValueError):
    pass


def _torch():
    import torch

    return torch


def _device():
    torch = _torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


def _dtype(name: str | None, default: str):
    torch = _torch()
    name = name or default
    mapping = {
        "float32": torch.float32, "float": torch.float32, "float16": torch.float16, "half": torch.float16,
        "bfloat16": torch.bfloat16, "int32": torch.int32, "int": torch.int32, "int64": torch.int64,
        "long": torch.int64, "bool": torch.bool, "float64": torch.float64,
    }
    if name not in mapping:
        raise CaseError(f"unknown dtype {name!r}; use one of {', '.join(sorted(mapping))}")
    return mapping[name]


def _shape(value, key: str) -> list[int]:
    shape = [value] if isinstance(value, int) else value
    if not isinstance(shape, list) or not all(isinstance(n, int) and n >= 0 for n in shape):
        raise CaseError(f'"{key}" needs a shape like [64, 128]')
    total = 1
    for n in shape:
        total *= n
    if total > 1 << 26:
        raise CaseError(f"tensor of shape {shape} is too large for a custom case")
    return shape


def is_tensor_spec(value: Any) -> bool:
    return isinstance(value, dict) and any(key in value for key in TENSOR_KEYS)


def decode_value(value: Any) -> Any:
    if isinstance(value, list):
        return [decode_value(item) for item in value]
    if not is_tensor_spec(value):
        if isinstance(value, dict):
            return {key: decode_value(item) for key, item in value.items()}
        return value
    torch = _torch()
    device = _device()
    generator = torch.Generator(device=device).manual_seed(int(value.get("seed", 0)))
    if "tensor" in value:
        default = "float32"
        data = value["tensor"]
        if isinstance(data, list) and data and all(isinstance(x, bool) for x in _flatten(data)):
            default = "bool"
        try:
            return torch.tensor(data, dtype=_dtype(value.get("dtype"), default), device=device)
        except (TypeError, ValueError) as exc:
            raise CaseError(f"bad tensor values: {exc}") from exc
    if "arange" in value:
        return torch.arange(int(value["arange"]), dtype=_dtype(value.get("dtype"), "int64"), device=device)
    for key in ("randn", "rand"):
        if key in value:
            shape = _shape(value[key], key)
            out = getattr(torch, key)(*shape, generator=generator, device=device, dtype=torch.float32)
            return out.to(_dtype(value.get("dtype"), "float32"))
    if "randint" in value:
        shape = _shape(value["randint"], "randint")
        low, high = int(value.get("low", 0)), int(value.get("high", 10))
        return torch.randint(low, high, shape, generator=generator, device=device, dtype=_dtype(value.get("dtype"), "int64"))
    for key in ("zeros", "ones", "empty"):
        if key in value:
            shape = _shape(value[key], key)
            return getattr(torch, key)(*shape, device=device, dtype=_dtype(value.get("dtype"), "float32"))
    raise CaseError(f"cannot read {value!r}")


def _flatten(data):
    if isinstance(data, list):
        for item in data:
            yield from _flatten(item)
    else:
        yield data


def encode_value(value: Any, output: bool = False) -> Any:
    """Turn a built-in case value into the editable JSON form."""
    try:
        torch = _torch()
    except ImportError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        dtype = str(value.dtype).replace("torch.", "")
        shape = list(value.shape)
        if output:
            spec: dict = {"empty": shape}
        elif value.numel() <= MAX_EXPLICIT:
            data = value.detach().cpu()
            values = data.tolist()
            if data.is_floating_point():
                values = _round(values)
            spec = {"tensor": values}
        elif value.is_floating_point():
            spec = {"randn": shape, "seed": 1}
        else:
            spec = {"randint": shape, "low": 0, "high": 10}
        if dtype != "float32" and not (dtype == "bool" and "tensor" in spec):
            spec["dtype"] = dtype
        return spec
    if isinstance(value, tuple):
        return [encode_value(item) for item in value]
    if isinstance(value, list):
        return [encode_value(item) for item in value]
    if isinstance(value, dict):
        return {key: encode_value(item) for key, item in value.items()}
    if isinstance(value, float):
        return round(value, 6)
    return value


def _round(values):
    if isinstance(values, list):
        return [_round(item) for item in values]
    return round(values, 5)


def template(case: dict, params: list[str], outputs: set[str] = frozenset()) -> dict:
    return {name: encode_value(case[name], output=name in outputs) for name in params if name in case}


def load_cases(path: str | Path, params: list[str]) -> list[dict]:
    """Read and decode a JSON list of cases. Raises CaseError with a message meant for the user."""
    try:
        raw = json.loads(Path(path).read_text())
    except json.JSONDecodeError as exc:
        raise CaseError(f"custom cases are not valid JSON: {exc}") from exc
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raise CaseError("custom cases must be a JSON list of objects, one per case")
    if len(raw) > MAX_CASES:
        raise CaseError(f"at most {MAX_CASES} custom cases")
    cases = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise CaseError(f"custom case {index} must be an object like {{{', '.join(f'{p!r}: ...' for p in params)}}}")
        missing = [name for name in params if name not in item]
        extra = [name for name in item if name not in params and not name.startswith("_")]
        if missing:
            raise CaseError(f"custom case {index} is missing {', '.join(missing)}")
        if extra:
            raise CaseError(f"custom case {index} has unknown argument(s) {', '.join(extra)}; expected {', '.join(params)}")
        try:
            case = {name: decode_value(item[name]) for name in params}
        except CaseError as exc:
            raise CaseError(f"custom case {index}: {exc}") from exc
        case["_name"] = str(item.get("_name") or f"custom {index}")
        cases.append(case)
    return cases
