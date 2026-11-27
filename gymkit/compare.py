"""Comparing a solution's result with the expected one: tolerant deep equality for numbers, lists,
dicts, NumPy arrays and PyTorch tensors.

Standard library only (NumPy and PyTorch are used when already imported), so the board can also run
it in the browser with Pyodide: see gymkit/examples.py.
"""

import math
import sys
from typing import Any


def _torch():
    return sys.modules.get("torch")


def _numpy():
    return sys.modules.get("numpy")  # only when the grader or the solution imported it


def _short(value: Any, limit: int = 160) -> str:
    torch = _torch()
    if torch is not None and isinstance(value, torch.Tensor):
        return f"tensor(shape={list(value.shape)}, dtype={str(value.dtype).replace('torch.', '')}, device={value.device.type})"
    np = _numpy()
    if np is not None and isinstance(value, np.ndarray) and value.size > 16:
        return f"array(shape={list(value.shape)}, dtype={value.dtype})"
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def default_compare(got: Any, want: Any, atol: float, rtol: float, where: str = "result") -> None:
    torch = _torch()
    if torch is not None and isinstance(want, torch.Tensor):
        if not isinstance(got, torch.Tensor):
            raise AssertionError(f"{where}: expected a tensor, got {type(got).__name__}")
        if tuple(got.shape) != tuple(want.shape):
            raise AssertionError(f"{where}: shape {list(got.shape)} != expected {list(want.shape)}")
        if got.dtype != want.dtype:
            raise AssertionError(f"{where}: dtype {got.dtype} != expected {want.dtype}")
        if torch.is_floating_point(want):
            g = got.detach().float().cpu()
            w = want.detach().float().cpu()
            bad = ~torch.isclose(g, w, atol=atol, rtol=rtol)
        else:
            # Integers and bools are compared in their own dtype: a float32 copy would make two
            # different int64 values past 2**24 look equal.
            g = got.detach().cpu()
            w = want.detach().cpu()
            bad = g != w
        count = int(bad.sum())
        if count:
            lines = [f"{where}: {count} of {w.numel()} values are wrong (atol {atol:g}, rtol {rtol:g})"]
            for flat in bad.flatten().nonzero().flatten()[:5].tolist():
                index = tuple(int(i) for i in torch.unravel_index(torch.tensor(flat), tuple(w.shape))) if w.dim() else ()
                lines.append(f"  {list(index)}  got {g.flatten()[flat].item():.6g}  expected {w.flatten()[flat].item():.6g}")
            if bool(torch.isnan(g).any()):
                lines.append("  the output contains NaN")
            raise AssertionError("\n".join(lines))
        return
    np = _numpy()
    if np is not None:
        if isinstance(want, np.ndarray):
            _compare_array(np, got, want, atol, rtol, where)
            return
        if isinstance(want, np.generic):
            want = want.item()
        if isinstance(got, np.generic):
            got = got.item()
        elif isinstance(got, np.ndarray) and isinstance(want, (list, tuple)):
            got = got.tolist()
    if isinstance(want, float) or (isinstance(want, int) and isinstance(got, float)):
        if not isinstance(got, (int, float)) or not math.isclose(got, want, rel_tol=rtol, abs_tol=atol):
            raise AssertionError(f"{where}: got {_short(got)}, expected {_short(want)}")
        return
    if isinstance(want, (list, tuple)):
        if not isinstance(got, (list, tuple)):
            raise AssertionError(f"{where}: expected a list, got {type(got).__name__}: {_short(got)}")
        if len(got) != len(want):
            raise AssertionError(f"{where}: length {len(got)} != expected {len(want)}\n  got:      {_short(got)}\n  expected: {_short(want)}")
        for index, (g, w) in enumerate(zip(got, want)):
            default_compare(g, w, atol, rtol, f"{where}[{index}]")
        return
    if isinstance(want, dict):
        if not isinstance(got, dict) or set(got) != set(want):
            raise AssertionError(f"{where}: got {_short(got)}, expected {_short(want)}")
        for key in want:
            default_compare(got[key], want[key], atol, rtol, f"{where}[{key!r}]")
        return
    if got != want:
        raise AssertionError(f"{where}: got {_short(got)}, expected {_short(want)}")


def _compare_array(np, got: Any, want, atol: float, rtol: float, where: str) -> None:
    """NumPy results: same shape; floats within tolerance (NaN matches NaN), everything else exact.
    Lists and tuples are accepted where an array is expected, so solutions may return either."""
    try:
        g = np.asarray(got)
    except Exception:
        raise AssertionError(f"{where}: expected an array, got {type(got).__name__}: {_short(got)}") from None
    if g.shape != want.shape:
        raise AssertionError(f"{where}: shape {list(g.shape)} != expected {list(want.shape)}")
    if want.dtype.kind in "fc":
        if g.dtype.kind not in "biufc":
            raise AssertionError(f"{where}: expected numbers, got dtype {g.dtype}")
        bad = ~np.isclose(g.astype(want.dtype), want, atol=atol, rtol=rtol, equal_nan=True)
    else:
        bad = g != want
    count = int(np.sum(bad))
    if count:
        lines = [f"{where}: {count} of {want.size} values are wrong (atol {atol:g}, rtol {rtol:g})"]
        for index in list(zip(*np.nonzero(bad)))[:5] if want.ndim else [()]:
            got_value, want_value = (getattr(v, "item", lambda v=v: v)() for v in (g[index], want[index]))
            lines.append(f"  {[int(i) for i in index]}  got {got_value!r}  expected {want_value!r}")
        raise AssertionError("\n".join(lines))
