"""Example cases a learner's browser can run: inputs and expected outputs, never the grader.

A machine that holds the private grader builds them (`build`) when it publishes or validates a gym:
the exercise's built-in cases, each with the reference's answer, as lossless JSON. Run mode already
shows these inputs and expected outputs to anyone who presses Run, so they reveal nothing new. Only
cases the reference solution passes here are kept, so an exercise whose grader calls the solution in
its own way (a class to drive, say) gets no examples rather than wrong ones.

The board runs them on the code in the editor with Pyodide in a Web Worker (`run_text`), for instant
feedback without a runner. That check is informal: it has no hidden cases, no time limits, and an
exercise with its own comparison may accept answers the plain comparison here rejects. Submit grades.

A reference that imports a module Pyodide does not have (PyTorch, Triton) gets no examples: the browser
could not run the learner's code either, so the exercise offers only Run and Submit.

Standard library only, plus gymkit/compare.py; NumPy is imported only when a case holds an array.
"""

import contextlib
import copy
import io
import json
import math
import re
import sys
import traceback
import types
from typing import Any

try:
    from gymkit.compare import _short, default_compare
except ImportError:  # in the browser both files sit side by side, outside a gymkit package
    from compare import _short, default_compare  # type: ignore[no-redef]

VERSION = 1
MAX_CASES = 20
MAX_CASE_BYTES = 8_000
MAX_TOTAL_BYTES = 40_000
SOLUTION_FILE = "solution.py"
# Modules a solution may import that Pyodide cannot load: an exercise that needs one gets no examples.
NOT_IN_BROWSER = ("torch", "triton", "jax", "tensorflow")
_IMPORTS_NOT_IN_BROWSER = re.compile(r"^\s*(?:import|from)\s+(?:%s)\b" % "|".join(NOT_IN_BROWSER), re.MULTILINE)


def needs_more_than_the_browser(source: str | None) -> bool:
    """Does this solution import a module the browser's Python (Pyodide) cannot load?"""
    return bool(source) and bool(_IMPORTS_NOT_IN_BROWSER.search(source))


class Unencodable(TypeError):
    pass


# ---- lossless JSON for case values ---------------------------------------------------------------


def encode(value: Any) -> Any:
    """A case value as JSON. Raises Unencodable for what the browser cannot rebuild (tensors, objects)."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"$": "float", "v": repr(value)}
    if isinstance(value, list):
        return [encode(item) for item in value]
    if isinstance(value, tuple):
        return {"$": "tuple", "items": [encode(item) for item in value]}
    if isinstance(value, (set, frozenset)):
        return {"$": "frozenset" if isinstance(value, frozenset) else "set", "items": [encode(item) for item in value]}
    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value) and "$" not in value:
            return {key: encode(item) for key, item in value.items()}
        return {"$": "dict", "items": [[encode(key), encode(item)] for key, item in value.items()]}
    np = sys.modules.get("numpy")
    if np is not None:
        if isinstance(value, np.ndarray):
            if value.dtype.kind not in "biufcU":
                raise Unencodable(f"array of dtype {value.dtype}")
            if value.dtype.kind == "c":
                raise Unencodable("complex array")
            return {"$": "ndarray", "dtype": value.dtype.str, "shape": list(value.shape), "data": encode(value.ravel().tolist())}
        if isinstance(value, np.generic):
            return {"$": "npscalar", "dtype": value.dtype.str, "v": encode(value.item())}
    raise Unencodable(f"cannot send a {type(value).__name__} to the browser")


def decode(value: Any) -> Any:
    if isinstance(value, list):
        return [decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    tag = value.get("$")
    if tag is None:
        return {key: decode(item) for key, item in value.items()}
    if tag == "float":
        return float(value["v"])
    if tag == "tuple":
        return tuple(decode(item) for item in value["items"])
    if tag == "set":
        return {decode(item) for item in value["items"]}
    if tag == "frozenset":
        return frozenset(decode(item) for item in value["items"])
    if tag == "dict":
        return {_hashable(decode(key)): decode(item) for key, item in value["items"]}
    if tag in ("ndarray", "npscalar"):
        import numpy as np

        if tag == "npscalar":
            return np.dtype(value["dtype"]).type(decode(value["v"]))
        return np.array(decode(value["data"]), dtype=np.dtype(value["dtype"])).reshape(value["shape"])
    raise ValueError(f"unknown value tag {tag!r}")


def _hashable(key: Any) -> Any:
    return tuple(_hashable(item) for item in key) if isinstance(key, list) else key


# ---- building (on a machine with the grader) ---------------------------------------------------


def build(problem, reference_source: str | None) -> str | None:
    """The examples of a gymkit exercise as a JSON string, or None when there are none to offer.

    With a Python reference solution, only the cases it passes when run here are kept. Without one,
    only an exercise that calls plain `solve(*args)` (no `call` of its own) gets examples."""
    if not reference_source and problem.call is not None:
        return None
    if needs_more_than_the_browser(reference_source):
        return None
    cases = []
    total = 0
    for index, case in enumerate(problem.functional_cases()):
        try:
            args = [case[name] for name in problem.params]
            want = problem.reference(*copy.deepcopy(args))
            item = {"name": str(case.get("_name") or f"case {index + 1}"), "args": [encode(arg) for arg in args],
                    "expected": encode(want)}
        except Exception:  # noqa: BLE001 - a case the browser cannot rebuild, or one the reference rejects
            continue
        size = len(json.dumps(item))
        if size > MAX_CASE_BYTES:
            continue
        if total + size > MAX_TOTAL_BYTES:
            break
        cases.append(item)
        total += size
        if len(cases) >= MAX_CASES:
            break
    if not cases:
        return None
    data = {"version": VERSION, "params": list(problem.params), "atol": problem.atol, "rtol": problem.rtol,
            "strict": problem.compare is None, "cases": cases}
    if not reference_source:
        return json.dumps(data, separators=(",", ":"))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            results = run(reference_source, json.dumps(data))
    except Exception:  # noqa: BLE001 - the reference does not even load this way
        return None
    data["cases"] = [case for case, result in zip(cases, results) if result["ok"]]
    return json.dumps(data, separators=(",", ":")) if data["cases"] else None


# ---- running (in the browser, or anywhere) -----------------------------------------------------


def _load(source: str) -> types.ModuleType:
    module = types.ModuleType("solution")
    module.__file__ = SOLUTION_FILE
    sys.modules["solution"] = module
    exec(compile(source, SOLUTION_FILE, "exec"), module.__dict__)  # noqa: S102 - the learner's own code, in their own browser
    return module


def _where(exc: BaseException) -> str:
    frames = [frame for frame in traceback.extract_tb(exc.__traceback__) if frame.filename == SOLUTION_FILE]
    if isinstance(exc, SyntaxError) and exc.filename == SOLUTION_FILE:
        return f"  at {SOLUTION_FILE}:{exc.lineno}  {(exc.text or '').strip()}".rstrip()
    if not frames:
        return ""
    return f"  at {SOLUTION_FILE}:{frames[-1].lineno}  {frames[-1].line or ''}".rstrip()


def run(source: str, examples: str) -> list[dict]:
    """Run `solve` from `source` on each example. Raises when the source does not load."""
    data = json.loads(examples)
    module = _load(source)
    solve = getattr(module, "solve", None)
    if not callable(solve):
        raise AttributeError("the solution does not define solve()")
    results = []
    for case in data["cases"]:
        args = [decode(arg) for arg in case["args"]]
        want = decode(case["expected"])
        inputs = ", ".join(f"{name}={_short(arg, 60)}" for name, arg in zip(data["params"], args))
        result = {"name": case["name"], "inputs": inputs, "expected": _short(want, 600), "ok": False,
                  "output": "", "stdout": "", "error": "", "message": ""}
        printed = io.StringIO()
        try:
            with contextlib.redirect_stdout(printed):
                got = solve(*args)
        except Exception as exc:  # noqa: BLE001
            if isinstance(want, str) and want == type(exc).__name__:  # "raise ValueError on bad input" exercises
                got = want
            else:
                result.update(stdout=printed.getvalue()[-16000:], error=f"{type(exc).__name__}: {exc}\n{_where(exc)}".rstrip())
                results.append(result)
                continue
        result["stdout"] = printed.getvalue()[-16000:]
        result["output"] = _short(got, 600)
        try:
            default_compare(got, want, data["atol"], data["rtol"])
            result["ok"] = True
        except AssertionError as exc:
            result["message"] = str(exc)
        results.append(result)
    return results


def run_text(source: str, examples: str) -> dict:
    """`run`, written out like the runner's Run mode: {"text", "passed", "total"}."""
    data = json.loads(examples)
    total = len(data["cases"])
    lines = ["Examples, run in this browser: not recorded, no time limits. Submit runs the full tests."]
    try:
        results = run(source, examples)
    except Exception as exc:  # noqa: BLE001
        lines += ["", "The code does not load:", f"  {type(exc).__name__}: {exc}"]
        where = _where(exc)
        if where:
            lines.append(where)
        return {"text": "\n".join(lines), "passed": 0, "total": total}
    passed = 0
    for index, result in enumerate(results):
        lines += ["", f"── example {index + 1}/{total}: {result['name']}", f"  inputs:   {result['inputs']}"]
        stdout = result["stdout"].rstrip("\n")
        if stdout:
            lines.append("  stdout:")
            lines += [f"  | {line}" for line in stdout.splitlines()]
        if result["error"]:
            lines.append(f"  ERROR  {result['error'].splitlines()[0]}")
            lines += [line for line in result["error"].splitlines()[1:]]
            continue
        lines += [f"  output:   {result['output']}", f"  expected: {result['expected']}"]
        if result["ok"]:
            passed += 1
            lines.append("  PASS")
        else:
            lines.append("  FAIL")
            lines += [f"    {line}" for line in result["message"].splitlines()]
    lines.append("")
    summary = f"{passed}/{total} examples match."
    if passed < total and not data.get("strict", True):
        summary += " This exercise checks answers its own way, so the full tests may accept an answer that differs here: Submit to be sure."
    elif passed == total:
        summary += " Submit to run the full tests and record it."
    lines.append(summary)
    return {"text": "\n".join(lines), "passed": passed, "total": total}
