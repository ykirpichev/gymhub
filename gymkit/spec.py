"""Exercise definition for gyms whose solutions are plain Python functions."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

Case = dict


@dataclass
class Problem:
    """One exercise. `solve(*args)` returns its answer; the checker compares it with `reference(*args)`.

    A case is a dict that maps each name in `params` to a value, plus an optional `_name`.
    Inputs are deep-copied before each call, so a solution that mutates its arguments cannot
    change what the reference sees.
    """

    slug: str
    title: str
    difficulty: str
    summary: str
    params: list[str]
    reference: Callable[..., Any]
    functional_cases: Callable[[], list[Case]]
    # Large input for `bench`. Seed varies per timed run. Return a new case each call (no cache):
    # a timed run gets it without a copy.
    performance_case: Callable[[int], Case] | None = None
    # Raise AssertionError with a readable message when `got` is wrong. Default: tolerant deep equality.
    compare: Callable[[Any, Any, Case], None] | None = None
    # How a solution module is called. Default: module.solve(*args). Use it for class-design exercises.
    call: Callable[[Any, list], Any] | None = None
    atol: float = 1e-5
    rtol: float = 1e-5
    # `bench` fails when the median exceeds this. Use it for complexity-sensitive exercises.
    # A number applies to every language; a dict sets it per language, e.g. {"python": 300, "cpp": 30}.
    time_limit_ms: float | dict[str, float] | None = None
    track: str = ""
    tags: list[str] = field(default_factory=list)
    directory: Path | None = None

    def limit_for(self, lang: str) -> float | None:
        if isinstance(self.time_limit_ms, dict):
            return self.time_limit_ms.get(lang)
        return self.time_limit_ms

    def __post_init__(self) -> None:
        if self.difficulty not in {"easy", "medium", "hard"}:
            raise ValueError(f"{self.slug}: difficulty must be easy, medium, or hard")
        if len(set(self.params)) != len(self.params):
            raise ValueError(f"{self.slug}: duplicate parameter names")
