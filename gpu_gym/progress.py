"""Record what you have solved and how fast, in gyms/gpu/progress.json.

The file is plain JSON so you can read it, diff it, or commit it. Layout:

{
  "version": 1,
  "exercises": {
    "vector_add": {
      "hints_seen": 1,
      "langs": {
        "cuda": {
          "attempts": 4,          # test runs that loaded the solution
          "passes": 2,
          "last_status": "pass",  # pass | fail
          "first_solved": "2026-11-20T10:31:02",
          "last_run": "2026-11-20T10:40:11",
          "best_ms": 0.0312,
          "best_at": "2026-11-20T10:40:11",
          "history": [{"at": "...", "ms": 0.04, "gbps": 300.1}, ...]
        }
      }
    }
  }
}
"""

import json
import os
from datetime import datetime
from pathlib import Path

from gpu_gym.paths import GYM

# GPU_GYM_PROGRESS_FILE points somewhere else, e.g. a scratch copy for another user's run on the board.
PROGRESS_FILE = Path(os.environ["GPU_GYM_PROGRESS_FILE"]) if os.environ.get("GPU_GYM_PROGRESS_FILE") else GYM / "progress.json"
_HISTORY_LIMIT = 50


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load() -> dict:
    if not PROGRESS_FILE.is_file():
        return {"version": 1, "exercises": {}}
    try:
        data = json.loads(PROGRESS_FILE.read_text())
    except json.JSONDecodeError:
        backup = PROGRESS_FILE.with_suffix(".json.broken")
        PROGRESS_FILE.replace(backup)
        print(f"progress.json was not valid JSON. Moved it to {backup.name} and started fresh.")
        return {"version": 1, "exercises": {}}
    data.setdefault("version", 1)
    data.setdefault("exercises", {})
    return data


def save(data: dict) -> None:
    tmp = PROGRESS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, PROGRESS_FILE)


def exercise(data: dict, slug: str) -> dict:
    entry = data["exercises"].setdefault(slug, {})
    entry.setdefault("hints_seen", 0)
    entry.setdefault("langs", {})
    return entry


def lang_entry(data: dict, slug: str, lang: str) -> dict:
    return exercise(data, slug)["langs"].setdefault(lang, {"attempts": 0, "passes": 0})


def record_test(slug: str, lang: str, passed: bool) -> None:
    data = load()
    entry = lang_entry(data, slug, lang)
    now = _now()
    entry["attempts"] = entry.get("attempts", 0) + 1
    entry["last_run"] = now
    entry["last_status"] = "pass" if passed else "fail"
    if passed:
        entry["passes"] = entry.get("passes", 0) + 1
        entry.setdefault("first_solved", now)
    save(data)


def record_bench(slug: str, lang: str, ms: float, gbps: float | None, tflops: float | None) -> float | None:
    """Store one benchmark result. Return the previous best, or None if this is the first."""
    data = load()
    entry = lang_entry(data, slug, lang)
    now = _now()
    previous = entry.get("best_ms")
    if previous is None or ms < previous:
        entry["best_ms"] = round(ms, 5)
        entry["best_at"] = now
    sample: dict = {"at": now, "ms": round(ms, 5)}
    if gbps is not None:
        sample["gbps"] = round(gbps, 1)
    if tflops is not None:
        sample["tflops"] = round(tflops, 3)
    history = entry.setdefault("history", [])
    history.append(sample)
    del history[:-_HISTORY_LIMIT]
    save(data)
    return previous


def is_solved(data: dict, slug: str, lang: str | None = None) -> bool:
    langs = data["exercises"].get(slug, {}).get("langs", {})
    if lang is not None:
        return "first_solved" in langs.get(lang, {})
    return any("first_solved" in item for item in langs.values())


def hints_seen(slug: str) -> int:
    return int(load()["exercises"].get(slug, {}).get("hints_seen", 0))


def set_hints_seen(slug: str, count: int) -> None:
    data = load()
    entry = exercise(data, slug)
    entry["hints_seen"] = max(entry.get("hints_seen", 0), count)
    save(data)


def reset(slug: str | None) -> None:
    data = load()
    if slug is None:
        data["exercises"] = {}
    else:
        data["exercises"].pop(slug, None)
    save(data)


def path() -> Path:
    return PROGRESS_FILE
