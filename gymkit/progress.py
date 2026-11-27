"""progress.json for a gymkit gym. Same layout as gpu_gym's progress.json, so the board reads both alike."""

import json
import os
from datetime import datetime
from pathlib import Path

_HISTORY_LIMIT = 50


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Progress:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> dict:
        if not self.path.is_file():
            return {"version": 1, "exercises": {}}
        try:
            data = json.loads(self.path.read_text())
        except json.JSONDecodeError:
            backup = self.path.with_suffix(".json.broken")
            self.path.replace(backup)
            return {"version": 1, "exercises": {}}
        data.setdefault("version", 1)
        data.setdefault("exercises", {})
        return data

    def save(self, data: dict) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, self.path)

    @staticmethod
    def _entry(data: dict, slug: str, lang: str) -> dict:
        exercise = data["exercises"].setdefault(slug, {"hints_seen": 0, "langs": {}})
        exercise.setdefault("hints_seen", 0)
        return exercise.setdefault("langs", {}).setdefault(lang, {"attempts": 0, "passes": 0})

    def record_test(self, slug: str, lang: str, passed: bool) -> bool:
        """Return True when this is the first pass."""
        data = self.load()
        entry = self._entry(data, slug, lang)
        now = _now()
        first = passed and "first_solved" not in entry
        entry["attempts"] = entry.get("attempts", 0) + 1
        entry["last_run"] = now
        entry["last_status"] = "pass" if passed else "fail"
        if passed:
            entry["passes"] = entry.get("passes", 0) + 1
            entry.setdefault("first_solved", now)
        self.save(data)
        return first

    def record_bench(self, slug: str, lang: str, ms: float, ref_ms: float | None) -> float | None:
        data = self.load()
        entry = self._entry(data, slug, lang)
        now = _now()
        previous = entry.get("best_ms")
        if previous is None or ms < previous:
            entry["best_ms"] = round(ms, 5)
            entry["best_at"] = now
        sample: dict = {"at": now, "ms": round(ms, 5)}
        if ref_ms is not None:
            sample["ref_ms"] = round(ref_ms, 5)
        history = entry.setdefault("history", [])
        history.append(sample)
        del history[:-_HISTORY_LIMIT]
        self.save(data)
        return previous

    def set_hints_seen(self, slug: str, count: int) -> None:
        data = self.load()
        exercise = data["exercises"].setdefault(slug, {"hints_seen": 0, "langs": {}})
        exercise["hints_seen"] = max(exercise.get("hints_seen", 0), count)
        self.save(data)

    def is_solved(self, data: dict, slug: str, lang: str | None = None) -> bool:
        langs = data["exercises"].get(slug, {}).get("langs", {})
        if lang is not None:
            return "first_solved" in langs.get(lang, {})
        return any("first_solved" in item for item in langs.values())

    def reset(self, slug: str | None) -> None:
        data = self.load()
        if slug is None:
            data["exercises"] = {}
        else:
            data["exercises"].pop(slug, None)
        self.save(data)
