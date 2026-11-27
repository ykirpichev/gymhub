"""Choice questions must not be solvable by position or length (#56).

Guards the course gyms against the patterns that made them guessable: the right answer always "a",
multi-select keys that are "everything but the last option", and a right answer that is reliably
the longest one.
"""

import json
from collections import Counter
from pathlib import Path

import pytest

from gymhub import packages

GYMS = Path(__file__).resolve().parents[1] / "gyms"


def _choice_items() -> list[dict]:
    items = []
    for meta_path in sorted(GYMS.glob("*/problems/*/problem.json")):
        meta = json.loads(meta_path.read_text())
        if meta.get("kind") != "choice":
            continue
        key = json.loads((meta_path.parent / "answer.json").read_text())
        items.append({"slug": meta_path.parent.name, "multiple": bool(meta.get("multiple")),
                      "choices": meta["choices"], "correct": set(key["correct"])})
    return items


ITEMS = _choice_items()
SINGLE = [item for item in ITEMS if not item["multiple"]]
MULTI = [item for item in ITEMS if item["multiple"]]


def test_there_are_enough_items_to_measure():
    assert len(SINGLE) >= 10 and len(MULTI) >= 10


def test_single_answers_are_spread_across_the_letters():
    counts = Counter(next(iter(item["correct"])) for item in SINGLE)
    assert max(counts.values()) <= 0.4 * len(SINGLE), counts
    assert len(counts) >= 4, counts


def test_the_right_answer_is_not_reliably_the_longest():
    strict, by_margin = [], []
    for item in SINGLE:
        lengths = {choice["id"]: len(choice["text"]) for choice in item["choices"]}
        right = lengths[next(iter(item["correct"]))]
        longest_wrong = max(length for cid, length in lengths.items() if cid not in item["correct"])
        if right > longest_wrong:
            strict.append(item["slug"])
        if right > 1.15 * longest_wrong:
            by_margin.append(item["slug"])
    assert len(strict) <= 0.5 * len(SINGLE), strict
    assert len(by_margin) <= 0.2 * len(SINGLE), by_margin


def test_multi_select_keys_do_not_follow_the_option_order():
    for item in MULTI:
        flags = [choice["id"] in item["correct"] for choice in item["choices"]]
        assert any(flags) and not all(flags), item["slug"]
        assert flags != sorted(flags, reverse=True), f"{item['slug']}: every right option comes before every wrong one"
    with_a = sum("a" in item["correct"] for item in MULTI)
    assert with_a <= 0.75 * len(MULTI)


def test_multi_select_varies_how_many_are_right():
    all_but_one = [item["slug"] for item in MULTI if len(item["correct"]) == len(item["choices"]) - 1]
    assert len(all_but_one) <= 0.5 * len(MULTI), all_but_one
    assert len({len(item["correct"]) for item in MULTI}) >= 3


@pytest.mark.parametrize("gym", ["llm-fundamentals", "ml-fundamentals"])
def test_the_quiz_gyms_still_validate(gym):
    assert packages.validate_upload(packages.load(GYMS / gym)) == []
