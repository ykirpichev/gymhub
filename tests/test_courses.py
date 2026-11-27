"""Courses: gym packages under gyms/ marked "publish": "upload", published by uploading them to a board.

Each must be a valid upload. Code exercises must pass the same checks a machine runs before
publishing (every grader loads, every starter fails, every reference solution passes), and every
non-code answer key must accept its own answer."""

import json
from pathlib import Path

import numpy as np
import pytest

from gymhub import kinds, packages

COURSES = Path(__file__).resolve().parents[1] / "gyms"
COURSE_IDS = sorted(
    path.name
    for path in COURSES.iterdir()
    if (path / "gym.json").is_file() and json.loads((path / "gym.json").read_text()).get("publish") == "upload"
)


def test_there_are_five_courses():
    assert COURSE_IDS == ["gpu-interview", "llm-fundamentals", "ml-coding", "ml-fundamentals", "ml-system-design", "pytorch-numpy"]


@pytest.mark.parametrize("course", COURSE_IDS)
def test_course_is_a_valid_upload(course):
    package = packages.load(COURSES / course)
    assert package.id == course
    assert packages.validate_upload(package) == []
    tracked = [slug for track in package.manifest["tracks"] for slug in track["exercises"]]
    assert sorted(tracked) == sorted(package.slugs()), "every exercise is in exactly one track"
    assert len(package.slugs()) >= 30


@pytest.mark.parametrize("course", COURSE_IDS)
def test_public_catalog_has_no_answers(course):
    package = packages.load(COURSES / course)
    catalog = json.dumps(packages.public_catalog(package))
    for slug in package.slugs():
        key = package.answer_key(slug)
        if key and key.get("explanation"):
            assert key["explanation"] not in catalog
        if key and key.get("rubric"):
            assert key["rubric"] not in catalog


def _correct_response(meta: dict, key: dict):
    if meta["kind"] == "choice":
        return key["correct"] if meta.get("multiple") else key["correct"][0]
    if meta["kind"] == "order":
        return key["order"]
    if meta["kind"] == "match":
        return key["pairs"]
    if meta["kind"] == "fill":
        return {blank: str(spec["answers"][0]) for blank, spec in key["blanks"].items()}
    return str(key["answers"][0])


@pytest.mark.parametrize("course", ["ml-fundamentals", "llm-fundamentals", "ml-system-design"])
def test_answer_keys_accept_their_own_answers(course):
    package = packages.load(COURSES / course)
    for slug in package.slugs():
        meta = package.meta(slug)
        if meta["kind"] not in {"choice", "answer", "order", "match", "fill"}:
            continue
        key = package.answer_key(slug)
        result = kinds.grade(meta["kind"], meta, key, _correct_response(meta, key))
        assert result["ok"], slug
        assert key.get("explanation"), f"{slug}: explain the answer"


def test_ml_coding_passes_machine_validation(tmp_path):
    """What a worker runs before publishing: graders load, starters fail, references pass."""
    from gymhub.runner.validate import validate

    result = validate(COURSES / "ml-coding")
    failed = [check for check in result["checks"] if not check["ok"]]
    assert result["ok"], failed
    assert not result["warnings"], result["warnings"]


def test_numpy_results_compare_by_shape_and_tolerance():
    from gymkit.runner import default_compare

    default_compare([1.0, 2.0], np.array([1.0, 2.0]), 1e-6, 1e-6)
    default_compare(np.array([[1, 2]]), [[1, 2]], 0, 0)
    default_compare(np.float64(0.5), 0.5, 1e-9, 1e-9)
    default_compare([np.nan], np.array([np.nan]), 1e-6, 1e-6)
    for bad in ([1.0, 2.1], [1.0], None):
        with pytest.raises(AssertionError):
            default_compare(bad, np.array([1.0, 2.0]), 1e-6, 1e-6)


def test_typed_answers_explain_once_right():
    meta = {"kind": "answer"}
    key = {"answers": ["0.5"], "match": "number", "tolerance": 0.01, "explanation": "Because."}
    assert kinds.grade("answer", meta, key, "0.501")["feedback"] == "Correct. Because."
    assert "Because" not in kinds.grade("answer", meta, key, "0.7")["feedback"]


def test_integer_tensors_are_compared_in_their_own_dtype():
    torch = pytest.importorskip("torch")
    from gymkit.runner import default_compare

    big = 2**40
    a, b = torch.tensor([big], dtype=torch.int64), torch.tensor([big + 1], dtype=torch.int64)
    assert a.float().item() == b.float().item()  # the float32 copies collide
    default_compare(a.clone(), a, 0, 0)
    with pytest.raises(AssertionError):
        default_compare(b, a, 0, 0)
    with pytest.raises(AssertionError):
        default_compare(torch.tensor([True, False]), torch.tensor([True, True]), 0, 0)
    default_compare(torch.tensor([1.0000001]), torch.tensor([1.0]), 1e-5, 1e-5)
