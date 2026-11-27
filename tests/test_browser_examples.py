"""Examples the browser runs (issue #109): built where the grader is, run anywhere, published without graders."""

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from gymkit import examples
from gymkit.spec import Problem
from tests.test_board_auth import H, board, client, sign_in  # noqa: F401  (fixture)

REPO = Path(__file__).resolve().parents[1]
HELLO = REPO / "examples" / "gyms" / "hello"


def _problem(**fields) -> Problem:
    defaults = dict(slug="p", title="P", difficulty="easy", summary="", params=["xs"],
                    reference=lambda xs: sorted(xs), functional_cases=lambda: [{"_name": "a", "xs": [3, 1, 2]}, {"xs": []}])
    return Problem(**{**defaults, **fields})


def _sum_list():
    from gymkit.gym import load_gym

    return load_gym(HELLO).load("sum_list")


@pytest.mark.parametrize("value", [
    None, True, 3, 2**80, -1.5, "text", [1, [2, 3]], (1, (2,)), {"a": 1, "b": [2.5]}, {1: "x", (2, 3): "y"},
    {"$": "a key that looks like a tag"}, {3, 4}, frozenset({5}), math.inf, -math.inf,
])
def test_values_round_trip_through_json(value):
    assert examples.decode(json.loads(json.dumps(examples.encode(value)))) == value


def test_arrays_round_trip_with_dtype_and_shape():
    for array in [np.arange(6, dtype=np.int32).reshape(2, 3), np.array([[0.1, np.nan]]), np.array([True, False]),
                  np.zeros((0, 4)), np.array(["ab", "c"])]:
        back = examples.decode(json.loads(json.dumps(examples.encode(array))))
        assert back.dtype == array.dtype and back.shape == array.shape
        assert np.array_equal(back, array, equal_nan=array.dtype.kind == "f")
    scalar = examples.decode(json.loads(json.dumps(examples.encode(np.float32(0.25)))))
    assert scalar == np.float32(0.25) and isinstance(scalar, np.float32)
    assert math.isnan(examples.decode(examples.encode(math.nan)))


def test_values_the_browser_cannot_rebuild_are_refused():
    with pytest.raises(examples.Unencodable):
        examples.encode(object())
    with pytest.raises(examples.Unencodable):
        examples.encode(np.array([object()]))


def test_examples_are_built_from_the_cases_and_the_reference():
    data = json.loads(examples.build(_sum_list(), "def solve(nums):\n    return sum(nums)\n"))
    assert data["params"] == ["nums"] and data["strict"]
    assert [case["name"] for case in data["cases"]] == ["example", "empty", "negatives", "random"]
    assert data["cases"][0] == {"name": "example", "args": [[1, 2, 3]], "expected": 6}


@pytest.mark.parametrize("source", [
    "import torch\n\ndef solve(nums):\n    return sum(nums)\n",
    "import numpy as np\nfrom torch import nn\n\ndef solve(nums):\n    return sum(nums)\n",
    "def solve(nums):\n    import triton\n    return sum(nums)\n",
])
def test_a_reference_that_needs_torch_offers_no_browser_examples(source):
    assert examples.build(_sum_list(), source) is None


def test_a_module_whose_name_only_starts_like_torch_is_fine():
    assert examples.needs_more_than_the_browser("import torchish\n") is False
    assert examples.needs_more_than_the_browser("# import torch in a comment\ndef solve(): pass\n") is False


def test_a_correct_solution_passes_and_a_wrong_one_fails():
    built = examples.build(_sum_list(), None)  # no reference: plain solve(*args), so no check needed
    good = examples.run_text("def solve(nums):\n    print('hi')\n    return sum(nums)\n", built)
    assert (good["passed"], good["total"]) == (4, 4)
    assert "  | hi" in good["text"] and "4/4 examples match" in good["text"]
    bad = examples.run_text("def solve(nums):\n    return 0\n", built)
    assert bad["passed"] == 1 and "FAIL" in bad["text"] and "expected 6" in bad["text"]


def test_errors_point_at_the_line_in_the_solution():
    built = examples.build(_sum_list(), None)
    crash = examples.run_text("def solve(nums):\n    return nums[0]\n", built)
    assert "ERROR  IndexError" in crash["text"] and "solution.py:2" in crash["text"]
    broken = examples.run_text("def solve(nums)\n    return 1\n", built)
    assert broken["passed"] == 0 and "does not load" in broken["text"] and "solution.py:1" in broken["text"]
    missing = examples.run_text("x = 1\n", built)
    assert "does not define solve()" in missing["text"]


def test_raising_the_expected_error_counts_as_the_answer():
    def reference(xs):
        return "ValueError" if not xs else max(xs)

    def call(module, args):
        try:
            return module.solve(*args)
        except ValueError:
            return "ValueError"

    problem = _problem(reference=reference, call=call)
    solution = "def solve(xs):\n    if not xs:\n        raise ValueError('empty')\n    return max(xs)\n"
    built = examples.build(problem, solution)
    assert len(json.loads(built)["cases"]) == 2
    assert examples.run_text(solution, built)["passed"] == 2
    assert examples.run_text("def solve(xs):\n    return max(xs, default=None)\n", built)["passed"] == 1


def test_an_exercise_with_its_own_call_needs_a_reference_that_passes():
    problem = _problem(call=lambda module, args: module.Sorter().sort(*args))
    assert examples.build(problem, None) is None  # cannot tell whether the browser calls it the same way
    driven = "class Sorter:\n    def sort(self, xs):\n        return sorted(xs)\n"
    assert examples.build(problem, driven) is None  # the reference fails plain solve(): no examples
    both = driven + "\ndef solve(xs):\n    return sorted(xs)\n"
    assert len(json.loads(examples.build(problem, both))["cases"]) == 2


def test_cases_too_large_or_unencodable_are_left_out(monkeypatch):
    cases = [{"xs": [object()]}, {"_name": "big", "xs": list(range(5000))}, {"_name": "small", "xs": [2, 1]}]
    data = json.loads(examples.build(_problem(functional_cases=lambda: cases), None))
    assert [case["name"] for case in data["cases"]] == ["small"]
    assert examples.build(_problem(functional_cases=lambda: cases[:2]), None) is None
    assert not json.loads(examples.build(_problem(compare=lambda got, want, case: None), None))["strict"]


def test_a_custom_comparison_says_other_answers_may_count():
    built = examples.build(_problem(compare=lambda got, want, case: None), None)
    text = examples.run_text("def solve(xs):\n    return xs\n", built)["text"]
    assert "FAIL" in text and "may accept an answer that differs" in text


def test_validation_and_the_local_catalog_add_examples(tmp_path):
    from gymhub.runner import catalog
    from gymhub.runner.validate import validate

    enrich = validate(HELLO)["enrich"]
    assert json.loads(enrich["sum_list"]["examples"])["cases"][0]["expected"] == 6
    assert json.loads(enrich["is_palindrome"]["examples"])["cases"]
    root = tmp_path / "hello"
    shutil.copytree(HELLO, root)
    problems = {item["slug"]: item for item in catalog.build(root)["problems"]}
    assert json.loads(problems["sum_list"]["examples"])["cases"][0]["expected"] == 6


def test_the_courses_offer_examples_for_most_python_exercises():
    from gymkit.gym import load_gym
    from gymhub import packages

    package = packages.load(REPO / "gyms" / "ml-coding")
    gym = load_gym(package.root)
    with_examples = 0
    for slug in package.slugs():
        if package.meta(slug).get("kind", "code") != "code":
            continue  # a lesson opening a track
        reference = package.references(slug).get("python")
        built = examples.build(gym.load(slug), reference.read_text() if reference else None)
        if built:
            assert len(built) <= examples.MAX_TOTAL_BYTES + 1000
            with_examples += 1
    assert with_examples >= 30


def test_published_examples_reach_the_exercise_and_only_as_text(board):
    from tests.test_gym_versions import _owner, _upload, _worker, complete_validation
    from gymhub import packages

    owner = _owner(board)
    _upload(owner, packages.pack(HELLO))
    worker = _worker(board, owner)
    job = worker.post("/v1/worker/lease", json={"gyms": ["algo"]}).json()["job"]
    built = examples.build(_sum_list(), None)
    enrich = {"sum_list": {"examples": built}, "is_palindrome": {"examples": {"cases": "not a string"}}}
    assert complete_validation(worker, job, enrich=enrich).status_code == 200
    assert owner.get("/api/gyms/hello/problems/sum_list").json()["problem"]["examples"] == built
    assert "examples" not in owner.get("/api/gyms/hello/problems/is_palindrome").json()["problem"]


def test_the_board_serves_the_browser_runner_and_nothing_else(board):
    c = client(board)
    sign_in(c, "owner")
    reply = c.get("/browser/examples.py")
    assert reply.status_code == 200 and "def run_text" in reply.text
    assert "def default_compare" in c.get("/browser/compare.py").text
    for name in ["runner.py", "problem.py", "..%2Fgymhub%2Fstore.py"]:
        reply = c.get(f"/browser/{name}")
        assert reply.status_code == 404 or reply.headers["content-type"].startswith("text/html")  # the app's page
        assert "import" not in reply.text
