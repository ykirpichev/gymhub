"""Gym Markdown (gymhub/gymmd.py): one file per gym, compiled into an ordinary package."""

import json
import re
import time
from pathlib import Path

import pytest

from gymhub import cli, gymmd, kinds, packages
from tests.test_board_auth import H, board, client, sign_in  # noqa: F401  (fixture)
from tests.test_gym_access import UIDS, people  # noqa: F401  (fixture)

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples" / "every-kind.gym.md"


def _gym(body: str, front: str = "id: test-gym\ntitle: Test") -> str:
    return f"---\n{front}\n---\n\nA gym for tests.\n\n# Track\n\n{body}"


# ---- the example and the template --------------------------------------------------------------


def test_the_example_compiles_and_its_grader_really_validates(tmp_path):
    package = gymmd.build(EXAMPLE.read_text(), tmp_path / "pkg")
    assert packages.validate_upload(package) == []
    kinds_used = {package.meta(slug)["kind"] for slug in package.slugs()}
    assert kinds_used == kinds.KINDS
    from gymhub.runner.validate import validate

    result = validate(package.root)
    assert result["ok"], [check for check in result["checks"] if not check["ok"]]


def test_the_template_checks_clean():
    gym, errors = gymmd.check(gymmd.TEMPLATE.format(id="my-gym", title="My gym"))
    assert errors == [] and [e.meta["kind"] for e in gym.exercises] == ["lesson", "choice", "code"]


def test_every_kind_grades_its_own_answer(tmp_path):
    package = gymmd.build(EXAMPLE.read_text(), tmp_path / "pkg")
    meta, key = package.meta("request_path"), package.answer_key("request_path")
    texts = {item["id"]: item["text"] for item in meta["items"]}
    assert [texts[i] for i in key["order"]][0] == "Resolve the host name with DNS"
    assert [item["id"] for item in meta["items"]] != key["order"], "the public list is shuffled"
    assert kinds.grade("order", meta, key, key["order"])["ok"]
    meta, key = package.meta("cost_of_operations"), package.answer_key("cost_of_operations")
    options = {o["id"]: o["text"] for o in meta["options"]}
    assert {p["text"]: options[key["pairs"][p["id"]]] for p in meta["prompts"]}["Binary search a sorted array"] == r"$O(\log n)$"
    assert len(meta["options"]) == 4  # one left over
    meta, key = package.meta("running_mean"), package.answer_key("running_mean")
    assert kinds.grade("fill", meta, key, {"b1": "5.0004", "b2": "N + 1"})["ok"]
    meta, key = package.meta("sum_loop"), package.answer_key("sum_loop")
    assert meta["code"] is True and meta["text"].startswith("total = [[b1]]")
    meta, key = package.meta("eighth"), package.answer_key("eighth")
    assert key == {"answers": ["0.125"], "match": "number", "tolerance": 0.0005, "explanation": "$0.25 / 2 = 0.125$."}
    assert meta["placeholder"] == "a decimal"
    meta, key = package.meta("primes"), package.answer_key("primes")
    assert meta["multiple"] is True and key["correct"] == ["a", "c"]
    assert package.meta("why_caches")["min_words"] == 30 and package.answer_key("why_caches")["pass_score"] == 0.75
    assert package.hints("two_sum")[1].startswith("A dictionary")
    assert package.editorial("two_sum").startswith("Learners see this") and "#### Pitfalls" in package.editorial("two_sum")
    assert package.manifest["tracks"][0]["summary"].startswith("A lesson")


def test_learners_never_get_keys_tests_or_solutions(tmp_path):
    package = gymmd.build(EXAMPLE.read_text(), tmp_path / "pkg")
    catalog = json.dumps(packages.public_catalog(package))
    for secret in ("Each step halves the range, so", "Mentions locality", "seen[target - x]", "[-4, 10, 1, 8]", "0.125"):
        assert secret not in catalog


# ---- every existing gym round-trips ------------------------------------------------------------


def _statement(text: str) -> str:
    lines = text.strip().split("\n")
    body = "\n".join(lines[1:] if lines and lines[0].startswith("# ") else lines)
    return re.sub(r"^#+ ", "", body, flags=re.M).strip()


def _spec(spec: dict) -> tuple:
    number = spec.get("match") == "number"
    answers = tuple(float(a) if number else str(a) for a in spec.get("answers", []))
    return answers, number, spec.get("tolerance", 0) if number else None, spec.get("rel_tolerance"), bool(spec.get("case_sensitive"))


def _canonical(package: packages.Package) -> dict:
    """What a learner sees and how each exercise grades, with generated ids and item order left out."""
    manifest = package.manifest
    out = {
        "gym": {key: manifest.get(key) for key in ("id", "title", "runtime", "order", "publish")},
        "description": " ".join(manifest.get("description", "").split()),
        "gpu": bool((manifest.get("requires") or {}).get("gpu")),
        "langs": [lang["id"] for lang in manifest.get("langs", [])] if any(
            package.meta(s).get("kind", "code") == "code" for s in package.slugs()) else [],
        "tracks": [(t["name"], t.get("summary", ""), t["exercises"]) for t in manifest.get("tracks", [])],
        "warmup": manifest.get("warmup", []),
        "sources": manifest.get("sources", []),
    }
    for slug in package.slugs():
        meta, key = package.meta(slug), package.answer_key(slug) or {}
        kind = meta.get("kind", "code")
        item = {
            "meta": {k: meta.get(k) for k in ("title", "difficulty", "summary", "minutes", "concepts", "placeholder", "min_words", "max_words")},
            "statement": _statement(package.statement(slug)),
            "hints": package.hints(slug),
            "editorial": package.editorial(slug),
            "explanation": str(key.get("explanation", "")).strip(),
        }
        if kind == "choice":
            item["choices"] = [(c["id"], c["text"], c["id"] in key["correct"]) for c in meta["choices"]]
            item["multiple"] = bool(meta.get("multiple"))
        elif kind == "order":
            text = {i["id"]: i["text"] for i in meta["items"]}
            item["order"] = [text[i] for i in key["order"]]
        elif kind == "match":
            options = {o["id"]: o["text"] for o in meta["options"]}
            item["pairs"] = [(p["text"], options[key["pairs"][p["id"]]]) for p in meta["prompts"]]
            item["options"] = sorted(options.values())
        elif kind == "fill":
            item["text"] = kinds.BLANK.sub("[[]]", meta["text"])
            item["blanks"] = [_spec(key["blanks"][b]) for b in kinds.BLANK.findall(meta["text"])]
            item["code"] = bool(meta.get("code"))
        elif kind == "answer":
            item["answer"] = _spec(key)
        elif kind == "essay":
            item["rubric"] = key["rubric"].strip()
            item["pass_score"] = key.get("pass_score", 0.7)
        elif kind == "checklist":
            item["items"] = meta["items"]
        elif kind == "code":
            directory = package.problem_dir(slug)
            item["files"] = {
                str(path.relative_to(directory)): path.read_text().replace("# GYM_STARTER\n", "", 1).rstrip()
                for path in sorted(directory.rglob("*"))
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
                and path.name not in {"problem.json", "PROBLEM.md", "HINTS.md", "answer.json", "EDITORIAL.md"}
            }
        out[slug] = (kind, item)
    return out


@pytest.mark.parametrize("path", [
    "examples/gyms/mixed", "examples/gyms/hello", "examples/gyms/gpu-scale", "gyms/ml-fundamentals", "gyms/llm-fundamentals",
    "gyms/ml-coding", "gyms/algo", "gyms/gpu", "gyms/llm",
])
def test_existing_gyms_round_trip(path, tmp_path):
    original = packages.load(REPO / path)
    text = gymmd.export(original)
    again = gymmd.build(text, tmp_path / original.id)
    assert _canonical(again) == _canonical(original)
    assert gymmd.export(again) == text, "exporting the compiled gym gives the same file"


def test_a_browser_draft_exports_as_tests(tmp_path):
    from gymhub import gymdraft

    original = gymdraft.write_package(gymdraft.EXAMPLE, tmp_path / "draft")
    text = gymmd.export(original)
    assert "solve(1, 2) == 3  # example" in text and "### Files" not in text
    again = gymmd.build(text, tmp_path / "again")
    assert json.loads((again.problem_dir("add_two") / "grader.json").read_text())["cases"] == \
        json.loads((original.problem_dir("add_two") / "grader.json").read_text())["cases"]


# ---- mistakes are explained, with lines --------------------------------------------------------


@pytest.mark.parametrize("body, line, message", [
    ("## Q\ndifficulty: easy\ncolour: red\n\nPick.\n\n### Choices\n- [x] a\n- [ ] b\n", 12, 'unknown field "colour"'),
    ("## Q\ndifficulty: easy\n\nPick.\n\n### Choices\n- [ ] a\n- [ ] b\n", 15, "mark the right choice"),
    ("## Q\ndifficulty: easy\n\nPick.\n\n### Choices\n- a\n- [x] b\n", 16, "- [x] right or - [ ] wrong"),
    ("## Q\ndifficulty: easy\n\nOrder.\n\n### Order\n1. a\n2. b\n", 10, "at least three"),
    ("## Q\ndifficulty: easy\n\nFill.\n\n### Blanks\nThe value $x = [[3]]$.\n", 10, "cannot sit inside $math$"),
    ("## Q\ndifficulty: easy\n\nType.\n\n### Answer\n- Paris ± 2\n", 15, "± works with numbers only"),
    ("## Q\ndifficulty: easy\n\nMatch.\n\n### Match\n| a | b |\n|---|---|\n| x | y |\n", 10, "at least two rows"),
    ("## Q\n\nNo difficulty.\n\n### Answer\n- 1\n", 10, "difficulty: easy, medium or hard"),
    ("## Q\ndifficulty: easy\n\nBoth.\n\n### Choices\n- [x] a\n\n### Answer\n- 1\n", 10, "sections of different kinds"),
    ("## Q\ndifficulty: easy\n\nCode.\n\n### Starter\n```python\ndef solve(a, b):\n    pass\n```\n\n### Tests\nsolve(1) == 2\n",
     22, "give exactly these inputs: solve(a, b)"),
    ("## Q\ndifficulty: easy\n\nCode.\n\n### Starter\n```python\ndef solve(a):\n    pass\n```\n\n### Tests\nsolve(x) == 2\n",
     22, "literal values only"),
    ("## Q\ndifficulty: easy\n\nCode.\n\n### Starter\n```python\ndef solve(a):\n    pass\n```\n\n### Tests\nsolve(1)\n",
     21, "needs a ### Solution"),
    ("## Q\ndifficulty: easy\n\nCode.\n\n### Starter\n```python\ndef solve(a):\n    pass\n```\n\n### Files\n```python file=../x.py\n```\n",
     22, "not a path this exercise may write"),
    ("## Q\ndifficulty: easy\n\nRead.\n\n### Rubric\nGood.\n\n### Explanation\nNo.\n", 18, "an essay exercise has no ### Explanation"),
])
def test_mistakes_are_explained_with_their_line(body, line, message):
    _, errors = gymmd.check(_gym(body))
    assert any(message in error["message"] and error["line"] == line for error in errors), errors


def test_gym_level_mistakes():
    assert "start with front matter" in gymmd.check("# Track\n\n## Q\n\nRead.\n")[1][0]["message"]
    errors = gymmd.check(_gym("## Q\n\nRead.\n", front="id: Bad_Id\ntitle: T"))[1]
    assert any("id must be 3-40 characters" in e["message"] for e in errors)
    errors = gymmd.check(_gym("## Q\n\nRead.\n\n## Q\n\nAgain.\n"))[1]
    assert errors[0]["line"] == 14 and "another exercise (line 10)" in errors[0]["message"]
    errors = gymmd.check(_gym("## Q\ndifficulty: easy\n\nCode.\n\n### Starter\n```cpp\nint solve();\n```\n\n### Files\n```python file=problem.py\n```\n"))[1]
    assert any("a cpp starter, but langs: is python" in e["message"] for e in errors)


def test_headings_inside_code_and_text_that_looks_like_a_field():
    gym, errors = gymmd.check(_gym(
        "## Q\ndifficulty: easy\n\nNote: this line is text.\n\n```python\n# not a track\n## not an exercise\n```\n\n### Steps\nA subheading.\n"))
    assert errors == []
    assert "Note: this line is text." in gym.exercises[0].statement and "## not an exercise" in gym.exercises[0].statement
    assert "### Steps" in gym.exercises[0].statement


def test_explicit_ids_are_kept():
    gym, errors = gymmd.check(_gym("## Q\ndifficulty: easy\n\nPick.\n\n### Choices\n- [x] Paris {#paris}\n- [ ] Lyon\n"
                                   "\n## F\ndifficulty: easy\n\nFill.\n\n### Blanks\nIt is [[5 {#five}]] and [[six]].\n"))
    assert errors == []
    assert gym.exercises[0].meta["choices"] == [{"id": "paris", "text": "Paris"}, {"id": "a", "text": "Lyon"}]
    assert gym.exercises[1].meta["text"] == "It is [[five]] and [[b1]]."


CODE = "## Q\ndifficulty: easy\n\nCode.\n\n### Starter\n```python\ndef solve(a):\n    pass\n```\n\n### Tests\n"


def test_an_unclosed_test_call_fails_fast():
    started = time.perf_counter()
    _, errors = gymmd.check(_gym(CODE + "solve([\n" + "1,\n" * 20_000))
    assert time.perf_counter() - started < 1
    assert any(e["line"] == 22 and "past 200 lines" in e["message"] for e in errors), errors
    started = time.perf_counter()
    _, errors = gymmd.check(_gym(CODE + "solve('''\n" + "x\n" * 20_000))
    assert time.perf_counter() - started < 1 and errors


def test_a_test_call_over_several_lines():
    gym, errors = gymmd.check(_gym(CODE + "solve([\n  1,  # one\n  2,\n]) == 3  # sum\nsolve(\"(\") == 1\nsolve('''a\n(]''') == 2\n"))
    assert errors == []
    assert gym.exercises[0].draft["cases"] == [
        {"name": "one", "args": {"a": [1, 2]}, "expected": 3},
        {"args": {"a": "("}, "expected": 1},
        {"args": {"a": "a\n(]"}, "expected": 2},
    ]
    _, errors = gymmd.check(_gym(CODE + "solve([1, 2) == 3\n"))
    assert any(e["line"] == 22 and "left open" in e["message"] for e in errors), errors


def test_big_sections_compile_in_linear_time():
    rows = "".join(f"| p{i} | o{i} |\n" for i in range(20_000))
    started = time.perf_counter()
    _, errors = gymmd.check(_gym(f"## Q\ndifficulty: easy\n\nMatch.\n\n### Match\n{rows}"))
    assert errors == [] and time.perf_counter() - started < 5
    started = time.perf_counter()
    _, errors = gymmd.check(_gym("## Q\ndifficulty: easy\n\nFill.\n\n### Blanks\n" + "[[" * 500_000 + "\n"))
    assert time.perf_counter() - started < 5


def test_list_items_keep_indented_code_and_nested_lists(tmp_path):
    hints = ["Try a loop:\n\n    for i in range(n):\n        total += i", "Two steps:\n\n- first\n  - deeper\n- second"]
    choice = "Paris, which:\n- is a capital\n- is in France"
    body = ("## Q\ndifficulty: easy\n\nPick.\n\n### Hints\n"
            "- Try a loop:\n\n      for i in range(n):\n          total += i\n"
            "- Two steps:\n\n  - first\n    - deeper\n  - second\n\n"
            "### Choices\n- [x] Paris, which:\n  - is a capital\n  - is in France\n- [ ] Lyon\n")
    gym, errors = gymmd.check(_gym(body))
    assert errors == [], errors
    assert gym.exercises[0].hints == hints
    assert gym.exercises[0].meta["choices"][0]["text"] == choice
    text = gymmd.export(gymmd.build(_gym(body), tmp_path / "a"))
    again = gymmd.build(text, tmp_path / "b")
    assert again.hints("q") == hints and again.meta("q")["choices"][0]["text"] == choice
    assert gymmd.export(again) == text


@pytest.mark.parametrize("heading, suggestion", [
    ("Explanations", "did you mean ### Explanation?"), ("Answer key", ""), ("Hint", "did you mean ### Hints?"),
])
def test_an_unknown_section_is_an_error_not_statement_text(heading, suggestion):
    body = f"## Q\ndifficulty: easy\n\nPick.\n\n### Choices\n- [x] a\n- [ ] b\n\n### {heading}\nThe answer is a.\n"
    _, errors = gymmd.check(_gym(body))
    error = next(e for e in errors if f"### {heading} is not a section" in e["message"])
    assert error["line"] == 19 and suggestion in error["message"] and "####" in error["message"]
    body = f"## Q\ndifficulty: easy\n\nPick.\n\n#### {heading}\nA sub-heading.\n\n### Choices\n- [x] a\n- [ ] b\n"
    gym, errors = gymmd.check(_gym(body))
    assert errors == [] and f"#### {heading}" in gym.exercises[0].statement


def test_an_unclosed_code_fence_is_an_error():
    _, errors = gymmd.check(_gym("## Q\n\nRead.\n\n```python\nx = 1\n\n## Another\n\nText.\n"))
    assert any(e["line"] == 14 and "never closes" in e["message"] for e in errors), errors
    _, errors = gymmd.check(_gym(CODE.replace("    pass\n```\n", "    pass\n") + "solve(1) == 1\n"))
    assert any(e["line"] == 16 and "never closes" in e["message"] for e in errors), errors


# ---- the CLI and the board ---------------------------------------------------------------------


def test_cli_check_json(tmp_path, capsys):
    path = tmp_path / "x.gym.md"
    path.write_text(_gym("## Q\ndifficulty: easy\n\nPick.\n\n### Choices\n- [ ] a\n- [ ] b\n"))
    assert cli.main(["gym", "check", str(path), "--json"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is False and result["errors"][0]["line"] == 15
    assert cli.main(["gym", "new", str(tmp_path / "fresh-gym.gym.md")]) == 0
    capsys.readouterr()
    assert cli.main(["gym", "check", str(tmp_path / "fresh-gym.gym.md"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["gym"] == "fresh-gym"
    assert cli.main(["gym", "export", str(REPO / "examples/gyms/hello"), "-o", str(tmp_path / "hello.gym.md")]) == 0
    assert cli.main(["gym", "build", str(tmp_path / "hello.gym.md"), "-o", str(tmp_path / "built")]) == 0
    assert packages.load(tmp_path / "built").id == "hello"


def test_the_board_takes_a_gym_markdown_upload(board, people):
    ann = people["ann"]
    text = EXAMPLE.read_text()
    reply = ann.post("/api/gym-versions", content=text.encode(), headers={**H, "Content-Type": "text/markdown"})
    assert reply.status_code == 200, reply.text
    assert reply.json()["version"]["gym"] == "every-kind"
    from tests.test_authoring import _validate_next

    _validate_next(board, people["owner"])  # published; its gym.json has no "order"
    assert board.store.get("gyms", "every-kind")["order"] == 99  # not None: lists of gyms sort by it
    assert ann.get("/api/home").status_code == 200
    assert [g for g in ann.get("/api/home").json()["gyms"] if g["id"] == "every-kind"]
    broken = text.replace("- [x] 2\n", "- 2\n")
    reply = ann.post("/api/gym-versions", content=broken.encode(), headers={**H, "Content-Type": "text/markdown"})
    assert reply.status_code == 400
    assert any(error.startswith("line ") and "primes" in error for error in reply.json()["errors"])
    guide = ann.get("/docs/gym-markdown")
    assert guide.status_code == 200 and guide.text.startswith("# Gym Markdown")


def test_sources_are_link_lines_under_the_description(tmp_path):
    text = ("---\nid: with-sources\ntitle: With sources\n---\n\nA gym about things.\n\n"
            "- [A book](https://example.com/book): where the topics come from\n- [A paper](https://arxiv.org/abs/1)\n\n"
            "# Track\n\n## Q\ndifficulty: easy\n\nPick.\n\n### Choices\n- [x] a\n- [ ] b\n")
    package = gymmd.build(text, tmp_path / "g")
    assert package.manifest["description"] == "A gym about things."
    assert package.manifest["sources"] == [
        {"title": "A book", "url": "https://example.com/book", "note": "where the topics come from"},
        {"title": "A paper", "url": "https://arxiv.org/abs/1"},
    ]
    assert gymmd.export(package).count("- [A") == 2
    with pytest.raises(gymmd.GymMarkdownError, match="source: url"):
        gymmd.compile_text(text.replace("https://arxiv.org/abs/1", "ftp://x"))
