"""What an isolated job sees of its gym (gymhub/runner/slices.py): no answer keys, no reference solutions."""

from gymhub.runner import slices


def test_the_slice_has_no_answers(tmp_path, monkeypatch):
    monkeypatch.setattr(slices, "SLICES", tmp_path / "slices")
    gym = slices.REPO / "gyms" / "ml-fundamentals"
    view = slices.gym_slice(gym)
    assert view.is_relative_to(tmp_path / "slices")
    assert (view / "gym.json").is_file() and (view / "problems").is_dir()
    assert not list(view.rglob("answer.json")) and not list(view.glob("problems/*/solution"))
    assert not (view / "solutions").exists() and not list(view.rglob("progress.json"))
    assert list(gym.rglob("answer.json"))  # the real gym still has its keys


def test_the_slice_follows_edits_and_keeps_the_checker(tmp_path, monkeypatch):
    monkeypatch.setattr(slices, "SLICES", tmp_path / "slices")
    gym = tmp_path / "g"
    (gym / "problems" / "p" / "solution").mkdir(parents=True)
    (gym / "problems" / "p" / "starter").mkdir()
    (gym / "gym.json").write_text("{}")
    (gym / "problems" / "p" / "problem.py").write_text("def check(): pass\n")
    (gym / "problems" / "p" / "starter" / "solution.py").write_text("# start\n")
    (gym / "problems" / "p" / "solution" / "solution.py").write_text("print('the answer')\n")
    (gym / "problems" / "p" / "answer.json").write_text('{"answers": ["42"]}')
    first = slices.gym_slice(gym)
    assert (first / "problems" / "p" / "problem.py").is_file() and (first / "problems" / "p" / "starter" / "solution.py").is_file()
    assert not (first / "problems" / "p" / "solution").exists() and not (first / "problems" / "p" / "answer.json").exists()
    assert slices.gym_slice(gym) == first  # unchanged: reused
    (gym / "problems" / "p" / "problem.py").write_text("def check(): return 1\n")
    second = slices.gym_slice(gym)
    assert second != first and "return 1" in (second / "problems" / "p" / "problem.py").read_text()


def test_the_worker_bundle_has_no_answer_keys_or_reference_solutions():
    from gymhub import bundle

    names = [str(path.relative_to(bundle.REPO)) for path in bundle.files()]
    assert any(name.endswith("/problem.py") for name in names)  # checkers still ship
    assert not [n for n in names if n.endswith("/answer.json")]
    assert not [n for n in names if n.startswith("gyms/") and "/solution/" in n]
