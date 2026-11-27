"""Gym packages: the three gyms are valid packages; the public catalog never carries grader content."""

import json
import shutil
from pathlib import Path

import pytest

from gymhub import packages

PUBLIC_PROBLEM_KEYS = {"slug", "title", "difficulty", "summary", "kind", "track", "order", "warmup", "statement", "hints", "editorial", "langs", "starters", "tests_hash"}


@pytest.mark.parametrize("gym_id", ["gpu", "llm", "algo"])
def test_repo_gyms_are_valid_packages(gym_id):
    package = packages.load(gym_id)
    assert packages.validate(package) == []
    assert len(package.slugs()) >= 7  # llm has 7 since bpe_train (pure Python) moved to algo (#61)


def test_problem_json_matches_each_grader():
    from gpu_gym.catalog import get_problem
    from gymkit.gym import load_gym

    for package in packages.discover():
        for slug in package.slugs():
            meta = package.meta(slug)
            declared = get_problem(slug) if package.runtime == "gpu_gym" else load_gym(package.id).load(slug)
            assert (meta["title"], meta["difficulty"], meta["summary"]) == (declared.title, declared.difficulty, declared.summary), slug


def test_public_catalog_has_no_grader_content():
    for package in packages.discover():
        catalog = packages.public_catalog(package)
        text = json.dumps(catalog)
        for problem in catalog["problems"]:
            assert set(problem) == PUBLIC_PROBLEM_KEYS
            for grader in package.grader_files(problem["slug"]):
                if grader.suffix in {".py", ".cpp"}:
                    body = grader.read_text()
                    # a distinctive chunk of every grader file must not appear anywhere in what the board gets
                    chunk = max(body.splitlines(), key=len).strip()
                    if len(chunk) > 30:
                        assert chunk not in text, f"{grader} leaked into the public catalog"
        assert all(f.name != "problem.py" for slug in package.slugs() for f in package.public_files(slug))


def test_validator_reports_broken_packages(tmp_path):
    source = packages.load("algo")
    root = tmp_path / "broken"
    shutil.copytree(source.root, root, ignore=shutil.ignore_patterns("solutions", "progress.json", "__pycache__"))
    manifest = json.loads((root / "gym.json").read_text())
    manifest["tracks"][0]["exercises"].append("does_not_exist")
    manifest["schema"] = 2
    (root / "gym.json").write_text(json.dumps(manifest))
    (root / "problems" / "merge_intervals" / "PROBLEM.md").unlink()
    (root / "problems" / "dijkstra" / "problem.py").unlink()
    (root / "problems" / "edit_distance" / "cpp" / "harness.cpp").unlink()
    meta = root / "problems" / "lru_cache" / "problem.json"
    meta.write_text(json.dumps({**json.loads(meta.read_text()), "difficulty": "brutal"}))
    shutil.rmtree(root / "problems" / "union_find" / "starter")
    errors = "\n".join(packages.validate(packages.load(root)))
    for expected in ("schema must be 1", "'does_not_exist'", "merge_intervals: PROBLEM.md", "dijkstra: no grader",
                     "edit_distance: has a C++ starter but no cpp/harness.cpp", "difficulty must be", "union_find: no starter"):
        assert expected in errors, expected


def test_gymkit_leaves_other_runtimes_alone():
    from gymkit.gym import gym_ids, load_gym

    assert "gpu" not in gym_ids()
    with pytest.raises(SystemExit, match="gpu_gym"):
        load_gym("gpu")


def test_a_gpu_gym_holds_no_pure_python_exercise():
    """#61: an exercise that never touches a tensor does not belong in a gym that requires a GPU."""
    for package in packages.discover():
        if not (package.manifest.get("requires") or {}).get("gpu"):
            continue
        for slug in package.slugs():
            grader = (package.problem_dir(slug) / "problem.py").read_text()
            assert "torch" in grader or "cuda" in grader, f"{package.id}/{slug} does not use the GPU"


SHARED_TOPICS = [
    ("llm", "causal_attention"), ("llm", "cross_entropy"), ("llm", "kv_cache_decode"), ("llm", "lora_linear"),
    ("llm", "sampling_filter"), ("algo", "bpe_train"), ("ml-coding", "causal_attention_mask"),
    ("ml-coding", "scaled_dot_product_attention"), ("ml-coding", "softmax_cross_entropy"), ("ml-coding", "kv_cache"),
    ("ml-coding", "lora_linear"), ("ml-coding", "top_k_top_p_sampling"), ("ml-coding", "byte_pair_encoding"),
    ("gpu", "cross_entropy"), ("gpu", "causal_attention"), ("gpu", "rms_norm"), ("gpu", "rope"),
]


@pytest.mark.parametrize("gym, slug", SHARED_TOPICS)
def test_each_shared_topic_says_how_its_version_differs(gym, slug):
    statement = (Path(__file__).resolve().parents[1] / "gyms" / gym / "problems" / slug / "PROBLEM.md").read_text()
    assert "## This version and the other one" in statement


def test_every_repo_gym_lists_its_sources_and_the_catalog_carries_them():
    for package in packages.discover():
        assert package.manifest.get("sources"), f"{package.id}: list what the gym is based on in gym.json sources"
        shown = packages.public_catalog(package)["gym"]["sources"]
        assert shown and all(set(item) == {"title", "url", "note"} for item in shown)


@pytest.mark.parametrize("source, message", [
    ({"title": "A", "url": "javascript:alert(1)"}, "url is an http"),
    ({"title": "", "url": "https://example.com"}, "title is 1-120"),
    ({"title": "A", "url": "https://example.com", "colour": "red"}, "unknown keys"),
    ("https://example.com", "each source is"),
])
def test_bad_sources_are_reported(tmp_path, source, message):
    root = tmp_path / "gym"
    shutil.copytree(packages.load("algo").root, root, ignore=shutil.ignore_patterns("solutions", "progress.json", "__pycache__"))
    manifest = json.loads((root / "gym.json").read_text())
    manifest["sources"] = [source]
    (root / "gym.json").write_text(json.dumps(manifest))
    package = packages.load(root)
    assert any(message in error for error in packages.validate(package))
    assert packages.public_catalog(package)["gym"]["sources"] == []
