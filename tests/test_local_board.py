"""A local board on a machine without PyTorch or a GPU: every gym is listed, and the courses are there.

    catalog   a gym whose graders import a module this machine lacks is still published, without the
              values only its grader computes, and with a note on the gym that says what to install
    courses   a direct runner on a local store seeds the repository's courses (gymhub/courses.py)
"""

import json
import shutil
from argparse import Namespace
from pathlib import Path

import pytest

from gymhub import courses, gymversions, packages
from gymhub.blobs import FileBlobs
from gymhub.runner import agent, catalog
from gymhub.store import SqliteStore

HELLO = Path(__file__).resolve().parents[1] / "examples" / "gyms" / "hello"


def _course(gyms: Path, gym_id: str = "hello") -> Path:
    """examples/gyms/hello as an upload-only course in a gyms/ folder of its own."""
    root = gyms / gym_id
    shutil.copytree(HELLO, root)
    manifest = json.loads((root / "gym.json").read_text())
    (root / "gym.json").write_text(json.dumps({**manifest, "id": gym_id, "publish": "upload"}))
    return root


@pytest.fixture
def local(tmp_path):
    return SqliteStore(str(tmp_path / "dev.db")), FileBlobs(tmp_path / "blobs")


# ---- catalog without a grader's dependencies ------------------------------------------------------


def test_catalog_without_a_module_still_lists_every_exercise(tmp_path):
    root = tmp_path / "hello"
    shutil.copytree(HELLO, root)
    grader = root / "problems" / "sum_list" / "problem.py"
    grader.write_text("import torch_that_is_not_installed  # noqa\n" + grader.read_text())

    built = catalog.build(root)

    slugs = {item["slug"]: item for item in built["problems"]}
    assert set(slugs) == {"sum_list", "is_palindrome"}
    assert "case_template" not in slugs["sum_list"]  # only its grader could have built one
    assert "case_template" in slugs["is_palindrome"]
    assert "torch_that_is_not_installed" in built["gym"]["machine_note"]


def test_catalog_with_every_module_has_no_note(tmp_path):
    root = tmp_path / "hello"
    shutil.copytree(HELLO, root)
    assert catalog.build(root)["gym"]["machine_note"] == ""


def test_machine_note_names_pytorch_and_the_missing_gpu():
    note = catalog.machine_note({"torch"}, needs_gpu=True, has_gpu=False)
    assert "PyTorch" in note and "restart the runner" in note
    assert "NVIDIA GPU" in note
    assert catalog.machine_note(set(), needs_gpu=True, has_gpu=True) == ""
    assert catalog.machine_note(set(), needs_gpu=False, has_gpu=False) == ""


def test_gpu_gym_is_published_without_pytorch():
    """Without PyTorch (as in CI), the GPU gym still has every exercise; with it, no note about it."""
    built = catalog.build("gpu")
    assert len(built["problems"]) == len(packages.load(packages.GYMS / "gpu").slugs())
    assert all(item["benchable"] for item in built["problems"])
    try:
        import torch  # noqa: F401
    except ImportError:
        assert "PyTorch" in built["gym"]["machine_note"]
    else:
        assert "PyTorch" not in built["gym"]["machine_note"]


# ---- the repository's courses on a local board ----------------------------------------------------


def test_seed_publishes_a_course_and_its_answer_keys(tmp_path, local):
    store, blobs = local
    _course(tmp_path / "gyms")

    assert courses.seed(store, blobs, log=lambda _: None, gyms_dir=tmp_path / "gyms") == {"hello": "published"}

    gym = store.get("gyms", "hello")
    assert gym["source"] == "bundle" and gym["version"] == 1 and gym["visibility"] == "public"
    assert gym["problems"] == 2
    version = store.get("gym_versions", gymversions.version_id("hello", 1))
    assert version["status"] == "published" and version["source"] == courses.SOURCE
    job = store.get("jobs", version["job"])
    assert job["status"] == "done" and job["ok"] is True
    problem = store.get("problems", "hello__sum_list")
    assert problem["case_template"] is not None  # what the validation computed reached the catalog


def test_seed_again_changes_nothing_until_the_files_change(tmp_path, local):
    store, blobs = local
    root = _course(tmp_path / "gyms")
    seed = lambda: courses.seed(store, blobs, log=lambda _: None, gyms_dir=tmp_path / "gyms")  # noqa: E731
    seed()

    assert seed() == {"hello": "skipped"}

    statement = root / "problems" / "sum_list" / "PROBLEM.md"
    statement.write_text(statement.read_text() + "\nOne more line.\n")
    assert seed() == {"hello": "published"}
    assert store.get("gyms", "hello")["version"] == 2


def test_seed_leaves_a_gym_someone_uploaded_alone(tmp_path, local):
    store, blobs = local
    root = _course(tmp_path / "gyms")
    mine = gymversions.upload(store, blobs, packages.pack(root), "u1", "a@example.com")
    gymversions.finish(store, blobs, store.get("jobs", mine["job"]), {
        "ok": True, "result": {"ok": True, "checks": [
            {"slug": slug, "lang": lang, "what": what, "ok": True}
            for slug, lang, what in packages.required_checks(packages.load(root))]},
    })
    assert store.get("gyms", "hello")["owner_uid"] == "u1"

    statement = root / "problems" / "sum_list" / "PROBLEM.md"
    statement.write_text(statement.read_text() + "\nChanged in the repository.\n")
    assert courses.seed(store, blobs, log=lambda _: None, gyms_dir=tmp_path / "gyms") == {"hello": "skipped"}
    assert store.get("gyms", "hello")["version"] == 1


def test_seed_records_a_course_that_fails_validation(tmp_path, local):
    store, blobs = local
    root = _course(tmp_path / "gyms")
    reference = root / "problems" / "sum_list" / "solution" / "solution.py"
    reference.write_text(reference.read_text().replace("return", "return 1 + ", 1))

    assert courses.seed(store, blobs, log=lambda _: None, gyms_dir=tmp_path / "gyms") == {"hello": "failed"}

    assert store.get("gyms", "hello") is None
    assert store.get("gym_versions", gymversions.version_id("hello", 1))["status"] == "failed"


def test_the_repository_has_courses_to_seed():
    assert sorted(package.id for package in courses.courses()) == ["gpu-interview", "llm-fundamentals", "ml-coding", "ml-fundamentals", "ml-system-design", "pytorch-numpy"]


@pytest.mark.parametrize(
    ("store", "api", "no_courses", "gyms", "expected"),
    [
        ("sqlite:/tmp/x.db", None, False, None, []),  # every course
        ("sqlite:/tmp/x.db", None, False, ["algo", "ml-coding"], ["ml-coding"]),
        ("sqlite:/tmp/x.db", None, False, ["algo"], None),
        ("sqlite:/tmp/x.db", None, True, None, None),  # --no-courses
        ("sqlite:/tmp/x.db", "https://board", False, None, None),  # worker API: a hosted board
        ("firestore:project", None, False, None, None),
    ],
)
def test_runner_seeds_courses_only_on_a_local_store(monkeypatch, store, api, no_courses, gyms, expected):
    monkeypatch.setenv("GYMHUB_STORE", store)
    assert agent._local_courses(Namespace(api=api, no_courses=no_courses), gyms) == expected


def test_machine_note_when_the_gpu_is_there_but_unusable():
    """A GPU machine whose PyTorch cannot see CUDA, or that has no nvcc, says so instead of failing runs."""
    note = catalog.machine_note(set(), needs_gpu=True, has_gpu=True, torch_sees_gpu=False)
    assert "PyTorch cannot use it" in note and "./setup.sh" in note
    note = catalog.machine_note(set(), needs_gpu=True, has_gpu=True, needs_nvcc=True)
    assert "CUDA toolkit (nvcc)" in note and "PyTorch and Triton" in note and "PyTorch cannot" not in note
    # No GPU at all: only the GPU note, not the toolkit or PyTorch-build ones.
    note = catalog.machine_note(set(), needs_gpu=True, has_gpu=False, torch_sees_gpu=False, needs_nvcc=True)
    assert "nvcc" not in note and "cannot use it" not in note


def test_gpu_gym_note_names_the_missing_toolkit(monkeypatch):
    """GPU present (nvidia-smi), no nvcc: the GPU gym's page says CUDA solutions wait for the toolkit."""
    fake = {"os": "Linux", "arch": "x86_64", "cpus": 4, "memory_gb": 16, "driver": "580.95", "cuda": None,
            "gpus": [{"name": "NVIDIA GeForce RTX 4090", "compute_capability": "8.9", "memory_gb": 24}], "tools": []}
    monkeypatch.setattr("gymhub.runner.capabilities.probe", lambda: fake)
    monkeypatch.setattr(catalog, "_torch_sees_gpu", lambda: True)
    note = catalog.build("gpu")["gym"]["machine_note"]
    if "PyTorch (" in note:  # no PyTorch on this machine (CI): that note comes first and is enough
        return
    assert "CUDA toolkit (nvcc)" in note
