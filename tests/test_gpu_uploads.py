"""Uploading a GPU gym (#60): accepted, validated and run only on machines with a GPU."""

import json
import shutil
import time
from pathlib import Path

import pytest

from gymhub import packages
from tests.test_board_auth import H, board, client, sign_in  # noqa: F401  (fixture)
from tests.test_gym_versions import GZ, _owner, complete_validation

REPO = Path(__file__).resolve().parents[1]
GPU_SCALE = REPO / "examples" / "gyms" / "gpu-scale"
CPU_BOX = {"gpus": [], "tools": ["g++"], "sandbox": True, "bundles": True}
GPU_BOX = {"gpus": [{"name": "NVIDIA GB10"}], "tools": ["g++", "nvcc"], "sandbox": True, "bundles": True}


def _machine(board, owner, caps):
    token = owner.post("/api/workers", json={"name": f"m{time.time_ns()}"}, headers=H).json()["token"]
    machine = client(board)
    machine.headers.update({"Authorization": f"Bearer {token}"})
    assert machine.post("/v1/worker/heartbeat", json={"gyms": [], "capabilities": caps}).status_code == 200
    import gymhub.board.workers as workers

    workers._capabilities.clear()
    return machine


def test_a_gpu_gym_package_is_accepted_for_upload():
    package = packages.load(GPU_SCALE)
    assert packages.validate_upload(package) == []
    assert packages.requirements(package) == {"gpu": True, "tools": ["nvcc"]}


def test_gpu_lang_files_match_gpu_gym():
    pytest.importorskip("torch")
    from gpu_gym.catalog import LANG_FILES

    assert packages.GPU_LANG_FILES == LANG_FILES


@pytest.mark.parametrize("change, message", [
    (lambda m: m.update(requires={"gpu": False}), '"gpu": true'),
    (lambda m: m["langs"].append({"id": "python", "label": "Python", "editor": "python", "file": "solution.py"}), "not 'python'"),
    (lambda m: m["langs"][0].update(file="kernel.cu"), "is solution.cu"),
])
def test_a_gpu_gym_must_declare_its_gpu_and_use_gpu_gym_files(tmp_path, change, message):
    root = tmp_path / "gpu-scale"
    shutil.copytree(GPU_SCALE, root)
    manifest = json.loads((root / "gym.json").read_text())
    change(manifest)
    (root / "gym.json").write_text(json.dumps(manifest))
    errors = packages.validate_upload(packages.load(root))
    assert any(message in error for error in errors), errors


def test_only_a_gpu_machine_validates_and_runs_a_gpu_gym(board):
    owner = _owner(board)
    reply = owner.post("/api/gym-versions", content=packages.pack(GPU_SCALE), headers=GZ)
    assert reply.status_code == 200, reply.text
    job = board.store.get("jobs", reply.json()["version"]["job"])
    assert job["requires"] == {"gpu": True, "tools": ["nvcc"]}
    cpu = _machine(board, owner, CPU_BOX)
    assert cpu.post("/v1/worker/lease", json={"gyms": []}).json()["job"] is None
    gpu = _machine(board, owner, GPU_BOX)
    leased = gpu.post("/v1/worker/lease", json={"gyms": []}).json()["job"]
    assert leased["action"] == "validate"
    assert complete_validation(gpu, leased).status_code == 200
    assert board.store.get("gyms", "gpu-scale")["version"] == 1
    # Learner jobs on the published gym need a GPU too.
    code = (GPU_SCALE / "problems/scale/solution/solution.py").read_text()
    queued = owner.post("/api/jobs", json={"gym": "gpu-scale", "slug": "scale", "lang": "pytorch", "action": "test", "code": code}, headers=H)
    assert queued.status_code == 200, queued.text
    assert board.store.get("jobs", queued.json()["id"])["requires"]["gpu"] is True
    assert cpu.post("/v1/worker/lease", json={"gyms": []}).json()["job"] is None
    assert gpu.post("/v1/worker/lease", json={"gyms": []}).json()["job"]["id"] == queued.json()["id"]


def _cuda_ready() -> bool:
    try:
        import torch

        return torch.cuda.is_available() and (shutil.which("nvcc") is not None or Path("/usr/local/cuda/bin/nvcc").is_file())
    except Exception:
        return False


@pytest.mark.skipif(not _cuda_ready(), reason="needs a CUDA GPU and nvcc")
def test_a_gpu_runner_validates_publishes_and_grades_an_uploaded_gpu_gym(board, tmp_path, monkeypatch):
    from gymhub.runner.agent import Runner
    from gymhub.runner.backends import DirectBackend
    import gymhub.runner.bundles as bundles

    monkeypatch.setenv("GYMHUB_TRUST_JOBS", "1")
    monkeypatch.setattr(bundles, "ROOT", tmp_path / "bundles")
    runner = Runner(DirectBackend(board.store, "spark"), {}, "spark")
    owner = _owner(board)
    assert owner.post("/api/gym-versions", content=packages.pack(GPU_SCALE), headers=GZ).status_code == 200
    job = runner.backend.lease([])
    runner.run_job(job)
    done = board.store.get("jobs", job["id"])
    assert done["ok"], done["output"]
    assert board.store.get("gym_versions", "gpu-scale__v0001")["status"] == "published"
    for lang, name in (("pytorch", "solution.py"), ("cuda", "solution.cu")):
        code = (GPU_SCALE / "problems/scale/solution" / name).read_text()
        reply = owner.post("/api/jobs", json={"gym": "gpu-scale", "slug": "scale", "lang": lang, "action": "test", "code": code}, headers=H)
        graded = runner.backend.lease([])
        assert graded["id"] == reply.json()["id"]
        runner.run_job(graded)
        result = board.store.get("jobs", graded["id"])
        assert result["ok"], result["output"]
