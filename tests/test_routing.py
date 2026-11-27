"""Routing: machines lease only jobs they can run; own machines first; best times per hardware."""

import time

from gymhub import jobqueue, routing
from tests.test_board_auth import H, board, client, shared_token, sign_in  # noqa: F401  (fixture)

GPU_BOX = {"arch": "aarch64", "gpus": [{"name": "NVIDIA GB10", "compute_capability": "12.1"}], "tools": ["nvcc", "g++"]}
CPU_BOX = {"arch": "x86_64", "gpus": [], "tools": ["g++"]}


def test_rules():
    gpu_cuda = routing.requirements("gpu", {"requires": {"gpu": True}}, "cuda")
    assert gpu_cuda == {"gpu": True, "tools": ["nvcc"]}
    assert routing.requirements("llm", None, "pytorch")["gpu"] is True  # fallback for older gym documents
    assert routing.requirements("algo", {"requires": {"gpu": False}}, "cpp") == {"gpu": False, "tools": ["g++"]}
    assert routing.unmet(CPU_BOX, gpu_cuda) == "an NVIDIA GPU"
    assert routing.unmet({**GPU_BOX, "tools": []}, gpu_cuda) == "the CUDA toolkit (nvcc)"
    assert routing.unmet(GPU_BOX, gpu_cuda) is None
    assert routing.hardware(GPU_BOX) == "NVIDIA GB10" and routing.hardware(CPU_BOX) == "CPU x86_64"


def _queue(board, **fields):
    job_id = f"j{time.time_ns()}"
    board.store.set("jobs", job_id, {"gym": "gpu", "slug": "vector_add", "lang": "cuda", "action": "bench", "status": "queued",
                                     "created_at": time.time() - 60, "uid": "g_9", "requires": {"gpu": True, "tools": ["nvcc"]}, **fields})
    return job_id


def test_cpu_machine_skips_gpu_jobs_and_hardware_is_recorded(board):
    job = _queue(board)
    assert jobqueue.lease(board.store, ["gpu"], "cpu-box", capabilities=CPU_BOX) is None
    leased = jobqueue.lease(board.store, ["gpu"], "gpu-box", capabilities=GPU_BOX)
    assert leased["id"] == job and leased["hardware"] == "NVIDIA GB10"


def test_best_times_are_kept_per_hardware(board):
    for ms, hardware in ((0.90, "NVIDIA GB10"), (0.85, "NVIDIA GB10"), (0.95, "NVIDIA GB10"), (0.40, "NVIDIA RTX 4090")):
        job = _queue(board)
        caps = {**GPU_BOX, "gpus": [{"name": hardware}]}
        leased = jobqueue.lease(board.store, ["gpu"], "w", capabilities=caps)
        jobqueue.complete(board.store, job, "w", {"status": "done", "ok": True, "output": f"  time    {ms:.4f} ms median   min ...\n"},
                          lease_id=leased["lease_id"])
    best = board.store.get("bests", "g_9__gpu__vector_add__cuda")["by_hardware"]
    assert best["NVIDIA GB10"]["ms"] == 0.85 and best["NVIDIA RTX 4090"]["ms"] == 0.40


def test_own_machine_gets_a_head_start(board):
    owner = client(board)
    sign_in(owner, "owner")
    owner.put("/api/admin/settings", json={"allowed_domains": ["example.com"], "members_can_run": True}, headers=H)
    ann = client(board)
    sign_in(ann, "ann")
    ann_uid = ann.get("/api/me").json()["user"]["uid"]
    shared_secret = shared_token(owner, "shared")
    code = ann.post("/api/my/workers/registration", json={"name": "anns"}, headers=H).json()["code"]
    own_token = client(board).post("/v1/register", json={"code": code}).json()["token"]
    shared, own = client(board), client(board)
    shared.headers.update({"Authorization": f"Bearer {shared_secret}"})
    own.headers.update({"Authorization": f"Bearer {own_token}"})
    shared.post("/v1/worker/heartbeat", json={"gyms": ["algo"], "capabilities": {**CPU_BOX, "sandbox": True}})
    own.post("/v1/worker/heartbeat", json={"gyms": ["algo"], "capabilities": CPU_BOX})  # own jobs need no sandbox
    job = ann.post("/api/jobs", json={"gym": "algo", "slug": "p", "lang": "python", "action": "run", "code": "x"}, headers=H).json()
    assert "warning" not in job
    assert shared.post("/v1/worker/lease", json={"gyms": ["algo"]}).json()["job"] is None  # Ann's machine is online
    assert own.post("/v1/worker/lease", json={"gyms": ["algo"]}).json()["job"]["id"] == job["id"]
    # after the head start, a shared machine takes it
    late = ann.post("/api/jobs", json={"gym": "algo", "slug": "p", "lang": "python", "action": "run", "code": "y"}, headers=H).json()["id"]
    board.store.set("jobs", late, {"created_at": time.time() - 60}, merge=True)
    assert shared.post("/v1/worker/lease", json={"gyms": ["algo"]}).json()["job"]["id"] == late
    del ann_uid


def test_warning_when_no_machine_can_run_it(board):
    owner = client(board)
    sign_in(owner, "owner")
    token = owner.post("/api/workers", json={"name": "cpu"}, headers=H).json()["token"]
    worker = client(board)
    worker.headers.update({"Authorization": f"Bearer {token}"})
    worker.post("/v1/worker/heartbeat", json={"gyms": ["gpu", "algo"], "capabilities": CPU_BOX})
    board.store.set("gyms", "gpu", {"title": "GPU", "updated_at": 0})
    board.store.set("problems", "gpu__v", {"gym": "gpu", "slug": "v", "title": "V", "langs": ["cuda"]})
    reply = owner.post("/api/jobs", json={"gym": "gpu", "slug": "v", "lang": "cuda", "action": "run", "code": "x"}, headers=H).json()
    assert "needs an NVIDIA GPU" in reply["warning"]
