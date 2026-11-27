"""The bridge between the worker API and pool grading (issue #88, M4).

A runner with GYMHUB_EXECUTOR=pool reports what the pool can grade (`capabilities["pool"]`), and the
board routes isolated jobs to it by the pool's capabilities. Existing workers send no "pool" and are
routed exactly as before, so both kinds of machine serve one queue through the same worker API.
"""

import time

from gymhub import routing
from gymhub.runner import pooled
from gymhub.runner.adapters import kit_adapter
from gymhub.runner.backends import ApiBackend
from tests.test_board_auth import H, board, client, shared_token, sign_in  # noqa: F401  (fixture)
from tests.test_pool import db_url, pool_env, start_agent  # noqa: F401  (fixtures)
from tests.test_pooled_gym import MERGE_INTERVALS, pool_settings  # noqa: F401  (fixture)

POOL = {"actions": ["bench", "run", "test"], "machines": 1, "gpus": [{"name": "NVIDIA GB10"}], "tools": ["g++", "nvcc"]}
BRIDGE = {"gpus": [], "tools": [], "sandbox": False, "bundles": True, "arch": "x86_64", "pool": POOL}  # no GPU, no Docker
LEGACY_CPU = {"gpus": [], "tools": [], "sandbox": False, "bundles": True, "arch": "x86_64"}  # what an existing worker sends


def test_routing_follows_the_pool_only_for_the_jobs_it_grades():
    member_test = {"isolated": True, "action": "test", "requires": {"gpu": True, "tools": ["nvcc"]}, "gym_public": True}
    assert routing.on_pool(BRIDGE, member_test)
    assert routing.unmet(routing.for_job(BRIDGE, member_test), member_test["requires"]) is None
    assert routing.for_job(BRIDGE, member_test)["sandbox"] is True
    # the owner's own job runs in place, on this machine's hardware
    own = {**member_test, "isolated": False}
    assert not routing.on_pool(BRIDGE, own) and routing.unmet(routing.for_job(BRIDGE, own), own["requires"]) == "an NVIDIA GPU"
    # an action the bridge keeps local; an uploaded gym's job goes to the pool even when not isolated
    assert not routing.on_pool({**BRIDGE, "pool": {**POOL, "actions": ["run", "test"]}}, {**member_test, "action": "bench"})
    assert routing.on_pool(BRIDGE, {"bundle": {"sha": "a" * 64}, "action": "test", "gym_public": True})
    # an existing worker, and an older one that reports no capabilities at all
    assert routing.for_job(LEGACY_CPU, member_test) is LEGACY_CPU and routing.for_job(None, member_test) is None


def test_a_gym_that_is_not_public_is_routed_by_the_bridges_own_pool_machines():
    """Its graders may reach only pool machines of the member the bridge submits as (#117)."""
    private = {"bundle": {"sha": "a" * 64}, "isolated": True, "action": "test", "requires": {"gpu": True, "tools": ["nvcc"]},
               "gym_public": False}
    unknown = {k: v for k, v in private.items() if k != "gym_public"}  # missing: not public (fail closed)
    # a bridge that reports no own machines (or an older one that does not say): not on the pool;
    # routed by the runner's own capabilities, so this one (no GPU, no sandbox) cannot take it
    for job in (private, unknown):
        assert not routing.on_pool(BRIDGE, job) and routing.for_job(BRIDGE, job) is BRIDGE
        assert not pooled.takes(BRIDGE["pool"], job)
    # own machines without a GPU: on the pool, but routed by those machines, so not a GPU job
    cpu_own = {**BRIDGE, "pool": {**POOL, "machines": 3, "own": {"machines": 1, "gpus": []}}}
    assert routing.on_pool(cpu_own, private) and routing.unmet(routing.for_job(cpu_own, private), private["requires"]) == "an NVIDIA GPU"
    gpu_own = {**BRIDGE, "pool": {**POOL, "own": {"machines": 1, "gpus": [{"name": "NVIDIA GB10"}]}}}
    assert routing.on_pool(gpu_own, unknown) and routing.unmet(routing.for_job(gpu_own, unknown), unknown["requires"]) is None
    assert routing.for_job(gpu_own, unknown)["machines"] == 1 and pooled.takes(gpu_own["pool"], unknown)
    # a public gym still uses the whole pool
    assert routing.for_job(cpu_own, {**private, "gym_public": True})["machines"] == 3
    # the runner: no report yet, so to the pool, where resources.machines: own keeps it on our machines
    assert pooled.takes(None, private) and pooled.own_machines_only(unknown) and pooled.own_machines_only({"gym_public": "yes"})
    assert not pooled.own_machines_only({"gym_public": True})


def _worker(board, token):
    c = client(board)
    c.headers.update({"Authorization": f"Bearer {token}"})
    return c


def _queue(board, **fields):
    job_id = f"job{time.time_ns()}"
    board.store.set("jobs", job_id, {"gym": "gpu", "slug": "p", "problem": "gpu__p", "lang": "cuda", "action": "test", "code": "x",
                                     "options": {}, "status": "queued", "created_at": time.time(), "output": "", "uid": "g_member",
                                     "isolated": True, "requires": {"gpu": True, "tools": ["nvcc"]}, **fields})
    return job_id


def test_the_worker_api_routes_a_bridge_by_the_pool_and_others_as_before(board):
    owner = client(board)
    sign_in(owner, "owner")
    bridge = _worker(board, shared_token(owner, "bridge"))
    legacy = _worker(board, shared_token(owner, "legacy"))
    bridge.post("/v1/worker/heartbeat", json={"capabilities": BRIDGE, "gyms": ["gpu"]})
    legacy.post("/v1/worker/heartbeat", json={"capabilities": LEGACY_CPU, "gyms": ["gpu"]})
    board.store.set("gyms", "gpu", {"id": "gpu"})  # a local gym: public

    member_gpu = _queue(board)
    # an existing worker: no GPU and no sandbox, so neither the GPU nor someone else's code
    assert legacy.post("/v1/worker/lease", json={"gyms": ["gpu"]}).json()["job"] is None
    # the bridge has neither either, but grades on the pool, which has both
    job = bridge.post("/v1/worker/lease", json={"gyms": ["gpu"]}).json()["job"]
    assert job["id"] == member_gpu and job["hardware"] is None  # the pool machine's arrives with the result
    done = bridge.post(f"/v1/worker/jobs/{member_gpu}/complete",
                       json={"final": {"status": "done", "ok": True, "output": "ok", "hardware": "NVIDIA GB10"}, "lease_id": job["lease_id"]})
    assert done.status_code == 200 and board.store.get("jobs", member_gpu)["hardware"] == "NVIDIA GB10"

    # the owner's own GPU job runs in place: not on a bridge without a GPU
    owner_uid = board.store.get("workers", bridge.headers["Authorization"].split("_")[1])["owner_uid"]
    own = _queue(board, uid=owner_uid, isolated=False)
    assert bridge.post("/v1/worker/lease", json={"gyms": ["gpu"]}).json()["job"] is None
    # a bench the bridge keeps local needs a local sandbox, like on any worker
    bridge.post("/v1/worker/heartbeat", json={"capabilities": {**BRIDGE, "pool": {**POOL, "actions": ["run", "test"]}}, "gyms": ["gpu"]})
    import gymhub.board.workers as workers

    workers._capabilities.clear()
    bench = _queue(board, action="bench", requires={"gpu": False, "tools": []})
    assert bridge.post("/v1/worker/lease", json={"gyms": ["gpu"]}).json()["job"] is None
    # an existing worker with a sandbox and a GPU still takes both, unchanged
    legacy.post("/v1/worker/heartbeat", json={"capabilities": {**LEGACY_CPU, "sandbox": True, "gpus": [{"name": "RTX 4090"}],
                                                               "tools": ["nvcc"]}, "gyms": ["gpu"]})
    workers._capabilities.clear()
    taken = {legacy.post("/v1/worker/lease", json={"gyms": ["gpu"]}).json()["job"]["id"] for _ in range(2)}
    assert taken == {own, bench}


def test_the_board_tells_a_bridge_whether_a_gym_is_public(board):
    """The board, not the runner, says whether a leased job's gym is public; a private uploaded gym
    goes to a bridge only by its own pool machines (none: left to a worker with a sandbox)."""
    import gymhub.board.workers as workers

    owner = client(board)
    sign_in(owner, "owner")
    bridge = _worker(board, shared_token(owner, "bridge"))
    bridge.post("/v1/worker/heartbeat", json={"capabilities": BRIDGE, "gyms": []})  # an admin's machine: may hold the gym
    board.store.set("gyms", "up", {"id": "up", "source": "bundle", "owner_uid": "g_member"})  # no visibility: private
    private = _queue(board, gym="up", bundle={"sha": "a" * 64}, gym_public=True)  # a stored claim counts for nothing
    assert bridge.post("/v1/worker/lease", json={"gyms": []}).json()["job"] is None  # the pool, but none of its own machines
    own = {**POOL, "own": {"machines": 1, "gpus": [{"name": "NVIDIA GB10"}]}}
    bridge.post("/v1/worker/heartbeat", json={"capabilities": {**BRIDGE, "pool": own}, "gyms": []})
    workers._capabilities.clear()
    job = bridge.post("/v1/worker/lease", json={"gyms": []}).json()["job"]
    assert job["id"] == private and job["gym_public"] is False
    board.store.set("gyms", "up", {"visibility": "public"}, merge=True)
    public = _queue(board, gym="up", bundle={"sha": "a" * 64})
    job = bridge.post("/v1/worker/lease", json={"gyms": []}).json()["job"]
    assert job["id"] == public and job["gym_public"] is True


def test_the_bridge_advertises_only_a_reachable_pool():
    class Down:
        def call(self, *args, **kwargs):
            raise OSError("no route")

    class Up:
        def call(self, *args, **kwargs):
            return {"machines": [
                {"name": "spark", "online": True, "drain": False, "profile": {"gpus": [{"model": "NVIDIA GB10"}], "arch": "aarch64"}},
                {"name": "jetson", "online": True, "drain": False, "mine": True, "profile": {"gpus": [{"model": "Orin"}]}},
                {"name": "gone", "online": False, "profile": {"gpus": [{"model": "H100"}]}},
                {"name": "spark2", "online": True, "drain": False, "profile": {"gpus": [{"model": "NVIDIA GB10"}]}},
            ]}

    class Empty:
        def call(self, *args, **kwargs):
            return {"machines": [{"name": "gone", "online": False, "profile": {}}]}

    assert pooled.advertised(Down()) is None and pooled.advertised(Empty()) is None
    found = pooled.advertised(Up())
    assert found["gpus"] == [{"name": "NVIDIA GB10"}, {"name": "Orin"}] and found["machines"] == 3
    assert found["actions"] == ["bench", "run", "test"] and found["tools"] == ["g++", "nvcc"]
    assert found["own"] == {"machines": 1, "gpus": [{"name": "Orin"}]}  # the pool member's own, for gyms not public


class BoardApi(ApiBackend):
    """The runner's worker-API client, unchanged, talking to the board in-process instead of over HTTP."""

    def __init__(self, board, token):
        super().__init__("http://board", token)
        self.http = _worker(board, token)

    def _call(self, path, body=None, timeout=60):
        reply = self.http.post(f"/v1/worker/{path}", json=body or {})
        if reply.status_code >= 400:
            raise RuntimeError(f"{path}: HTTP {reply.status_code}: {reply.text[:300]}")
        return reply.json() if reply.content else None


def test_a_bridge_and_an_existing_worker_serve_one_board(board, pool_env, pool_settings, monkeypatch):
    """Board -> worker API -> a runner in pool mode (no GPU, no Docker) -> pool agent; and, on the same
    board and queue, an existing worker that knows nothing of the pool leases and completes as before."""
    from gymhub.runner.agent import Runner

    monkeypatch.setenv("GYMHUB_TRUST_JOBS", "1")
    start_agent(pool_env, "grader")
    owner = client(board)
    sign_in(owner, "owner")
    owner.put("/api/admin/settings", json={"allowed_domains": ["example.com"], "members_can_run": True}, headers=H)
    bridge_token = shared_token(owner, "bridge")
    legacy_token = shared_token(owner, "legacy")

    runner = Runner(BoardApi(board, bridge_token), {"algo": kit_adapter("algo")}, "bridge")
    runner.gpu = "none"  # and no sandbox: other users' code goes to the pool
    runner.heartbeat()
    sent = board.store.get("runners", bridge_token.split("_")[1])["capabilities"]
    assert sent["pool"]["machines"] == 1 and not sent["sandbox"]
    runner.publish_catalog(runner.adapters["algo"])

    legacy = BoardApi(board, legacy_token)
    legacy.heartbeat({"name": "legacy", "gyms": ["algo"], "capabilities": {**LEGACY_CPU, "sandbox": True}})

    ann = client(board)
    sign_in(ann, "ann")

    def submit():
        reply = ann.post("/api/jobs", json={"gym": "algo", "slug": "merge_intervals", "lang": "python", "action": "test",
                                            "code": MERGE_INTERVALS}, headers=H)
        assert reply.status_code == 200, reply.text
        return reply.json()["id"]

    first = submit()
    job = runner.backend.lease(["algo"])
    assert job["id"] == first and job["isolated"]
    runner.run_job(job)
    done = board.store.get("jobs", first)
    assert done["ok"] and done["status"] == "done", done["output"]
    assert "[graded on grader]" in done["output"] and done["runner"] == bridge_token.split("_")[1]

    second = submit()
    job = legacy.lease(["algo"])  # the existing worker's own client, its own sandbox
    assert job["id"] == second and job["user_progress"]["exercises"]["merge_intervals"]
    assert not legacy.update(second, "running in my sandbox\n")
    legacy.complete(second, {"status": "done", "ok": True, "exit_code": 0, "output": "graded locally\n"}, job["user_progress"])
    done = board.store.get("jobs", second)
    assert done["ok"] and done["runner"] == legacy_token.split("_")[1] and done["output"] == "graded locally\n"
