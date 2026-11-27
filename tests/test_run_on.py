"""Run on: where a job ran (host, via) and choosing where it runs (job["target"]).

A target only narrows who leases a job: {"runner": id} only that runner, {"pool_machine": name} only a
pool bridge that lists that machine and would grade the job there, pinned by resources.machine.
"""

import time

from gymhub import jobqueue, routing
from gymhub.runner import pooled
from gymhub.runner.adapters import kit_adapter
from pool import scheduler
from tests.test_board_auth import H, board, client, shared_token, sign_in  # noqa: F401  (fixture)
from tests.test_pool import FREE, db_url, job_spec, pool_env, start_agent, wait_for  # noqa: F401  (fixtures)
from tests.test_pool_bridge import BoardApi
from tests.test_pooled_gym import MERGE_INTERVALS, pool_settings  # noqa: F401  (fixture)

CPU_BOX = {"arch": "x86_64", "gpus": [], "tools": ["g++"], "sandbox": True, "bundles": True}
LISTED = [{"name": "spark", "gpus": [{"name": "NVIDIA GB10"}], "own": True},
          {"name": "jetson", "gpus": [{"name": "Orin"}], "own": False}]
POOL = {"actions": ["bench", "run", "test"], "machines": 2, "gpus": [{"name": "NVIDIA GB10"}, {"name": "Orin"}], "tools": ["g++", "nvcc"],
        "own": {"machines": 1, "gpus": [{"name": "NVIDIA GB10"}]}, "list": LISTED}
BRIDGE = {"gpus": [], "tools": [], "sandbox": False, "bundles": True, "arch": "x86_64", "pool": POOL}


def test_targets_narrow_routing():
    job = {"isolated": True, "action": "test", "requires": {"gpu": True, "tools": ["nvcc"]}, "gym_public": True}
    assert not routing.misses_target("any", BRIDGE, job)  # Auto
    assert routing.misses_target("a", CPU_BOX, {**job, "target": {"runner": "b"}})
    assert not routing.misses_target("b", CPU_BOX, {**job, "target": {"runner": "b"}})
    assert routing.misses_target("b", CPU_BOX, {**job, "target": {"what": "x"}})  # not understood: nobody
    jetson = {**job, "target": {"pool_machine": "jetson"}}
    assert not routing.misses_target("bridge", BRIDGE, jetson) and routing.misses_target("cpu", CPU_BOX, jetson)
    assert routing.misses_target("bridge", BRIDGE, {**job, "target": {"pool_machine": "elsewhere"}})
    # routed by that machine's GPU only
    assert routing.for_job(BRIDGE, jetson)["gpus"] == [{"name": "Orin"}]
    # a gym that is not public: only the bridge's own machines, even when asked for another
    private = {**jetson, "gym_public": False}
    assert routing.misses_target("bridge", BRIDGE, private)
    assert not routing.misses_target("bridge", BRIDGE, {**private, "target": {"pool_machine": "spark"}})
    # the owner's own job goes to the pool when pinned to a pool machine
    assert routing.on_pool(BRIDGE, {**jetson, "isolated": False}) and not routing.on_pool(BRIDGE, {**job, "isolated": False})
    # an older bridge lists no machines: a pool machine target never reaches it
    old = {**BRIDGE, "pool": {k: v for k, v in POOL.items() if k != "list"}}
    assert routing.misses_target("bridge", old, jetson)


def test_the_pool_pins_a_job_to_one_machine():
    machine = {"id": "m1", "name": "jetson", "owner": "ann", "labels": {}, "policy": {}, "profile": {"gpus": [{"index": 0, "memory_gb": 24}]}}
    assert scheduler.fits(job_spec(machine="jetson"), machine, FREE) == []
    assert scheduler.fits(job_spec(machine="spark"), machine, FREE) is None
    assert "machine" not in job_spec()["resources"]  # older specs unchanged
    assert scheduler.fits(job_spec(machine="jetson", machines="own"), machine, FREE, "bob") is None  # never widens own
    from pool.spec import SpecError, normalize

    for bad in ("", "a b", 5):
        try:
            job_spec(machine=bad)
        except SpecError:
            continue
        raise AssertionError(f"accepted resources.machine {bad!r}")
    try:
        normalize({"image": "x", "command": ["x"], "resources": {"machine": "jetson"}, "gang": {"size": 2}})
        raise AssertionError("a gang pinned to one machine")
    except SpecError:
        pass
    spec = pooled.spec_for(kit_adapter("algo"), "test", ["true"], False, True, "jetson")
    assert spec["resources"]["machine"] == "jetson" and spec["resources"]["machines"] == "own"
    assert "machine" not in pooled.spec_for(kit_adapter("algo"), "test", ["true"], False)["resources"]


def _worker(board, token):
    c = client(board)
    c.headers.update({"Authorization": f"Bearer {token}"})
    return c


def _setup(board):
    """An admin's two machines (cpu-a, cpu-b), a pool bridge, and Eve's own machine, not shared."""
    owner = client(board)
    sign_in(owner, "owner")
    owner.put("/api/admin/settings", json={"allowed_domains": ["example.com"], "members_can_run": True}, headers=H)
    machines = {}
    for name, caps in (("cpu-a", CPU_BOX), ("cpu-b", CPU_BOX), ("bridge", BRIDGE)):
        token = shared_token(owner, name)
        machines[name] = (_worker(board, token), token.split("_")[1])
        machines[name][0].post("/v1/worker/heartbeat", json={"capabilities": caps, "gyms": ["algo"]})
    bob = client(board)
    sign_in(bob, "eve")
    code = bob.post("/api/my/workers/registration", json={"name": "bobs"}, headers=H).json()["code"]
    token = client(board).post("/v1/register", json={"code": code}).json()["token"]
    machines["bobs"] = (_worker(board, token), token.split("_")[1])
    machines["bobs"][0].post("/v1/worker/heartbeat", json={"capabilities": CPU_BOX, "gyms": ["algo"]})
    board.store.set("problems", "algo__p", {"gym": "algo", "slug": "p", "langs": ["python"]})
    board.store.set("gyms", "algo", {"id": "algo", "title": "Algo", "requires": {"gpu": False}})
    ann = client(board)
    sign_in(ann, "ann")
    return ann, machines


def _submit(ann, **extra):
    return ann.post("/api/jobs", json={"gym": "algo", "slug": "p", "lang": "python", "action": "test", "code": "x", **extra}, headers=H)


def _lease(machines, name):
    return machines[name][0].post("/v1/worker/lease", json={"gyms": ["algo"]}).json()["job"]


def test_choices_and_a_runner_target(board):
    ann, machines = _setup(board)
    choices = ann.get("/api/run-targets", params={"gym": "algo", "slug": "p", "lang": "python"}).json()["choices"]
    names = {(c["kind"], c["name"]) for c in choices}
    # the admin's machines and the pool's machines (a public gym), never Eve's unshared machine
    assert names == {("runner", "cpu-a"), ("runner", "cpu-b"), ("runner", "bridge"), ("pool", "spark"), ("pool", "jetson")}
    assert next(c for c in choices if c["name"] == "jetson") == {"target": {"pool_machine": "jetson"}, "name": "jetson", "kind": "pool",
                                                                 "hardware": "Orin", "via": "bridge"}

    reply = _submit(ann, target={"runner": machines["cpu-b"][1]})
    assert reply.status_code == 200 and "warning" not in reply.json()
    job_id = reply.json()["id"]
    assert _lease(machines, "cpu-a") is None and _lease(machines, "bridge") is None
    job = _lease(machines, "cpu-b")
    assert job["id"] == job_id and board.store.get("jobs", job_id)["host"] == "cpu-b"  # "Running on cpu-b"
    machines["cpu-b"][0].post(f"/v1/worker/jobs/{job_id}/complete",
                              json={"final": {"status": "done", "ok": True, "output": "ok"}, "lease_id": job["lease_id"]})
    done = ann.get(f"/api/jobs/{job_id}").json()
    assert done["host"] == "cpu-b" and done["via"] is None and done["hardware"] == "CPU x86_64"


def test_an_unauthorized_target_is_refused(board):
    ann, machines = _setup(board)
    reply = _submit(ann, target={"runner": machines["bobs"][1]})
    assert reply.status_code == 403 and "cannot run this on" in reply.text
    assert _submit(ann, target={"pool_machine": "nowhere"}).status_code == 403
    assert _submit(ann, target={"runner": "x", "pool_machine": "y"}).status_code == 403
    # a target stored on a job some other way never widens who leases it
    job_id = f"j{time.time_ns()}"
    ann_uid = ann.get("/api/me").json()["user"]["uid"]
    board.store.set("jobs", job_id, {"gym": "algo", "slug": "p", "lang": "python", "action": "test", "status": "queued", "code": "x",
                                     "created_at": time.time(), "uid": ann_uid, "isolated": True, "requires": {"gpu": False, "tools": []},
                                     "target": {"runner": machines["bobs"][1]}})
    assert _lease(machines, "bobs") is None and _lease(machines, "cpu-a") is None
    # a gym that is not public: never another member's pool machine
    board.store.set("gyms", "algo", {"visibility": "shared", "learners": [ann_uid]}, merge=True)
    board.store.delete("jobs", job_id)
    assert _submit(ann, target={"pool_machine": "jetson"}).status_code == 403
    choices = ann.get("/api/run-targets", params={"gym": "algo", "slug": "p", "lang": "python"}).json()["choices"]
    assert "jetson" not in {c["name"] for c in choices}


def test_a_pool_machine_target_and_an_offline_one(board):
    ann, machines = _setup(board)
    reply = _submit(ann, target={"pool_machine": "jetson"})
    assert reply.status_code == 200 and "warning" not in reply.json()
    job_id = reply.json()["id"]
    assert _lease(machines, "cpu-a") is None
    job = _lease(machines, "bridge")
    assert job["id"] == job_id and job["target"] == {"pool_machine": "jetson"}
    assert board.store.get("jobs", job_id)["host"] == "bridge"
    # the bridge says where the pool placed it, then completes
    bridge = machines["bridge"][0]
    bridge.post(f"/v1/worker/jobs/{job_id}/update", json={"output": "…", "lease_id": job["lease_id"], "host": "jetson"})
    running = board.store.get("jobs", job_id)
    assert running["host"] == "jetson" and running["via"] == "bridge"
    bridge.post(f"/v1/worker/jobs/{job_id}/complete", json={"final": {"status": "done", "ok": True, "output": "ok", "hardware": "Orin",
                                                                      "host": "jetson", "via": "bridge"}, "lease_id": job["lease_id"]})
    done = board.store.get("jobs", job_id)
    assert (done["host"], done["via"], done["hardware"]) == ("jetson", "bridge", "Orin")
    # the bridge goes offline: the target is still allowed, but the job waits, and says so
    board.store.set("runners", machines["bridge"][1], {"heartbeat_at": 0}, merge=True)
    reply = _submit(ann, target={"pool_machine": "jetson"})
    assert reply.status_code == 200 and "offline" in reply.json()["warning"] and "Auto" in reply.json()["warning"]
    assert _lease(machines, "cpu-a") is None


def test_host_is_recorded_for_local_and_direct_runs(board):
    job_id = f"j{time.time_ns()}"
    board.store.set("jobs", job_id, {"gym": "algo", "slug": "p", "lang": "python", "action": "test", "status": "queued",
                                     "created_at": time.time(), "uid": "g_1", "requires": {"gpu": False, "tools": []}})
    leased = jobqueue.lease(board.store, ["algo"], "w1", capabilities=CPU_BOX, name="home-box")
    assert board.store.get("jobs", job_id)["host"] == "home-box"
    jobqueue.complete(board.store, job_id, "w1", {"status": "done", "ok": True, "output": "", "host": "x" * 200},
                      lease_id=leased["lease_id"])  # too long: ignored
    assert board.store.get("jobs", job_id)["host"] == "home-box"


def test_a_pooled_job_records_the_pool_machine_and_honours_the_target(board, pool_env, pool_settings, monkeypatch):
    """Board -> bridge -> pool with two machines: Auto records whichever ran it; a target pins it."""
    from gymhub.runner.agent import Runner

    start_agent(pool_env, "spark")
    start_agent(pool_env, "jetson")
    owner = client(board)
    sign_in(owner, "owner")
    owner.put("/api/admin/settings", json={"allowed_domains": ["example.com"], "members_can_run": True}, headers=H)
    token = shared_token(owner, "bridge")
    runner = Runner(BoardApi(board, token), {"algo": kit_adapter("algo")}, "bridge")
    runner.gpu = "none"
    wait_for(lambda: len((pooled.advertised(runner._pool()) or {}).get("list", [])) == 2)
    runner.heartbeat()
    runner.publish_catalog(runner.adapters["algo"])
    ann = client(board)
    sign_in(ann, "ann")
    choices = ann.get("/api/run-targets", params={"gym": "algo", "slug": "merge_intervals", "lang": "python"}).json()["choices"]
    assert {c["name"] for c in choices if c["kind"] == "pool"} == {"spark", "jetson"}

    for target in ("jetson", "spark"):
        reply = ann.post("/api/jobs", json={"gym": "algo", "slug": "merge_intervals", "lang": "python", "action": "test",
                                            "code": MERGE_INTERVALS, "target": {"pool_machine": target}}, headers=H)
        assert reply.status_code == 200, reply.text
        job = runner.backend.lease(["algo"])
        assert job["id"] == reply.json()["id"]
        runner.run_job(job)
        done = board.store.get("jobs", job["id"])
        assert done["ok"], done["output"]
        assert (done["host"], done["via"]) == (target, "bridge") and f"[graded on {target}]" in done["output"]
    # Auto: wherever the pool put it, recorded
    reply = ann.post("/api/jobs", json={"gym": "algo", "slug": "merge_intervals", "lang": "python", "action": "test",
                                        "code": MERGE_INTERVALS}, headers=H)
    runner.run_job(runner.backend.lease(["algo"]))
    done = board.store.get("jobs", reply.json()["id"])
    assert done["host"] in {"spark", "jetson"} and done["via"] == "bridge"
