"""Worker v2 (#135): the gymhub runner as a dispatcher, and pool machines on the board's Compute page.

A dispatcher (GYMHUB_EXECUTOR=pool, GYMHUB_DISPATCH_ONLY=1) sends every job to the pool, the owner's
own and every action (bench and the validation of an uploaded version too), and executes nothing on
its own machine. The board lists the pool's machines and gives admins the pool's install command.
"""

import dataclasses
import json
import shutil
import tempfile
from pathlib import Path

import pytest

from gymhub import packages, routing
from gymhub.runner import pooled
from gymhub.runner.adapters import kit_adapter
from tests.test_board_auth import H, board, client, sign_in  # noqa: F401  (fixture)
from tests.test_gym_versions import HELLO, _upload
from tests.test_pool import db_url, pool_env, start_agent  # noqa: F401  (fixtures)
from tests.test_pooled_gym import MERGE_INTERVALS

POOL = {"actions": ["bench", "run", "test", "validate"], "machines": 1, "gpus": [{"name": "NVIDIA GB10"}],
        "tools": ["g++", "nvcc"], "own": {"machines": 1, "gpus": [{"name": "NVIDIA GB10"}]}, "dispatch_only": True}
DISPATCHER = {"gpus": [], "tools": [], "sandbox": False, "bundles": True, "arch": "x86_64", "dispatch_only": True, "pool": POOL}


@pytest.fixture()
def dispatch(monkeypatch):
    monkeypatch.setenv("GYMHUB_EXECUTOR", "pool")
    monkeypatch.setenv("GYMHUB_DISPATCH_ONLY", "1")
    monkeypatch.setenv("GYMHUB_POOL_IMAGE", "unused-by-the-process-runtime")
    monkeypatch.setenv("GYMHUB_TRUST_JOBS", "1")


@pytest.fixture()
def owner_files():
    """The owner's solutions and progress for a test: inside the repo (the runner names them relative
    to it), never the real ones."""
    from gymhub.runner.adapters import REPO

    (REPO / ".cache").mkdir(exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="test-dispatcher-", dir=REPO / ".cache"))
    yield folder
    shutil.rmtree(folder, ignore_errors=True)


# ---- routing -------------------------------------------------------------------------------------


def test_a_dispatcher_is_routed_by_the_pool_for_every_job_and_takes_nothing_else():
    owner_bench = {"isolated": False, "action": "bench", "requires": {"gpu": True, "tools": ["nvcc"]}, "gym_public": True}
    assert routing.on_pool(DISPATCHER, owner_bench) and not routing.refuses(DISPATCHER, owner_bench)
    assert routing.unmet(routing.for_job(DISPATCHER, owner_bench), owner_bench["requires"]) is None
    validate = {"isolated": True, "action": "validate", "bundle": {"sha": "a" * 64}, "requires": {"gpu": False, "tools": []}}
    assert routing.on_pool(DISPATCHER, validate)  # not public: by its own pool machines, of which it has one
    # a gym that is not public with none of its own machines online: it has nowhere to send it
    none_own = {**DISPATCHER, "pool": {**POOL, "own": {"machines": 0, "gpus": []}}}
    assert routing.refuses(none_own, {**owner_bench, "gym_public": False})
    # the pool cannot be reached (no report): a job without requirements is still not for it
    alone = {key: value for key, value in DISPATCHER.items() if key != "pool"}
    assert routing.refuses(alone, {"isolated": False, "action": "test", "requires": {"gpu": False, "tools": []}})
    # the owner's save on a GPU gym: a file edit on the dispatcher's machine, which needs no GPU
    save = {"isolated": False, "action": "save", "requires": {"gpu": True, "tools": ["nvcc"]}}
    assert not routing.refuses(DISPATCHER, save) and routing.unmet(routing.for_job(DISPATCHER, save), save["requires"]) is None
    assert routing.refuses(DISPATCHER, {**save, "isolated": True})  # a member's save is the board's own business
    # a bridge that is not a dispatcher keeps the owner's jobs in place, as before
    bridge = {**DISPATCHER, "dispatch_only": False, "pool": {**POOL, "dispatch_only": False}}
    assert not routing.on_pool(bridge, owner_bench) and not routing.refuses(bridge, owner_bench)


def test_dispatch_only_settings(monkeypatch):
    monkeypatch.setenv("GYMHUB_POOL_ACTIONS", "test")
    monkeypatch.setenv("GYMHUB_DISPATCH_ONLY", "1")
    monkeypatch.setenv("GYMHUB_EXECUTOR", "local")
    assert not pooled.dispatch_only()  # only in pool mode
    monkeypatch.delenv("GYMHUB_EXECUTOR")  # pool is the default
    assert pooled.dispatch_only() and pooled.actions() == {"run", "test", "bench", "validate"}


# ---- the runner refuses to execute anything locally --------------------------------------------


def test_a_dispatcher_never_starts_a_process(dispatch, owner_files, tmp_path, monkeypatch):
    from gymhub.runner.agent import Runner

    monkeypatch.setenv("POOL_CONFIG_DIR", str(tmp_path / "no-pool-login"))
    monkeypatch.delenv("POOL_SERVER", raising=False)
    monkeypatch.delenv("POOL_TOKEN", raising=False)

    class Backend:
        can_publish = True

        def __init__(self):
            self.done = {}

        def complete(self, job_id, final, user_progress=None):
            self.done[job_id] = final

        def publish_code(self, *args):
            pass

        def publish_progress(self, *args):
            pass

    adapter = dataclasses.replace(kit_adapter("algo"), solutions_dir=owner_files / "solutions", progress_file=owner_files / "progress.json")
    runner = Runner(Backend(), {"algo": adapter}, "dispatcher")
    with pytest.raises(RuntimeError, match="dispatcher"):
        runner._stream("j", ["true"], 10)
    caps = runner.capabilities()  # no pool member configured: nothing of this machine's own either
    assert caps["dispatch_only"] and caps["gpus"] == [] and caps["tools"] == [] and not caps["sandbox"] and "pool" not in caps
    started = []
    monkeypatch.setattr(Runner, "_stream", lambda self, *args, **kwargs: started.append(args) or ("", True, 0))
    # the owner's test, someone else's bench: with no pool to send them to, they fail; nothing runs here
    for job_id, isolated, action in (("j1", False, "test"), ("j2", True, "bench")):
        runner.run_job({"id": job_id, "gym": "algo", "slug": "merge_intervals", "lang": "python", "action": action,
                        "code": MERGE_INTERVALS, "isolated": isolated, "uid": "g_x"})
        assert runner.backend.done[job_id]["status"] == "error" and "not logged in" in runner.backend.done[job_id]["output"]
    assert started == []
    # saving is a file edit, not execution: still done here
    runner.run_job({"id": "j3", "gym": "algo", "slug": "merge_intervals", "lang": "python", "action": "save", "code": "x = 1\n"})
    assert (owner_files / "solutions/merge_intervals/solution.py").read_text() == "x = 1\n" and started == []


def test_a_dispatcher_grades_the_owners_bench_and_an_uploaded_version_on_the_pool(board, pool_env, dispatch, owner_files, tmp_path,
                                                                                 monkeypatch):
    """Board -> dispatcher -> pool agent: the owner's own test and bench (recorded under the pool
    machine's hardware class), and the validation of an uploaded gym version. Nothing runs locally."""
    from gymhub.runner import bundles
    from gymhub.runner.agent import Runner
    from gymhub.runner.backends import DirectBackend

    monkeypatch.setattr(bundles, "ROOT", tmp_path / "bundles")
    started = []
    monkeypatch.setattr(Runner, "_stream", lambda self, *args, **kwargs: started.append(args) or ("", False, 1))
    start_agent(pool_env, "grader")
    adapter = dataclasses.replace(kit_adapter("algo"), solutions_dir=owner_files / "solutions", progress_file=owner_files / "progress.json")
    runner = Runner(DirectBackend(board.store, "dispatcher"), {"algo": adapter}, "dispatcher")
    runner.gpu = "none"
    runner.heartbeat()
    sent = board.store.get("runners", "dispatcher")["capabilities"]
    assert sent["dispatch_only"] and sent["pool"]["dispatch_only"] and sent["pool"]["own"]["machines"] == 1
    runner.publish_catalog(adapter)
    owner = client(board)
    sign_in(owner, "owner")

    for action in ("test", "bench"):
        reply = owner.post("/api/jobs", json={"gym": "algo", "slug": "merge_intervals", "lang": "python", "action": action,
                                              "code": MERGE_INTERVALS}, headers=H)
        assert reply.status_code == 200, reply.text
        job = runner.backend.lease(["algo"])
        assert job["id"] == reply.json()["id"] and not job["isolated"] and job["hardware"] is None
        runner.run_job(job)
        done = board.store.get("jobs", job["id"])
        assert done["ok"], done["output"]
        assert "[graded on grader]" in done["output"]
    uid = owner.get("/api/me").json()["user"]["uid"]
    best = board.store.get("bests", f"{uid}__algo__merge_intervals__python")["by_hardware"]
    assert list(best) == [done["hardware"]] and done["hardware"].startswith("CPU")  # the pool machine's class
    # the owner's progress came back from the pool into the owner's file, and on to the board
    progress = json.loads((owner_files / "progress.json").read_text())
    assert progress["exercises"]["merge_intervals"]["langs"]["python"]["first_solved"]
    assert board.store.get("progress", "algo")["data"]["exercises"]["merge_intervals"]
    assert (owner_files / "solutions/merge_intervals/solution.py").read_text() == MERGE_INTERVALS

    # An uploaded version is validated on the pool too, and published from its result.
    _upload(owner, packages.pack(HELLO))
    job = runner.backend.lease([])
    assert job["action"] == "validate"
    runner.run_job(job)
    done = board.store.get("jobs", job["id"])
    assert done["ok"] and "[graded on grader]" in done["output"], done["output"]
    assert board.store.get("gym_versions", "hello__v0001")["status"] == "published"
    assert board.store.get("problems", "hello__sum_list")["benchable"] is False
    assert started == []


def test_the_validation_folder_carries_the_package_and_the_validator(tmp_path):
    folder = tmp_path / "work"
    folder.mkdir()
    shutil.copytree(HELLO, tmp_path / "hello")
    command = pooled.stage_validation(tmp_path / "hello", folder, graders=False)
    assert (folder / "package/gym.json").is_file() and (folder / "gymhub/runner/validate.py").is_file()
    assert not (folder / "gymkit").exists() and "gymhub.runner.validate package" in command[-1]


# ---- the board's Compute page --------------------------------------------------------------------


class FakePool:
    def __init__(self):
        self.calls = []

    def call(self, method, path, body=None, timeout=60):
        self.calls.append((method, path, body))
        if path.startswith("/v1/machines"):
            return {"pool": "default", "machines": [
                {"id": "m1", "name": "spark", "owner": "admin", "online": True, "last_seen": 1.0, "drain": False,
                 "labels": {"gpu": "unified", "arch": "aarch64"}, "running": ["j1"],
                 "profile": {"gpus": [{"model": "NVIDIA GB10", "memory_gb": 119.7}], "arch": "aarch64", "cpus": 20}},
                {"id": "m2", "name": "jetson", "owner": "admin", "online": False, "last_seen": None, "drain": False,
                 "labels": {}, "running": [], "profile": {}},
            ]}
        if path.startswith("/v1/links"):
            return {"pool": "default", "links": [], "islands": [["jetson", "spark"]]}
        if path == "/v1/machine-codes":
            return {"code": "pm_abc", "pool": "default", "expires_at": 2.0, "install": "curl -fsSL http://pool/install.sh | sudo sh -s -- --code pm_abc",
                    "sha256": "f" * 64, "other": "dropped"}
        raise AssertionError(path)


def test_the_compute_page_lists_pool_machines_and_admins_add_one(board, monkeypatch):
    import gymhub.board.poolmachines as poolmachines

    owner, member = client(board), client(board)
    sign_in(owner, "owner")
    owner.put("/api/admin/settings", json={"allowed_domains": ["example.com"]}, headers=H)
    sign_in(member, "ann")
    # no pool configured: nothing to list, nothing to add
    assert owner.get("/api/pool/machines").json() == {"configured": False, "machines": [], "can_add": False, "bridge": None}
    assert owner.post("/api/pool/machine-codes", json={"name": "x"}, headers=H).status_code == 409

    fake = FakePool()
    monkeypatch.setattr(poolmachines, "pool_client", lambda: fake)
    listed = owner.get("/api/pool/machines").json()
    assert listed["configured"] and listed["can_add"] and listed["pool"] == "default"
    spark, jetson = listed["machines"]
    assert spark["name"] == "spark" and spark["online"] and spark["running"] == 1
    assert spark["gpus"] == [{"model": "NVIDIA GB10", "memory_gb": 119.7}] and spark["labels"]["gpu"] == "unified"
    assert not jetson["online"] and jetson["gpus"] == []
    assert listed["islands"] == [{"machines": ["jetson", "spark"]}] and listed["source"] == "board"
    assert member.get("/api/pool/machines").json()["can_add"] is False

    # Add a machine: admins only, like worker tokens
    assert member.post("/api/pool/machine-codes", json={"name": "box"}, headers=H).status_code == 403
    reply = owner.post("/api/pool/machine-codes", json={"name": "box"}, headers=H)
    assert reply.status_code == 200, reply.text
    code = reply.json()
    assert code["install"].endswith("--code pm_abc") and code["sha256"] == "f" * 64 and "other" not in code
    assert fake.calls[-1] == ("POST", "/v1/machine-codes", {"pool": None, "name": "box"})
    assert owner.post("/api/pool/machine-codes", json={"name": "bad name"}, headers=H).status_code == 422

    # a pool server that does not answer: the page says so instead of failing
    class Down:
        def call(self, *args, **kwargs):
            raise OSError("connection refused")

    monkeypatch.setattr(poolmachines, "pool_client", lambda: Down())
    down = owner.get("/api/pool/machines").json()
    assert down["configured"] and down["machines"] == [] and "connection refused" in down["error"]
    assert owner.post("/api/pool/machine-codes", json={}, headers=H).status_code == 502


def test_the_board_reads_the_pool_settings(monkeypatch):
    import gymhub.board.poolmachines as poolmachines

    monkeypatch.delenv("GYMHUB_POOL_SERVER", raising=False)
    monkeypatch.setenv("GYMHUB_POOL_TOKEN", "t")
    assert poolmachines.pool_client() is None
    monkeypatch.setenv("GYMHUB_POOL_SERVER", "http://pool:8765/")
    found = poolmachines.pool_client()
    assert found.server == "http://pool:8765" and found.token == "t"


def test_my_machines_say_which_runner_is_a_dispatcher(board):
    owner = client(board)
    sign_in(owner, "owner")
    token = owner.post("/api/workers", json={"name": "d"}, headers=H).json()["token"]
    worker = client(board)
    worker.headers.update({"Authorization": f"Bearer {token}"})
    assert worker.post("/v1/worker/heartbeat", json={"gyms": [], "capabilities": DISPATCHER}).status_code == 200
    (mine,) = owner.get("/api/my/workers").json()["workers"]
    assert mine["dispatcher"] is True


def test_the_pool_report_lists_every_machine_and_its_lan_islands():
    from gymhub import poolview

    found = poolview.report(FakePool())
    assert found["pool"] == "default" and found["islands"] == [{"machines": ["jetson", "spark"]}]
    spark, jetson = found["machines"]
    assert spark["name"] == "spark" and spark["online"] and spark["running"] == 1 and spark["gpus"][0]["model"] == "NVIDIA GB10"
    assert not jetson["online"] and jetson["gpus"] == []  # offline machines are listed too, unlike the routing report
    assert "profile" not in spark and "id" not in spark

    class Down:
        def call(self, *args, **kwargs):
            raise OSError("down")

    assert poolview.report(Down()) is None
    assert poolview.clean_report({"machines": "x"}) is None
    cleaned = poolview.clean_report({"pool": "p", "at": 5, "machines": [{"name": "a", "gpus": [{"model": "G" * 500}], "running": "3"}, {"gpus": []}],
                                     "islands": [{"machines": ["a", "ghost"]}, ["a"], {"machines": ["ghost"]}]})
    (only,) = cleaned["machines"]
    assert only["name"] == "a" and len(only["gpus"][0]["model"]) == 80 and only["running"] == 0
    assert cleaned["islands"] == [{"machines": ["a"]}]
    gyms = [("algo", False, True), ("gpu", True, True), ("mine", False, False)]
    assert poolview.gyms_for({"gpus": [{"model": "G"}], "own": True}, gyms) == ["algo", "gpu", "mine"]
    assert poolview.gyms_for({"gpus": [], "own": False}, gyms) == ["algo"]


def test_a_board_that_cannot_reach_the_pool_shows_what_the_dispatcher_reports(board, monkeypatch):
    """#162: the board on Cloud Run, the pool server on a LAN. The dispatcher's heartbeat carries the
    pool's machines; the page lists them, says the board is connected through it, and which gyms each takes."""
    import gymhub.board.poolmachines as poolmachines
    from gymhub import poolview

    monkeypatch.setattr(poolmachines, "pool_client", lambda: None)
    board.store.set("gyms", "gpu", {"title": "GPU", "updated_at": 1.0, "requires": {"gpu": True}, "visibility": "public"})
    owner = client(board)
    sign_in(owner, "owner")
    token = owner.post("/api/workers", json={"name": "spark-10dc"}, headers=H).json()["token"]
    worker = client(board)
    worker.headers.update({"Authorization": f"Bearer {token}"})
    report = poolview.report(FakePool())
    for machine in report["machines"]:
        machine["own"] = True
    assert worker.post("/v1/worker/heartbeat", json={"gyms": [], "capabilities": DISPATCHER, "pool_report": report}).status_code == 200

    listed = owner.get("/api/pool/machines").json()
    assert listed["configured"] and listed["source"] == "bridge" and not listed["stale"] and listed["can_add"] is False
    assert listed["bridge"]["name"] == "spark-10dc" and listed["bridge"]["online"] and listed["bridge"]["dispatcher"]
    spark, jetson = listed["machines"]
    assert spark["online"] and spark["gpus"][0]["model"] == "NVIDIA GB10" and "gpu" in spark["gyms"]
    assert "gpu" not in jetson["gyms"] and "algo" in jetson["gyms"]  # CPU only: no GPU gym
    assert listed["islands"] == [{"machines": ["jetson", "spark"]}]

    # a quiet bridge: its last report is shown, marked stale
    (worker_id,) = [doc_id for doc_id, _doc in board.store.recent("runners", "heartbeat_at", 5)]
    board.store.set("runners", worker_id, {"heartbeat_at": 1.0}, merge=True)
    assert owner.get("/api/pool/machines").json()["stale"] is True

    # only an admin's machine may report the pool
    board.store.set("workers", worker_id, {"owner_uid": "x", "owner_email": "ann@example.com"}, merge=True)
    import gymhub.board.workers as workers_module

    workers_module._cache.clear()
    worker.post("/v1/worker/heartbeat", json={"gyms": [], "capabilities": DISPATCHER, "pool_report": report})
    assert "pool_report" not in board.store.get("runners", worker_id)
    assert owner.get("/api/pool/machines").json()["machines"] == []


def test_the_runner_heartbeat_carries_the_pool_report(dispatch, monkeypatch):
    from gymhub.runner.agent import Runner

    class Backend:
        def heartbeat(self, info):
            self.info = info

    runner = Runner(Backend(), {}, "dispatcher")
    runner.gpu = "none (dispatcher)"
    runner.pool_client = FakePool()
    runner.heartbeat()
    sent = runner.backend.info
    assert sent["capabilities"]["pool"]["machines"] == 1  # routing: the online machine
    assert [m["name"] for m in sent["pool_report"]["machines"]] == ["spark", "jetson"]  # the page: all of them
    assert sum(path.startswith("/v1/machines") for _method, path, _body in runner.pool_client.calls) == 1


def test_an_older_dispatcher_without_a_pool_report_still_lists_the_machines_that_are_up(board, monkeypatch):
    """A dispatcher from before `pool_report` sends only its routing report (online machines and their
    GPUs): the page lists those, marked partial, rather than no pool at all."""
    import gymhub.board.poolmachines as poolmachines

    monkeypatch.setattr(poolmachines, "pool_client", lambda: None)
    owner = client(board)
    sign_in(owner, "owner")
    assert owner.get("/api/pool/machines").json()["bridge"] is None  # nothing connects the board to a pool
    token = owner.post("/api/workers", json={"name": "spark-10dc"}, headers=H).json()["token"]
    worker = client(board)
    worker.headers.update({"Authorization": f"Bearer {token}"})
    pool = {**POOL, "list": [{"name": "spark", "gpus": [{"name": "NVIDIA GB10"}], "own": True}, {"name": "jetson", "gpus": [{"name": "Orin"}], "own": True}]}
    assert worker.post("/v1/worker/heartbeat", json={"gyms": [], "capabilities": {**DISPATCHER, "pool": pool}}).status_code == 200
    listed = owner.get("/api/pool/machines").json()
    assert listed["configured"] and listed["partial"] and listed["bridge"]["name"] == "spark-10dc"
    assert [(m["name"], m["online"], m["gpus"][0]["model"]) for m in listed["machines"]] == [("spark", True, "NVIDIA GB10"), ("jetson", True, "Orin")]
    # a bridge that cannot reach the pool: no machines, and the page says why
    worker.post("/v1/worker/heartbeat", json={"gyms": [], "capabilities": {k: v for k, v in DISPATCHER.items() if k != "pool"}})
    nothing = owner.get("/api/pool/machines").json()
    assert not nothing["configured"] and nothing["bridge"]["online"] and not nothing["bridge"]["reaches_pool"]
