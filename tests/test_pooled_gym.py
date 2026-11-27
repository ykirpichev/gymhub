"""Grading board jobs on the compute pool (gymhub/runner/pooled.py, issue #45 M4 in small).

A real pool server and an agent with the process runtime grade an algo exercise the way the runner
would hand it over: the submitted code, the user's progress, and the gym as a slice without answers.
"""

import json

import pytest

from pathlib import Path
from types import SimpleNamespace

from gymhub.runner import pooled
from gymhub.runner.adapters import bundle_adapter, kit_adapter
from pool.client import Client
from tests.test_board_auth import board  # noqa: F401  (fixture)
from tests.test_pool import db_url, pool_env, start_agent  # noqa: F401  (fixtures)

EDIT_DISTANCE = """
def solve(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
"""


MERGE_INTERVALS = """
def solve(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged
"""


@pytest.fixture()
def pool_settings(monkeypatch):
    monkeypatch.setenv("GYMHUB_EXECUTOR", "pool")
    monkeypatch.setenv("GYMHUB_POOL_IMAGE", "unused-by-the-process-runtime")


def _grade(code: str, action: str = "test"):
    seen = []
    job = {"code": code, "options": {}, "user_progress": None, "requires": {"gpu": False}}
    output, ok, exit_code, progress, hardware = pooled.run(Client.member(), kit_adapter("algo"), job, "edit_distance", "python",
                                                           action, lambda text: seen.append(text) and False, 300)
    assert hardware and hardware.startswith("CPU")  # the pool machine's class (no GPU in tests)
    return output, ok, progress, seen


def test_a_correct_solution_passes_on_the_pool(pool_env, pool_settings):
    start_agent(pool_env, "grader")
    output, ok, progress, seen = _grade(EDIT_DISTANCE)
    assert ok, output
    assert "[graded on grader]" in output and seen  # output streamed back while it ran
    # a CPU gym's job says why it has no GPU, not "GPUs none" (#163)
    assert "algo [python] runs on the CPU: no GPU requested" in output and "[pool] on grader, no GPU requested" in output
    assert "GPUs none" not in output
    entry = progress["exercises"]["edit_distance"]
    assert entry["langs"]["python"]["last_status"] == "pass" and entry["langs"]["python"]["passes"] == 1


def test_a_wrong_solution_fails_and_is_recorded(pool_env, pool_settings):
    start_agent(pool_env, "grader")
    output, ok, progress, _ = _grade("def solve(a, b):\n    return 0\n")
    assert not ok
    assert progress["exercises"]["edit_distance"]["langs"]["python"]["last_status"] == "fail"


def test_the_job_folder_has_no_answers(tmp_path):
    adapter = kit_adapter("algo")
    command = pooled.stage(adapter, "edit_distance", "python", EDIT_DISTANCE, {"action": "test"}, None, tmp_path)
    files = {str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file()}
    assert "job/solution.py" in files and "gyms/algo/problems/edit_distance/problem.py" in files
    assert not any("/solution/" in f or f.endswith("answer.json") or "progress.json" in f and not f.startswith("job/") for f in files)
    assert "gymkit gyms/algo test edit_distance --lang python --file job/solution.py" in command[-1]
    assert json.loads((tmp_path / "job" / "progress.json").read_text())["exercises"] == {}


def test_a_grader_image_job_carries_only_the_gym_slice_and_the_job(tmp_path):
    """With a grader image (graders at /opt/gym in the image) /work has no gymkit/ or gpu_gym/."""
    command = pooled.stage(kit_adapter("algo"), "edit_distance", "python", EDIT_DISTANCE, {"action": "test"}, None, tmp_path,
                           graders=False)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["gyms", "job"]
    assert sorted(p.name for p in (tmp_path / "gyms").iterdir()) == ["algo"]
    assert sorted(p.name for p in (tmp_path / "job").iterdir()) == ["progress.json", "solution.py"]
    # the image's own PYTHONPATH (/opt/gym) is kept, after /work
    assert 'export PYTHONPATH="$POOL_WORK${PYTHONPATH:+:$PYTHONPATH}"' in command[-1]


def test_a_legacy_image_job_also_ships_the_graders(tmp_path):
    pooled.stage(kit_adapter("algo"), "edit_distance", "python", EDIT_DISTANCE, {"action": "test"}, None, tmp_path, graders=True)
    assert (tmp_path / "gymkit" / "cli.py").is_file() and (tmp_path / "gpu_gym" / "cli.py").is_file()


@pytest.fixture()
def no_image_settings(monkeypatch):
    for name in ("GYMHUB_POOL_IMAGE", "GYMHUB_POOL_IMAGES", "GYMHUB_GRADER_IMAGE", "GYMHUB_GRADER_VERSION",
                 "GYMHUB_POOL_SHIP_GRADERS"):
        monkeypatch.delenv(name, raising=False)


def test_the_image_follows_the_machine_profile(no_image_settings):
    from pool import spec as specs

    repo = "ghcr.io/ykirpichev/gymhub/grader"
    cpu = specs.normalize(pooled.spec_for(kit_adapter("algo"), "test", ["x"], needs_gpu=False))
    gpu = specs.normalize(pooled.spec_for(kit_adapter("algo"), "test", ["x"], needs_gpu=True))
    # labels as pool/profile.py sets them on each kind of machine
    x86_card = {"arch": "x86_64", "gpu": "discrete"}
    spark = {"arch": "aarch64", "gpu": "unified"}
    jetson = {"arch": "aarch64", "gpu": "tegra"}
    laptop = {"arch": "arm64", "gpu": "none"}
    assert {specs.image_for(cpu, m) for m in (x86_card, spark, jetson, laptop)} == {f"{repo}:cpu"}
    assert specs.image_for(gpu, x86_card) == specs.image_for(gpu, spark) == f"{repo}:cuda"
    assert specs.image_for(gpu, jetson) == f"{repo}:l4t-r36"
    assert not pooled.ships_graders(False) and not pooled.ships_graders(True)


def test_image_overrides_and_pins(no_image_settings, monkeypatch):
    monkeypatch.setenv("GYMHUB_GRADER_VERSION", "abc1234")
    assert pooled.choose_images(True) == ("ghcr.io/ykirpichev/gymhub/grader:cuda-abc1234",
                                          {"gpu=tegra": "ghcr.io/ykirpichev/gymhub/grader:l4t-r36-abc1234"})
    assert not pooled.ships_graders(True)
    monkeypatch.setenv("GYMHUB_GRADER_IMAGE", "registry.lan:5000/grader")
    assert pooled.choose_images(False) == ("registry.lan:5000/grader:cpu-abc1234", {})
    assert pooled.is_grader_image("registry.lan:5000/grader@sha256:00")
    assert not pooled.is_grader_image("registry.lan:5000/other:cpu")
    # GYMHUB_POOL_IMAGES overrides one platform, GYMHUB_POOL_IMAGE the rest; a legacy image gets the graders in /work
    monkeypatch.setenv("GYMHUB_POOL_IMAGES", "gpu=tegra=my-l4t")
    image, per_platform = pooled.choose_images(True)
    assert image == "registry.lan:5000/grader:cuda-abc1234" and per_platform == {"gpu=tegra": "my-l4t"}
    assert pooled.ships_graders(True)  # a Jetson may run my-l4t, which has no graders inside
    monkeypatch.delenv("GYMHUB_POOL_IMAGES")
    monkeypatch.setenv("GYMHUB_POOL_IMAGE", "nvcr.io/nvidia/pytorch:25.01-py3")
    assert pooled.choose_images(False)[0] == "nvcr.io/nvidia/pytorch:25.01-py3" and pooled.ships_graders(False)
    monkeypatch.setenv("GYMHUB_POOL_SHIP_GRADERS", "0")
    assert not pooled.ships_graders(True)


def test_settings(monkeypatch):
    monkeypatch.setenv("GYMHUB_POOL_IMAGES", "gpu=tegra=l4t-img, gpu=unified=spark-img")
    assert pooled.images() == {"gpu=tegra": "l4t-img", "gpu=unified": "spark-img"}
    assert pooled.actions() == {"run", "test", "bench"}
    monkeypatch.delenv("GYMHUB_EXECUTOR", raising=False)
    assert pooled.enabled()  # the default: other users' code runs only on the pool
    monkeypatch.setenv("GYMHUB_EXECUTOR", "local")
    assert not pooled.enabled()


def test_board_jobs_never_get_a_network(monkeypatch):
    """Pool jobs from the board run with network: none; one that asks for peers (the host's
    network) or island is refused before anything reaches the pool."""
    monkeypatch.setenv("GYMHUB_POOL_IMAGE", "grader:1")
    adapter = kit_adapter("algo")
    assert pooled.spec_for(adapter, "test", ["true"], False)["network"] == "none"
    for network in ("peers", "island"):
        with pytest.raises(RuntimeError, match="may not use network"):
            pooled.check_network({"network": network})

    calls = []
    client = SimpleNamespace(upload_folder=lambda *a, **k: {"artifact": "a1"}, call=lambda *a, **k: calls.append(a))
    real = pooled.spec_for
    monkeypatch.setattr(pooled, "spec_for", lambda *a, **k: {**real(*a, **k), "network": "peers"})
    monkeypatch.setattr(pooled, "stage", lambda *a, **k: ["true"])
    job = {"code": "x", "requires": {"gpu": False}}
    with pytest.raises(RuntimeError, match="peers"):
        pooled.run(client, adapter, job, "edit_distance", "python", "test", lambda text: False, 60)
    assert calls == []


def test_an_uploaded_gpu_gym_is_graded_from_its_own_folder(tmp_path):
    """gpu_gym grades gyms/gpu unless GPU_GYM_ROOT says otherwise; an uploaded GPU gym has its own id."""
    root = Path(__file__).resolve().parents[1] / "examples" / "gyms" / "gpu-scale"
    adapter = bundle_adapter(root)
    command = pooled.stage(adapter, "scale", "pytorch", "def solve(X, Y, N):\n    pass\n", {"action": "test"}, None, tmp_path)
    assert (tmp_path / "gyms" / "gpu-scale" / "problems" / "scale" / "problem.py").is_file()
    assert not (tmp_path / "gyms" / "gpu-scale" / "problems" / "scale" / "solution").exists()
    assert 'GPU_GYM_ROOT="$POOL_WORK/gyms/gpu-scale"' in command[-1]
    assert "-m gpu_gym test scale --lang pytorch" in command[-1]


def test_the_board_grades_on_the_pool_end_to_end(board, pool_env, pool_settings, monkeypatch):
    """Board -> runner (GYMHUB_EXECUTOR=pool) -> pool agent: test and bench, with progress and a best
    time recorded under the pool machine's hardware."""
    from gymhub.runner.agent import Runner
    from gymhub.runner.backends import DirectBackend
    from tests.test_board_auth import H, client, sign_in

    start_agent(pool_env, "grader")
    runner = Runner(DirectBackend(board.store, "spark"), {"algo": kit_adapter("algo")}, "spark")
    runner.gpu = "none"
    runner.heartbeat()  # reports the pool, so the direct backend leases ann's isolated jobs
    runner.publish_catalog(runner.adapters["algo"])
    owner = client(board)
    sign_in(owner, "owner")
    owner.put("/api/admin/settings", json={"allowed_domains": ["example.com"], "members_can_run": True}, headers=H)
    ann = client(board)
    sign_in(ann, "ann")  # a member: her code is graded isolated, so on the pool
    for action in ("test", "bench"):  # merge_intervals: no time limit, so a slow CI machine still passes the bench
        reply = ann.post("/api/jobs", json={"gym": "algo", "slug": "merge_intervals", "lang": "python", "action": action,
                                            "code": MERGE_INTERVALS}, headers=H)
        assert reply.status_code == 200, reply.text
        job = runner.backend.lease(["algo"])
        assert job["id"] == reply.json()["id"] and job["isolated"]
        runner.run_job(job)
        done = board.store.get("jobs", job["id"])
        assert done["ok"], done["output"]
        assert "[graded on grader]" in done["output"]
    uid = ann.get("/api/me").json()["user"]["uid"]
    progress = board.store.get("user_progress", f"{uid}__algo")["data"]["exercises"]["merge_intervals"]["langs"]["python"]
    assert progress["first_solved"]
    best = board.store.get("bests", f"{uid}__algo__merge_intervals__python")["by_hardware"]
    assert list(best) == [done["hardware"]] and done["hardware"].startswith("CPU")  # the pool machine's class


def test_a_pool_machines_gpu_model_is_its_hardware_class():
    assert pooled.profile_hardware({"gpus": [{"index": 0, "model": "NVIDIA GB10"}], "arch": "aarch64"}) == "NVIDIA GB10"
    assert pooled.profile_hardware({"gpus": [], "arch": "aarch64"}) == "CPU aarch64"


def test_a_gym_that_is_not_public_reaches_only_the_bridges_own_pool_machines(board, pool_env, pool_settings, monkeypatch):
    """#117 on the pool path: another pool member's machine never gets a job (graders, hidden tests)
    of a gym that is not public; with no own machine online the runner grades it itself (here only
    because the test trusts it, GYMHUB_TRUST_JOBS; otherwise it is left queued); a
    public gym's job may still go to any machine of the pool."""
    import threading
    import time

    from gymhub.runner.agent import Runner
    from gymhub.runner.backends import DirectBackend
    from tests.test_board_auth import H, client, sign_in
    from tests.test_pool import invite, wait_for

    monkeypatch.setenv("GYMHUB_TRUST_JOBS", "1")
    db = pool_env["db"]
    start_agent(pool_env, "theirs", member=invite(pool_env, name="bob"))  # the bridge submits as ann
    runner = Runner(DirectBackend(board.store, "spark"), {"algo": kit_adapter("algo")}, "spark")
    runner.gpu = "none"
    runner.publish_catalog(runner.adapters["algo"])
    owner = client(board)
    sign_in(owner, "owner")
    owner.put("/api/admin/settings", json={"allowed_domains": ["example.com"], "members_can_run": True}, headers=H)
    member = client(board)
    sign_in(member, "ann")
    uid = member.get("/api/me").json()["user"]["uid"]
    board.store.set("gyms", "algo", {"visibility": "unlisted", "learners": [uid]}, merge=True)

    def submit() -> dict:
        reply = member.post("/api/jobs", json={"gym": "algo", "slug": "merge_intervals", "lang": "python", "action": "test",
                                               "code": MERGE_INTERVALS}, headers=H)
        assert reply.status_code == 200, reply.text
        job = runner.backend.lease(["algo"])
        assert job["id"] == reply.json()["id"]
        return job

    def pool_jobs() -> list[dict]:
        return [{**row, "spec": json.loads(row["spec"])} for row in db.all("select * from jobs order by created_at")]

    # The bridge reports its own pool machines: none online. The board says the gym is not public,
    # so the job is graded here (trusted, see above), and nothing reaches the pool.
    runner.heartbeat()
    assert runner.pool_bridge["machines"] == 1 and runner.pool_bridge["own"] == {"machines": 0, "gpus": []}
    job = submit()
    assert job["gym_public"] is False
    runner.run_job(job)
    done = board.store.get("jobs", job["id"])
    assert done["ok"] and "[pool job" not in done["output"], done["output"]
    assert pool_jobs() == []

    # Sent to the pool (the bridge has not reported yet): only for its own machines. The other
    # member's machine is online and idle and still never gets it; the bridge's own machine does.
    runner.pool_bridge = None
    job = submit()
    worker = threading.Thread(target=runner.run_job, args=(job,), daemon=True)
    worker.start()
    wait_for(pool_jobs)
    time.sleep(2)  # several polls of the other member's agent
    (queued,) = pool_jobs()
    assert queued["status"] == "queued" and queued["machine"] is None and queued["spec"]["resources"]["machines"] == "own"
    start_agent(pool_env, "mine")
    worker.join(timeout=120)
    done = board.store.get("jobs", job["id"])
    assert done["ok"] and "[graded on mine]" in done["output"], done["output"]

    # A public gym: any machine of the pool.
    board.store.set("gyms", "algo", {"visibility": "public"}, merge=True)
    job = submit()
    assert job["gym_public"] is True
    runner.run_job(job)
    assert board.store.get("jobs", job["id"])["ok"]
    assert "machines" not in pool_jobs()[-1]["spec"]["resources"]


def test_a_pool_image_failure_names_the_machine_and_the_fix():
    """#163: the machine could not pull a private grader image. Not a failed grade: what to build where."""
    image = pooled.grader_image("l4t-r36")
    final = {"status": "failed", "machine": "jetson", "exit_code": None,
             "error": f"ImageUnavailable: image {image} is not on this machine and could not be pulled: unauthorized"}
    with pytest.raises(pooled.ImageUnavailable) as caught:
        pooled.check_image(final, "[pool job j1]\n")
    assert str(caught.value) == f"Not graded: jetson does not have the image {image} and could not pull it."
    assert caught.value.output.startswith("[pool job j1]\n")
    assert "On jetson, from the repo root: sh deploy/grader/build-local.sh l4t-r36" in caught.value.output


def test_image_failures_with_a_versioned_or_foreign_image(monkeypatch):
    monkeypatch.setenv("GYMHUB_GRADER_VERSION", "v3")
    final = {"status": "failed", "machine": "spark", "error": f"ImageUnavailable: image {pooled.grader_image('cuda')} is not on ..."}
    with pytest.raises(pooled.ImageUnavailable, match="cuda-v3") as caught:
        pooled.check_image(final, "")
    assert "build-local.sh cuda" in caught.value.output
    final["error"] = "ImageUnavailable: image example.org/mine:1 is not on this machine and could not be pulled: no"
    with pytest.raises(pooled.ImageUnavailable) as caught:
        pooled.check_image(final, "")
    assert "Pull or build example.org/mine:1 on spark." in caught.value.output


def test_other_pool_failures_are_left_to_the_grade():
    pooled.check_image({"status": "failed", "exit_code": 1, "error": None}, "")  # a failing solution
    pooled.check_image({"status": "failed", "error": "machine jetson stopped reporting (no retries left)"}, "")
    pooled.check_image({"status": "succeeded", "error": "ImageUnavailable: image x is not"}, "")


def test_the_board_reports_a_missing_image_as_a_machine_error(board, pool_env, pool_settings):
    """Board -> runner -> a pool machine without the image: the job ends in error with the fix, and
    nothing is recorded as a failed attempt."""
    from gymhub.runner.agent import Runner
    from gymhub.runner.backends import DirectBackend
    from pool.agent import ImageUnavailable
    from tests.test_board_auth import H, client, sign_in

    agent = start_agent(pool_env, "jetson")

    def no_image(image, say):
        raise ImageUnavailable(f"image {image} is not on this machine and could not be pulled: unauthorized")

    agent.runtime.prepare = no_image
    runner = Runner(DirectBackend(board.store, "spark"), {"algo": kit_adapter("algo")}, "spark")
    runner.gpu = "none"
    runner.heartbeat()
    runner.publish_catalog(runner.adapters["algo"])
    owner = client(board)
    sign_in(owner, "owner")
    owner.put("/api/admin/settings", json={"allowed_domains": ["example.com"], "members_can_run": True}, headers=H)
    ann = client(board)
    sign_in(ann, "ann")
    reply = ann.post("/api/jobs", json={"gym": "algo", "slug": "merge_intervals", "lang": "python", "action": "test",
                                        "code": MERGE_INTERVALS}, headers=H)
    job = runner.backend.lease(["algo"])
    assert job["id"] == reply.json()["id"]
    runner.run_job(job)
    done = board.store.get("jobs", job["id"])
    assert done["status"] == "error" and not done["ok"], done
    assert "Not graded: jetson does not have the image unused-by-the-process-runtime" in done["output"]
    assert "[pool] ImageUnavailable" in done["output"]  # the pool's own log line is kept
    uid = ann.get("/api/me").json()["user"]["uid"]
    assert board.store.get("user_progress", f"{uid}__algo") is None
