"""Uploaded gyms (issue #14): bundles, validation on a worker, publishing, and "tests changed" markers."""

import io
import json
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

import pytest

from gymhub import jobqueue, packages
from tests.test_board_auth import H, board, client, shared_token, sign_in  # noqa: F401  (fixture)

REPO = Path(__file__).resolve().parents[1]
HELLO = REPO / "examples" / "gyms" / "hello"
GZ = {**H, "Content-Type": "application/gzip"}


def _tar(files: dict[str, bytes], links: dict[str, str] | None = None) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for name, target in (links or {}).items():
            info = tarfile.TarInfo(name)
            info.type, info.linkname = tarfile.SYMTYPE, target
            tar.addfile(info)
    return buffer.getvalue()


def _hello_copy(tmp_path: Path, name: str = "hello") -> Path:
    target = tmp_path / name
    shutil.copytree(HELLO, target)
    return target


def _owner(board):
    c = client(board)
    sign_in(c, "owner")
    return c


def _worker(board, owner, bundles=True):
    token = shared_token(owner, f"w{time.time_ns()}")  # it validates members' uploads too
    w = client(board)
    w.headers.update({"Authorization": f"Bearer {token}"})
    caps = {"gpus": [], "tools": ["g++"], "sandbox": True, **({"bundles": True} if bundles else {})}
    assert w.post("/v1/worker/heartbeat", json={"gyms": ["algo"], "capabilities": caps}).status_code == 200
    import gymhub.board.workers as workers

    workers._capabilities.clear()
    return w


def passing_report(worker, job) -> dict:
    """The report of a validation run in which every check the stored bundle requires passed."""
    data = worker.post(f"/v1/worker/jobs/{job['id']}/bundle").content
    with tempfile.TemporaryDirectory() as tmp:
        required = sorted(packages.required_checks(packages.unpack(data, Path(tmp))))
    return {"ok": True, "warnings": [], "enrich": {},
            "checks": [{"slug": slug, "lang": lang, "what": what, "ok": True} for slug, lang, what in required]}


def complete_validation(worker, job, **result):
    """The machine finishes the validation job with a clean report (override fields with keywords)."""
    final = {"status": "done", "ok": True, "output": "", "result": {**passing_report(worker, job), **result}}
    return worker.post(f"/v1/worker/jobs/{job['id']}/complete", json={"final": final, "lease_id": job["lease_id"]})


def _upload(owner, data: bytes):
    return owner.post("/api/gym-versions", content=data, headers=GZ)


# ---- bundles --------------------------------------------------------------------------------


def test_pack_is_deterministic_and_round_trips(tmp_path):
    first, second = packages.pack(HELLO), packages.pack(HELLO)
    assert first == second
    package = packages.unpack(first, tmp_path / "out")
    assert package.id == "hello" and package.slugs() == ["sum_list", "is_palindrome"]
    assert packages.validate_upload(package) == []
    assert package.references("sum_list")  # solution/ is packed: workers need it, learners never see it
    catalog = packages.public_catalog(package)
    assert "return sum(nums)" not in json.dumps(catalog)


@pytest.mark.parametrize(
    "files, links, message",
    [
        ({"../evil.py": b"x", "gym.json": b"{}"}, None, "unsafe path"),
        ({"/abs.py": b"x", "gym.json": b"{}"}, None, "unsafe path"),
        ({"gym.json": b'{"id": "x"}'}, {"problems/link": "/etc/passwd"}, "plain files"),
        ({"problems/a.txt": b"x"}, None, "no gym.json"),
        ({"gym.json": b"not json"}, None, "not valid JSON"),
        ({"gym.json": b"{}"}, None, "must name the gym"),
    ],
)
def test_unpack_refuses_unsafe_or_broken_bundles(tmp_path, files, links, message):
    with pytest.raises(packages.BundleError, match=message):
        packages.unpack(_tar(files, links), tmp_path / "out")
    assert not (tmp_path / "evil.py").exists()


def test_unpack_refuses_garbage_and_oversize(tmp_path, monkeypatch):
    with pytest.raises(packages.BundleError, match="not a .tar.gz"):
        packages.unpack(b"hello", tmp_path / "a")
    monkeypatch.setattr(packages, "MAX_UNPACKED_BYTES", 10)
    with pytest.raises(packages.BundleError, match="unpacks to more"):
        packages.unpack(_tar({"gym.json": b'{"id": "abc", "padding": "....."}'}), tmp_path / "b")


def test_tests_hash_follows_the_grader_not_the_statement(tmp_path):
    root = _hello_copy(tmp_path)
    before = packages.load(root).tests_hash("sum_list")
    (root / "problems/sum_list/PROBLEM.md").write_text("# Reworded\n")
    (root / "problems/sum_list/solution/solution.py").write_text("def solve(nums):\n    return sum(nums)  # tidier\n")
    assert packages.load(root).tests_hash("sum_list") == before
    (root / "problems/sum_list/problem.py").write_text((root / "problems/sum_list/problem.py").read_text() + "\n# more cases\n")
    assert packages.load(root).tests_hash("sum_list") != before


# ---- upload and publish through the board -------------------------------------------------------


def test_only_admins_upload_and_bad_bundles_are_rejected_with_reasons(board, tmp_path):
    member = client(board)
    board.store.set("settings", "auth", {"allowed_domains": ["example.com"]})
    sign_in(member, "ann")
    assert member.post("/api/gym-versions", content=packages.pack(HELLO), headers=GZ).status_code == 403

    owner = _owner(board)
    root = _hello_copy(tmp_path)
    (root / "problems/sum_list/PROBLEM.md").unlink()
    reply = _upload(owner, packages.pack(root))
    assert reply.status_code == 400 and any("PROBLEM.md" in error for error in reply.json()["errors"])
    assert _upload(owner, b"garbage").status_code == 400
    assert board.store.where("gym_versions", "gym", "hello") == []
    # A gym published from a machine's local files cannot be replaced by an upload.
    algo = _hello_copy(tmp_path, "algo")
    manifest = json.loads((algo / "gym.json").read_text())
    (algo / "gym.json").write_text(json.dumps({**manifest, "id": "algo"}))
    assert "local files" in _upload(owner, packages.pack(algo)).json()["detail"]


def test_upload_validate_publish(board):
    owner = _owner(board)
    data = packages.pack(HELLO)
    version = _upload(owner, data).json()["version"]
    assert version["status"] == "validating" and version["version"] == 1 and version["exercises"] == 2
    assert _upload(owner, data).json()["version"]["duplicate"]  # the same bytes again: no new version
    assert board.store.get("gyms", "hello") is None  # nothing visible before validation

    old_worker = _worker(board, owner, bundles=False)
    assert old_worker.post("/v1/worker/lease", json={"gyms": ["algo"]}).json()["job"] is None
    worker = _worker(board, owner)
    job = worker.post("/v1/worker/lease", json={"gyms": ["algo"]}).json()["job"]
    assert job["action"] == "validate" and job["bundle"]["version"] == 1
    assert worker.post(f"/v1/worker/jobs/{job['id']}/bundle").content == data
    assert old_worker.post(f"/v1/worker/jobs/{job['id']}/bundle").status_code == 409  # only the lease holder

    enrich = {"sum_list": {"benchable": False, "case_template": {"nums": [1]}, "statement": "hacked"}}
    assert complete_validation(worker, job, enrich=enrich).status_code == 200

    versions = owner.get("/api/gym-versions").json()["versions"]
    assert versions[0]["status"] == "published" and versions[0]["live"]
    gym = owner.get("/api/gyms/hello").json()
    assert gym["gym"]["source"] == "bundle" and gym["gym"]["version"] == 1
    assert [problem["slug"] for problem in gym["problems"]] == ["sum_list", "is_palindrome"]
    problem = board.store.get("problems", "hello__sum_list")
    assert problem["case_template"] == {"nums": [1]} and problem["tests_hash"]
    assert "hacked" not in json.dumps(problem)  # a worker adds only grader-derived fields
    # A machine's local gym with the same id cannot overwrite the uploaded one.
    reply = worker.post("/v1/worker/catalog", json={"gym": "hello", "meta": {}, "problems": []})
    assert reply.status_code == 409


def test_a_failed_validation_is_never_published(board):
    owner = _owner(board)
    _upload(owner, packages.pack(HELLO))
    worker = _worker(board, owner)
    job = worker.post("/v1/worker/lease", json={"gyms": []}).json()["job"]
    checks = [{"slug": "sum_list", "lang": "python", "what": "starter fails", "ok": False, "detail": "the starter already passes"}]
    final = {"status": "done", "ok": False, "output": "", "result": {"ok": False, "checks": checks}}
    worker.post(f"/v1/worker/jobs/{job['id']}/complete", json={"final": final, "lease_id": job["lease_id"]})
    version = owner.get("/api/gym-versions?gym=hello").json()["versions"][0]
    assert version["status"] == "failed" and "starter already passes" in version["errors"][0]
    assert board.store.get("gyms", "hello") is None
    # A worker that crashed or sent nothing also fails the version.
    _upload(owner, packages.pack(HELLO) + b"")  # same bytes, but v1 failed: a new version
    job = worker.post("/v1/worker/lease", json={"gyms": []}).json()["job"]
    worker.post(f"/v1/worker/jobs/{job['id']}/complete", json={"final": {"status": "error", "ok": False, "output": "boom"}, "lease_id": job["lease_id"]})
    assert owner.get("/api/gym-versions?gym=hello").json()["versions"][0]["status"] == "failed"
    assert board.store.get("gyms", "hello") is None


# ---- a real runner validates and grades (sandbox off) -----------------------------------------


def _runner(board, monkeypatch):
    from gymhub.runner.agent import Runner
    from gymhub.runner.backends import DirectBackend

    monkeypatch.setenv("GYMHUB_TRUST_JOBS", "1")
    import gymhub.runner.bundles as bundles

    return Runner(DirectBackend(board.store, "spark"), {}, "spark"), bundles


def _run_next(runner) -> dict:
    job = runner.backend.lease([])
    assert job is not None
    runner.run_job(job)
    return runner.backend.store.get("jobs", job["id"])


def test_runner_validates_bundles_and_grades_against_the_published_version(board, tmp_path, monkeypatch):
    runner, bundles = _runner(board, monkeypatch)
    monkeypatch.setattr(bundles, "ROOT", tmp_path / "bundles")
    owner = _owner(board)

    # Broken: a starter that already passes. Validation fails and nothing is published.
    broken = _hello_copy(tmp_path, "broken")
    shutil.copy(broken / "problems/sum_list/solution/solution.py", broken / "problems/sum_list/starter/solution.py")
    _upload(owner, packages.pack(broken))
    job = _run_next(runner)
    assert job["status"] == "done" and not job["ok"]
    version = board.store.get("gym_versions", "hello__v0001")
    assert version["status"] == "failed" and any("starter fails" in error for error in version["errors"])
    assert board.store.get("gyms", "hello") is None

    # The real thing passes and is published.
    _upload(owner, packages.pack(HELLO))
    job = _run_next(runner)
    assert job["ok"], job["output"]
    assert board.store.get("gym_versions", "hello__v0002")["status"] == "published"
    assert board.store.get("problems", "hello__sum_list")["benchable"] is False

    # The owner's jobs in an uploaded gym run from the bundle, isolated, with the owner's own workspace.
    code = (HELLO / "problems/sum_list/solution/solution.py").read_text()
    reply = owner.post("/api/jobs", json={"gym": "hello", "slug": "sum_list", "lang": "python", "action": "test", "code": code}, headers=H)
    queued = board.store.get("jobs", reply.json()["id"])
    assert queued["isolated"] and queued["bundle"]["version"] == 2 and queued["tests_hash"]
    done = _run_next(runner)
    assert done["ok"], done["output"]
    detail = owner.get("/api/gyms/hello/problems/sum_list").json()
    assert detail["problem"]["current"]["python"] == code and not detail["owner"]
    assert detail["tests_changed"] == {}

    # v3 changes the grader: the earlier solve is now marked "tests changed".
    changed = _hello_copy(tmp_path, "changed")
    grader = changed / "problems/sum_list/problem.py"
    grader.write_text(grader.read_text().replace('{"_name": "empty", "nums": []},', '{"_name": "empty", "nums": []},\n        {"_name": "one", "nums": [7]},'))
    _upload(owner, packages.pack(changed))
    assert _run_next(runner)["ok"]
    detail = owner.get("/api/gyms/hello/problems/sum_list").json()
    assert detail["tests_changed"]["python"]["version"] == 2
    assert owner.get("/api/gyms/hello").json()["tests_changed"] == {"sum_list": ["python"]}
    # Untouched exercises keep their solves valid.
    assert "is_palindrome" not in owner.get("/api/gyms/hello").json()["tests_changed"]


def test_local_solves_are_recorded_too(board):
    board.store.set("problems", "algo__p", {"tests_hash": "aaa"}, merge=True)
    job_id = "j1"
    board.store.set("jobs", job_id, {"gym": "algo", "slug": "p", "lang": "python", "action": "test", "uid": "u1",
                                     "tests_hash": "aaa", "runner": "w", "status": "running"})
    jobqueue.complete(board.store, job_id, "w", {"status": "done", "ok": True, "output": ""})
    assert board.store.get("solves", "u1__algo__p__python")["tests_hash"] == "aaa"


# ---- the board decides whether a version publishes (#53) ---------------------------------------


def _validating(board):
    owner = _owner(board)
    _upload(owner, packages.pack(HELLO))
    worker = _worker(board, owner)
    job = worker.post("/v1/worker/lease", json={"gyms": []}).json()["job"]
    return owner, worker, job


def test_a_partial_check_list_does_not_publish(board):
    owner, worker, job = _validating(board)
    report = passing_report(worker, job)
    assert len(report["checks"]) >= 6  # two exercises: grader, starter and reference each
    one = {"checks": [report["checks"][0]]}  # a single passing check used to publish the whole gym
    assert complete_validation(worker, job, **one).status_code == 200
    version = owner.get("/api/gym-versions?gym=hello").json()["versions"][0]
    assert version["status"] == "failed" and any("no result for" in error for error in version["errors"])
    assert board.store.get("gyms", "hello") is None


def test_a_check_for_something_the_package_does_not_have_does_not_make_up_for_a_missing_one(board):
    owner, worker, job = _validating(board)
    checks = passing_report(worker, job)["checks"]
    checks = [c for c in checks if not (c["slug"] == "is_palindrome" and c["what"] == "reference passes")]
    checks.append({"slug": "invented", "lang": "python", "what": "reference passes", "ok": True})
    complete_validation(worker, job, checks=checks)
    assert owner.get("/api/gym-versions?gym=hello").json()["versions"][0]["status"] == "failed"


def test_a_dead_validator_fails_the_version_after_the_last_try(board):
    owner = _owner(board)
    _upload(owner, packages.pack(HELLO))
    worker = _worker(board, owner)
    for attempt in range(jobqueue.MAX_TRIES):
        job = worker.post("/v1/worker/lease", json={"gyms": []}).json()["job"]
        assert job["tries"] == attempt + 1
        board.store.set("jobs", job["id"], {"lease_expires_at": time.time() - 1}, merge=True)
        jobqueue.sweep(board.store)
    assert board.store.get("jobs", job["id"])["status"] == "error"
    version = owner.get("/api/gym-versions?gym=hello").json()["versions"][0]
    assert version["status"] == "failed" and "stopped responding" in version["errors"][0]


def test_a_cancelled_validation_fails_its_version(board):
    owner, worker, job = _validating(board)
    board.store.set("jobs", job["id"], {"cancel": True, "lease_expires_at": time.time() - 1}, merge=True)
    jobqueue.sweep(board.store)
    assert board.store.get("jobs", job["id"])["status"] == "cancelled"
    assert owner.get("/api/gym-versions?gym=hello").json()["versions"][0]["status"] == "failed"


def test_only_an_admins_machine_validates_and_publishes(board):
    owner = _owner(board)
    owner.put("/api/admin/settings", json={"allowed_domains": ["example.com"], "members_can_run": True}, headers=H)
    ann = client(board)
    sign_in(ann, "ann")
    code = ann.post("/api/my/workers/registration", json={"name": "anns"}, headers=H).json()["code"]
    own_token = client(board).post("/v1/register", json={"code": code}).json()["token"]
    own = client(board)
    own.headers.update({"Authorization": f"Bearer {own_token}"})
    caps = {"gpus": [], "tools": [], "sandbox": True, "bundles": True}
    assert own.post("/v1/worker/heartbeat", json={"gyms": [], "capabilities": caps}).status_code == 200
    import gymhub.board.workers as workers

    workers._capabilities.clear()
    # Ann uploads (her gym, her uid on the job); her own machine would report on her own code.
    uid = ann.get("/api/me").json()["user"]["uid"]
    board.store.set("jobs", "v1", {"gym": "hello", "action": "validate", "status": "queued", "uid": uid, "created_at": time.time(),
                                   "isolated": True, "bundle": {"sha": "0" * 64, "key": "k", "version": 1}, "output": ""})
    assert own.post("/v1/worker/lease", json={"gyms": []}).json()["job"] is None
    admin_machine = _worker(board, owner)
    assert admin_machine.post("/v1/worker/lease", json={"gyms": []}).json()["job"]["id"] == "v1"
