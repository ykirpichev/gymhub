"""Adding a machine: one-time code -> bundle + token; a member's machine runs only that member's jobs."""

import io
import tarfile
import time

from tests.test_board_auth import H, board, client, sign_in  # noqa: F401  (fixture)


def _member(board):
    owner = client(board)
    sign_in(owner, "owner")
    owner.put("/api/admin/settings", json={"allowed_domains": ["example.com", "school.edu"], "members_can_run": True}, headers=H)
    ann = client(board)
    sign_in(ann, "ann")
    return owner, ann


def test_registration_bundle_and_token(board):
    _owner, ann = _member(board)
    reply = ann.post("/api/my/workers/registration", json={"name": "home-4090"}, headers=H).json()
    code = reply["code"]
    assert len(code) == 9 and reply["expires_in"] == 600
    anon = client(board)
    assert anon.get("/v1/register/bundle.tar.gz?code=WRONG-CODE").status_code == 403
    bundle = anon.get(f"/v1/register/bundle.tar.gz?code={code}")
    assert bundle.status_code == 200
    names = tarfile.open(fileobj=io.BytesIO(bundle.content)).getnames()
    assert "gpu_gym/cli.py" in names and "gymkit/cli.py" in names and "gymhub/runner/agent.py" in names
    assert not [n for n in names if "solutions" in n.split("/") or n.endswith("progress.json") or n.startswith("gymhub/board")]
    registered = anon.post("/v1/register", json={"code": code, "host": "box"}).json()
    assert registered["name"] == "home-4090" and registered["token"].startswith("gw_")
    assert anon.post("/v1/register", json={"code": code}).status_code == 403  # single use
    worker = client(board)
    worker.headers.update({"Authorization": f"Bearer {registered['token']}"})
    beat = worker.post("/v1/worker/heartbeat", json={"gpu": "RTX 4090"}).json()
    assert beat["can_publish"] is False
    assert worker.post("/v1/worker/catalog", json={"gym": "algo", "meta": {}, "problems": []}).status_code == 403
    mine = ann.get("/api/my/workers").json()
    assert [w["name"] for w in mine["workers"]] == ["home-4090"] and mine["runs"] == "only your jobs"


def test_expired_code_and_bad_names(board):
    _owner, ann = _member(board)
    assert ann.post("/api/my/workers/registration", json={"name": "has space"}, headers=H).status_code == 422
    code = ann.post("/api/my/workers/registration", json={"name": "box"}, headers=H).json()["code"]
    board.store.set("registrations", code, {"expires_at": time.time() - 1}, merge=True)
    assert client(board).post("/v1/register", json={"code": code}).status_code == 403


def test_member_machine_runs_only_its_owner_jobs(board):
    owner, ann = _member(board)
    code = ann.post("/api/my/workers/registration", json={"name": "box"}, headers=H).json()["code"]
    token = client(board).post("/v1/register", json={"code": code}).json()["token"]
    worker = client(board)
    worker.headers.update({"Authorization": f"Bearer {token}"})
    bob = client(board)
    sign_in(bob, "bob")
    job = {"gym": "algo", "slug": "p", "lang": "python", "action": "run", "code": "x"}
    bob_job = bob.post("/api/jobs", json=job, headers=H).json()["id"]
    owner_job = owner.post("/api/jobs", json=job, headers=H).json()["id"]
    assert worker.post("/v1/worker/lease", json={"gyms": ["algo"]}).json()["job"] is None
    ann_job = ann.post("/api/jobs", json=job, headers=H).json()["id"]
    assert worker.post("/v1/worker/lease", json={"gyms": ["algo"]}).json()["job"]["id"] == ann_job
    assert board.store.get("jobs", bob_job)["status"] == "queued" and board.store.get("jobs", owner_job)["status"] == "queued"


def test_install_script_and_removal(board):
    _owner, ann = _member(board)
    script = client(board).get("/install.sh").text
    assert script.startswith("#!/bin/sh") and "__BOARD__" not in script and "http://testserver" in script
    code = ann.post("/api/my/workers/registration", json={"name": "box"}, headers=H).json()["code"]
    token = client(board).post("/v1/register", json={"code": code}).json()["token"]
    worker_id = token.split("_")[1]
    bob = client(board)
    sign_in(bob, "bob")
    assert bob.delete(f"/api/my/workers/{worker_id}", headers=H).status_code == 404
    assert ann.delete(f"/api/my/workers/{worker_id}", headers=H).status_code == 200
    import gymhub.board.workers as workers

    workers._cache.clear()
    worker = client(board)
    worker.headers.update({"Authorization": f"Bearer {token}"})
    assert worker.post("/v1/worker/heartbeat", json={}).status_code == 401


def test_adding_a_machine_needs_the_run_permission(board):
    owner = client(board)
    sign_in(owner, "owner")
    owner.put("/api/admin/settings", json={"allowed_domains": ["example.com"], "members_can_run": False}, headers=H)
    ann = client(board)
    sign_in(ann, "ann")
    assert ann.post("/api/my/workers/registration", json={"name": "box"}, headers=H).status_code == 403
    owner.put("/api/admin/settings", json={"allowed_domains": ["example.com"], "members_can_run": True}, headers=H)
    import gymhub.board.auth as auth

    auth._user_cache.clear()
    assert ann.post("/api/my/workers/registration", json={"name": "box"}, headers=H).status_code == 200


def test_can_add_machine_behind_iap_only_with_worker_service(board, monkeypatch):
    from gymhub.board import workers
    _owner, ann = _member(board)
    assert ann.get("/api/my/workers").json()["can_add"] is True
    monkeypatch.setenv("GYMHUB_AUTH", "iap")
    monkeypatch.delenv("GYMHUB_WORKER_URL", raising=False)
    assert workers.can_add_machines() is False
    monkeypatch.setenv("GYMHUB_WORKER_URL", "https://board-workers.example")
    assert workers.can_add_machines() is True
