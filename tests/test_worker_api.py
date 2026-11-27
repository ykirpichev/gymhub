"""Worker tokens and the worker API: leases, output, cancel, completion, isolation, expiry."""

import time

from gymhub import jobqueue
from tests.test_board_auth import H, board, client, shared_token, sign_in  # noqa: F401  (fixture)


def _admin_and_token(board, name="spark", shared=True):
    owner = client(board)
    sign_in(owner, "owner")
    token = shared_token(owner, name) if shared else owner.post("/api/workers", json={"name": name}, headers=H).json()["token"]
    return owner, token


def _worker(board, token):
    c = client(board)
    c.headers.update({"Authorization": f"Bearer {token}"})
    return c


def _queue(board, **fields):
    job_id = f"job{time.time_ns()}"
    board.store.set("jobs", job_id, {"gym": "algo", "slug": "p", "problem": "algo__p", "lang": "python", "action": "test",
                                     "code": "x", "options": {}, "status": "queued", "created_at": time.time(), "output": "", **fields})
    return job_id


def test_tokens_are_checked_and_revocable(board):
    owner, token = _admin_and_token(board)
    worker = _worker(board, token)
    assert worker.post("/v1/worker/heartbeat", json={"gpu": "GB10", "gyms": ["algo"]}).status_code == 200
    assert _worker(board, token[:-3] + "xyz").post("/v1/worker/heartbeat", json={}).status_code == 401
    assert client(board).post("/v1/worker/heartbeat", json={}).status_code == 401
    listed = owner.get("/api/workers").json()["workers"]
    assert len(listed) == 1 and "token_hash" not in listed[0]
    owner.delete(f"/api/workers/{listed[0]['id']}", headers=H)
    import gymhub.board.workers as workers

    workers._cache.clear()
    assert worker.post("/v1/worker/heartbeat", json={}).status_code == 401


def test_only_admin_workers_publish_catalogs(board):
    _owner, token = _admin_and_token(board)
    worker = _worker(board, token)
    problems = [{"slug": "q", "title": "Q", "langs": ["python"], "current": {"python": "c"}}]
    assert worker.post("/v1/worker/catalog", json={"gym": "algo", "meta": {"title": "Algo"}, "problems": problems}).status_code == 200
    assert board.store.get("problems", "algo__q")["title"] == "Q"
    assert board.store.get("problems", "algo__p") is None  # stale exercise removed
    # The same token after its owner stops being an admin.
    worker_id = token.split("_")[1]
    owner_uid = board.store.get("workers", worker_id)["owner_uid"]
    board.store.set("users", owner_uid, {"role": "member"}, merge=True)
    import gymhub.board.auth as auth
    import gymhub.board.workers as workers

    auth._user_cache.clear()
    workers._cache.clear()
    board.app  # noqa: B018
    import os

    os.environ["GYMHUB_ADMIN_EMAILS"] = "someone-else@example.com"
    assert worker.post("/v1/worker/catalog", json={"gym": "algo", "meta": {}, "problems": problems}).status_code == 403


def test_lease_update_cancel_complete(board):
    _owner, token = _admin_and_token(board)
    worker = _worker(board, token)
    job_id = _queue(board)
    reply = worker.post("/v1/worker/lease", json={"gyms": ["algo"]}).json()
    assert reply["job"]["id"] == job_id and reply["job"]["code"] == "x"
    lease_id = reply["job"]["lease_id"]
    assert worker.post("/v1/worker/lease", json={"gyms": ["algo"]}).json()["job"] is None
    assert worker.post("/v1/worker/lease", json={"gyms": ["gpu"]}).json()["job"] is None
    assert worker.post(f"/v1/worker/jobs/{job_id}/update", json={"output": "line 1\n", "lease_id": lease_id}).json() == {"stop": False}
    assert board.store.get("jobs", job_id)["output"] == "line 1\n"
    board.store.set("jobs", job_id, {"cancel": True}, merge=True)
    assert worker.post(f"/v1/worker/jobs/{job_id}/update", json={"lease_id": lease_id}).json() == {"stop": True}
    assert worker.post(f"/v1/worker/jobs/{job_id}/complete", json={"final": {"status": "done", "ok": True, "output": "ok"}, "lease_id": lease_id}).status_code == 200
    doc = board.store.get("jobs", job_id)
    assert doc["status"] == "done" and doc["ok"] is True and doc["lease_expires_at"] is None
    # another worker cannot complete or update a job it does not hold
    owner2, token2 = _admin_and_token(board, "other")
    other = _worker(board, token2)
    assert other.post(f"/v1/worker/jobs/{job_id}/complete", json={"final": {"status": "done"}}).status_code == 409
    assert other.post(f"/v1/worker/jobs/{job_id}/update", json={"output": "x"}).json() == {"stop": True}


def test_isolated_job_carries_and_returns_user_progress(board):
    _owner, token = _admin_and_token(board)
    worker = _worker(board, token)
    board.store.set("user_progress", "g_9__algo", {"data": {"version": 1, "exercises": {"old": {}}}})
    job_id = _queue(board, isolated=True, uid="g_9")
    # someone else's code: only a machine with a working sandbox may take it
    assert worker.post("/v1/worker/lease", json={"gyms": ["algo"]}).json()["job"] is None
    worker.post("/v1/worker/heartbeat", json={"capabilities": {"sandbox": True, "gpus": [], "tools": []}})
    job = worker.post("/v1/worker/lease", json={"gyms": ["algo"]}).json()["job"]
    assert job["user_progress"]["exercises"] == {"old": {}}
    new = {"version": 1, "exercises": {"old": {}, "p": {"langs": {"python": {"first_solved": "now"}}}}}
    worker.post(f"/v1/worker/jobs/{job_id}/complete", json={"final": {"status": "done", "ok": True}, "user_progress": new, "lease_id": job["lease_id"]})
    assert board.store.get("user_progress", "g_9__algo")["data"] == new
    assert board.store.get("progress", "algo") is None  # the owner's progress is untouched


def test_expired_lease_is_requeued_then_failed(board, monkeypatch):
    _owner, token = _admin_and_token(board)
    worker = _worker(board, token)
    job_id = _queue(board)
    import gymhub.board.workers as workers

    for attempt in range(1, jobqueue.MAX_TRIES + 1):
        workers._sweep["at"] = 0  # the expired-lease sweep runs once a minute; pretend a minute passed
        leased = worker.post("/v1/worker/lease", json={"gyms": ["algo"]}).json()["job"]
        assert leased and leased["id"] == job_id and leased["tries"] == attempt
        board.store.set("jobs", job_id, {"lease_expires_at": time.time() - 1}, merge=True)  # the machine vanished
    jobqueue.sweep(board.store)
    doc = board.store.get("jobs", job_id)
    assert doc["status"] == "error" and "stopped responding" in doc["output"]


def test_poll_interval_follows_activity(board):
    import gymhub.board.workers as workers

    _owner, token = _admin_and_token(board)
    worker = _worker(board, token)
    workers._activity.update(written=0, read=0, at=0)
    assert worker.post("/v1/worker/heartbeat", json={}).json()["poll_ms"] == workers.IDLE_POLL_MS
    owner = client(board)
    sign_in(owner, "owner")
    owner.get("/api/gyms/algo/problems/p")  # someone opens an exercise
    workers._activity["read"] = 0
    assert worker.post("/v1/worker/heartbeat", json={}).json()["poll_ms"] == workers.ACTIVE_POLL_MS


# ---- a job's outcome is decided by the board (#53) ----------------------------------------------


def _lease(board, **fields):
    _owner, token = _admin_and_token(board)
    worker = _worker(board, token)
    job_id = _queue(board, **fields)
    job = worker.post("/v1/worker/lease", json={"gyms": ["algo"]}).json()["job"]
    assert job["id"] == job_id and job["lease_id"]
    return worker, job


def _expire(board, job_id):
    board.store.set("jobs", job_id, {"lease_expires_at": time.time() - 1}, merge=True)


def test_a_cancelled_job_whose_lease_expires_is_not_queued_again(board):
    _worker_client, job = _lease(board)
    board.store.set("jobs", job["id"], {"cancel": True}, merge=True)
    _expire(board, job["id"])
    jobqueue.sweep(board.store)
    doc = board.store.get("jobs", job["id"])
    assert doc["status"] == "cancelled" and not doc["ok"]
    jobqueue.sweep(board.store)
    assert board.store.where("jobs", "status", "queued") == []


def test_an_uncancelled_job_is_still_requeued_with_a_new_lease(board):
    worker, job = _lease(board)
    _expire(board, job["id"])
    jobqueue.sweep(board.store)
    assert board.store.get("jobs", job["id"])["status"] == "queued"
    again = worker.post("/v1/worker/lease", json={"gyms": ["algo"]}).json()["job"]
    assert again["id"] == job["id"] and again["lease_id"] != job["lease_id"] and again["tries"] == 2


def test_a_result_for_an_earlier_lease_is_ignored(board):
    worker, job = _lease(board)
    _expire(board, job["id"])
    jobqueue.sweep(board.store)
    newer = worker.post("/v1/worker/lease", json={"gyms": ["algo"]}).json()["job"]
    late = {"final": {"status": "done", "ok": True, "output": "late"}, "lease_id": job["lease_id"]}
    assert worker.post(f"/v1/worker/jobs/{job['id']}/complete", json=late).status_code == 409
    assert worker.post(f"/v1/worker/jobs/{job['id']}/update", json={"output": "late", "lease_id": job["lease_id"]}).json() == {"stop": True}
    assert board.store.get("jobs", job["id"])["status"] == "running"  # the newer attempt is untouched
    ok = {"final": {"status": "done", "ok": True, "output": "fresh"}, "lease_id": newer["lease_id"]}
    assert worker.post(f"/v1/worker/jobs/{job['id']}/complete", json=ok).status_code == 200
    # A repeat of the same complete changes nothing.
    assert worker.post(f"/v1/worker/jobs/{job['id']}/complete", json=ok).status_code == 409
    assert board.store.get("jobs", job["id"])["output"] == "fresh"
    # A result without any lease id does not apply either.
    assert jobqueue.complete(board.store, job["id"], "x", {"status": "done"}) is False
