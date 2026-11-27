"""Machines are private by default: a machine (an admin's too, a pool bridge too) takes only its
owner's jobs until its owner shares it, with everyone or by email (gymhub/board/workers.py)."""

import time

from tests.test_board_auth import H, board, client, sign_in  # noqa: F401  (fixture)

BRIDGE = {"gpus": [], "tools": [], "pool": {"actions": ["run", "test", "bench"], "machines": 1, "gpus": [], "tools": ["g++"],
                                           "own": {"machines": 1, "gpus": []}}}


def _people(board):
    owner = client(board)
    sign_in(owner, "owner")
    owner.put("/api/admin/settings", json={"allowed_domains": ["example.com"], "members_can_run": True}, headers=H)
    ann, eve = client(board), client(board)
    sign_in(ann, "ann")
    sign_in(eve, "eve")
    return owner, ann, eve


def _bridge(board, owner):
    token = owner.post("/api/workers", json={"name": "bridge"}, headers=H).json()["token"]
    worker = client(board)
    worker.headers.update({"Authorization": f"Bearer {token}"})
    assert worker.post("/v1/worker/heartbeat", json={"gyms": ["algo"], "capabilities": BRIDGE}).status_code == 200
    import gymhub.board.workers as workers

    workers._capabilities.clear()
    return worker, token.split("_")[1]


def _submit(person, code="x"):
    return person.post("/api/jobs", json={"gym": "algo", "slug": "p", "lang": "python", "action": "test", "code": code}, headers=H).json()


def _lease(worker):
    return worker.post("/v1/worker/lease", json={"gyms": ["algo"]}).json()["job"]


def _share(board, owner, worker_id, **body):
    import gymhub.board.workers as workers

    assert owner.post(f"/api/my/workers/{worker_id}/sharing", json=body, headers=H).status_code == 200
    workers._cache.clear()


def test_an_admins_pool_bridge_grades_only_the_admins_jobs_until_shared(board):
    owner, ann, eve = _people(board)
    bridge, bridge_id = _bridge(board, owner)
    anns = _submit(ann)
    assert "only their owners' jobs" in anns["warning"]  # Ann is told why it waits
    assert _lease(bridge) is None  # not shared: Ann's job waits
    mine = _submit(owner, "mine")
    assert _lease(bridge)["id"] == mine["id"]

    _share(board, owner, bridge_id, emails=["ann@example.com"])
    assert _lease(bridge)["id"] == anns["id"]
    eves = _submit(eve)
    assert "only their owners' jobs" in eves["warning"]
    assert _lease(bridge) is None  # shared with Ann, not Eve

    _share(board, owner, bridge_id, everyone=True)
    assert "warning" not in _submit(eve, "again")
    assert _lease(bridge)["id"] == eves["id"]

    _share(board, owner, bridge_id)  # back to private
    _submit(eve, "third")
    assert _lease(bridge) is None


def test_the_compute_page_shows_admin_machines_as_private(board):
    owner, _ann, _eve = _people(board)
    _bridge(board, owner)
    reply = owner.get("/api/my/workers").json()
    assert reply["runs"] == "only your jobs"
    assert reply["workers"][0]["shared_everyone"] is False and reply["workers"][0]["shared_emails"] == []


def test_a_job_with_no_submitter_still_goes_to_an_admins_machine(board):
    owner, _ann, _eve = _people(board)
    bridge, _ = _bridge(board, owner)
    board.store.set("jobs", "old", {"gym": "algo", "slug": "p", "problem": "algo__p", "lang": "python", "action": "test",
                                    "code": "x", "options": {}, "status": "queued", "created_at": time.time(), "output": ""})
    assert _lease(bridge)["id"] == "old"


def test_the_pool_panel_says_whose_jobs_the_pool_grades(board, monkeypatch):
    import gymhub.board.poolmachines as poolmachines

    monkeypatch.setattr(poolmachines, "pool_client", lambda: None)
    owner, ann, eve = _people(board)
    _bridge_worker, bridge_id = _bridge(board, owner)

    def grades(person):
        return person.get("/api/pool/machines").json()["bridge"]["grades_for"]

    assert grades(owner) == {"who": "owner", "yours": True, "mine": True, "worker": bridge_id, "emails": []}
    assert grades(ann) == {"who": "owner", "yours": False, "mine": False}  # no worker id, no emails
    _share(board, owner, bridge_id, emails=["ann@example.com"])
    assert grades(owner)["emails"] == ["ann@example.com"] and grades(owner)["who"] == "some"
    assert grades(ann) == {"who": "some", "yours": True, "mine": False}
    assert grades(eve) == {"who": "some", "yours": False, "mine": False}
    _share(board, owner, bridge_id, everyone=True)
    assert grades(eve) == {"who": "everyone", "yours": True, "mine": False}
