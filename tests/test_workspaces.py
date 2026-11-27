"""Per-user workspaces and progress: members never see the owner's code, jobs or progress, nor each other's."""

import time

from tests.test_board_auth import H, board, client, sign_in  # noqa: F401  (fixture)


def _setup(board):
    store = board.store
    store.set("problems", "algo__p", {
        "gym": "algo", "slug": "p", "title": "P", "langs": ["python", "cpp"], "order": 0,
        "starters": {"python": "# GYM_STARTER\ndef solve(): pass\n", "cpp": "// GYM_STARTER\n"},
        "current": {"python": "OWNER SECRET SOLUTION", "cpp": "owner cpp"},
    })
    store.set("progress", "algo", {"gym": "algo", "data": {"version": 1, "exercises": {"p": {"langs": {"python": {"first_solved": "2026-01-01T00:00:00"}}}}}})
    owner = client(board)
    sign_in(owner, "owner")
    owner.put("/api/admin/settings", json={"allowed_domains": ["example.com", "school.edu"], "members_can_run": True}, headers=H)
    ann, bob = client(board), client(board)
    sign_in(ann, "ann")
    sign_in(bob, "bob")
    return owner, ann, bob


def test_member_never_sees_owner_code_or_progress(board):
    owner, ann, _ = _setup(board)
    assert owner.get("/api/gyms/algo/problems/p").json()["problem"]["current"]["python"] == "OWNER SECRET SOLUTION"
    data = ann.get("/api/gyms/algo/problems/p").json()
    assert data["problem"]["current"]["python"].startswith("# GYM_STARTER")
    assert data["progress"] == {} and data["owner"] is False
    home = ann.get("/api/home").json()
    assert home["gyms"][0]["progress"]["exercises"] == {}


def test_save_reset_and_hint_stay_in_the_member_workspace(board):
    owner, ann, bob = _setup(board)
    job = {"gym": "algo", "slug": "p", "lang": "python", "action": "save", "code": "ann's code"}
    jid = ann.post("/api/jobs", json=job, headers=H).json()["id"]
    assert ann.get(f"/api/jobs/{jid}").json()["status"] == "done"  # the API did it; no runner needed
    assert ann.get("/api/gyms/algo/problems/p").json()["problem"]["current"]["python"] == "ann's code"
    assert bob.get("/api/gyms/algo/problems/p").json()["problem"]["current"]["python"].startswith("# GYM_STARTER")
    assert owner.get("/api/gyms/algo/problems/p").json()["problem"]["current"]["python"] == "OWNER SECRET SOLUTION"
    ann.post("/api/jobs", json={**job, "action": "reset", "code": None}, headers=H)
    assert ann.get("/api/gyms/algo/problems/p").json()["problem"]["current"]["python"].startswith("# GYM_STARTER")
    ann.post("/api/jobs", json={"gym": "algo", "slug": "p", "action": "hint", "options": {"count": 2}}, headers=H)
    assert ann.get("/api/gyms/algo/problems/p").json()["progress"]["hints_seen"] == 2
    assert "hints_seen" not in owner.get("/api/gyms/algo/problems/p").json()["progress"]


def test_member_jobs_are_isolated_and_private(board):
    owner, ann, bob = _setup(board)
    job = {"gym": "algo", "slug": "p", "lang": "python", "action": "test", "code": "ann's attempt"}
    jid = ann.post("/api/jobs", json=job, headers=H).json()["id"]
    doc = board.store.get("jobs", jid)
    assert doc["isolated"] is True and doc["uid"].startswith("g_")
    # the submitted code is also saved to Ann's workspace
    assert ann.get("/api/gyms/algo/problems/p").json()["problem"]["current"]["python"] == "ann's attempt"
    board.store.set("jobs", jid, {"status": "done", "ok": False, "output": "FAIL"}, merge=True)
    assert [j["id"] for j in ann.get("/api/gyms/algo/problems/p").json()["jobs"]] == [jid]
    assert bob.get(f"/api/jobs/{jid}").status_code == 404
    assert bob.get(f"/api/jobs/{jid}/code").status_code == 404
    assert bob.post(f"/api/jobs/{jid}/cancel", headers=H).status_code == 404
    assert owner.get("/api/gyms/algo/problems/p").json()["jobs"] == []
    owner_job = owner.post("/api/jobs", json={**job, "code": "owner try"}, headers=H).json()["id"]
    assert board.store.get("jobs", owner_job)["isolated"] is False
    assert ann.get(f"/api/jobs/{owner_job}").status_code == 404


def test_member_cannot_run_without_permission_but_can_save(board):
    owner, ann, _ = _setup(board)
    owner.put("/api/admin/settings", json={"allowed_domains": ["example.com"], "members_can_run": False}, headers=H)
    base = {"gym": "algo", "slug": "p", "lang": "python", "code": "x"}
    assert ann.post("/api/jobs", json={**base, "action": "run"}, headers=H).status_code == 403
    assert ann.post("/api/jobs", json={**base, "action": "save"}, headers=H).status_code == 200
