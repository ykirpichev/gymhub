"""Behind IAP: IAP identifies the visitor; invites, roles and workspaces work exactly as with in-app sign-in."""

import importlib
import time

import pytest
from fastapi.testclient import TestClient

H = {"X-Gymhub": "1"}
PEOPLE = {
    "owner": {"sub": "111", "email": "owner@example.com", "email_verified": True},
    "second": {"sub": "222", "email": "second@example.com", "email_verified": True},
}


@pytest.fixture()
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("GYMHUB_STORE", f"sqlite:{tmp_path / 'db.sqlite'}")
    monkeypatch.setenv("GYMHUB_AUTH", "iap")
    monkeypatch.setenv("GYMHUB_IAP_AUDIENCE", "/projects/1/locations/r/services/s")
    monkeypatch.setenv("GYMHUB_ALLOWED_EMAILS", "owner@example.com")
    monkeypatch.delenv("GYMHUB_ADMIN_EMAILS", raising=False)
    import gymhub.board.app as app_module
    import gymhub.board.auth as auth

    app_module = importlib.reload(app_module)
    auth._user_cache.clear()
    # Stand-in for IAP's signed header: x-test-user names who IAP says this is.
    monkeypatch.setattr(auth, "iap_claims", lambda request: dict(PEOPLE[request.headers["x-test-user"]]))
    app_module.store.set("problems", "algo__p", {"gym": "algo", "slug": "p", "title": "P", "langs": ["python"],
                                                 "starters": {"python": "# GYM_STARTER\n"}, "current": {"python": "OWNER CODE"}})
    app_module.store.set("gyms", "algo", {"title": "Algo", "order": 1, "updated_at": time.time(), "langs": []})
    return app_module


def as_(board, who):
    c = TestClient(board.app)
    c.headers.update({"x-test-user": who, **H})
    return c


def test_owner_is_admin_and_owner(board):
    me = as_(board, "owner").get("/api/me").json()
    assert me["user"]["is_admin"] and me["owner"] and me["auth"] == "iap"


def test_stranger_is_not_invited(board):
    response = as_(board, "second").get("/api/me")
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "not_invited"
    assert response.json()["detail"]["email"] == "second@example.com"


def test_invite_turns_stranger_into_member_with_own_workspace(board):
    owner = as_(board, "owner")
    code = owner.post("/api/admin/invites", json={"email": "second@example.com", "uses": 1}).json()["code"]
    second = as_(board, "second")
    assert second.post("/api/auth/accept-invite", json={"invite": code}).status_code == 200
    me = second.get("/api/me").json()
    assert me["user"]["role"] == "member" and me["owner"] is False and me["user"]["uid"] == "g_222"
    assert second.get("/api/gyms/algo/problems/p").json()["problem"]["current"]["python"].startswith("# GYM_STARTER")
    owner.put("/api/admin/settings", json={"allowed_domains": [], "members_can_run": True})
    job = second.post("/api/jobs", json={"gym": "algo", "slug": "p", "lang": "python", "action": "test", "code": "mine"}).json()
    assert board.store.get("jobs", job["id"])["isolated"] is True
    # used up: nobody else can use it
    assert as_(board, "owner").post("/api/auth/accept-invite", json={"invite": code}).status_code in (200, 403)


def test_bad_invite_is_refused_and_disabled_user_is_locked_out(board):
    second = as_(board, "second")
    assert second.post("/api/auth/accept-invite", json={"invite": "nope"}).status_code == 403
    owner = as_(board, "owner")
    code = owner.post("/api/admin/invites", json={"uses": 1}).json()["code"]
    second.post("/api/auth/accept-invite", json={"invite": code})
    owner.post("/api/admin/users/g_222", json={"status": "disabled"})
    assert second.get("/api/me").status_code == 403
