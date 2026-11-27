"""Sign-in, invites, roles and permissions of the board API (app auth mode, SQLite store)."""

import importlib
import time

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

H = {"X-Gymhub": "1"}


@pytest.fixture()
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("GYMHUB_STORE", f"sqlite:{tmp_path / 'db.sqlite'}")
    monkeypatch.setenv("GYMHUB_AUTH", "app")
    monkeypatch.setenv("GYMHUB_GOOGLE_CLIENT_ID", "test-client")
    monkeypatch.setenv("GYMHUB_SESSION_KEY", "k" * 48)
    monkeypatch.setenv("GYMHUB_ADMIN_EMAILS", "owner@example.com")
    import gymhub.board.app as app_module
    import gymhub.board.auth as auth_module

    app_module = importlib.reload(app_module)
    auth_module._user_cache.clear()

    # Google's verification replaced by a table of fake tokens: "token-<name>" -> claims.
    people = {
        "owner": {"sub": "1", "email": "owner@example.com", "email_verified": True, "name": "Owner"},
        "ann": {"sub": "2", "email": "ann@example.com", "email_verified": True, "name": "Ann"},
        "bob": {"sub": "3", "email": "bob@school.edu", "email_verified": True, "name": "Bob"},
        "eve": {"sub": "4", "email": "eve@example.com", "email_verified": True, "name": "Eve"},
        "unverified": {"sub": "5", "email": "x@example.com", "email_verified": False},
        # Not signed in by most tests: people a gym is shared with before they join the board.
        "new": {"sub": "6", "email": "new@example.com", "email_verified": True, "name": "New"},
        "co": {"sub": "7", "email": "co@school.edu", "email_verified": True, "name": "Co"},
        "late": {"sub": "8", "email": "late@example.com", "email_verified": True, "name": "Late"},
    }

    def fake_verify(credential):
        name = credential.removeprefix("token-")
        if name not in people:
            raise HTTPException(401, "bad token")
        return people[name]

    monkeypatch.setattr(auth_module, "verify_google_token", fake_verify)
    store = app_module.store
    store.set("gyms", "algo", {"title": "Algo", "order": 1, "updated_at": time.time(), "langs": []})
    store.set("problems", "algo__p", {"gym": "algo", "slug": "p", "title": "P", "langs": ["python"], "order": 0})
    return app_module


def client(board):
    return TestClient(board.app)


def sign_in(c, who, invite=None):
    return c.post("/api/auth/google", json={"credential": f"token-{who}", "invite": invite}, headers=H)


def shared_token(owner, name):
    """A worker token whose machine takes everyone's jobs: machines are private (their owner's jobs
    only) until shared, an admin's too (gymhub/board/workers.py may_run_for)."""
    token = owner.post("/api/workers", json={"name": name}, headers=H).json()["token"]
    reply = owner.post(f"/api/my/workers/{token.split('_')[1]}/sharing", json={"everyone": True}, headers=H)
    assert reply.status_code == 200, reply.text
    return token


def test_signed_out_gets_401_but_page_and_config_load(board):
    c = client(board)
    assert c.get("/api/me").status_code == 401
    assert c.get("/api/home").status_code == 401
    assert c.get("/").status_code == 200  # the SPA, which shows the sign-in page
    assert c.get("/api/auth/config").json()["google_client_id"] == "test-client"


def test_admin_email_signs_in_and_is_admin(board):
    c = client(board)
    assert sign_in(c, "owner").status_code == 200
    me = c.get("/api/me").json()
    assert me["user"]["email"] == "owner@example.com" and me["user"]["is_admin"] and me["can_run"]


def test_uninvited_and_unverified_are_refused(board):
    c = client(board)
    assert sign_in(c, "ann").status_code == 403
    assert sign_in(c, "unverified").status_code == 403
    assert sign_in(c, "nobody").status_code == 401


def test_invite_admits_once_and_member_cannot_run(board):
    owner = client(board)
    sign_in(owner, "owner")
    code = owner.post("/api/admin/invites", json={"uses": 1, "days": 7}, headers=H).json()["code"]
    ann = client(board)
    assert sign_in(ann, "ann", code).status_code == 200
    me = ann.get("/api/me").json()
    assert me["user"]["role"] == "member" and not me["can_run"]
    assert ann.get("/api/home").status_code == 200
    job = {"gym": "algo", "slug": "p", "lang": "python", "action": "run", "code": "x"}
    assert ann.post("/api/jobs", json=job, headers=H).status_code == 403
    # single use: Eve cannot reuse it
    assert sign_in(client(board), "eve", code).status_code == 403
    # members cannot reach admin endpoints
    assert ann.get("/api/admin/users").status_code == 403


def test_invite_for_specific_email(board):
    owner = client(board)
    sign_in(owner, "owner")
    code = owner.post("/api/admin/invites", json={"email": "ann@example.com", "uses": 5}, headers=H).json()["code"]
    assert sign_in(client(board), "eve", code).status_code == 403
    assert sign_in(client(board), "ann", code).status_code == 200


def test_expired_and_revoked_invites(board):
    owner = client(board)
    sign_in(owner, "owner")
    code = owner.post("/api/admin/invites", json={"uses": 5}, headers=H).json()["code"]
    board.store.set("invites", code, {"expires_at": time.time() - 1}, merge=True)
    assert sign_in(client(board), "ann", code).status_code == 403
    code2 = owner.post("/api/admin/invites", json={"uses": 5}, headers=H).json()["code"]
    owner.delete(f"/api/admin/invites/{code2}", headers=H)
    assert sign_in(client(board), "ann", code2).status_code == 403


def test_allowed_domain_and_members_can_run_setting(board):
    owner = client(board)
    sign_in(owner, "owner")
    assert owner.put("/api/admin/settings", json={"allowed_domains": ["School.edu"], "members_can_run": True}, headers=H).status_code == 200
    bob = client(board)
    assert sign_in(bob, "bob").status_code == 200
    assert bob.get("/api/me").json()["can_run"]
    job = {"gym": "algo", "slug": "p", "lang": "python", "action": "run", "code": "x"}
    assert bob.post("/api/jobs", json=job, headers=H).status_code == 200


def test_disable_user_ends_their_session(board):
    owner = client(board)
    sign_in(owner, "owner")
    code = owner.post("/api/admin/invites", json={"uses": 1}, headers=H).json()["code"]
    ann = client(board)
    sign_in(ann, "ann", code)
    uid = next(u["uid"] for u in owner.get("/api/admin/users").json()["users"] if u["email"] == "ann@example.com")
    assert owner.post(f"/api/admin/users/{uid}", json={"status": "disabled"}, headers=H).status_code == 200
    assert ann.get("/api/me").status_code == 401
    assert sign_in(ann, "ann").status_code == 403


def test_admin_cannot_demote_self(board):
    owner = client(board)
    sign_in(owner, "owner")
    uid = owner.get("/api/me").json()["user"]["uid"]
    assert owner.post(f"/api/admin/users/{uid}", json={"role": "member"}, headers=H).status_code == 400


def test_tampered_cookie_and_missing_csrf_header(board):
    c = client(board)
    sign_in(c, "owner")
    cookie = c.cookies.get("gymhub_session")
    c.cookies.clear()  # replace, not add: newer httpx keeps both cookies otherwise
    c.cookies.set("gymhub_session", cookie[:-2] + ("AA" if not cookie.endswith("AA") else "BB"))
    assert c.get("/api/me").status_code == 401
    c2 = client(board)
    sign_in(c2, "owner")
    assert c2.post("/api/auth/logout").status_code == 403  # no X-Gymhub header
    assert c2.post("/api/auth/logout", headers=H).status_code == 200
    assert c2.get("/api/me").status_code == 401
