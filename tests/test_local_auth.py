"""GYMHUB_AUTH=local: name-and-password accounts, created only from invite links (gymhub/board/auth.py)."""

import importlib
import time

import pytest
from fastapi.testclient import TestClient

from gymhub import local

H = {"X-Gymhub": "1"}


@pytest.fixture()
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("GYMHUB_STORE", f"sqlite:{tmp_path / 'db.sqlite'}")
    monkeypatch.setenv("GYMHUB_AUTH", "local")
    monkeypatch.setenv("GYMHUB_SESSION_KEY", "k" * 48)
    monkeypatch.setenv("GYMHUB_ADMIN_EMAILS", "owner")
    import gymhub.board.app as app_module
    import gymhub.board.auth as auth_module

    app_module = importlib.reload(app_module)
    auth_module._user_cache.clear()
    return app_module


def invite(board, **fields):
    code = board.auth.new_invite_code()
    board.store.set("invites", code, {"uses_left": 1, "uses_total": 1, "expires_at": time.time() + 3600,
                                      "created_at": time.time(), **fields})
    return code


def register(c, username, code, password="correct horse", name=""):
    return c.post("/api/auth/register", json={"username": username, "password": password, "name": name, "invite": code},
                  headers=H)


def owner_client(board):
    c = TestClient(board.app)
    assert register(c, "owner", invite(board, email="owner", role="admin")).status_code == 200
    return c


def test_owner_link_makes_the_admin_and_owner(board):
    c = owner_client(board)
    me = c.get("/api/me").json()
    assert me["user"]["role"] == "admin" and me["owner"] and me["auth"] == "local"
    # The hash lives apart from users/, so no user listing carries it.
    assert "hash" not in str(c.get("/api/admin/users").json())
    assert board.store.get("passwords", "l_owner")["hash"].startswith("scrypt$")


def test_member_joins_by_invite_signs_out_and_back_in(board):
    owner = owner_client(board)
    code = owner.post("/api/admin/invites", json={"uses": 1, "days": 1}, headers=H).json()["code"]
    ann = TestClient(board.app)
    info = ann.get(f"/api/auth/invite/{code}").json()
    assert info == {"valid": True, "for": None, "reset": False, "role": "member"}
    assert register(ann, "Ann", code, name="Ann A").status_code == 200
    me = ann.get("/api/me").json()["user"]
    assert me["email"] == "ann" and me["name"] == "Ann A" and me["role"] == "member"
    assert ann.post("/api/auth/logout", headers=H).status_code == 200
    ann.cookies.clear()
    assert ann.get("/api/me").status_code == 401
    assert ann.post("/api/auth/password", json={"username": "ann", "password": "wrong pass"}, headers=H).status_code == 401
    assert ann.post("/api/auth/password", json={"username": "ann", "password": "correct horse"}, headers=H).status_code == 200
    assert ann.get("/api/me").status_code == 200
    # The invite was single-use.
    assert register(TestClient(board.app), "bob", code).status_code == 403


def test_no_invite_no_account_and_names_are_checked(board):
    c = TestClient(board.app)
    assert register(c, "eve", "nope").status_code == 403
    code = invite(board)
    assert register(c, "a b", code).status_code == 400
    assert register(c, "eve", code, password="short").status_code == 400
    assert register(c, "eve", code).status_code == 200
    assert register(TestClient(board.app), "eve", invite(board)).status_code == 409  # taken


def test_admin_name_needs_an_invite_made_for_it(board):
    # Before the owner signs up, a member's invite cannot claim the owner's name (and admin role).
    assert register(TestClient(board.app), "owner", invite(board)).status_code == 403
    assert register(TestClient(board.app), "ann", invite(board, email="bob")).status_code == 403  # for someone else


def test_wrong_passwords_pause_sign_in_for_that_name(board):
    owner_client(board)
    c = TestClient(board.app)
    for _ in range(board.auth.SIGNIN_FAILURES):
        assert c.post("/api/auth/password", json={"username": "owner", "password": "x" * 9}, headers=H).status_code == 401
    assert c.post("/api/auth/password", json={"username": "owner", "password": "correct horse"}, headers=H).status_code == 429


def test_the_pause_does_not_reset_at_a_clock_boundary(board, monkeypatch):
    """The window opens at the first wrong password: a burst that crosses :00, :15, :30 or :45 still counts."""
    owner_client(board)
    c = TestClient(board.app)
    boundary = (time.time() // board.auth.SIGNIN_WINDOW + 1) * board.auth.SIGNIN_WINDOW
    clock = [boundary - 5]
    monkeypatch.setattr(board.auth.time, "time", lambda: clock[0])
    for i in range(board.auth.SIGNIN_FAILURES):
        clock[0] = boundary - 5 + i  # the burst straddles the boundary
        assert c.post("/api/auth/password", json={"username": "owner", "password": "x" * 9}, headers=H).status_code == 401
    assert c.post("/api/auth/password", json={"username": "owner", "password": "correct horse"}, headers=H).status_code == 429
    clock[0] = boundary - 5 + board.auth.SIGNIN_WINDOW + 1  # a full window after the first failure
    assert c.post("/api/auth/password", json={"username": "owner", "password": "correct horse"}, headers=H).status_code == 200


def test_password_reset_link_and_change(board):
    owner = owner_client(board)
    ann = TestClient(board.app)
    register(ann, "ann", invite(board))
    reset = owner.post("/api/admin/users/l_ann/password-reset", headers=H).json()["code"]
    assert TestClient(board.app).get(f"/api/auth/invite/{reset}").json()["reset"] is True
    assert register(TestClient(board.app), "owner", reset).status_code == 403  # bound to ann
    fresh = TestClient(board.app)
    assert register(fresh, "ann", reset, password="new password").status_code == 200
    assert fresh.get("/api/me").json()["user"]["role"] == "member"
    assert fresh.post("/api/me/password", json={"old": "wrong one", "new": "another one"}, headers=H).status_code == 403
    assert fresh.post("/api/me/password", json={"old": "new password", "new": "another one"}, headers=H).status_code == 200
    assert fresh.post("/api/auth/password", json={"username": "ann", "password": "another one"}, headers=H).status_code == 200


def test_disabled_member_cannot_sign_in(board):
    owner = owner_client(board)
    register(TestClient(board.app), "ann", invite(board))
    owner.post("/api/admin/users/l_ann", json={"status": "disabled"}, headers=H)
    reply = TestClient(board.app).post("/api/auth/password", json={"username": "ann", "password": "correct horse"}, headers=H)
    assert reply.status_code == 403


def test_deleting_the_account_removes_the_password(board):
    c = TestClient(board.app)
    register(c, "ann", invite(board))
    assert c.request("DELETE", "/api/me", json={"confirm": "ann"}, headers=H).status_code == 200
    assert board.store.get("passwords", "l_ann") is None


def test_other_modes_refuse_local_endpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("GYMHUB_STORE", f"sqlite:{tmp_path / 'db.sqlite'}")
    monkeypatch.setenv("GYMHUB_AUTH", "app")
    import gymhub.board.app as app_module

    app_module = importlib.reload(app_module)
    c = TestClient(app_module.app)
    assert c.post("/api/auth/password", json={"username": "a1", "password": "x"}, headers=H).status_code == 400


# ---- gymhub-board --lan ---------------------------------------------------------------------------


def test_lan_environment(tmp_path):
    env = local.environment({"GYMHUB_STORE": f"sqlite:{tmp_path / 'b.db'}"}, lan=True, owner="yury")
    assert env["GYMHUB_AUTH"] == "local" and env["GYMHUB_ADMIN_EMAILS"] == "yury"
    assert "GYMHUB_TRUST_JOBS" not in env  # members' code is not run here unisolated by default
    key = (tmp_path / "session.key")
    assert key.read_text().strip() == env["GYMHUB_SESSION_KEY"] and oct(key.stat().st_mode & 0o777) == "0o600"
    again = local.environment({"GYMHUB_STORE": f"sqlite:{tmp_path / 'b.db'}"}, lan=True)
    assert again["GYMHUB_SESSION_KEY"] == env["GYMHUB_SESSION_KEY"]  # kept across restarts


def test_owner_link_once_then_only_on_request(tmp_path):
    from gymhub.store import SqliteStore

    env = local.environment({"GYMHUB_STORE": f"sqlite:{tmp_path / 'b.db'}"}, lan=True, owner="yury")
    code = local.owner_link(env)
    store = SqliteStore(str(tmp_path / "b.db"))
    assert store.get("invites", code)["role"] == "admin" and store.get("invites", code)["email"] == "yury"
    store.set("users", "l_yury", {"email": "yury", "role": "admin", "status": "active"})
    assert local.owner_link(env) is None
    assert store.get("invites", local.owner_link(env, force=True))["reset_uid"] == "l_yury"


def test_lan_may_listen_on_every_address(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("GYMHUB_AUTH", raising=False)
    monkeypatch.setenv("GYMHUB_STORE", f"sqlite:{tmp_path / 'b.db'}")
    monkeypatch.setattr(local, "readiness", lambda: (["  ok    GPU          test"], True))
    assert local.main(["--lan", "--check"]) == 0  # not refused, unlike GYMHUB_AUTH=dev on 0.0.0.0
