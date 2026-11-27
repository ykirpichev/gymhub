"""Notification settings (issue #23): stored per user, returned by /api/me."""

from tests.test_board_auth import H, board, client, sign_in  # noqa: F401  (fixture)


def test_settings_round_trip(board):
    c = client(board)
    sign_in(c, "owner")
    assert c.get("/api/me").json()["notifications"] == {"jobs": True, "machines": True}
    assert c.put("/api/me/notifications", json={"jobs": False, "machines": True}, headers=H).status_code == 200
    assert c.get("/api/me").json()["notifications"] == {"jobs": False, "machines": True}
    assert c.put("/api/me/notifications", json={"jobs": True}).status_code == 403  # CSRF header required
    assert client(board).put("/api/me/notifications", json={}, headers=H).status_code == 401
