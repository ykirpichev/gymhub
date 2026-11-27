"""CLI sync for local editors (issue #22): pull, push, keep local edits, submit, and the token's limits."""

import json
import time

import pytest

from gymhub import sync
from tests.test_board_auth import H, board, client, sign_in  # noqa: F401  (fixture)
from tests.test_gym_access import UIDS, _gym, people  # noqa: F401  (fixture)


@pytest.fixture()
def cli(board, people, monkeypatch, tmp_path):
    token = people["eve"].post("/api/my/tokens", json={"name": "laptop"}, headers=H).json()["token"]
    api = client(board)

    def request(_board, _token, method, path, body=None):
        reply = api.request(method, path, json=body, headers={"Authorization": f"Bearer {token}", "X-Gymhub": "1"})
        if reply.status_code >= 400:
            detail = reply.json().get("detail")
            detail = detail.get("message") if isinstance(detail, dict) else detail
            raise (sync.Conflict if reply.status_code == 409 else SystemExit)(f"HTTP {reply.status_code}: {detail}")
        return reply.json()

    monkeypatch.setattr(sync, "request", request)
    monkeypatch.setattr(sync, "CONFIG", tmp_path / "cli.json")
    monkeypatch.setenv("GYMHUB_BOARD", "http://board")
    monkeypatch.setenv("GYMHUB_TOKEN", token)
    _gym(board, visibility="public", langs=[{"id": "python", "label": "Python", "file": "solution.py"}])
    board.store.set("problems", "g1__p", {"statement": "# P\n", "kind": "code"}, merge=True)
    return tmp_path, token


def _args(**kw):
    return type("Args", (), {"board": None, "token": None, "dir": None, "force": False, **kw})()


def test_pull_push_and_back(board, cli):
    work, _token = cli
    assert sync.pull(_args(gym="g1", dir=str(work))) == 0
    file = work / "g1" / "p" / "solution.py"
    assert file.read_text() == "# GYM_STARTER\n" and (work / "g1" / "p" / "PROBLEM.md").read_text() == "# P\n"
    file.write_text("def solve(): return 1\n")
    assert sync._push_root(work / "g1", "b", "t") == 1
    assert sync._push_root(work / "g1", "b", "t", quiet=True) == 0  # nothing changed since
    saved = board.store.get("workspaces", f"{UIDS['eve']}__g1__p__python")
    assert saved["code"] == "def solve(): return 1\n"
    # Edited on the board: the next pull brings it, unless the local file has unpushed changes.
    board.store.set("workspaces", f"{UIDS['eve']}__g1__p__python", {"code": "from the board\n"}, merge=True)
    sync.pull(_args(gym="g1", dir=str(work)))
    assert file.read_text() == "from the board\n"
    file.write_text("local work\n")
    board.store.set("workspaces", f"{UIDS['eve']}__g1__p__python", {"code": "newer on the board\n"}, merge=True)
    sync.pull(_args(gym="g1", dir=str(work)))
    assert file.read_text() == "local work\n"
    sync.pull(_args(gym="g1", dir=str(work), force=True))
    assert file.read_text() == "newer on the board\n"
    meta = json.loads((work / "g1" / ".gymhub.json").read_text())
    assert meta["files"]["p/solution.py"]["lang"] == "python"


def test_submit_streams_until_done(board, cli, capsys, monkeypatch):
    work, _token = cli
    sync.pull(_args(gym="g1", dir=str(work)))
    file = work / "g1" / "p" / "solution.py"
    file.write_text("def solve(): return 2\n")
    real = sync.request

    def finishing(board_url, token, method, path, body=None):
        reply = real(board_url, token, method, path, body)
        if method == "POST" and path == "/api/jobs":
            board.store.set("jobs", reply["id"], {"status": "done", "ok": True, "output": "p  [python]  pass  3/3\n", "finished_at": time.time()}, merge=True)
        return reply

    monkeypatch.setattr(sync, "request", finishing)
    assert sync.submit(_args(file=str(file))) == 0
    assert "pass  3/3" in capsys.readouterr().out
    job = next(doc for _id, doc in board.store.where("jobs", "slug", "p") if doc.get("action") == "test")
    assert job["code"] == "def solve(): return 2\n" and job["uid"] == UIDS["eve"]


def test_tokens_reach_only_learning_endpoints(board, cli):
    _work, token = cli
    api = client(board)
    headers = {"Authorization": f"Bearer {token}", "X-Gymhub": "1"}
    assert api.get("/api/gyms/g1", headers=headers).status_code == 200
    assert api.get("/api/gyms/g1/problems/p", headers=headers).status_code == 200
    assert api.get("/api/home", headers=headers).status_code == 401
    assert api.put("/api/gyms/g1/settings", json={"visibility": "public"}, headers=headers).status_code == 401
    assert api.get("/api/admin/users", headers=headers).status_code == 401
    board.store.set("gyms", "g1", {"visibility": "private"}, merge=True)
    assert api.get("/api/gyms/g1", headers=headers).status_code == 404  # visibility still applies


def test_a_stale_push_is_refused_and_the_board_keeps_the_newer_code(board, cli, capsys):
    work, _token = cli
    sync.pull(_args(gym="g1", dir=str(work)))
    root, file = work / "g1", work / "g1" / "p" / "solution.py"
    workspace = f"{UIDS['eve']}__g1__p__python"
    # Edited in the browser after the pull (the browser saves through the same API, without a base).
    people_eve = client(board)
    sign_in(people_eve, "eve")
    assert people_eve.post("/api/jobs", json={"gym": "g1", "slug": "p", "lang": "python", "action": "save", "code": "from the browser\n"}, headers=H).status_code == 200
    file.write_text("older local edit\n")
    conflicts = {}
    assert sync._push_root(root, "b", "t", conflicts=conflicts) == 0
    assert "conflict p/solution.py" in capsys.readouterr().out and list(conflicts) == ["p/solution.py"]
    assert board.store.get("workspaces", workspace)["code"] == "from the browser\n"  # the board still has the newer code
    assert file.read_text() == "older local edit\n"  # and so does the local file
    # A watcher reports it once, not on every tick.
    assert sync._push_root(root, "b", "t", quiet=True, conflicts=conflicts) == 0
    assert "conflict" not in capsys.readouterr().out
    # On purpose: --force replaces the board's copy.
    assert sync._push_root(root, "b", "t", force=True) == 1
    assert board.store.get("workspaces", workspace)["code"] == "older local edit\n"


def test_an_unchanged_remote_still_saves_every_time(board, cli):
    work, _token = cli
    sync.pull(_args(gym="g1", dir=str(work)))
    root, file = work / "g1", work / "g1" / "p" / "solution.py"
    for n in range(3):  # each push moves the revision forward, so the next one is still accepted
        file.write_text(f"v{n}\n")
        assert sync._push_root(root, "b", "t") == 1
    assert board.store.get("workspaces", f"{UIDS['eve']}__g1__p__python")["code"] == "v2\n"


def test_submit_does_not_replace_a_newer_copy(board, cli):
    work, _token = cli
    sync.pull(_args(gym="g1", dir=str(work)))
    file = work / "g1" / "p" / "solution.py"
    board.store.set("workspaces", f"{UIDS['eve']}__g1__p__python", {"uid": UIDS["eve"], "code": "newer\n"})
    file.write_text("def solve(): return 3\n")
    with pytest.raises(sync.Conflict):
        sync.submit(_args(file=str(file)))
    assert board.store.get("workspaces", f"{UIDS['eve']}__g1__p__python")["code"] == "newer\n"
