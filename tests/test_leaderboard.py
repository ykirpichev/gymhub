"""Leaderboards (issue #21): opt-in, trusted machines only, per language and hardware."""

import time

from gymhub import jobqueue
from tests.test_board_auth import H, board, client, sign_in  # noqa: F401  (fixture)
from tests.test_gym_access import UIDS, _gym, people  # noqa: F401  (fixture)


def _bench(board, uid, runner, ms, hardware="NVIDIA GB10", lang="python"):
    job_id = f"b{time.time_ns()}"
    board.store.set("jobs", job_id, {"gym": "g1", "slug": "p", "problem": "g1__p", "lang": lang, "action": "bench", "uid": uid,
                                     "created_by": "x@example.com", "isolated": True, "status": "running", "runner": runner,
                                     "hardware": hardware, "created_at": time.time()})
    jobqueue.complete(board.store, job_id, runner, {"status": "done", "ok": True, "output": f"  time    {ms} ms median   min"})


def test_trusted_machines_only(board, people, monkeypatch):
    _gym(board, visibility="public", leaderboard=True)
    monkeypatch.setenv("GYMHUB_ADMIN_EMAILS", "owner@example.com")
    board.store.set("workers", "w-ann", {"owner_uid": UIDS["ann"], "owner_email": "ann@example.com"})      # the gym owner's machine
    board.store.set("workers", "w-admin", {"owner_uid": UIDS["owner"], "owner_email": "owner@example.com"})
    board.store.set("workers", "w-eve", {"owner_uid": UIDS["eve"], "owner_email": "eve@example.com"})      # a member's own machine
    board.store.set("workers", "w-gone", {"owner_uid": UIDS["ann"], "owner_email": "ann@example.com", "revoked": True})
    _bench(board, UIDS["bob"], "w-ann", 2.0)
    _bench(board, UIDS["eve"], "w-admin", 1.5)
    _bench(board, UIDS["eve"], "w-eve", 0.001)     # untrusted: ignored
    _bench(board, UIDS["bob"], "w-gone", 0.002)    # revoked: ignored
    _bench(board, UIDS["owner"], "spark-10dc", 1.0)  # a direct-mode runner (project credentials)
    _bench(board, UIDS["bob"], "w-ann", 3.0)       # slower than bob's best: no change
    board_view = people["eve"].get("/api/gyms/g1/problems/p/leaderboard").json()
    assert [(e["rank"], e["ms"]) for e in board_view["entries"]] == [(1, 1.0), (2, 1.5), (3, 2.0)]
    assert board_view["me"]["rank"] == 2 and board_view["hardware"] == "NVIDIA GB10"
    assert [e["name"] for e in board_view["entries"]] == ["Owner", "Eve", "Bob"]


def test_per_hardware_and_language(board, people):
    _gym(board, visibility="public", leaderboard=True)
    _bench(board, UIDS["bob"], "direct", 1.0, hardware="NVIDIA H100")
    _bench(board, UIDS["eve"], "direct", 5.0, hardware="NVIDIA GB10")
    _bench(board, UIDS["eve"], "direct", 0.5, hardware="NVIDIA GB10", lang="cuda")
    eve = people["eve"]
    default = eve.get("/api/gyms/g1/problems/p/leaderboard?lang=python").json()
    assert default["hardware"] == "NVIDIA GB10" and default["hardwares"] == ["NVIDIA GB10", "NVIDIA H100"]  # her own GPU first
    h100 = eve.get("/api/gyms/g1/problems/p/leaderboard?lang=python&hardware=NVIDIA H100").json()
    assert [e["ms"] for e in h100["entries"]] == [1.0] and h100["me"] is None
    assert eve.get("/api/gyms/g1/problems/p/leaderboard?lang=cuda").json()["entries"][0]["ms"] == 0.5


def test_opt_in_and_visibility(board, people):
    _gym(board, visibility="public")
    _bench(board, UIDS["bob"], "direct", 1.0)
    assert people["eve"].get("/api/gyms/g1/problems/p/leaderboard").status_code == 404
    assert board.store.where("leaderboard", "board", "g1__p") == []  # nothing recorded while off
    ann = people["ann"]
    assert ann.put("/api/gyms/g1/settings", json={"visibility": "private", "leaderboard": True}, headers=H).status_code == 200
    assert people["eve"].get("/api/gyms/g1/problems/p/leaderboard").status_code == 404  # private gym
    assert ann.get("/api/gyms/g1/settings").json()["leaderboard"] is True
