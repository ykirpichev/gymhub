"""Per-user limits on running code; admins exempt; saving free; machine time counted from real duration."""

import threading
import time

from gymhub import jobqueue
from tests.test_board_auth import H, board, client, sign_in  # noqa: F401  (fixture)

JOB = {"gym": "algo", "slug": "p", "lang": "python", "action": "run", "code": "x"}


def _member(board, **limits):
    owner = client(board)
    sign_in(owner, "owner")
    owner.put("/api/admin/settings", json={"allowed_domains": ["example.com"], "members_can_run": True}, headers=H)
    owner.put("/api/admin/limits", json={"jobs_per_hour": 0, "minutes_per_day": 0, "max_waiting": 0, **limits}, headers=H)
    ann = client(board)
    sign_in(ann, "ann")
    return owner, ann


def test_waiting_jobs_cap(board):
    _owner, ann = _member(board, max_waiting=2)
    assert ann.post("/api/jobs", json=JOB, headers=H).status_code == 200
    assert ann.post("/api/jobs", json=JOB, headers=H).status_code == 200
    third = ann.post("/api/jobs", json=JOB, headers=H)
    assert third.status_code == 429 and "waiting or running" in third.json()["detail"]
    # saving is never limited
    assert ann.post("/api/jobs", json={**JOB, "action": "save"}, headers=H).status_code == 200


def test_hourly_cap_and_admin_exemption(board):
    owner, ann = _member(board, jobs_per_hour=2)
    for _ in range(2):
        job = ann.post("/api/jobs", json=JOB, headers=H).json()["id"]
        board.store.set("jobs", job, {"status": "done"}, merge=True)
    assert ann.post("/api/jobs", json=JOB, headers=H).status_code == 429
    assert ann.get("/api/me").json()["usage"]["jobs_this_hour"] == 2
    for _ in range(4):
        assert owner.post("/api/jobs", json=JOB, headers=H).status_code == 200  # admins are exempt
    assert "usage" not in owner.get("/api/me").json()


def test_machine_minutes_are_counted_when_jobs_finish(board):
    _owner, ann = _member(board, minutes_per_day=1)
    job = ann.post("/api/jobs", json=JOB, headers=H).json()["id"]
    leased = jobqueue.lease(board.store, ["algo"], "w1")
    assert leased["id"] == job
    board.store.set("jobs", job, {"started_at": time.time() - 75}, merge=True)  # it ran 75 s
    jobqueue.complete(board.store, job, "w1", {"status": "done", "ok": True, "output": ""}, lease_id=leased["lease_id"])
    assert ann.get("/api/me").json()["usage"]["seconds_today"] >= 75
    blocked = ann.post("/api/jobs", json=JOB, headers=H)
    assert blocked.status_code == 429 and "minutes of machine time" in blocked.json()["detail"]


def test_write_throttle(board, monkeypatch):
    _owner, ann = _member(board)
    monkeypatch.setattr(board, "WRITES_PER_MINUTE", 3)
    codes = [ann.post("/api/jobs", json={**JOB, "action": "save"}, headers=H).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3] == 429


def test_write_throttle_ignores_a_forged_forwarded_header(board, monkeypatch):
    monkeypatch.setattr(board, "WRITES_PER_MINUTE", 3)
    caller = client(board)
    # The platform appends the real address last; whatever comes before it is the caller's claim.
    codes = [
        caller.post("/api/jobs", json=JOB, headers={**H, "X-Forwarded-For": f"10.0.0.{n}, 203.0.113.9"}).status_code
        for n in range(6)
    ]
    assert 429 in codes[3:] and 429 not in codes[:3]
    other = caller.post("/api/jobs", json=JOB, headers={**H, "X-Forwarded-For": "10.0.0.1, 198.51.100.7"})
    assert other.status_code != 429  # a different real address has its own allowance


def test_write_throttle_is_kept_in_the_store(board, monkeypatch):
    """The counter is a document, so a second instance of the board sees the same count."""
    monkeypatch.setattr(board, "WRITES_PER_MINUTE", 2)
    monkeypatch.setattr(board.time, "time", lambda: 1_700_000_010.0)  # one minute window, however slow the run
    assert [board._count_write("ip:1.2.3.4") for _ in range(3)] == [True, True, False]
    assert [board._count_write("user:7") for _ in range(2)] == [True, True]


def test_parallel_jobs_stay_inside_max_waiting(board):
    _owner, ann = _member(board, max_waiting=2)
    codes = []

    def submit():
        codes.append(ann.post("/api/jobs", json=JOB, headers=H).status_code)

    threads = [threading.Thread(target=submit) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert codes.count(200) == 2 and codes.count(429) == 6
    assert len([1 for _id, job in board.store.where("jobs", "status", "queued")]) == 2


def test_parallel_jobs_stay_inside_the_hourly_limit(board):
    _owner, ann = _member(board, jobs_per_hour=3)
    codes = []

    def submit():
        codes.append(ann.post("/api/jobs", json=JOB, headers=H).status_code)

    threads = [threading.Thread(target=submit) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert codes.count(200) == 3


def test_a_finished_job_frees_its_waiting_slot(board):
    _owner, ann = _member(board, max_waiting=1)
    first = ann.post("/api/jobs", json=JOB, headers=H)
    assert first.status_code == 200
    assert ann.post("/api/jobs", json=JOB, headers=H).status_code == 429
    board.store.set("jobs", first.json()["id"], {"status": "done"}, merge=True)
    assert ann.post("/api/jobs", json=JOB, headers=H).status_code == 200


def test_attempt_delay_grows_and_resets(board):
    from gymhub import quotas

    assert [quotas.attempt_delay(n) for n in (1, 3, 4, 5, 6, 20)] == [0, 0, 5, 10, 20, 300]
    store = board.store
    for _ in range(3):
        quotas.begin_attempt(store, "u", "g__p")  # three misses in a row are free
    quotas.begin_attempt(store, "u", "g__p")  # the fourth is allowed, and starts a delay
    try:
        quotas.begin_attempt(store, "u", "g__p")
        raise AssertionError("expected a delay")
    except quotas.QuotaExceeded as exc:
        assert "Try again" in str(exc)
    quotas.finish_attempt(store, "u", "g__p", ok=True)
    quotas.begin_attempt(store, "u", "g__p")  # a correct answer cleared the misses
