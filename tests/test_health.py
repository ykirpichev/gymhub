"""Operational numbers (issue #24): structured job logs and the Admin health summary."""

import json
import logging
import time

from gymhub import health, jobqueue
from tests.test_board_auth import H, board, client, sign_in  # noqa: F401  (fixture)


def _events(caplog) -> list[dict]:
    return [record.fields for record in caplog.records if record.name == "gymhub.jobs"]


def test_job_lifecycle_is_logged_without_code_or_emails(board, caplog):
    caplog.set_level(logging.INFO, logger="gymhub")
    owner = client(board)
    sign_in(owner, "owner")
    token = owner.post("/api/workers", json={"name": "spark"}, headers=H).json()["token"]
    worker = client(board)
    worker.headers.update({"Authorization": f"Bearer {token}"})
    board.store.set("gyms", "algo", {"title": "Algo", "updated_at": 0})
    board.store.set("problems", "algo__p", {"gym": "algo", "slug": "p", "title": "P", "langs": ["python"]})
    secret_code = "print('my secret solution')"
    job_id = owner.post("/api/jobs", json={"gym": "algo", "slug": "p", "lang": "python", "action": "test", "code": secret_code},
                        headers=H).json()["id"]
    job = worker.post("/v1/worker/lease", json={"gyms": ["algo"]}).json()["job"]
    worker.post(f"/v1/worker/jobs/{job_id}/complete",
                json={"final": {"status": "error", "ok": False, "output": "Traceback: secret"}, "lease_id": job["lease_id"]})
    cancelled = owner.post("/api/jobs", json={"gym": "algo", "slug": "p", "lang": "python", "action": "bench", "code": "x"},
                           headers=H).json()["id"]
    owner.post(f"/api/jobs/{cancelled}/cancel", headers=H)

    events = _events(caplog)
    assert [(e["event"], e["job"]) for e in events] == [
        ("job_created", job_id), ("job_leased", job_id), ("job_finished", job_id),
        ("job_created", cancelled), ("job_finished", cancelled),
    ]
    leased, finished = events[1], events[2]
    assert leased["queue_wait_s"] >= 0 and leased["tries"] == 1 and leased["runner"]
    assert finished["status"] == "error" and finished["ok"] is False and finished["run_s"] >= 0
    assert finished["gym"] == "algo" and finished["lang"] == "python" and finished["action"] == "test" and finished["uid"]
    assert events[4]["status"] == "cancelled"
    assert [r.levelname for r in caplog.records if r.name == "gymhub.jobs"][2] == "WARNING"
    text = caplog.text + json.dumps(events)
    assert "secret" not in text and "@example.com" not in text


def test_requeue_and_expiry_are_logged(board, caplog):
    caplog.set_level(logging.INFO, logger="gymhub")
    store = board.store
    store.set("jobs", "j", {"gym": "algo", "slug": "p", "action": "test", "status": "queued", "created_at": time.time()})
    for _ in range(jobqueue.MAX_TRIES):
        assert jobqueue.lease(store, ["algo"], "spark")
        store.set("jobs", "j", {"lease_expires_at": time.time() - 1}, merge=True)
        jobqueue.sweep(store)
    assert [e["event"] for e in _events(caplog)] == ["job_leased", "job_requeued"] * 2 + ["job_leased", "job_finished"]
    assert _events(caplog)[-1]["status"] == "error"


def test_log_formats(monkeypatch):
    record = logging.LogRecord("gymhub.jobs", logging.WARNING, __file__, 1, "job finished: test algo/p error (j1)", None, None)
    record.fields = {"event": "job_finished", "job": "j1", "run_s": 2.5, "runner": None}
    entry = json.loads(health.CloudFormatter().format(record))
    assert entry["severity"] == "WARNING" and entry["message"].startswith("job finished")
    assert entry["event"] == "job_finished" and entry["run_s"] == 2.5 and entry["time"].endswith("Z")
    plain = health.PlainFormatter().format(record)
    assert "WARNING" in plain and "run_s=2.5" in plain and "runner" not in plain and not plain.startswith("{")

    monkeypatch.delenv("GYMHUB_LOG_FORMAT", raising=False)
    monkeypatch.delenv("K_SERVICE", raising=False)
    assert not health.json_logs()
    monkeypatch.setenv("K_SERVICE", "gymhub-board")
    assert health.json_logs()
    monkeypatch.setenv("GYMHUB_LOG_FORMAT", "plain")
    assert not health.json_logs()
    monkeypatch.delenv("K_SERVICE")
    monkeypatch.setenv("GYMHUB_LOG_FORMAT", "json")
    assert health.json_logs()


def test_summary_over_the_last_hour_and_day(board):
    store = board.store
    t = time.time()
    for i, (status, wait, run) in enumerate([("done", 1.0, 2.0), ("done", 3.0, 4.0), ("error", 5.0, 10.0)]):
        store.set("jobs", f"j{i}", {"gym": "algo", "slug": "p", "action": "test", "uid": "u", "status": "running", "runner": "spark",
                                    "hardware": "NVIDIA GB10", "created_at": t - 60, "started_at": t - 60 + wait})
        jobqueue.complete(store, f"j{i}", "spark", {"status": status, "ok": status == "done", "output": ""})
        store.set("jobs", f"j{i}", {"finished_at": t - 60 + wait + run}, merge=True)
    store.set("jobs", "q", {"gym": "algo", "action": "bench", "status": "queued", "created_at": t})
    store.set("jobs", "c", {"gym": "algo", "action": "bench", "status": "cancelled", "created_at": t, "finished_at": t})
    # three hours ago: only in the day
    store.set("jobs", "old", {"gym": "algo", "action": "bench", "status": "done", "runner": "spark", "created_at": t - 3 * 3600,
                              "started_at": t - 3 * 3600 + 100, "finished_at": t - 3 * 3600 + 400})
    store.set("runners", "spark", {"name": "Spark", "heartbeat_at": t, "online_slots": {str(int(t - 2 * 3600)): 600.0,
                                                                                          str(int(t - 300)): 300.0}})

    owner = client(board)
    sign_in(owner, "owner")
    reply = owner.get("/api/admin/health").json()
    assert reply["queued_now"] == 1 and reply["running_now"] == 0 and reply["complete"] is True
    hour, day = reply["windows"]
    assert (hour["hours"], day["hours"]) == (1, 24)
    assert hour["jobs"] == 5 and hour["by_action"] == {"test": 3, "bench": 2} and day["jobs"] == 6
    assert hour["queue_wait_s"] == {"p50": 3.0, "p95": 5.0, "n": 3} and hour["run_s"] == {"p50": 4.0, "p95": 10.0, "n": 3}
    assert hour["error_rate"] == round(1 / 3, 3) and day["error_rate"] == 0.25 and hour["cancelled"] == 1
    spark = hour["machines"]["spark"]
    assert spark["name"] == "Spark" and spark["jobs"] == 3 and spark["errors"] == 1 and spark["hardware"] == "NVIDIA GB10"
    assert spark["busy_s"] == 16.0 and spark["online_s"] == 300.0 and spark["utilization"] == round(16 / 300, 3)
    assert day["machines"]["spark"]["busy_s"] == 316.0 and day["machines"]["spark"]["online_s"] == 900.0
    assert owner.get("/api/admin/health?hours=72").json()["windows"][1]["hours"] == 72

    board.store.set("settings", "auth", {"allowed_domains": ["example.com"]})
    member = client(board)
    sign_in(member, "ann")
    assert member.get("/api/admin/health").status_code == 403


def test_heartbeats_record_online_time(board):
    owner = client(board)
    sign_in(owner, "owner")
    token = owner.post("/api/workers", json={"name": "spark"}, headers=H).json()["token"]
    worker = client(board)
    worker.headers.update({"Authorization": f"Bearer {token}"})
    worker.post("/v1/worker/heartbeat", json={"gyms": ["algo"]})
    worker_id = token.split("_")[1]
    doc = board.store.get("runners", worker_id)
    assert doc["online_slots"] == {}  # nothing before the first heartbeat
    board.store.set("runners", worker_id, {"heartbeat_at": doc["heartbeat_at"] - 30}, merge=True)
    worker.post("/v1/worker/heartbeat", json={"gyms": ["algo"]})
    assert 29 <= sum(board.store.get("runners", worker_id)["online_slots"].values()) <= 32
    assert "online_slots" not in owner.get("/api/overview").json()["runners"][0]

    # a gap longer than a heartbeat timeout is time offline; slots older than a day are dropped
    at = 1_000_000_000.0
    slots = health.add_online({"heartbeat_at": at - 600, "online_slots": {str(int(at - 2 * 86400)): 600}}, at)
    assert slots == {}
    slots = health.add_online({"heartbeat_at": at - 30, "online_slots": slots}, at)
    assert slots == {str(int(at // 600 * 600)): 30}
    assert health.online_seconds({str(int(at - 900)): 600}, at - 600) == 300  # half the slot is inside
