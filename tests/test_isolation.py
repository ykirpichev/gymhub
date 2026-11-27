"""Other users' code runs only on the compute pool: a runner has no sandbox of its own (#132)."""

from gymhub.runner.backends import DirectBackend
from gymhub.store import SqliteStore, now


def _queue(store, job_id, **fields):
    store.set("jobs", job_id, {"gym": "algo", "slug": "merge_intervals", "lang": "python", "action": "test",
                               "status": "queued", "created_at": now(), "uid": "g_1", **fields})


def test_direct_mode_leaves_isolated_jobs_it_would_run_itself(tmp_path, monkeypatch):
    """No board in between to say whose job it is, and no pool: an isolated job stays queued for a
    machine that can isolate it; the owner's own jobs are leased as before."""
    monkeypatch.delenv("GYMHUB_TRUST_JOBS", raising=False)
    store = SqliteStore(tmp_path / "db.sqlite")
    backend = DirectBackend(store, "w")
    _queue(store, "theirs", isolated=True)
    _queue(store, "mine")
    assert backend.lease(["algo"])["id"] == "mine"
    assert backend.lease(["algo"]) is None
    assert store.get("jobs", "theirs")["status"] == "queued"
    monkeypatch.setenv("GYMHUB_TRUST_JOBS", "1")  # local development: run it here anyway
    assert backend.lease(["algo"])["id"] == "theirs"


def test_a_runner_never_reports_a_sandbox(monkeypatch):
    from gymhub.runner.agent import Runner

    monkeypatch.setenv("GYMHUB_EXECUTOR", "local")
    runner = Runner(DirectBackend(SqliteStore(":memory:"), "w"), {}, "w")
    assert runner.capabilities()["sandbox"] is False
