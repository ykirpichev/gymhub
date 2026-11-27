"""Retention (issue #24): scratch documents carry expire_at, a datetime Firestore's TTL acts on."""

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from gymhub import jobqueue, quotas
from gymhub.store import EXPIRING, PrefixedStore, SqliteStore, expires_in
from tests.test_board_auth import H, board, client, sign_in  # noqa: F401  (fixture)

REPO = Path(__file__).resolve().parents[1]


def test_sqlite_round_trips_datetimes_and_deletes_expired(tmp_path):
    store = SqliteStore(tmp_path / "db.sqlite")
    soon = expires_in(60)
    store.set("usage", "a", {"n": 1, "expire_at": soon})
    store.set("usage", "b", {"n": 2, "expire_at": datetime.now(timezone.utc) - timedelta(seconds=1)})
    store.set("usage", "c", {"n": 3})
    store.set("jobs", "old", {"expire_at": datetime.now(timezone.utc) - timedelta(days=1)})
    assert store.get("usage", "a")["expire_at"] == soon and store.get("usage", "a")["expire_at"].tzinfo is not None
    store.set("usage", "a", {"m": 1}, merge=True)
    assert store.claim("usage", "a", {"m": 1}, {"m": 2})
    assert store.transact("usage", "a", lambda doc: {**doc, "k": 1})["expire_at"] == soon
    assert store.get("usage", "a")["expire_at"] == soon
    assert store.delete_expired("usage") == 1
    assert sorted(doc_id for doc_id, _ in store.recent("usage", "n", 10)) == ["a", "c"]
    assert store.get("jobs", "old") is not None  # only the collection asked for
    assert PrefixedStore(store, "").delete_expired("jobs") == 1


def test_only_scratch_jobs_expire(board):
    owner = client(board)
    sign_in(owner, "owner")
    board.store.set("gyms", "algo", {"title": "Algo", "updated_at": 0})
    board.store.set("problems", "algo__p", {"gym": "algo", "slug": "p", "title": "P", "langs": ["python"]})
    ids = {}
    for action in ("run", "test", "bench", "save", "hint"):
        reply = owner.post("/api/jobs", json={"gym": "algo", "slug": "p", "lang": "python", "action": action, "code": "x"}, headers=H)
        ids[action] = reply.json()["id"]
    jobs = {action: board.store.get("jobs", job_id) for action, job_id in ids.items()}
    assert "expire_at" not in jobs["test"] and "expire_at" not in jobs["bench"]  # submission history stays
    day = datetime.now(timezone.utc) + timedelta(days=1)
    assert abs((jobs["run"]["expire_at"] - day).total_seconds()) < 60
    assert jobs["save"]["expire_at"] > day + timedelta(days=28) and jobs["hint"]["expire_at"] > day
    # the job's own API still serves it (FastAPI encodes the datetime)
    assert owner.get(f"/api/jobs/{ids['run']}").status_code == 200

    # The board's sweep deletes what has expired, like Firestore's TTL would.
    board.store.set("jobs", ids["save"], {"expire_at": datetime.now(timezone.utc) - timedelta(seconds=1)}, merge=True)
    jobqueue.sweep(board.store)
    assert board.store.get("jobs", ids["save"]) is None and board.store.get("jobs", ids["test"]) is not None


def test_usage_attempts_throttle_and_registrations_expire(board):
    store = board.store
    quotas.check_and_count(store, "u1", "job1")
    quotas.add_machine_time(store, "u1", 5)
    quotas.begin_attempt(store, "u1", "algo__q")
    for doc in (store.get("usage", "u1"), store.get("attempts", "u1__algo__q")):
        assert isinstance(doc["expire_at"], datetime) and doc["expire_at"] > datetime.now(timezone.utc)
    owner = client(board)
    sign_in(owner, "owner")
    owner.post("/api/my/workers/registration", json={"name": "box"}, headers=H)
    assert all("expire_at" in doc for _id, doc in store.recent("registrations", "created_at", 5))
    assert [doc for _id, doc in store.recent("throttle", "window", 5)][0]["expire_at"] > datetime.now(timezone.utc)
    assert quotas.usage(store, "u1")["jobs_this_hour"] == 1  # reads still work


def test_gcp_ttl_covers_the_expiring_collections():
    script = (REPO / "deploy" / "gcp.sh").read_text()
    listed = re.search(r'^TTL_COLLECTIONS="([^"]+)"', script, re.M).group(1).split()
    assert sorted(listed) == sorted(EXPIRING)
    assert "ttl) cmd_ttl" in script
