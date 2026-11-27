"""Staging shares the production database through a collection prefix (issue #24)."""

import pytest

from gymhub import blobs, store


def test_prefix_keeps_staging_apart(tmp_path, monkeypatch):
    path = tmp_path / "board.db"
    monkeypatch.setenv("GYMHUB_STORE", f"sqlite:{path}")
    monkeypatch.setenv("GYMHUB_STORE_PREFIX", "staging-")
    staging = store.open_store()
    monkeypatch.delenv("GYMHUB_STORE_PREFIX")
    production = store.open_store()

    staging.set("jobs", "a", {"status": "queued", "t": 1})
    production.set("jobs", "b", {"status": "queued", "t": 2})
    assert [doc_id for doc_id, _ in staging.where("jobs", "status", "queued")] == ["a"]
    assert [doc_id for doc_id, _ in production.recent("jobs", "t", 10)] == ["b"]
    assert production.get("staging-jobs", "a") == {"status": "queued", "t": 1}
    assert staging.claim("jobs", "a", {"status": "queued"}, {"status": "running"})
    assert not staging.claim("jobs", "b", {"status": "queued"}, {"status": "running"})
    staging.delete("jobs", "a")
    assert staging.get("jobs", "a") is None


def test_bad_prefix_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("GYMHUB_STORE", f"sqlite:{tmp_path / 'x.db'}")
    monkeypatch.setenv("GYMHUB_STORE_PREFIX", "a/b")
    with pytest.raises(SystemExit):
        store.open_store()


def test_staging_bundles_live_under_a_folder(monkeypatch):
    monkeypatch.delenv("GYMHUB_BLOBS", raising=False)
    monkeypatch.setenv("GYMHUB_STORE_PREFIX", "staging-")
    gcs = blobs.open_blobs(store_spec="firestore:proj")
    assert (gcs.bucket, gcs.prefix) == ("proj-gym-bundles", "staging/")
    plain = blobs.open_blobs("gs://proj-gym-bundles")
    assert (plain.bucket, plain.prefix) == ("proj-gym-bundles", "")
