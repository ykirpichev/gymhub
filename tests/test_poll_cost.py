"""An idle machine polling for work should cost about one Firestore read per poll, not four."""

from gymhub.store import SqliteStore
from tests.test_board_auth import H, board, client, sign_in  # noqa: F401  (fixture)


class CountingStore:
    """Wraps a store and counts reads (documents fetched; an empty query still costs one)."""

    def __init__(self, inner: SqliteStore):
        self.inner, self.reads, self.writes = inner, 0, 0

    def get(self, *args, **kwargs):
        self.reads += 1
        return self.inner.get(*args, **kwargs)

    def where(self, *args, **kwargs):
        rows = self.inner.where(*args, **kwargs)
        self.reads += max(1, len(rows))
        return rows

    def recent(self, *args, **kwargs):
        rows = self.inner.recent(*args, **kwargs)
        self.reads += max(1, len(rows))
        return rows

    def set(self, *args, **kwargs):
        self.writes += 1
        return self.inner.set(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def test_idle_polls_cost_about_one_read(board, monkeypatch):
    import gymhub.board.workers as workers

    owner = client(board)
    sign_in(owner, "owner")
    token = owner.post("/api/workers", json={"name": "box"}, headers=H).json()["token"]
    machine = client(board)
    machine.headers.update({"Authorization": f"Bearer {token}"})
    machine.post("/v1/worker/heartbeat", json={"gyms": ["algo"], "capabilities": {"gpus": [], "tools": []}})
    machine.post("/v1/worker/lease", json={"gyms": ["algo"]})  # warm the caches
    counter = CountingStore(board.store)
    monkeypatch.setattr(board, "store", counter)
    for _ in range(20):
        assert machine.post("/v1/worker/lease", json={"gyms": ["algo"]}).json()["job"] is None
    assert counter.reads / 20 <= 1.2, f"{counter.reads / 20:.1f} reads per idle poll"
    assert counter.writes == 0
    # heartbeats write the machine's status, but last_seen only every few minutes
    before = counter.writes
    for _ in range(5):
        machine.post("/v1/worker/heartbeat", json={"gyms": ["algo"]})
    assert counter.writes - before == 5
    del workers
