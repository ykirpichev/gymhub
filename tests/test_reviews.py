"""Spaced review (issue #20): the queue on Home, re-solving from scratch, rescheduling."""

import time

from gymhub import jobqueue, reviews
from tests.test_board_auth import H, board, client, sign_in  # noqa: F401  (fixture)
from tests.test_gym_access import UIDS, _gym, people  # noqa: F401  (fixture)

DAY = 86400


def _iso(ts):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts))


def _solved(board, uid, gym, slug, days_ago):
    key = f"{uid}__{gym}"
    doc = board.store.get("user_progress", key) or {"uid": uid, "gym": gym, "data": {"version": 1, "exercises": {}}}
    doc["data"]["exercises"][slug] = {"hints_seen": 0, "langs": {"python": {"attempts": 1, "first_solved": _iso(time.time() - days_ago * DAY)}}}
    board.store.set("user_progress", key, doc)


def test_schedule():
    gyms = [{"id": "g", "title": "G", "problem_list": [{"slug": "a", "title": "A"}, {"slug": "b", "title": "B"}, {"slug": "w", "warmup": True}],
             "progress": {"exercises": {
                 "a": {"langs": {"python": {"first_solved": _iso(time.time() - 2 * DAY)}}},
                 "b": {"langs": {"python": {"first_solved": _iso(time.time() - 3600)}}},
                 "w": {"langs": {"python": {"first_solved": _iso(time.time() - 9 * DAY)}}},
                 "c": {"langs": {"python": {"attempts": 3}}}}}}]

    class Store(dict):
        def where(self, _c, _f, _v):
            return list(self.items())

        def get(self, _c, key):
            return dict.get(self, key)

        def set(self, _c, key, doc):
            self[key] = doc

    store = Store()
    queue = reviews.queue(store, "u", gyms)
    assert [item["slug"] for item in queue["due"]] == ["a"] and queue["upcoming"] == 1  # b: tomorrow; warm-ups never
    first = reviews.advance(store, "u", "g", "a")
    assert first["stage"] == 1 and round((first["due_at"] - time.time()) / DAY) == 7
    assert reviews.queue(store, "u", gyms)["due"] == []
    later = {item["slug"]: item for item in reviews.queue(store, "u", gyms, now=time.time() + 8 * DAY)["due"]}
    assert later["a"]["stage"] == 1 and later["b"]["stage"] == 0
    reviews.advance(store, "u", "g", "a")
    last = reviews.advance(store, "u", "g", "a")
    assert last["stage"] == 3 and last["due_at"] is None
    assert all(item["slug"] != "a" for item in reviews.queue(store, "u", gyms, now=time.time() + 400 * DAY)["due"])


def test_resolve_from_scratch_on_the_board(board, people):
    _gym(board, visibility="public")
    eve = people["eve"]
    _solved(board, UIDS["eve"], "g1", "p", days_ago=2)
    board.store.set("workspaces", f"{UIDS['eve']}__g1__p__python", {"code": "MY SAVED SOLUTION"})
    home = eve.get("/api/home").json()
    assert [item["slug"] for item in home["reviews"]["due"]] == ["p"]
    reply = eve.post("/api/jobs", json={"gym": "g1", "slug": "p", "lang": "python", "action": "test",
                                        "code": "from scratch", "options": {"review": True}}, headers=H)
    job = board.store.get("jobs", reply.json()["id"])
    assert job["isolated"] and job["options"] == {"review": True}
    assert board.store.get("workspaces", f"{UIDS['eve']}__g1__p__python")["code"] == "MY SAVED SOLUTION"
    board.store.set("jobs", reply.json()["id"], {"status": "running", "runner": "w"}, merge=True)
    jobqueue.complete(board.store, reply.json()["id"], "w", {"status": "done", "ok": True, "output": "pass"})
    assert eve.get("/api/home").json()["reviews"]["due"] == []
    assert board.store.get("reviews", f"{UIDS['eve']}__g1__p")["stage"] == 1


def test_owner_reviews_run_isolated_too(board, people):
    owner = people["owner"]
    reply = owner.post("/api/jobs", json={"gym": "algo", "slug": "p", "lang": "python", "action": "test", "code": "x",
                                          "options": {"review": True}}, headers=H)
    assert board.store.get("jobs", reply.json()["id"])["isolated"]  # never writes the owner's solution file
    plain = owner.post("/api/jobs", json={"gym": "algo", "slug": "p", "lang": "python", "action": "test", "code": "x"}, headers=H)
    assert not board.store.get("jobs", plain.json()["id"])["isolated"]
