"""Gym visibility, roles, join links, Explore, forks and owner stats (issue #15).

Every visibility level is checked against every kind of user, on every endpoint that exposes a gym."""

import time

import pytest

from gymhub import packages
from tests.test_board_auth import H, board, client, sign_in  # noqa: F401  (fixture)
from tests.test_gym_versions import GZ, HELLO, _worker, complete_validation

UIDS = {"owner": "g_1", "ann": "g_2", "bob": "g_3", "eve": "g_4"}


@pytest.fixture()
def people(board):
    board.store.set("settings", "auth", {"allowed_domains": ["example.com", "school.edu"], "members_can_run": True})
    clients = {}
    for name in UIDS:
        c = client(board)
        assert sign_in(c, name).status_code == 200
        clients[name] = c
    return clients


def _gym(board, gym_id="g1", **fields):
    """An uploaded gym owned by ann, with one exercise."""
    doc = {"title": gym_id.upper(), "source": "bundle", "owner_uid": UIDS["ann"], "visibility": "private", "updated_at": time.time(),
           "langs": [{"id": "python", "label": "Python"}], "bundle": {"sha": "0" * 64, "key": "k", "version": 1}, "version": 1, **fields}
    board.store.set("gyms", gym_id, doc)
    board.store.set("problems", f"{gym_id}__p", {"gym": gym_id, "slug": "p", "title": "P", "langs": ["python"], "order": 0,
                                                 "starters": {"python": "# GYM_STARTER\n"}})
    return gym_id


def _sees(c, gym_id) -> dict:
    """What one user can do with a gym, through every read path."""
    listed = any(g["id"] == gym_id for g in c.get("/api/home").json()["gyms"])
    overview = any(g["id"] == gym_id for g in c.get("/api/overview").json()["gyms"])
    assert listed == overview
    return {
        "listed": listed,
        "explore": any(g["id"] == gym_id for g in c.get("/api/explore").json()["gyms"]),
        "open": c.get(f"/api/gyms/{gym_id}").status_code == 200,
        "problem": c.get(f"/api/gyms/{gym_id}/problems/p").status_code == 200,
        "run": c.post("/api/jobs", json={"gym": gym_id, "slug": "p", "lang": "python", "action": "save", "code": "x"}, headers=H).status_code == 200,
    }


NOTHING = {"listed": False, "explore": False, "open": False, "problem": False, "run": False}


def test_private_gym_is_invisible_to_everyone_else(board, people):
    _gym(board, visibility="private", editors=[UIDS["bob"]])
    assert _sees(people["eve"], "g1") == NOTHING
    assert _sees(people["ann"], "g1") == {**NOTHING, "listed": True, "open": True, "problem": True, "run": True}
    assert _sees(people["bob"], "g1")["open"]  # editor
    assert _sees(people["owner"], "g1")["open"]  # admins see every gym
    assert people["eve"].get("/api/gyms/g1/settings").status_code == 404


def test_shared_gym_only_for_people_it_is_shared_with(board, people):
    _gym(board, visibility="shared", learners=[UIDS["bob"]])
    bob = _sees(people["bob"], "g1")
    assert bob == {**NOTHING, "listed": True, "open": True, "problem": True, "run": True}
    assert _sees(people["eve"], "g1") == NOTHING


def test_unlisted_gym_opens_by_link_and_is_then_listed(board, people):
    _gym(board, visibility="unlisted", share_code="c0de")
    eve = people["eve"]
    assert _sees(eve, "g1") == NOTHING  # the id alone opens nothing (#116)
    assert eve.post("/api/gyms/g1/join", json={"code": "wrong"}, headers=H).status_code == 404
    assert eve.post("/api/gyms/g1/join", json={"code": "c0de"}, headers=H).status_code == 200  # following the link
    after = _sees(eve, "g1")
    assert after == {**NOTHING, "listed": True, "open": True, "problem": True, "run": True}  # but never in Explore


def test_public_gym_is_in_explore_for_everyone(board, people):
    _gym(board, visibility="public", tags=["graphs"])
    for name in ("bob", "eve"):
        assert _sees(people[name], "g1") == {"listed": True, "explore": True, "open": True, "problem": True, "run": True}
    found = people["eve"].get("/api/explore?q=G1").json()
    assert found["gyms"][0]["by"] == "Ann" and found["tags"] == ["graphs"]
    assert people["eve"].get("/api/explore?tag=strings").json()["gyms"] == []


def test_local_gyms_default_to_public_and_admins_can_hide_them(board, people):
    assert _sees(people["eve"], "algo")["open"]  # the fixture's machine-published gym
    owner = people["owner"]
    assert owner.put("/api/gyms/algo/settings", json={"visibility": "private"}, headers=H).status_code == 200
    assert not _sees(people["eve"], "algo")["open"]
    # A machine republishing the catalog keeps the setting.
    _owner_token = owner.post("/api/workers", json={"name": "m"}, headers=H).json()["token"]
    worker = client(board)
    worker.headers.update({"Authorization": f"Bearer {_owner_token}"})
    worker.post("/v1/worker/catalog", json={"gym": "algo", "meta": {"title": "Algo"}, "problems": [{"slug": "p", "title": "P", "langs": ["python"]}]})
    assert board.store.get("gyms", "algo")["visibility"] == "private"


def test_only_the_owner_changes_settings_and_editors_upload(board, people):
    _gym(board, visibility="private", editors=[UIDS["bob"]])
    bob, ann = people["bob"], people["ann"]
    assert bob.get("/api/gyms/g1/settings").json()["role"] == "editor"
    assert "share_code" not in bob.get("/api/gyms/g1/settings").json()
    assert bob.put("/api/gyms/g1/settings", json={"visibility": "public"}, headers=H).status_code == 403
    reply = ann.put("/api/gyms/g1/settings", json={"visibility": "shared", "editor_emails": ["bob@school.edu"],
                                                    "learner_emails": ["eve@example.com"], "tags": ["Graphs", "bad tag!!!"]}, headers=H)
    assert reply.status_code == 200
    settings = reply.json()
    assert settings["learners"] == ["eve@example.com"] and settings["tags"] == ["graphs"] and settings["share_code"]
    unknown = ann.put("/api/gyms/g1/settings", json={"visibility": "shared", "learner_emails": ["eve@example.com", "nobody@x.org"]}, headers=H)
    assert unknown.status_code == 200 and unknown.json()["pending_learners"] == ["nobody@x.org"]  # invited (#118)
    assert ann.put("/api/gyms/g1/settings", json={"visibility": "secret"}, headers=H).status_code == 400
    # Sharing lists and the join code never reach learners.
    eve_view = people["eve"].get("/api/gyms/g1").json()["gym"]
    assert eve_view["role"] == "learner" and "learners" not in eve_view and "share_code" not in eve_view


def test_join_link(board, people):
    _gym(board, visibility="shared", share_code="abc123")
    eve = people["eve"]
    assert eve.post("/api/gyms/g1/join", json={"code": "wrong"}, headers=H).status_code == 404
    assert eve.post("/api/gyms/g1/join", json={"code": "abc123"}, headers=H).status_code == 200
    assert _sees(eve, "g1")["open"]
    new = people["ann"].post("/api/gyms/g1/share-code", headers=H).json()["share_code"]
    assert new != "abc123"
    assert people["bob"].post("/api/gyms/g1/join", json={"code": "abc123"}, headers=H).status_code == 404
    board.store.set("gyms", "g1", {"visibility": "private"}, merge=True)
    assert people["bob"].post("/api/gyms/g1/join", json={"code": new}, headers=H).status_code == 404


def test_members_create_gyms_and_cannot_take_others_ids(board, people):
    ann, eve = people["ann"], people["eve"]
    assert ann.post("/api/gym-versions", content=packages.pack(HELLO), headers=GZ).status_code == 200
    assert eve.get("/api/gym-versions?gym=hello").json()["versions"] == []  # not hers
    assert len(ann.get("/api/gym-versions").json()["versions"]) == 1
    worker = _worker(board, people["owner"])
    job = worker.post("/v1/worker/lease", json={"gyms": []}).json()["job"]
    complete_validation(worker, job)
    gym = board.store.get("gyms", "hello")
    assert gym["owner_uid"] == UIDS["ann"] and gym["visibility"] == "private"
    assert not _sees(eve, "hello")["open"]
    # eve cannot publish a version of ann's gym, even one she can see.
    board.store.set("gyms", "hello", {"visibility": "public"}, merge=True)
    reply = eve.post("/api/gym-versions", content=packages.pack(HELLO), headers=GZ)
    assert reply.status_code == 400 and "taken" in reply.json()["detail"]
    # Members cannot create gyms while running code is off for them.
    board.store.set("settings", "auth", {"members_can_run": False}, merge=True)
    assert eve.post("/api/gym-versions", content=packages.pack(HELLO), headers=GZ).status_code == 403


def test_an_editors_version_keeps_the_owner_and_settings(board, people):
    _gym(board, "hello", visibility="shared", editors=[UIDS["bob"]], learners=[UIDS["eve"]], tags=["intro"])
    reply = people["bob"].post("/api/gym-versions", content=packages.pack(HELLO), headers=GZ)
    assert reply.status_code == 200, reply.text
    worker = _worker(board, people["owner"])
    job = worker.post("/v1/worker/lease", json={"gyms": []}).json()["job"]
    complete_validation(worker, job)
    gym = board.store.get("gyms", "hello")
    assert gym["version"] == 2 and gym["owner_uid"] == UIDS["ann"]
    assert gym["visibility"] == "shared" and gym["learners"] == [UIDS["eve"]] and gym["tags"] == ["intro"]


def test_fork_needs_permission_and_makes_a_private_copy(board, people):
    ann, eve = people["ann"], people["eve"]
    ann.post("/api/gym-versions", content=packages.pack(HELLO), headers=GZ)
    worker = _worker(board, people["owner"])
    job = worker.post("/v1/worker/lease", json={"gyms": []}).json()["job"]
    complete_validation(worker, job)
    board.store.set("gyms", "hello", {"visibility": "public"}, merge=True)
    assert eve.post("/api/gyms/hello/fork", json={"id": "eves-hello"}, headers=H).status_code == 403  # forks not allowed
    assert eve.get("/api/gyms/hello").json()["gym"]["can_fork"] is False
    board.store.set("gyms", "hello", {"allow_fork": True}, merge=True)
    assert eve.post("/api/gyms/hello/fork", json={"id": "Bad Id"}, headers=H).status_code == 400
    reply = eve.post("/api/gyms/hello/fork", json={"id": "eves-hello", "title": "Eve's hello"}, headers=H)
    assert reply.status_code == 200, reply.text
    version = reply.json()["version"]
    assert version["gym"] == "eves-hello" and version["forked_from"] == {"gym": "hello", "version": 1}
    job = worker.post("/v1/worker/lease", json={"gyms": []}).json()["job"]
    assert job["uid"] == UIDS["eve"]  # her code now: it runs as her job
    complete_validation(worker, job)
    fork = board.store.get("gyms", "eves-hello")
    assert fork["owner_uid"] == UIDS["eve"] and fork["visibility"] == "private" and fork["title"] == "Eve's hello"
    assert fork["forked_from"]["gym"] == "hello"
    assert not _sees(ann, "eves-hello")["open"]  # the original's owner does not see a private fork
    # Local gyms have no bundle: nothing to fork.
    assert people["owner"].post("/api/gyms/algo/fork", json={"id": "algo-copy"}, headers=H).status_code == 403


def test_stats_are_aggregates_for_owner_and_editors_only(board, people):
    _gym(board, visibility="public", editors=[UIDS["bob"]])
    progress = {
        UIDS["eve"]: {"exercises": {"p": {"hints_seen": 2, "langs": {"python": {"attempts": 5}}}}},
        UIDS["owner"]: {"exercises": {"p": {"hints_seen": 0, "langs": {"python": {"attempts": 2, "first_solved": "2026-11-23T10:00:00"}}}}},
    }
    for uid, data in progress.items():
        board.store.set("user_progress", f"{uid}__g1", {"uid": uid, "gym": "g1", "data": data})
    assert people["eve"].get("/api/gyms/g1/stats").status_code == 403
    stats = people["bob"].get("/api/gyms/g1/stats").json()
    row = stats["problems"][0]
    assert stats["learners"] == 2
    assert (row["started"], row["solved"], row["stuck"], row["solve_rate"]) == (2, 1, 1, 0.5)
    assert row["attempts_to_solve"] == 2 and row["attempts_while_stuck"] == 5 and row["hints_per_learner"] == 1
    assert "eve" not in str(stats) and UIDS["eve"] not in str(stats)


def test_share_with_someone_not_on_the_board_yet(board, people):
    """#118: an unknown email becomes a pending invitation, claimed at that person's first sign-in."""
    _gym(board, visibility="shared")
    ann = people["ann"]
    reply = ann.put("/api/gyms/g1/settings", json={"visibility": "shared", "learner_emails": ["New@Example.com", "bob@school.edu"],
                                                    "editor_emails": ["co@school.edu"]}, headers=H)
    assert reply.status_code == 200, reply.text
    body = reply.json()
    assert body["pending_learners"] == ["new@example.com"] and body["pending_editors"] == ["co@school.edu"]
    assert body["learners"] == ["bob@school.edu"]
    assert ann.put("/api/gyms/g1/settings", json={"visibility": "shared", "learner_emails": ["not an email"]}, headers=H).status_code == 400

    newcomer, co = client(board), client(board)
    assert sign_in(newcomer, "new").status_code == 200 and sign_in(co, "co").status_code == 200
    assert _sees(newcomer, "g1")["open"]
    assert co.get("/api/gyms/g1/settings").status_code == 200  # an editor
    after = ann.get("/api/gyms/g1/settings").json()
    assert after["pending_learners"] == [] and after["pending_editors"] == []
    assert "new@example.com" in after["learners"] and "co@school.edu" in after["editors"]

    # Removing a pending email before they sign in withdraws the invitation.
    ann.put("/api/gyms/g1/settings", json={"visibility": "shared", "learner_emails": ["late@example.com"]}, headers=H)
    ann.put("/api/gyms/g1/settings", json={"visibility": "shared", "learner_emails": []}, headers=H)
    late = client(board)
    sign_in(late, "late")
    assert not _sees(late, "g1")["open"]


def test_leave_a_gym(board, people):
    """#119: a learner leaves a shared gym; the owner cannot leave."""
    _gym(board, visibility="shared", learners=[UIDS["bob"]], editors=[UIDS["eve"]])
    bob, eve, ann = people["bob"], people["eve"], people["ann"]
    assert bob.post("/api/gyms/g1/leave", headers=H).json() == {"ok": True, "still_open": False}
    assert _sees(bob, "g1") == NOTHING
    assert eve.post("/api/gyms/g1/leave", headers=H).status_code == 200  # editors may leave too
    assert eve.get("/api/gyms/g1/settings").status_code == 404
    assert ann.post("/api/gyms/g1/leave", headers=H).status_code == 400
    assert people["owner"].post("/api/gyms/nope/leave", headers=H).status_code == 404


def test_narrowing_a_public_gym_keeps_people_practising(board, people):
    """#119: public -> shared keeps everyone with progress, unless the owner says otherwise."""
    _gym(board, visibility="public")
    for name in ("bob", "eve"):
        board.store.set("user_progress", f"{UIDS[name]}__g1", {"uid": UIDS[name], "gym": "g1",
                                                               "data": {"exercises": {"p": {"langs": {"python": {"attempts": 1}}}}}})
    ann = people["ann"]
    reply = ann.put("/api/gyms/g1/settings", json={"visibility": "shared"}, headers=H).json()
    assert reply["kept_learners"] == 2 and set(reply["learners"]) == {"bob@school.edu", "eve@example.com"}
    assert _sees(people["bob"], "g1")["open"]

    ann.put("/api/gyms/g1/settings", json={"visibility": "public"}, headers=H)
    reply = ann.put("/api/gyms/g1/settings", json={"visibility": "shared", "keep_learners": False}, headers=H).json()
    assert reply["kept_learners"] == 0 and reply["learners"] == []
    assert not _sees(people["bob"], "g1")["open"]


def test_transfer_ownership(board, people):
    """#120: the new owner runs the gym; the old one stays on as an editor."""
    _gym(board, visibility="shared", learners=[UIDS["bob"]])
    ann, bob = people["ann"], people["bob"]
    assert bob.post("/api/gyms/g1/transfer", json={"email": "bob@school.edu"}, headers=H).status_code == 403
    assert ann.post("/api/gyms/g1/transfer", json={"email": "nobody@x.org"}, headers=H).status_code == 400
    assert ann.post("/api/gyms/g1/transfer", json={"email": "ann@example.com"}, headers=H).status_code == 400
    reply = ann.post("/api/gyms/g1/transfer", json={"email": "Bob@School.edu"}, headers=H)
    assert reply.json() == {"ok": True, "owner": "bob@school.edu"}
    settings = bob.get("/api/gyms/g1/settings").json()
    assert settings["role"] == "owner" and settings["editors"] == ["ann@example.com"] and settings["learners"] == []
    assert ann.put("/api/gyms/g1/settings", json={"visibility": "public"}, headers=H).status_code == 403
    assert people["owner"].post("/api/gyms/algo/transfer", json={"email": "bob@school.edu"}, headers=H).status_code == 400  # local gym


def test_delete_a_gym(board, people):
    """#120: gone for everyone, queued jobs cancelled, progress kept, the id kept for its owner."""
    _gym(board, visibility="public")
    ann, bob = people["ann"], people["bob"]
    job = bob.post("/api/jobs", json={"gym": "g1", "slug": "p", "lang": "python", "action": "test", "code": "x"}, headers=H).json()["id"]
    board.store.set("user_progress", f"{UIDS['bob']}__g1", {"uid": UIDS["bob"], "gym": "g1", "data": {"exercises": {"p": {}}}})
    assert bob.request("DELETE", "/api/gyms/g1", json={"confirm": "g1"}, headers=H).status_code == 403
    assert ann.request("DELETE", "/api/gyms/g1", json={"confirm": "nope"}, headers=H).status_code == 400
    assert ann.request("DELETE", "/api/gyms/g1", json={"confirm": "g1"}, headers=H).json() == {"ok": True}
    for name in ("ann", "bob", "eve"):
        assert _sees(people[name], "g1") == NOTHING
    assert board.store.get("jobs", job)["status"] == "cancelled"
    assert board.store.get("problems", "g1__p") is None
    assert board.store.get("user_progress", f"{UIDS['bob']}__g1") is not None
    taken = board._may_publish(board.auth.User(uid=UIDS["bob"], email="bob@school.edu", name="Bob"))
    assert "deleted gym" in taken(None, "g1")
    mine = board._may_publish(board.auth.User(uid=UIDS["ann"], email="ann@example.com", name="Ann"))
    assert mine(None, "g1") is None
    assert people["owner"].request("DELETE", "/api/gyms/algo", json={"confirm": "algo"}, headers=H).status_code == 400  # local gym


def test_admins_review_gyms_before_they_go_public(board, people):
    """#121: with review on, a member's request waits for an admin; admins' own gyms and boards without review go straight."""
    _gym(board, visibility="shared")
    ann, owner, eve = people["ann"], people["owner"], people["eve"]
    settings = {"allowed_domains": ["example.com", "school.edu"], "members_can_run": True}
    assert owner.put("/api/admin/settings", json={**settings, "public_gyms_need_review": True}, headers=H).status_code == 200

    reply = ann.put("/api/gyms/g1/settings", json={"visibility": "public"}, headers=H).json()
    assert reply["visibility"] == "shared" and reply["visibility_requested"] == "public" and reply["public_needs_review"]
    assert not _sees(eve, "g1")["open"]
    assert eve.get("/api/admin/gym-requests").status_code == 403
    requests = owner.get("/api/admin/gym-requests").json()["requests"]
    assert [r["id"] for r in requests] == ["g1"] and requests[0]["requested_by"] == "ann@example.com"

    assert owner.post("/api/admin/gym-requests/g1", json={"approve": False}, headers=H).json()["visibility"] == "shared"
    assert owner.get("/api/admin/gym-requests").json()["requests"] == []
    assert owner.post("/api/admin/gym-requests/g1", json={"approve": True}, headers=H).status_code == 404  # nothing pending

    ann.put("/api/gyms/g1/settings", json={"visibility": "public"}, headers=H)
    assert owner.post("/api/admin/gym-requests/g1", json={"approve": True}, headers=H).json()["visibility"] == "public"
    assert _sees(eve, "g1")["explore"]
    assert ann.put("/api/gyms/g1/settings", json={"visibility": "public"}, headers=H).json()["visibility"] == "public"  # stays

    # Withdrawing: asking for something else drops the request.
    ann.put("/api/gyms/g1/settings", json={"visibility": "private"}, headers=H)
    ann.put("/api/gyms/g1/settings", json={"visibility": "public"}, headers=H)
    assert ann.put("/api/gyms/g1/settings", json={"visibility": "private"}, headers=H).json()["visibility_requested"] is None

    _gym(board, "g2", owner_uid=UIDS["owner"])
    assert owner.put("/api/gyms/g2/settings", json={"visibility": "public"}, headers=H).json()["visibility"] == "public"
    owner.put("/api/admin/settings", json={**settings, "public_gyms_need_review": False}, headers=H)
    assert ann.put("/api/gyms/g1/settings", json={"visibility": "public"}, headers=H).json()["visibility"] == "public"


def test_review_is_on_unless_an_admin_turns_it_off(board, people):
    """Public beta: a new board holds members' gyms for review; the setting it had stored wins."""
    board.store.set("settings", "auth", {"allowed_domains": ["example.com", "school.edu"], "members_can_run": True})
    _gym(board, visibility="private")
    reply = people["ann"].put("/api/gyms/g1/settings", json={"visibility": "public"}, headers=H).json()
    assert reply["visibility"] == "private" and reply["visibility_requested"] == "public"
    assert people["owner"].get("/api/admin/users").json()["settings"]["public_gyms_need_review"] is True


def _report(c, gym="g1", reason="spam", details=""):
    return c.post(f"/api/gyms/{gym}/report", json={"reason": reason, "details": details}, headers=H)


def test_report_a_gym(board, people):
    _gym(board, visibility="public")
    ann, bob, eve, owner = people["ann"], people["bob"], people["eve"], people["owner"]
    assert _report(ann).status_code == 400  # its own owner
    assert _report(bob, reason="nonsense").status_code == 400
    assert _report(bob, reason="other").status_code == 400  # "other" needs a word
    assert _report(bob, "nope").status_code == 404
    _gym(board, "g2", visibility="private")
    assert _report(eve, "g2").status_code == 404  # a gym you cannot open cannot be reported
    assert _report(bob).json() == {"ok": True}
    assert _report(bob, reason="offensive", details="slurs in lesson 2").status_code == 200  # replaces bob's first
    assert _report(eve, reason="offensive").status_code == 200
    assert eve.get("/api/admin/gym-reports").status_code == 403
    rows = owner.get("/api/admin/gym-reports").json()["reports"]
    assert [row["id"] for row in rows] == ["g1"]
    assert rows[0]["count"] == 2 and rows[0]["reasons"] == {"offensive": 2}
    assert {r["by"] for r in rows[0]["reports"]} == {"bob@school.edu", "eve@example.com"}

    assert owner.post("/api/admin/gym-reports/g1", json={"action": "shrug"}, headers=H).status_code == 400
    assert owner.post("/api/admin/gym-reports/g1", json={"action": "dismiss"}, headers=H).json()["status"] == "dismissed"
    assert owner.get("/api/admin/gym-reports").json()["reports"] == []
    assert _sees(eve, "g1")["explore"]  # dismissing leaves the gym as it was


def test_admin_hides_a_reported_gym_until_they_open_it(board, people):
    _gym(board, visibility="public", editors=[UIDS["bob"]])
    ann, bob, eve, owner = people["ann"], people["bob"], people["eve"], people["owner"]
    _report(eve, reason="harmful", details="steals tokens")
    assert owner.post("/api/admin/gym-reports/g1", json={"action": "hide", "note": "credential stealer"}, headers=H).json()["status"] == "hidden"
    assert _sees(eve, "g1") == NOTHING
    assert _sees(ann, "g1")["open"] and _sees(bob, "g1")["open"]  # owner and editors still see it
    assert ann.get("/api/gyms/g1/settings").json()["hidden"]["note"] == "credential stealer"
    assert owner.get("/api/admin/gym-reports").json()["reports"] == []
    assert board.store.get("gym_reports", f"g1__{UIDS['eve']}")["status"] == "hidden"

    # The owner cannot share it again; asking for public goes to the admins even if review is off.
    for visibility in ("shared", "unlisted"):
        assert ann.put("/api/gyms/g1/settings", json={"visibility": visibility}, headers=H).status_code == 403
    settings = {"allowed_domains": ["example.com", "school.edu"], "members_can_run": True, "public_gyms_need_review": False}
    owner.put("/api/admin/settings", json=settings, headers=H)
    reply = ann.put("/api/gyms/g1/settings", json={"visibility": "public"}, headers=H).json()
    assert reply["visibility"] == "private" and reply["visibility_requested"] == "public"
    assert owner.post("/api/admin/gym-requests/g1", json={"approve": True}, headers=H).json()["visibility"] == "public"
    assert board.store.get("gyms", "g1").get("hidden") is None
    assert ann.put("/api/gyms/g1/settings", json={"visibility": "shared"}, headers=H).status_code == 200  # free again


def test_admin_removes_a_reported_gym(board, people):
    _gym(board, visibility="public")
    eve, owner = people["eve"], people["owner"]
    _report(eve, reason="copyright")
    assert owner.post("/api/admin/gym-reports/g1", json={"action": "remove"}, headers=H).json()["status"] == "removed"
    assert _sees(eve, "g1") == NOTHING and board.store.get("gyms", "g1") is None
    assert board.store.get("gym_tombstones", "g1")["deleted_by"] == "owner@example.com"
    assert owner.post("/api/admin/gym-reports/nope", json={"action": "hide"}, headers=H).status_code == 404
    _report(eve, "algo", reason="broken")  # a machine's local gym: hidden, never deleted
    assert owner.post("/api/admin/gym-reports/algo", json={"action": "remove"}, headers=H).status_code == 400
    assert owner.post("/api/admin/gym-reports/algo", json={"action": "hide"}, headers=H).status_code == 200

def test_showcase_lists_public_gyms_to_signed_out_visitors(board):
    """The landing page's gym list: no sign-in, public gyms only, and nobody's name or email."""
    from gymhub.board import gyms

    gyms._showcase_cache.clear()
    _gym(board, "g1", visibility="public", title="Kernels", description="CUDA practice", problems=5, requires={"gpu": True})
    _gym(board, "g2", visibility="unlisted", problems=3)
    _gym(board, "g3", visibility="private", problems=2)
    _gym(board, "g4", visibility="shared", problems=4)
    reply = client(board).get("/api/showcase")
    assert reply.status_code == 200
    data = reply.json()
    shown = {g["id"]: g for g in data["gyms"]}
    assert "g1" in shown and not {"g2", "g3", "g4"} & set(shown)  # the board's own gyms may be listed too
    assert shown["g1"] == {"id": "g1", "title": "Kernels", "description": "CUDA practice", "problems": 5, "gpu": True, "langs": ["Python"]}
    assert data["total_gyms"] == len(shown) and data["total_exercises"] == sum(g["problems"] for g in shown.values())
    assert "owner" not in str(data) and "@" not in str(data)


def test_auth_config_names_the_source(board):
    assert client(board).get("/api/auth/config").json()["source_url"].startswith("https://")
