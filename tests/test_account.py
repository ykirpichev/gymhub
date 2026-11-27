"""A member's own data: the privacy and terms pages, data export, and self-service account deletion."""

import json
import time

from gymhub.blobs import bundle_key, open_blobs
from tests.test_board_auth import H, board, client, sign_in  # noqa: F401  (fixture)
from tests.test_gym_access import UIDS, _gym, people  # noqa: F401  (fixture)

ANN = UIDS["ann"]


def _ann_has_data(board):
    """Ann's footprint across the board: her own gym, a role on someone else's, work, a machine and a token."""
    store = board.store
    _gym(board, "mine", visibility="public", learners=[UIDS["bob"]])  # owned by ann
    _gym(board, "bobs", owner_uid=UIDS["bob"], visibility="shared", editors=[ANN], learners=[ANN, UIDS["eve"]])
    store.set("gym_versions", "mine__v0001", {"gym": "mine", "version": 1, "sha": "a" * 64, "uid": ANN, "created_by": "ann@example.com"})
    store.set("gym_versions", "bobs__v0002", {"gym": "bobs", "version": 2, "sha": "b" * 64, "uid": ANN, "created_by": "ann@example.com"})
    blobs = open_blobs()
    blobs.put(bundle_key("mine", "a" * 64), b"bundle")
    blobs.put(bundle_key("bobs", "b" * 64), b"other bundle")
    store.set("gym_invites", "ann@example.com__other", {"email": "ann@example.com", "gym": "other", "role": "learner"})
    store.set("user_progress", f"{ANN}__bobs", {"uid": ANN, "gym": "bobs", "data": {"exercises": {"p": {"hints_seen": 1}}}})
    store.set("user_progress", f"{UIDS['eve']}__bobs", {"uid": UIDS["eve"], "gym": "bobs", "data": {}})
    store.set("workspaces", f"{ANN}__bobs__p__python", {"uid": ANN, "gym": "bobs", "slug": "p", "lang": "python", "code": "print(1)"})
    store.set("jobs", "j1", {"uid": ANN, "gym": "bobs", "slug": "p", "action": "test", "status": "done", "code": "print(1)",
                             "created_by": "ann@example.com", "created_at": time.time()})
    store.set("jobs", "j2", {"uid": UIDS["eve"], "gym": "bobs", "slug": "p", "action": "test", "status": "done"})
    store.set("solves", f"{ANN}__bobs__p__python", {"uid": ANN, "gym": "bobs"})
    store.set("bests", f"{ANN}__bobs__p__python", {"uid": ANN, "gym": "bobs", "by_hardware": {}})
    store.set("reviews", f"{ANN}__bobs__p", {"uid": ANN, "gym": "bobs", "slug": "p", "stage": 0})
    store.set("leaderboard", f"bobs__p__python__cpu__{ANN}", {"board": "bobs__p", "uid": ANN, "name": "Ann", "ms": 1.0})
    store.set("api_tokens", "t1", {"uid": ANN, "name": "laptop", "hash": "secret-hash", "created_at": time.time()})
    store.set("tutor_usage", f"{ANN}__2026-11", {"uid": ANN, "month": "2026-11", "usd": 0.1})
    store.set("usage", ANN, {"hour": "x", "jobs_this_hour": 1})
    store.set("workers", "w1", {"name": "ann-box", "owner_uid": ANN, "owner_email": "ann@example.com", "token_hash": "h"})
    store.set("runners", "w1", {"worker_id": "w1", "owner_email": "ann@example.com", "heartbeat_at": time.time()})
    store.set("gym_drafts", "draft1", {"owner_uid": ANN, "spec": {}})
    store.set("gym_reports", f"bobs__{ANN}", {"gym": "bobs", "uid": ANN, "email": "ann@example.com", "reason": "spam"})


def test_privacy_and_terms_are_public_and_name_the_operator(board, monkeypatch):
    c = client(board)  # not signed in
    for page, heading in (("privacy", "Privacy policy"), ("terms", "Terms of use")):
        reply = c.get(f"/{page}")
        assert reply.status_code == 200 and heading in reply.text and "Draft." in reply.text
        assert "{{" not in reply.text
    monkeypatch.setenv("GYMHUB_OPERATOR", "Ada <Lovelace>")
    monkeypatch.setenv("GYMHUB_CONTACT_EMAIL", "privacy@example.org")
    text = c.get("/privacy").text
    assert "Ada &lt;Lovelace&gt;" in text and 'href="mailto:privacy@example.org"' in text
    page = c.get("/").text  # the board itself links them from every page
    assert 'href="/privacy"' in page and 'href="/terms"' in page


def test_export_has_everything_of_mine_and_nothing_of_others_or_secrets(board, people):
    _ann_has_data(board)
    reply = people["ann"].get("/api/me/export")
    assert reply.status_code == 200 and "attachment" in reply.headers["content-disposition"]
    data = reply.json()
    assert data["account"]["email"] == "ann@example.com"
    assert set(data["jobs"]) == {"j1"} and data["workspaces"][f"{ANN}__bobs__p__python"]["code"] == "print(1)"
    assert set(data["user_progress"]) == {f"{ANN}__bobs"}
    assert set(data["gyms_owned"]) == {"mine"} and data["gym_roles"] == {"bobs": ["editor", "learner"]}
    assert data["api_tokens"]["t1"]["name"] == "laptop" and "hash" not in data["api_tokens"]["t1"]
    assert "token_hash" not in data["machines"]["w1"]
    assert set(data["gym_invites"]) == {"ann@example.com__other"}
    assert set(data["gym_reports"]) == {f"bobs__{ANN}"}
    assert "secret-hash" not in json.dumps(data)


def test_deleting_my_account_removes_my_data_and_my_gyms_but_not_others(board, people):
    _ann_has_data(board)
    store = board.store
    ann = people["ann"]
    preview = ann.get("/api/me/deletion").json()
    assert [gym["id"] for gym in preview["gyms"]] == ["mine"] and preview["gyms"][0]["learners"] == 1
    assert [m["name"] for m in preview["machines"]] == ["ann-box"] and preview["blocked"] is None

    wrong = ann.request("DELETE", "/api/me", json={"confirm": "bob@school.edu"}, headers=H)
    assert wrong.status_code == 400 and store.get("users", ANN)
    assert ann.request("DELETE", "/api/me", json={"confirm": "ANN@example.com "}, headers=H).status_code == 200

    assert store.get("users", ANN) is None
    assert ann.get("/api/me").status_code == 401  # the session ended with the account
    for collection in ("user_progress", "workspaces", "jobs", "solves", "bests", "reviews", "leaderboard", "api_tokens", "tutor_usage",
                       "gym_reports"):
        assert store.where(collection, "uid", ANN) == [], collection
    assert store.get("usage", ANN) is None
    assert store.get("workers", "w1") is None and store.get("runners", "w1") is None
    assert store.get("gym_drafts", "draft1") is None
    assert store.where("gym_invites", "email", "ann@example.com") == []
    # Her gym is gone for everyone, its id kept for admins; its bundle is deleted.
    assert store.get("gyms", "mine") is None and store.get("problems", "mine__p") is None
    assert store.get("gym_tombstones", "mine")["owner_uid"] is None
    assert store.get("gym_versions", "mine__v0001") is None
    blobs = open_blobs()
    assert not any((blobs.root / "bundles" / "mine").glob("*"))
    # Bob's gym stays, without her on it; the version she uploaded stays, without her email.
    bobs = store.get("gyms", "bobs")
    assert ANN not in bobs["editors"] and bobs["learners"] == [UIDS["eve"]]
    assert store.get("gym_versions", "bobs__v0002")["created_by"] == "deleted account"
    assert blobs.get(bundle_key("bobs", "b" * 64)) == b"other bundle"
    # Other people's data is untouched.
    assert store.get("jobs", "j2") and store.get("user_progress", f"{UIDS['eve']}__bobs")
    assert store.get("users", UIDS["bob"])
    assert "ann@example.com" not in json.dumps([store.get("gyms", "bobs"), store.get("gym_versions", "bobs__v0002")])


def test_a_deleted_member_needs_a_new_invite(board, people):
    board.store.set("settings", "auth", {"allowed_domains": [], "members_can_run": True})
    ann = people["ann"]
    assert ann.request("DELETE", "/api/me", json={"confirm": "ann@example.com"}, headers=H).status_code == 200
    assert sign_in(client(board), "ann").status_code == 403


def test_admins_on_the_admin_list_cannot_delete_themselves(board, people):
    owner = people["owner"]
    assert owner.get("/api/me/deletion").json()["blocked"]
    reply = owner.request("DELETE", "/api/me", json={"confirm": "owner@example.com"}, headers=H)
    assert reply.status_code == 400 and board.store.get("users", UIDS["owner"])


def test_deletion_needs_the_same_origin_header(board, people):
    reply = people["ann"].request("DELETE", "/api/me", json={"confirm": "ann@example.com"})
    assert reply.status_code == 403 and board.store.get("users", ANN)
