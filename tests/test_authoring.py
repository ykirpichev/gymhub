"""Creating gyms without the repo (issue #16): browser drafts, API tokens, the CLI and GitHub pushes."""

import copy
import hashlib
import hmac
import io
import json
import tarfile
import time
from pathlib import Path

import pytest

from gymhub import gymdraft, packages
from gymhub.board import authoring
from tests.test_board_auth import H, board, client, sign_in  # noqa: F401  (fixture)
from tests.test_gym_access import UIDS, people  # noqa: F401  (fixture)
from tests.test_gym_versions import HELLO, _worker, complete_validation

def _validate_next(board, owner_client):
    worker = _worker(board, owner_client)
    job = worker.post("/v1/worker/lease", json={"gyms": []}).json()["job"]
    assert complete_validation(worker, job).status_code == 200
    return job


# ---- the generator ----------------------------------------------------------------------------


def test_draft_round_trips_and_its_grader_really_validates(tmp_path):
    package = gymdraft.write_package(gymdraft.EXAMPLE, tmp_path / "pkg")
    assert packages.validate_upload(package) == []
    again = gymdraft.from_package(package)
    assert again["exercises"][0]["cases"] == gymdraft.EXAMPLE["exercises"][0]["cases"]
    assert again["exercises"][0]["starter"] == gymdraft.EXAMPLE["exercises"][0]["starter"]
    # The learner-facing catalog has the statement and starter, not the tests or the solution.
    text = json.dumps(packages.public_catalog(package))
    assert "10**12" not in text and "1000000000000" not in text and "return a + b" not in text
    from gymhub.runner.validate import validate

    result = validate(package.root)
    assert result["ok"], result["checks"]


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda s: s.update(id="Bad"), "gym id"),
        (lambda s: s["exercises"][0].update(params=["a", "class"]), "Python names"),
        (lambda s: s["exercises"][0]["cases"][0].update(args={"a": 1}), "exactly these inputs"),
        (lambda s: s["exercises"][0].update(solution=""), "needs an expected output"),
        (lambda s: s["exercises"][0].update(starter="x = 1"), "define solve"),
        (lambda s: s.update(exercises=[]), "at least one exercise"),
    ],
)
def test_bad_drafts_are_explained(change, message):
    spec = copy.deepcopy(gymdraft.EXAMPLE)
    change(spec)
    assert any(message in error for error in gymdraft.check(spec))


# ---- drafts on the board ----------------------------------------------------------------------


def test_publish_a_gym_from_the_browser(board, people):
    ann = people["ann"]
    spec = {**copy.deepcopy(gymdraft.EXAMPLE), "id": "anns-gym"}
    assert ann.get("/api/drafts/new").json()["spec"]["id"] == "my-first-gym"
    broken = copy.deepcopy(spec)
    broken["exercises"][0]["params"] = []
    reply = ann.post("/api/drafts/anns-gym/publish", json={"spec": broken}, headers=H)
    assert reply.status_code == 400 and reply.json()["detail"]["errors"]
    reply = ann.post("/api/drafts/anns-gym/publish", json={"spec": spec}, headers=H)
    assert reply.status_code == 200, reply.text
    assert reply.json()["version"]["source"] == "editor"
    _validate_next(board, people["owner"])
    gym = board.store.get("gyms", "anns-gym")
    assert gym["owner_uid"] == UIDS["ann"] and gym["visibility"] == "private"
    assert ann.get("/api/drafts/anns-gym").json()["spec"] == spec
    assert [d["id"] for d in ann.get("/api/my/drafts").json()["drafts"]] == ["anns-gym"]
    # Nobody else reads or overwrites it; an editor may.
    eve = people["eve"]
    assert eve.get("/api/drafts/anns-gym").status_code == 404
    assert eve.put("/api/drafts/anns-gym", json={"spec": spec}, headers=H).status_code == 404
    board.store.set("gyms", "anns-gym", {"editors": [UIDS["bob"]]}, merge=True)
    assert people["bob"].put("/api/drafts/anns-gym", json={"spec": spec}, headers=H).status_code == 200


def test_unpublished_drafts_belong_to_their_author(board, people):
    spec = {**copy.deepcopy(gymdraft.EXAMPLE), "id": "draft-only"}
    assert people["ann"].put("/api/drafts/draft-only", json={"spec": spec}, headers=H).json()["errors"] == []
    assert people["eve"].get("/api/drafts/draft-only").status_code == 404
    reply = people["eve"].put("/api/drafts/draft-only", json={"spec": spec}, headers=H)
    assert reply.status_code == 409 and "taken" in reply.json()["detail"]
    assert people["ann"].put("/api/drafts/other-id", json={"spec": spec}, headers=H).status_code == 400  # id mismatch


def test_a_new_gym_never_overwrites_or_clashes(board, people):
    ann, eve = people["ann"], people["eve"]
    first = ann.get("/api/drafts/new").json()["spec"]
    assert first["id"] == "my-first-gym"
    first["title"] = "Ann's real work"
    assert ann.put("/api/drafts/my-first-gym", json={"spec": first, "create": True}, headers=H).status_code == 200
    # Starting another gym: the example comes with an id nobody uses, for ann and for anyone else.
    assert ann.get("/api/drafts/new").json()["spec"]["id"] == "my-first-gym-2"
    assert eve.get("/api/drafts/new").json()["spec"]["id"] == "my-first-gym-2"
    # Creating with a taken id is refused, never a silent overwrite of ann's earlier draft.
    again = {**copy.deepcopy(gymdraft.EXAMPLE), "id": "my-first-gym"}
    for who in (ann, eve):
        reply = who.put("/api/drafts/my-first-gym", json={"spec": again, "create": True}, headers=H)
        assert reply.status_code == 409 and "taken" in reply.json()["detail"]
        reply = who.post("/api/drafts/my-first-gym/publish", json={"spec": again, "create": True}, headers=H)
        assert reply.status_code == 409
    assert ann.get("/api/drafts/my-first-gym").json()["spec"]["title"] == "Ann's real work"
    # Later saves of ann's own draft are updates.
    assert ann.put("/api/drafts/my-first-gym", json={"spec": first}, headers=H).status_code == 200
    # An id that only has uploaded versions (say a failed first publish) is taken too.
    _gym_version(board, "uploaded-only", "eve")
    assert ann.get("/api/drafts/new").json()["spec"]["id"] == "my-first-gym-2"
    clash = {**copy.deepcopy(gymdraft.EXAMPLE), "id": "uploaded-only"}
    assert ann.put("/api/drafts/uploaded-only", json={"spec": clash, "create": True}, headers=H).status_code == 409


def _gym_version(board, gym, name):
    board.store.set("gym_versions", f"{gym}__v1", {"gym": gym, "version": 1, "status": "failed", "uid": UIDS[name]})


def test_an_upload_cannot_take_an_id_held_by_a_draft(board, people, tmp_path):
    spec = {**copy.deepcopy(gymdraft.EXAMPLE), "id": "held-id"}
    assert people["ann"].put("/api/drafts/held-id", json={"spec": spec, "create": True}, headers=H).status_code == 200
    data = packages.pack(gymdraft.write_package(spec, tmp_path / "pkg").root)
    reply = people["eve"].post("/api/gym-versions", content=data, headers={**H, "Content-Type": "application/gzip"})
    assert reply.status_code == 400 and "taken" in reply.json()["detail"]
    reply = people["ann"].post("/api/gym-versions", content=data, headers={**H, "Content-Type": "application/gzip"})
    assert reply.status_code == 200, reply.text  # the draft's author may (e.g. from the CLI)


def test_a_refused_publish_still_saved_the_draft(board, people):
    ann = people["ann"]
    spec = {**copy.deepcopy(gymdraft.EXAMPLE), "id": "half-done"}
    spec["exercises"][0]["statement"] = ""
    reply = ann.post("/api/drafts/half-done/publish", json={"spec": spec, "create": True}, headers=H)
    assert reply.status_code == 400
    detail = reply.json()["detail"]
    assert detail["saved"] is True and any("statement" in error for error in detail["errors"])
    assert ann.get("/api/drafts/half-done").json()["spec"] == spec
    # Refused before saving (a taken id): not saved, and the reply does not say so.
    reply = people["eve"].post("/api/drafts/half-done/publish", json={"spec": spec, "create": True}, headers=H)
    assert reply.status_code == 409 and "saved" not in str(reply.json()["detail"])


def test_my_drafts_shows_ones_that_are_not_live(board, people):
    ann = people["ann"]
    spec = {**copy.deepcopy(gymdraft.EXAMPLE), "id": "not-live"}
    assert ann.put("/api/drafts/not-live", json={"spec": spec, "create": True}, headers=H).status_code == 200
    [draft] = ann.get("/api/my/drafts").json()["drafts"]
    assert draft["id"] == "not-live" and draft["published"] is False and draft["latest"] is None
    assert ann.post("/api/drafts/not-live/publish", json={"spec": spec}, headers=H).status_code == 200
    worker = _worker(board, people["owner"])
    job = worker.post("/v1/worker/lease", json={"gyms": []}).json()["job"]
    assert complete_validation(worker, job, ok=False).status_code == 200  # its checks failed
    [draft] = ann.get("/api/my/drafts").json()["drafts"]
    assert draft["published"] is False and draft["latest"] == {"version": 1, "status": "failed"}
    assert board.store.get("gyms", "not-live") is None
    assert people["eve"].get("/api/my/drafts").json()["drafts"] == []


def test_a_fork_of_an_editor_gym_can_be_edited_in_the_browser(board, people):
    ann, eve = people["ann"], people["eve"]
    spec = {**copy.deepcopy(gymdraft.EXAMPLE), "id": "anns-editor-gym"}
    assert ann.post("/api/drafts/anns-editor-gym/publish", json={"spec": spec, "create": True}, headers=H).status_code == 200
    _validate_next(board, people["owner"])
    board.store.set("gyms", "anns-editor-gym", {"visibility": "public", "allow_fork": True}, merge=True)
    reply = eve.post("/api/gyms/anns-editor-gym/fork", json={"id": "eves-copy", "title": "Eve's copy"}, headers=H)
    assert reply.status_code == 200, reply.text
    forked = eve.get("/api/drafts/eves-copy").json()["spec"]
    assert forked["id"] == "eves-copy" and forked["title"] == "Eve's copy"
    assert forked["exercises"][0]["cases"] == spec["exercises"][0]["cases"]
    assert forked["exercises"][0]["solution"].strip() == spec["exercises"][0]["solution"].strip()
    assert ann.get("/api/drafts/eves-copy").status_code == 404  # eve's, not ann's


# ---- API tokens -------------------------------------------------------------------------------


def test_api_tokens_publish_and_nothing_else(board, people):
    from fastapi.testclient import TestClient

    ann = people["ann"]
    token = ann.post("/api/my/tokens", json={"name": "laptop"}, headers=H).json()["token"]
    assert [t["name"] for t in ann.get("/api/my/tokens").json()["tokens"]] == ["laptop"]
    cli = TestClient(board.app)
    cli.headers.update({"Authorization": f"Bearer {token}", "X-Gymhub": "1"})
    reply = cli.post("/api/gym-versions", content=packages.pack(HELLO), headers={"Content-Type": "application/gzip"})
    assert reply.status_code == 200, reply.text
    assert cli.get("/api/gym-versions?gym=hello").json()["versions"][0]["created_by"] == "ann@example.com"
    assert cli.get("/api/home").status_code == 401  # a token is not a session
    assert cli.post("/api/my/tokens", json={}).status_code == 401
    bad = TestClient(board.app)
    bad.headers.update({"Authorization": f"Bearer {token[:-2]}xx", "X-Gymhub": "1"})
    assert bad.get("/api/gym-versions").status_code == 401
    token_id = token.split("_")[1]
    assert people["eve"].delete(f"/api/my/tokens/{token_id}", headers=H).status_code == 404
    assert ann.delete(f"/api/my/tokens/{token_id}", headers=H).status_code == 200
    assert cli.get("/api/gym-versions").status_code == 401


# ---- GitHub -----------------------------------------------------------------------------------


def _github_tarball(package_dir: Path, subdir: str, sha: str) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        top = f"anns-gym-{sha[:7]}"
        info = tarfile.TarInfo(f"{top}/README.md")
        info.size = 2
        tar.addfile(info, io.BytesIO(b"hi"))
        link = tarfile.TarInfo(f"{top}/{subdir}/problems/evil")
        link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
        tar.addfile(link)
        for path in sorted(package_dir.rglob("*")):
            if path.is_file():
                data = path.read_bytes()
                info = tarfile.TarInfo(f"{top}/{subdir}/{path.relative_to(package_dir)}")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def _signed(c, gym, secret, event, payload):
    body = json.dumps(payload).encode()
    signature = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return c.post(f"/v1/github/{gym}", content=body, headers={"X-GitHub-Event": event, "X-Hub-Signature-256": signature,
                                                             "Content-Type": "application/json"})


def test_github_push_publishes_a_version(board, people, tmp_path, monkeypatch):
    ann = people["ann"]
    spec = {**copy.deepcopy(gymdraft.EXAMPLE), "id": "anns-gym"}
    ann.post("/api/drafts/anns-gym/publish", json={"spec": spec}, headers=H)
    _validate_next(board, people["owner"])
    assert people["eve"].put("/api/gyms/anns-gym/github", json={"repo": "ann/gyms"}, headers=H).status_code == 404
    assert ann.put("/api/gyms/anns-gym/github", json={"repo": "not a repo"}, headers=H).status_code == 400
    source = ann.put("/api/gyms/anns-gym/github", json={"repo": "https://github.com/ann/gyms.git", "path": "gyms/mine",
                                                        "ref": "refs/tags/v*"}, headers=H).json()["source"]
    assert source["repo"] == "ann/gyms" and source["webhook_url"].endswith("/v1/github/anns-gym")

    spec["exercises"][0]["cases"].append({"name": "zeros", "args": {"a": 0, "b": 0}, "expected": 0})
    package_dir = gymdraft.write_package(spec, tmp_path / "pkg").root
    sha = "a" * 40
    fetched = []
    monkeypatch.setattr(authoring, "_download", lambda url: fetched.append(url) or _github_tarball(package_dir, "gyms/mine", sha))
    c = client(board)
    secret = source["secret"]
    assert _signed(c, "anns-gym", "wrong", "push", {"ref": "refs/tags/v1", "after": sha}).status_code == 401
    assert _signed(c, "anns-gym", secret, "ping", {}).json()["pong"]
    assert _signed(c, "anns-gym", secret, "push", {"ref": "refs/heads/main", "after": sha}).json()["ignored"] == "refs/heads/main"
    reply = _signed(c, "anns-gym", secret, "push", {"ref": "refs/tags/v1", "after": sha})
    assert reply.status_code == 200, reply.text
    assert reply.json()["version"] == 2 and fetched == [f"https://codeload.github.com/ann/gyms/tar.gz/{sha}"]
    version = board.store.get("gym_versions", "anns-gym__v0002")
    assert version["commit"]["sha"] == sha and version["created_by"] == "ann@example.com"
    assert not (tmp_path / "evil").exists()

    # The repository's gym.json must name the connected gym.
    other = gymdraft.write_package({**spec, "id": "someone-else"}, tmp_path / "other").root
    monkeypatch.setattr(authoring, "_download", lambda url: _github_tarball(other, "gyms/mine", "b" * 40))
    assert _signed(c, "anns-gym", secret, "push", {"ref": "refs/tags/v2", "after": "b" * 40}).status_code == 400
    assert "someone-else" in ann.get("/api/gyms/anns-gym/github").json()["source"]["last"]["error"]


def test_extract_subtree_refuses_escapes(tmp_path):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        info = tarfile.TarInfo("top/../../evil.txt")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
        info = tarfile.TarInfo("top/gym.json")
        info.size = 2
        tar.addfile(info, io.BytesIO(b"{}"))
    root = packages.extract_subtree(buffer.getvalue(), tmp_path / "out")
    assert (root / "gym.json").is_file() and not (tmp_path / "evil.txt").exists()
    with pytest.raises(packages.BundleError, match="inside the repository"):
        packages.extract_subtree(buffer.getvalue(), tmp_path / "x", "../up")


# ---- the CLI ----------------------------------------------------------------------------------


def test_cli_validate(capfd, tmp_path):
    from gymhub import cli

    assert cli.main(["gym", "validate", str(HELLO), "--quick"]) == 0
    broken = gymdraft.write_package(gymdraft.EXAMPLE, tmp_path / "pkg").root
    (broken / "problems/add_two/starter/solution.py").write_text("# GYM_STARTER\ndef solve(a, b):\n    return a + b\n")
    assert cli.main(["gym", "validate", str(broken)]) == 1
    assert "FAIL add_two [python] starter fails" in capfd.readouterr().out
