"""Sharing a machine: with chosen people or everyone; someone else's code only where the sandbox works."""

from tests.test_board_auth import H, board, client, sign_in  # noqa: F401  (fixture)

JOB = {"gym": "algo", "slug": "p", "lang": "python", "action": "run", "code": "x"}


def _setup(board, sandbox=True):
    owner = client(board)
    sign_in(owner, "owner")
    owner.put("/api/admin/settings", json={"allowed_domains": ["example.com", "school.edu"], "members_can_run": True}, headers=H)
    ann, bob = client(board), client(board)
    sign_in(ann, "ann")
    sign_in(bob, "bob")
    code = ann.post("/api/my/workers/registration", json={"name": "anns"}, headers=H).json()["code"]
    token = client(board).post("/v1/register", json={"code": code}).json()["token"]
    machine = client(board)
    machine.headers.update({"Authorization": f"Bearer {token}"})
    machine.post("/v1/worker/heartbeat", json={"gyms": ["algo"], "capabilities": {"gpus": [], "tools": [], "sandbox": sandbox}})
    return owner, ann, bob, machine, token.split("_")[1]


def _lease(machine):
    return machine.post("/v1/worker/lease", json={"gyms": ["algo"]}).json()["job"]


def test_not_shared_by_default(board):
    _owner, _ann, bob, machine, _id = _setup(board)
    bob.post("/api/jobs", json=JOB, headers=H)
    assert _lease(machine) is None


def test_share_with_one_person(board):
    _owner, ann, bob, machine, worker_id = _setup(board)
    assert ann.post(f"/api/my/workers/{worker_id}/sharing", json={"emails": ["nobody@x.org"]}, headers=H).status_code == 400
    assert bob.post(f"/api/my/workers/{worker_id}/sharing", json={"emails": ["bob@school.edu"]}, headers=H).status_code == 404
    assert ann.post(f"/api/my/workers/{worker_id}/sharing", json={"emails": ["Bob@School.edu"]}, headers=H).status_code == 200
    job = bob.post("/api/jobs", json=JOB, headers=H).json()["id"]
    assert _lease(machine)["id"] == job
    listed = ann.get("/api/my/workers").json()["workers"][0]
    assert listed["shared_emails"] == ["bob@school.edu"] and listed["jobs_for_others"][0]["who"] == "bob@school.edu"


def test_share_with_everyone(board):
    _owner, ann, bob, machine, worker_id = _setup(board)
    ann.post(f"/api/my/workers/{worker_id}/sharing", json={"everyone": True}, headers=H)
    job = bob.post("/api/jobs", json=JOB, headers=H).json()["id"]
    assert _lease(machine)["id"] == job


def test_never_unsandboxed_for_others(board):
    _owner, ann, bob, machine, worker_id = _setup(board, sandbox=False)
    ann.post(f"/api/my/workers/{worker_id}/sharing", json={"everyone": True}, headers=H)
    bob.post("/api/jobs", json=JOB, headers=H)
    assert _lease(machine) is None  # shared, but no sandbox: Bob's code does not run here
    own = ann.post("/api/jobs", json=JOB, headers=H).json()["id"]
    assert _lease(machine)["id"] == own  # the owner's own code runs without one


def test_members_see_only_machines_they_can_use(board):
    _owner, ann, bob, machine, worker_id = _setup(board)
    names = lambda client_: [r["name"] for r in client_.get("/api/overview").json()["runners"]]  # noqa: E731
    assert "anns" in names(ann) and "anns" not in names(bob)
    ann.post(f"/api/my/workers/{worker_id}/sharing", json={"emails": ["bob@school.edu"]}, headers=H)
    assert "anns" in names(bob)


def test_a_non_public_gym_reaches_only_machines_whose_owner_may_see_it(board):
    """#117: a shared machine gets an uploaded gym's bundle only if the gym is public or its owner has a role on it."""
    import time

    _owner, ann, bob, machine, worker_id = _setup(board)
    machine.post("/v1/worker/heartbeat", json={"gyms": ["algo"], "capabilities": {"gpus": [], "tools": [], "sandbox": True, "bundles": True}})
    ann.post(f"/api/my/workers/{worker_id}/sharing", json={"everyone": True}, headers=H)
    bob_uid = bob.get("/api/me").json()["user"]["uid"]
    ann_uid = ann.get("/api/me").json()["user"]["uid"]
    board.store.set("gyms", "bobs", {"title": "Bob's", "source": "bundle", "owner_uid": bob_uid, "visibility": "private",
                                     "updated_at": time.time(), "bundle": {"sha": "0" * 64, "key": "k", "version": 1}, "version": 1})
    board.store.set("problems", "bobs__p", {"gym": "bobs", "slug": "p", "title": "P", "langs": ["python"], "order": 0})
    job = bob.post("/api/jobs", json={**JOB, "gym": "bobs"}, headers=H).json()["id"]
    assert _lease(machine) is None  # private: Ann's machine never sees Bob's graders

    board.store.set("gyms", "bobs", {"visibility": "shared", "learners": [ann_uid]}, merge=True)
    assert _lease(machine)["id"] == job  # Ann may see the gym, so her machine may grade it

    board.store.set("gyms", "bobs", {"visibility": "public", "learners": []}, merge=True)
    again = bob.post("/api/jobs", json={**JOB, "gym": "bobs"}, headers=H).json()["id"]
    assert _lease(machine)["id"] == again  # public gyms go anywhere
