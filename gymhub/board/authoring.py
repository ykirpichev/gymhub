"""Creating gyms without the repo: the browser editor, API tokens for the CLI, and GitHub pushes.

    GET  /api/drafts/new                 the example to start from, with an id nobody uses yet
    GET  /api/drafts/{gym}               a gym's editor draft (owner and editors; your own for a new gym)
    PUT  /api/drafts/{gym}               save a draft; {"spec", "create": true} the first time (409 if the id is taken)
    POST /api/drafts/{gym}/publish       save, build the package, upload it as a new version (takes "create" too)
    GET  /api/my/drafts                  your drafts, with their latest version's status

    GET/POST /api/my/tokens, DELETE /api/my/tokens/{id}     personal API tokens (gymhub.cli gym push)

    GET/PUT/DELETE /api/gyms/{gym}/github                    owner: connect a public GitHub repo
    POST /v1/github/{gym}                                    GitHub's push webhook (HMAC-signed)
"""

import copy
import fnmatch
import hashlib
import hmac
import json
import secrets
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from gymhub import gymdraft, gymversions, packages
from gymhub.board import auth
from gymhub.board.auth import User, current_user

router = APIRouter()
GITHUB_REPO = __import__("re").compile(r"^[A-Za-z0-9_.-]{1,100}/[A-Za-z0-9_.-]{1,100}$")


def _board():
    from gymhub.board import app

    return app


def _upload(user: User, data: bytes, extra: dict | None = None) -> dict:
    board = _board()
    if not auth.can_run(user):
        raise HTTPException(403, "Creating gyms runs their code on a machine, which is not enabled for members yet.")
    try:
        return board._start_upload(user, data, extra)
    except gymversions.UploadRejected as exc:
        raise HTTPException(400, {"message": str(exc), "errors": exc.errors}) from exc


# ---- drafts --------------------------------------------------------------------------------------


def id_in_use(store, gym: str) -> bool:
    """A gym, a draft or an uploaded version (maybe still validating, maybe failed) already uses this id."""
    return (store.get("gyms", gym) is not None or store.get("gym_drafts", gym) is not None
            or bool(gymversions.versions(store, gym)))


def _taken(gym: str) -> HTTPException:
    return HTTPException(409, f"The id {gym!r} is taken; choose another")


def _draft_access(user: User, gym: str, write: bool = False, create: bool = False) -> dict | None:
    """The existing draft doc (or None). Raises unless this user may edit the gym's draft.

    Someone else's unpublished draft is hidden (404) from reads; a write to its id is told the id is
    taken (409). `create`: the author is starting a new gym, so any existing use of the id is a clash,
    never a silent overwrite of an earlier draft."""
    board = _board()
    store = board.store
    if create and id_in_use(store, gym):
        raise _taken(gym)
    if store.get("gyms", gym) is not None:
        board.require_gym(user, gym, "editor")
    draft = store.get("gym_drafts", gym)
    if draft and store.get("gyms", gym) is None and draft.get("owner_uid") != user.uid and not user.is_admin:
        raise _taken(gym) if write else HTTPException(404, "No such draft")
    return draft


def _free_id(store, base: str) -> str:
    for number in range(1, 100):
        gym = base if number == 1 else f"{base}-{number}"
        if not id_in_use(store, gym):
            return gym
    return f"{base}-{secrets.token_hex(3)}"


@router.get("/api/drafts/{gym}")
def get_draft(gym: str, user: User = Depends(current_user)):
    if gym == "new":
        spec = copy.deepcopy(gymdraft.EXAMPLE)
        return {"spec": {**spec, "id": _free_id(_board().store, spec["id"])}, "new": True}
    draft = _draft_access(user, gym)
    if draft is None:
        raise HTTPException(404, "This gym was not made in the browser editor")
    return {"spec": draft["spec"], "updated_at": draft.get("updated_at")}


class Draft(BaseModel):
    spec: dict
    create: bool = False  # the first save of a new gym: refuse an id that is already used


def _save(user: User, gym: str, spec: dict, create: bool = False) -> dict:
    if spec.get("id") != gym:
        raise HTTPException(400, "The draft's id does not match")
    if len(json.dumps(spec)) > 900_000:
        raise HTTPException(413, "The draft is too large for the editor; pack the gym and upload it instead")
    existing = _draft_access(user, gym, write=True, create=create)
    store = _board().store
    store.set("gym_drafts", gym, {
        "spec": spec, "updated_at": time.time(), "updated_by": user.email,
        "owner_uid": (existing or {}).get("owner_uid") or user.uid,
    })
    return {"ok": True, "errors": gymdraft.check(spec)}


@router.put("/api/drafts/{gym}")
def save_draft(gym: str, body: Draft, user: User = Depends(current_user)):
    if not packages.GYM_ID.match(gym):
        raise HTTPException(400, "gym id: 3-40 characters, lowercase letters, digits and -")
    return _save(user, gym, body.spec, body.create)


@router.post("/api/drafts/{gym}/publish")
def publish_draft(gym: str, body: Draft, user: User = Depends(current_user)):
    if not packages.GYM_ID.match(gym):
        raise HTTPException(400, "gym id: 3-40 characters, lowercase letters, digits and -")
    _save(user, gym, body.spec, body.create)
    # From here on the draft is saved: say so in any refusal, so the editor treats the id as taken by it.
    try:
        with tempfile.TemporaryDirectory(prefix="gym-draft-") as tmp:
            try:
                package = gymdraft.write_package(body.spec, Path(tmp) / "pkg")
            except gymdraft.DraftError as exc:
                raise HTTPException(400, {"message": f"{len(exc.errors)} problem(s) in the draft", "errors": exc.errors}) from exc
            data = packages.pack(package.root)
        return _upload(user, data, {"source": "editor"})
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
        raise HTTPException(exc.status_code, {**detail, "saved": True}) from exc


@router.get("/api/my/drafts")
def my_drafts(user: User = Depends(current_user)):
    store = _board().store
    drafts = []
    for gym, doc in store.where("gym_drafts", "owner_uid", user.uid):
        latest = next(iter(gymversions.versions(store, gym)), None)
        drafts.append({"id": gym, "title": doc["spec"].get("title", gym), "updated_at": doc.get("updated_at"),
                       "published": store.get("gyms", gym) is not None,
                       "latest": {"version": latest["version"], "status": latest["status"]} if latest else None})
    return {"drafts": sorted(drafts, key=lambda item: item.get("updated_at") or 0, reverse=True)}


# ---- personal API tokens -------------------------------------------------------------------------


class NewToken(BaseModel):
    name: str = Field(default="cli", max_length=60)


@router.post("/api/my/tokens")
def create_token(body: NewToken, user: User = Depends(current_user)):
    store = _board().store
    if len([1 for _id, doc in store.where("api_tokens", "uid", user.uid) if not doc.get("revoked")]) >= 10:
        raise HTTPException(400, "You have 10 tokens already; revoke one first")
    token_id, secret = secrets.token_hex(6), secrets.token_urlsafe(24)
    store.set("api_tokens", token_id, {"uid": user.uid, "name": body.name.strip() or "cli", "hash": auth.token_hash(secret),
                                       "created_at": time.time()})
    return {"id": token_id, "token": f"gp_{token_id}_{secret}"}


@router.get("/api/my/tokens")
def list_tokens(user: User = Depends(current_user)):
    store = _board().store
    tokens = [{"id": token_id, "name": doc.get("name"), "created_at": doc.get("created_at")}
              for token_id, doc in store.where("api_tokens", "uid", user.uid) if not doc.get("revoked")]
    return {"tokens": sorted(tokens, key=lambda item: item["created_at"] or 0)}


@router.delete("/api/my/tokens/{token_id}")
def revoke_token(token_id: str, user: User = Depends(current_user)):
    store = _board().store
    doc = store.get("api_tokens", token_id)
    if not doc or doc.get("uid") != user.uid:
        raise HTTPException(404, "No such token")
    store.set("api_tokens", token_id, {"revoked": True}, merge=True)
    return {"ok": True}


# ---- GitHub --------------------------------------------------------------------------------------
# gym_sources/{gym}   kind "github", repo "owner/name", path, ref (glob: refs/heads/main, refs/tags/v*),
#                     secret, owner_uid (whose name versions are published under), last {sha, at, result}


class GitHubSource(BaseModel):
    repo: str = Field(max_length=200)
    path: str = Field(default="", max_length=200)
    ref: str = Field(default="refs/heads/main", max_length=200)


def _source_view(request: Request, gym: str, doc: dict) -> dict:
    base = str(request.base_url).rstrip("/")
    return {"repo": doc["repo"], "path": doc.get("path", ""), "ref": doc.get("ref"), "secret": doc["secret"],
            "webhook_url": f"{base}/v1/github/{gym}", "last": doc.get("last")}


@router.get("/api/gyms/{gym}/github")
def get_github(gym: str, request: Request, user: User = Depends(current_user)):
    board = _board()
    board.require_gym(user, gym, "owner")
    doc = board.store.get("gym_sources", gym)
    return {"source": _source_view(request, gym, doc) if doc else None}


@router.put("/api/gyms/{gym}/github")
def set_github(gym: str, body: GitHubSource, request: Request, user: User = Depends(current_user)):
    board = _board()
    gym_doc = board.require_gym(user, gym, "owner")
    if gym_doc.get("source") != "bundle":
        raise HTTPException(400, "Only uploaded gyms can publish from GitHub")
    repo = body.repo.strip().removeprefix("https://github.com/").removesuffix(".git").strip("/")
    if not GITHUB_REPO.match(repo):
        raise HTTPException(400, "repo: owner/name, like octocat/my-gym")
    if ".." in Path(body.path).parts:
        raise HTTPException(400, "path must stay inside the repository")
    existing = board.store.get("gym_sources", gym) or {}
    doc = {"kind": "github", "repo": repo, "path": body.path.strip("/"), "ref": body.ref.strip() or "refs/heads/main",
           "secret": existing.get("secret") or secrets.token_urlsafe(24), "owner_uid": user.uid, "updated_at": time.time()}
    board.store.set("gym_sources", gym, doc)
    return {"source": _source_view(request, gym, doc)}


@router.delete("/api/gyms/{gym}/github")
def delete_github(gym: str, user: User = Depends(current_user)):
    board = _board()
    board.require_gym(user, gym, "owner")
    board.store.delete("gym_sources", gym)
    return {"ok": True}


def _download(url: str) -> bytes:
    import requests

    with requests.get(url, stream=True, timeout=30) as reply:
        if reply.status_code != 200:
            raise HTTPException(502, f"GitHub answered HTTP {reply.status_code} for the archive (is the repository public?)")
        chunks, total = [], 0
        for chunk in reply.iter_content(1 << 16):
            total += len(chunk)
            if total > packages.MAX_BUNDLE_BYTES:
                raise HTTPException(413, "The repository archive is over 20 MB")
            chunks.append(chunk)
    return b"".join(chunks)


@router.post("/v1/github/{gym}")
async def github_webhook(gym: str, request: Request):
    import asyncio

    store = _board().store
    source = store.get("gym_sources", gym)
    body = await request.body()
    if not source:
        raise HTTPException(404, "No GitHub connection for this gym")
    expected = "sha256=" + hmac.new(source["secret"].encode(), body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, request.headers.get("x-hub-signature-256", "")):
        raise HTTPException(401, "Bad signature")
    event = request.headers.get("x-github-event", "")
    if event == "ping":
        return {"ok": True, "pong": True}
    if event != "push":
        return {"ok": True, "ignored": event}
    payload = json.loads(body)
    ref, sha = payload.get("ref", ""), payload.get("after", "")
    if payload.get("deleted") or not fnmatch.fnmatch(ref, source.get("ref", "refs/heads/main")):
        return {"ok": True, "ignored": ref}
    if not __import__("re").fullmatch(r"[0-9a-f]{40}", sha):
        raise HTTPException(400, "No commit in the push")
    return await asyncio.to_thread(_publish_commit, gym, source, ref, sha)


def _publish_commit(gym: str, source: dict, ref: str, sha: str) -> dict:
    store = _board().store
    owner = auth.load_user(source["owner_uid"])
    if not owner or owner.get("status") != "active":
        raise HTTPException(403, "The gym owner who connected GitHub is no longer active")
    user = auth._user_from(source["owner_uid"], owner)
    data = _download(f"https://codeload.github.com/{source['repo']}/tar.gz/{sha}")
    with tempfile.TemporaryDirectory(prefix="gym-github-") as tmp:
        try:
            root = packages.extract_subtree(data, Path(tmp) / "pkg", source.get("path", ""))
            package = packages.load(root)
        except packages.BundleError as exc:
            result = {"error": str(exc)}
            store.set("gym_sources", gym, {"last": {"sha": sha, "ref": ref, "at": time.time(), **result}}, merge=True)
            raise HTTPException(400, str(exc)) from exc
        if package.id != gym:
            message = f"gym.json in the repository has id {package.id!r}, but it is connected to {gym!r}"
            store.set("gym_sources", gym, {"last": {"sha": sha, "ref": ref, "at": time.time(), "error": message}}, merge=True)
            raise HTTPException(400, message)
        bundle = packages.pack(root)
    try:
        reply = _upload(user, bundle, {"source": "github", "commit": {"repo": source["repo"], "sha": sha, "ref": ref}})
    except HTTPException as exc:
        store.set("gym_sources", gym, {"last": {"sha": sha, "ref": ref, "at": time.time(), "error": str(exc.detail)}}, merge=True)
        raise
    version = reply["version"]
    store.set("gym_sources", gym, {"last": {"sha": sha, "ref": ref, "at": time.time(), "version": version["version"]}}, merge=True)
    return {"ok": True, "version": version["version"], "status": version["status"]}
