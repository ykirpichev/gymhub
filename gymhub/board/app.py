"""Control board: lists gyms and exercises, takes submissions, shows results.

It never runs user code. It writes job documents; the runner on your GPU machine picks them up.

    GYMHUB_STORE=sqlite:.cache/gymhub/dev.db GYMHUB_AUTH=dev uvicorn gymhub.board.app:app --port 8080
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from gymhub import accounts, gymaccess, gymmd, gymversions, health, jobqueue, packages, quotas, reviews, routing, tutor
from gymhub.blobs import open_blobs
from gymhub.board import auth, authoring, gyms, legal, poolmachines, runtargets, workers
from gymhub.board.auth import User, current_user, mode, require_admin
from gymhub.store import expires_in, new_id, now, open_store, problem_id

STATIC = Path(__file__).parent / "static"
RUNNER_ONLINE_SECONDS = 90
ACTIONS = {"run", "test", "bench", "save", "reset", "hint"}
LIST_FIELDS = ("slug", "title", "difficulty", "summary", "track", "order", "warmup", "langs", "benchable", "tests_hash", "kind", "minutes")

app = FastAPI(title="Human Gym", docs_url=None, redoc_url=None, openapi_url=None)
store = open_store()
health.setup_logging()
app.include_router(workers.router)
app.include_router(gyms.router)
app.include_router(authoring.router)
app.include_router(poolmachines.router)
app.include_router(legal.router)


WORKER_PATHS = ("/v1/", "/install.sh")


@app.middleware("http")
async def worker_surface(request: Request, call_next):
    """GYMHUB_SURFACE=workers: serve only the worker API and the installer.

    Behind IAP the board is reachable only with a Google login, which a machine should not hold. The
    same image is deployed a second time without IAP and with this surface, so machines reach the
    board with their worker token alone (deploy/gcp.sh)."""
    if os.environ.get("GYMHUB_SURFACE") == "workers" and not request.url.path.startswith(WORKER_PATHS):
        return JSONResponse({"detail": "Not found"}, status_code=404)
    return await call_next(request)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    elif request.url.path.startswith("/static/"):
        # Always revalidate; the versioned URLs from index.html change on every deploy anyway.
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.exception_handler(HTTPException)
async def http_error(_request: Request, exc: HTTPException):
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


def _runners() -> list[dict]:
    runners = []
    for _doc_id, doc in store.recent("runners", "heartbeat_at", 20):
        doc.pop("online_slots", None)  # Admin health only
        doc["online"] = time.time() - doc.get("heartbeat_at", 0) < RUNNER_ONLINE_SECONDS
        runners.append(doc)
    return runners


def _job_summary(doc_id: str, doc: dict) -> dict:
    keys = ("gym", "slug", "lang", "action", "status", "ok", "created_at", "started_at", "finished_at", "runner", "hardware",
            "host", "via")
    summary = {key: doc.get(key) for key in keys}
    summary["id"] = doc_id
    summary["has_code"] = doc.get("code") is not None
    output = doc.get("output") or ""
    summary["headline"] = next((line for line in output.splitlines() if line.strip()), "")[:200]
    return summary


WRITES_PER_MINUTE = int(os.environ.get("GYMHUB_WRITES_PER_MINUTE", "120"))


def _client_key(request: Request) -> str:
    """Who a write counts against: the signed-in user, else the address the platform appended.

    Cloud Run appends the caller's real address to X-Forwarded-For, so the last entry is the one a
    caller cannot choose; the entries before it are whatever the caller sent.
    """
    token = request.cookies.get(auth.SESSION_COOKIE)
    uid = auth.read_session(token) if token else None
    if uid:
        return f"user:{uid}"
    hops = [hop.strip() for hop in (request.headers.get("x-forwarded-for") or "").split(",") if hop.strip()]
    return "ip:" + (hops[-1] if hops else (request.client.host if request.client else "?"))


def _count_write(key: str) -> bool:
    """Count one write in the shared store (every instance sees it). False when over the limit."""
    window = int(time.time() // 60)
    name = hashlib.sha256(key.encode()).hexdigest()[:32]

    def apply(doc: dict | None) -> dict:
        count = doc["count"] + 1 if doc and doc.get("window") == window else 1
        # One document per signed-in user or address; a day after its last write it is gone.
        return {"window": window, "count": count, "expire_at": expires_in(86400)}

    return store.transact("throttle", name, apply)["count"] <= WRITES_PER_MINUTE


@app.middleware("http")
async def throttle_writes(request: Request, call_next):
    """A flood guard on API writes, per signed-in user or caller address, shared by all instances."""
    if request.method in {"POST", "PUT", "PATCH", "DELETE"} and request.url.path.startswith("/api/"):
        if not await run_in_threadpool(_count_write, _client_key(request)):
            return JSONResponse({"detail": "Too many requests. Slow down and try again in a minute."}, status_code=429)
    return await call_next(request)


@app.middleware("http")
async def same_origin_writes(request: Request, call_next):
    """Cookie sessions: refuse state-changing API calls that come from another site."""
    if request.method in {"POST", "PUT", "PATCH", "DELETE"} and request.url.path.startswith("/api/"):
        if request.headers.get("x-gymhub") != "1":
            return JSONResponse({"detail": "Missing X-Gymhub header"}, status_code=403)
    return await call_next(request)


# ---- sign-in -------------------------------------------------------------------------------------


@app.get("/api/auth/config")
def auth_config():
    """Public: what the sign-in page needs."""
    return {
        "mode": mode(), "google_client_id": os.environ.get("GYMHUB_GOOGLE_CLIENT_ID", ""),
        "source_url": os.environ.get("GYMHUB_SOURCE_URL", "https://github.com/ykirpichev/human-gym"),
    }


class GoogleSignIn(BaseModel):
    credential: str = Field(max_length=8000)
    invite: str | None = Field(default=None, max_length=100)


@app.post("/api/auth/google")
def sign_in_google(body: GoogleSignIn, request: Request, response: Response):
    if mode() != "app":
        raise HTTPException(400, "Sign-in is not used in this mode")
    uid, _doc = auth.sign_in_google(body.credential, body.invite)
    auth.set_session_cookie(response, request, uid)
    return {"ok": True}


class AcceptInvite(BaseModel):
    invite: str = Field(max_length=100)


@app.post("/api/auth/accept-invite")
def accept_invite(body: AcceptInvite, request: Request):
    """Behind IAP: the visitor is already identified by Google; joining only needs the invite."""
    if mode() != "iap":
        raise HTTPException(400, "Use Sign in with Google to accept an invite")
    auth.admit(auth.iap_claims(request), body.invite)
    return {"ok": True}


class LocalSignIn(BaseModel):
    username: str = Field(max_length=64)
    password: str = Field(max_length=200)


class LocalRegister(LocalSignIn):
    name: str = Field(default="", max_length=100)
    invite: str = Field(max_length=100)


def _local_only() -> None:
    if mode() != "local":
        raise HTTPException(400, "This board does not use name-and-password sign-in")


@app.get("/api/auth/invite/{code}")
def invite_details(code: str):
    """Local mode: whom an invite link is for and whether it still works (the account form asks first)."""
    _local_only()
    return auth.invite_info(code)


@app.post("/api/auth/register")
def register_local(body: LocalRegister, request: Request, response: Response):
    """Local mode: create an account from an invite link, or set a new password from a reset link."""
    _local_only()
    uid, _doc = auth.register_local(body.username, body.password, body.name, body.invite)
    auth.set_session_cookie(response, request, uid)
    return {"ok": True}


@app.post("/api/auth/password")
def sign_in_local(body: LocalSignIn, request: Request, response: Response):
    _local_only()
    uid, _doc = auth.sign_in_local(body.username, body.password)
    auth.set_session_cookie(response, request, uid)
    return {"ok": True}


class PasswordChange(BaseModel):
    old: str = Field(max_length=200)
    new: str = Field(max_length=200)


@app.post("/api/me/password")
def change_my_password(body: PasswordChange, user: User = Depends(current_user)):
    _local_only()
    auth.change_password(user.uid, body.old, body.new)
    return {"ok": True}


@app.post("/api/auth/logout")
def sign_out(response: Response):
    auth.clear_session_cookie(response)
    return {"ok": True}


@app.get("/api/me")
def me(user: User = Depends(current_user)):
    me = {"user": user.public(), "auth": mode(), "can_run": auth.can_run(user), "owner": is_owner(user),
          "notifications": (store.get("users", user.uid) or {}).get("notifications") or {"jobs": True, "machines": True}}
    if not user.is_admin:
        me["usage"] = quotas.usage(store, user.uid)
        me["limits"] = quotas.limits(store)
    return me


@app.get("/api/admin/health")
def admin_health(hours: float = 24, _admin: User = Depends(require_admin)):
    return health.summary(store, hours=max(1.0, min(hours, 24 * 14)))


class NotificationSettings(BaseModel):
    jobs: bool = True
    machines: bool = True


@app.put("/api/me/notifications")
def my_notifications(body: NotificationSettings, user: User = Depends(current_user)):
    store.set("users", user.uid, {"notifications": body.model_dump()}, merge=True)
    auth.forget_user(user.uid)
    return {"notifications": body.model_dump()}


# ---- your data: export and account deletion (gymhub/accounts.py) ---------------------------------


def _own_account(user: User) -> None:
    if mode() == "dev":
        raise HTTPException(400, "A local dev board has no accounts to export or delete")


@app.get("/api/me/export")
def export_my_data(user: User = Depends(current_user)):
    """Everything the board holds about you, as one JSON file."""
    _own_account(user)
    body = json.dumps(accounts.export(store, user.uid), indent=2, default=str)
    return Response(body, media_type="application/json",
                    headers={"Content-Disposition": 'attachment; filename="human-gym-my-data.json"'})


@app.get("/api/me/deletion")
def account_deletion_preview(user: User = Depends(current_user)):
    """What deleting your account removes that other people would notice."""
    _own_account(user)
    blocked = "You are on the board's admin list (GYMHUB_ADMIN_EMAILS). Remove yourself from it first." \
        if user.email in auth.admin_emails() else None
    return {**accounts.preview(store, user.uid), "email": user.email, "blocked": blocked}


class DeleteAccount(BaseModel):
    confirm: str = Field(max_length=200)


@app.delete("/api/me")
def delete_my_account(body: DeleteAccount, response: Response, user: User = Depends(current_user)):
    """Delete your account and your data, now. Gyms you own are deleted for everyone."""
    _own_account(user)
    if user.email in auth.admin_emails():
        raise HTTPException(400, "You are on the board's admin list (GYMHUB_ADMIN_EMAILS); you would be an admin "
                                 "again at your next sign-in. Remove yourself from it first.")
    if body.confirm.strip().lower() != user.email.lower():
        raise HTTPException(400, f"Type your email ({user.email}) to confirm")
    counts = accounts.delete(store, user.uid, open_blobs())
    auth.forget_user(user.uid)
    auth.clear_session_cookie(response)
    logging.getLogger("gymhub.board").info("account %s deleted by its owner: %s", user.uid, counts)
    return {"ok": True, "deleted": counts}


# ---- admin -------------------------------------------------------------------------------------


@app.get("/api/admin/users")
def admin_users(_admin: User = Depends(require_admin)):
    users = [{"uid": uid, **doc} for uid, doc in store.recent("users", "created_at", 500)]
    for user in users:
        user["usage"] = quotas.usage(store, user["uid"])
    return {"users": users, "settings": auth.auth_settings(), "admin_emails": sorted(auth.admin_emails()), "limits": quotas.limits(store),
            "ai": {"enabled": tutor.enabled(), "monthly_usd": tutor.board_budget(store), "spent": tutor.board_usage(store)}}


class UserUpdate(BaseModel):
    role: str | None = None
    status: str | None = None


@app.post("/api/admin/users/{uid}")
def admin_update_user(uid: str, body: UserUpdate, admin: User = Depends(require_admin)):
    doc = store.get("users", uid)
    if doc is None:
        raise HTTPException(404, "No such user")
    update = {}
    if body.role is not None:
        if body.role not in {"admin", "member"}:
            raise HTTPException(400, "Role must be admin or member")
        update["role"] = body.role
    if body.status is not None:
        if body.status not in {"active", "disabled"}:
            raise HTTPException(400, "Status must be active or disabled")
        update["status"] = body.status
    if uid == admin.uid and (update.get("role") == "member" or update.get("status") == "disabled"):
        raise HTTPException(400, "You cannot demote or disable yourself")
    store.set("users", uid, update, merge=True)
    auth.forget_user(uid)
    return {"ok": True}


@app.get("/api/admin/invites")
def admin_invites(_admin: User = Depends(require_admin)):
    invites = [{"code": code, **doc} for code, doc in store.recent("invites", "created_at", 200)]
    return {"invites": invites}


class InviteRequest(BaseModel):
    email: str | None = Field(default=None, max_length=200)  # local mode: the account name it is for
    role: str = Field(default="member", pattern="^(member|admin)$")
    uses: int = Field(default=1, ge=1, le=1000)
    days: int = Field(default=14, ge=1, le=365)
    note: str = Field(default="", max_length=200)


@app.post("/api/admin/invites")
def admin_create_invite(body: InviteRequest, admin: User = Depends(require_admin)):
    code = auth.new_invite_code()
    doc = {
        "created_by": admin.email,
        "email": (body.email or "").strip().lower() or None,
        "uses_left": body.uses,
        "uses_total": body.uses,
        "expires_at": now() + body.days * 86400,
        "created_at": now(),
        "note": body.note,
    }
    if body.role == "admin":
        doc["role"] = "admin"
    store.set("invites", code, doc)
    return {"code": code, **doc}


@app.post("/api/admin/users/{uid}/password-reset")
def admin_password_reset(uid: str, admin: User = Depends(require_admin)):
    """Local mode: a one-time link, valid for a day, that lets this person pick a new password."""
    _local_only()
    doc = store.get("users", uid)
    if doc is None or not uid.startswith("l_"):
        raise HTTPException(404, "No such account")
    code = auth.new_invite_code()
    invite = {"created_by": admin.email, "email": doc["email"], "uses_left": 1, "uses_total": 1, "reset_uid": uid,
              "expires_at": now() + 86400, "created_at": now(), "note": "password reset"}
    store.set("invites", code, invite)
    return {"code": code, **invite}


@app.delete("/api/admin/invites/{code}")
def admin_revoke_invite(code: str, _admin: User = Depends(require_admin)):
    store.delete("invites", code)
    return {"ok": True}


class Limits(BaseModel):
    jobs_per_hour: int = Field(ge=0, le=10_000)
    minutes_per_day: int = Field(ge=0, le=24 * 60)
    max_waiting: int = Field(ge=0, le=100)


@app.put("/api/admin/limits")
def admin_limits(body: Limits, _admin: User = Depends(require_admin)):
    """0 means no limit. Admins are never limited."""
    store.set("settings", "limits", body.model_dump())
    return {"ok": True}


class AiSettings(BaseModel):
    monthly_usd: float = Field(ge=0, le=100_000)


@app.put("/api/admin/ai")
def admin_ai(body: AiSettings, _admin: User = Depends(require_admin)):
    """The board-wide monthly cap on Claude spend (tutor and essay grading). 0 means no cap."""
    store.set("settings", "ai", body.model_dump())
    return {"ok": True}


class AuthSettings(BaseModel):
    allowed_domains: list[str] = Field(default_factory=list, max_length=50)
    members_can_run: bool = False
    public_gyms_need_review: bool = True


@app.put("/api/admin/settings")
def admin_settings(body: AuthSettings, _admin: User = Depends(require_admin)):
    domains = sorted({d.strip().lower().lstrip("@") for d in body.allowed_domains if d.strip()})
    if any("." not in d or " " in d for d in domains):
        raise HTTPException(400, "Domains look like example.com")
    store.set("settings", "auth", {"allowed_domains": domains, "members_can_run": body.members_can_run,
                                   "public_gyms_need_review": body.public_gyms_need_review})
    return {"ok": True}


# ---- whose data -----------------------------------------------------------------------------------
#
# The owner (the person whose machine runs the jobs) keeps working on the runner's local files:
# solutions/, progress.json, edits from a local editor. Everyone else gets their own workspace and
# progress in Firestore, and their jobs run from a scratch copy that never touches the owner's files.
#
#   workspaces/{uid}__{gym}__{slug}__{lang}   uid, gym, slug, lang, code, updated_at
#   user_progress/{uid}__{gym}                uid, gym, data (same shape as progress.json)


# ---- uploaded gyms (gymhub/gymversions.py) ----------------------------------------------------
# Admins only for now; who else may create gyms, and who may see them, comes with issue #15.


MAX_GYMS_PER_MEMBER = int(os.environ.get("GYMHUB_MAX_GYMS_PER_MEMBER", "10"))


def gym_role(user: User, doc: dict | None) -> str | None:
    return gymaccess.role(user.uid, user.is_admin, doc)


def require_gym(user: User, gym: str, needed: str = "learner") -> dict:
    """The gym's document if this user has at least `needed`. A gym they cannot see does not exist (404)."""
    doc = store.get("gyms", gym)
    current = gym_role(user, doc)
    if current is None:
        raise HTTPException(404, f"No gym {gym!r}")
    if not gymaccess.at_least(current, needed):
        raise HTTPException(403, f"You need to be this gym's {needed} for that")
    return doc


def _may_publish(user: User):
    def check(existing: dict | None, gym_id: str) -> str | None:
        if existing is None:
            draft = store.get("gym_drafts", gym_id)  # someone's unpublished draft in the browser editor holds the id
            if draft and draft.get("owner_uid") != user.uid and not user.is_admin:
                return f"{gym_id!r} is taken by another gym; choose another id"
            deleted = store.get("gym_tombstones", gym_id)  # a deleted gym's id stays its owner's (#120)
            if deleted and deleted.get("owner_uid") != user.uid and not user.is_admin:
                return f"{gym_id!r} belonged to a deleted gym; choose another id"
            if not user.is_admin and len(store.where("gyms", "owner_uid", user.uid)) >= MAX_GYMS_PER_MEMBER:
                return f"You already own {MAX_GYMS_PER_MEMBER} gyms, the most a member may have"
            return None
        if existing.get("source") != "bundle":
            return f"{existing.get('id')!r} is a gym published by a machine's local files; choose another id"
        if not gymaccess.at_least(gym_role(user, existing), "editor"):
            return f"{existing.get('id')!r} is taken by another gym; choose another id"
        return None

    return check


def _start_upload(user: User, data: bytes, extra: dict | None = None) -> dict:
    if not user.is_admin:
        try:
            quotas.check_and_count(store, user.uid)  # validation runs on a machine like any other job
        except quotas.QuotaExceeded as exc:
            raise HTTPException(429, str(exc)) from exc
    version = gymversions.upload(store, open_blobs(), data, user.uid, user.email, _may_publish(user), extra)
    reply = {"version": version}
    if not version.get("duplicate"):
        warning = _no_machine_warning(user, version["gym"], {"gpu": False, "tools": []}, bundle=True,
                                      job={"action": "validate", "isolated": True})
        if warning:
            reply["warning"] = warning
    return reply


@app.post("/api/gym-versions")
async def upload_gym_version(request: Request, user: User = Depends(auth.publisher)):
    if not auth.can_run(user):
        raise HTTPException(403, "Creating gyms runs their code on a machine, which is not enabled for members yet.")
    limit = packages.MAX_BUNDLE_BYTES
    if int(request.headers.get("content-length") or 0) > limit:
        raise HTTPException(413, f"The bundle is over {limit // 1024 // 1024} MB")
    data = await request.body()
    if len(data) > limit:
        raise HTTPException(413, f"The bundle is over {limit // 1024 // 1024} MB")
    if gymmd.is_gym_markdown(data):  # one .gym.md file: compiled here (parsing only, no code runs)
        try:
            data = await asyncio.to_thread(gymmd.bundle, data)
        except gymmd.GymMarkdownError as exc:
            return JSONResponse({"detail": f"{len(exc.errors)} problem(s) in the Gym Markdown",
                                 "errors": gymmd.format_errors(exc.errors).split("\n")}, status_code=400)
    try:
        return await asyncio.to_thread(_start_upload, user, data)
    except gymversions.UploadRejected as exc:
        return JSONResponse({"detail": str(exc), "errors": exc.errors}, status_code=400)


@app.get("/api/gym-versions")
def gym_versions(gym: str | None = None, user: User = Depends(auth.publisher)):
    """One gym's versions (its owner and editors), or recent uploads: everyone's for admins, else your own."""
    if gym:
        existing = store.get("gyms", gym)
        if existing is not None:
            require_gym(user, gym, "editor")
        docs = gymversions.versions(store, gym)
        if existing is None and not user.is_admin:  # never published: only your own attempts
            docs = [doc for doc in docs if doc.get("uid") == user.uid]
    elif user.is_admin:
        docs = [{**doc, "id": doc_id} for doc_id, doc in store.recent("gym_versions", "created_at", 50)]
    else:
        docs = [{**doc, "id": doc_id} for doc_id, doc in store.where("gym_versions", "uid", user.uid)]
        docs.sort(key=lambda doc: doc.get("created_at", 0), reverse=True)
        docs = docs[:50]
    live = {}
    for doc in docs:
        if doc["gym"] not in live:
            live[doc["gym"]] = (store.get("gyms", doc["gym"]) or {}).get("version")
        doc["live"] = live[doc["gym"]] == doc.get("version")
    return {"versions": docs}


def _gym_order(doc: dict) -> float:
    """Where a gym sorts among gyms. Versions published before gym.json's "order" was optional may hold None."""
    order = doc.get("order")
    return order if isinstance(order, (int, float)) and not isinstance(order, bool) else 99


def is_owner(user: User) -> bool:
    return mode() == "dev" or user.email in auth.admin_emails()


def _workspace_id(uid: str, gym: str, slug: str, lang: str) -> str:
    return f"{uid}__{gym}__{slug}__{lang}"


def uses_workspace(user: User, gym_doc: dict | None) -> bool:
    """Code and progress in the board's store rather than the runner's files: everyone but the owner,
    and everyone (the owner too) in uploaded gyms, which have no local files."""
    return not is_owner(user) or gymversions.is_bundle_gym(gym_doc)


def progress_for(user: User, gym: str, gym_doc: dict | None = None) -> dict:
    if gym_doc is None and is_owner(user):
        gym_doc = store.get("gyms", gym)
    if not uses_workspace(user, gym_doc):
        data = (store.get("progress", gym) or {}).get("data")
    else:
        data = (store.get("user_progress", f"{user.uid}__{gym}") or {}).get("data")
    data = data or {}
    data.setdefault("version", 1)
    data.setdefault("exercises", {})
    return data


def _save_user_progress(user: User, gym: str, data: dict) -> None:
    store.set("user_progress", f"{user.uid}__{gym}", {"uid": user.uid, "gym": gym, "data": data, "updated_at": now()})


def _visible(user: User, job: dict) -> bool:
    """The owner sees the owner's jobs; everyone else sees only their own."""
    if job.get("isolated"):
        return job.get("uid") == user.uid
    return is_owner(user)


def _solved_count(progress: dict) -> int:
    return sum(
        1 for entry in progress.get("exercises", {}).values() if any("first_solved" in lang for lang in entry.get("langs", {}).values())
    )


def _usable_runners(user: User) -> list[dict]:
    """Admins see every machine; members see the machines that can run their jobs."""
    runners = _runners()
    if user.is_admin:
        return runners
    visible = []
    for runner in runners:
        owner_email = runner.get("owner_email")
        if not owner_email or owner_email in auth.admin_emails() or owner_email == user.email:
            visible.append(runner)
            continue
        worker = store.get("workers", runner.get("worker_id", "")) or {}
        if worker.get("shared_everyone") or user.uid in (worker.get("shared_uids") or []):
            visible.append(runner)
    return visible


@app.get("/api/overview")
def overview(user: User = Depends(current_user)):
    gyms = []
    for doc_id, doc in sorted(store.recent("gyms", "updated_at", 200), key=lambda item: _gym_order(item[1])):
        if gymaccess.listed(user.uid, user.is_admin, doc):
            gyms.append({**_public_gym(doc, user), "id": doc_id, "solved": _solved_count(progress_for(user, doc_id, doc))})
    # Run-mode jobs are scratch work: never listed.
    jobs = [
        _job_summary(doc_id, doc)
        for doc_id, doc in store.recent("jobs", "created_at", 100)
        if doc.get("action") != "run" and _visible(user, doc)
    ][:15]
    return {"gyms": gyms, "runners": _usable_runners(user), "jobs": jobs}


@app.get("/api/home")
def home(user: User = Depends(current_user)):
    """Everything the Home page needs in one request. Dates are left raw: the browser computes
    streaks and "this week" in the user's own time zone."""
    gyms = []
    runners = store.recent("runners", "heartbeat_at", 50)
    for gym_id, doc in sorted(store.recent("gyms", "updated_at", 200), key=lambda item: _gym_order(item[1])):
        if not gymaccess.listed(user.uid, user.is_admin, doc):
            continue
        problems = [
            {key: problem.get(key) for key in LIST_FIELDS} for _id, problem in store.where("problems", "gym", gym_id)
        ]
        problems.sort(key=lambda problem: problem.get("order", 0))
        gyms.append({**_public_gym(doc, user), "id": gym_id, "problem_list": problems, "progress": progress_for(user, gym_id, doc),
                     "gradable": _gradable(user, gym_id, doc, runners)})
    jobs = [
        _job_summary(doc_id, doc)
        for doc_id, doc in store.recent("jobs", "created_at", 100)
        if doc.get("action") in {"test", "bench", "answer"} and _visible(user, doc)
    ][:10]
    return {"gyms": gyms, "runners": _usable_runners(user), "jobs": jobs, "reviews": reviews.queue(store, user.uid, gyms)}


def _public_gym(doc: dict, user: User) -> dict:
    """A gym document as this user may see it: sharing lists and the join code only for its owner."""
    current = gym_role(user, doc)
    shown = {key: value for key, value in doc.items() if key not in {"editors", "learners", "share_code"}}
    shown.update({
        "role": current, "visibility": gymaccess.visibility(doc), "can_fork": gymaccess.can_fork(current, doc),
        "tags": doc.get("tags") or [],
        "tutor": bool(doc.get("tutor", True)) and tutor.enabled(),
    })
    return shown


@app.get("/api/gyms/{gym}")
def gym_detail(gym: str, user: User = Depends(auth.publisher)):  # also personal API tokens (gymhub.cli)
    doc = require_gym(user, gym)  # unlisted gyms open only after joining with the link's code (POST .../join)
    problems = [{key: problem.get(key) for key in LIST_FIELDS} for _id, problem in store.where("problems", "gym", gym)]
    problems.sort(key=lambda problem: problem.get("order", 0))
    current = {problem["slug"]: problem.get("tests_hash") for problem in problems}
    return {"gym": {**_public_gym(doc, user), "id": gym}, "problems": problems, "progress": progress_for(user, gym, doc),
            "tests_changed": _tests_changed(user, gym, current)}


def _tests_changed(user: User, gym: str, current: dict[str, str | None]) -> dict[str, list[str]]:
    """slug -> languages whose last passing submission ran against tests that have changed since."""
    changed: dict[str, list[str]] = {}
    for _id, solve in store.where("solves", "uid", user.uid):
        slug = solve.get("slug")
        if solve.get("gym") == gym and current.get(slug) and solve.get("tests_hash") != current[slug]:
            changed.setdefault(slug, []).append(solve.get("lang") or "")
    return changed


@app.get("/api/gyms/{gym}/problems/{slug}")
def problem_detail(gym: str, slug: str, user: User = Depends(auth.publisher)):  # also personal API tokens (gymhub.cli)
    workers.record_activity()
    gym_doc = require_gym(user, gym)
    doc = store.get("problems", problem_id(gym, slug))
    if doc is None:
        raise HTTPException(404, f"No exercise {gym}/{slug}")
    if uses_workspace(user, gym_doc):
        # Never show the owner's code: each user sees their own saved code, or the starter.
        current = {}
        for lang in doc.get("langs", []):
            saved = store.get("workspaces", _workspace_id(user.uid, gym, slug, lang))
            current[lang] = saved["code"] if saved else (doc.get("starters") or {}).get(lang, "")
        doc["current"] = current
        doc["revisions"] = {lang: _revision(code) for lang, code in current.items()}
    progress = progress_for(user, gym, gym_doc)["exercises"].get(slug, {})
    jobs = [
        _job_summary(doc_id, job)
        for doc_id, job in store.where("jobs", "problem", problem_id(gym, slug))
        if job.get("action") in {"test", "bench", "answer"} and _visible(user, job)
    ]
    jobs.sort(key=lambda job: job.get("created_at") or 0, reverse=True)
    # The editorial (worked solution) opens after a first attempt, or when the learner asks for it.
    editorial = doc.pop("editorial", "") or ""
    if editorial:
        if _editorial_open(user, gym, slug, progress, jobs):
            doc["editorial"] = editorial
        else:
            doc["editorial_locked"] = True
    if doc.get("kind", "code") != "code":
        # The learner's latest answer, to show it again (their own only, like everything in jobs).
        last = next((store.get("jobs", job["id"]) for job in jobs if job.get("action") == "answer"), None)
        try:
            doc["last_answer"] = json.loads(last["code"]) if last and last.get("code") else None
        except (TypeError, ValueError):
            doc["last_answer"] = None
    bests, tests_changed = {}, {}
    kind = doc.get("kind", "code")
    for lang in doc.get("langs", []) if kind == "code" else [kind]:  # non-code exercises record under their kind
        record = store.get("bests", f"{user.uid}__{gym}__{slug}__{lang}")
        if record:
            bests[lang] = record.get("by_hardware", {})
        solve = store.get("solves", f"{user.uid}__{gym}__{slug}__{lang}")
        if solve and doc.get("tests_hash") and solve.get("tests_hash") != doc["tests_hash"]:
            tests_changed[lang] = {"version": solve.get("version"), "at": solve.get("at")}
    return {
        "problem": doc, "gym": {**_public_gym(gym_doc, user), "id": gym}, "progress": progress, "jobs": jobs[:20],
        "owner": not uses_workspace(user, gym_doc), "bests": bests, "tests_changed": tests_changed,
    }


def _editorial_id(user: User, gym: str, slug: str) -> str:
    return f"{user.uid}__{gym}__{slug}"


def _editorial_open(user: User, gym: str, slug: str, progress: dict, jobs: list[dict]) -> bool:
    """Has this learner tried the exercise (a submission, a recorded attempt) or chosen to read the editorial?"""
    if any(job.get("action") in {"test", "answer"} for job in jobs):
        return True
    if any(int(lang.get("attempts") or 0) > 0 or "first_solved" in lang for lang in (progress.get("langs") or {}).values()):
        return True
    return store.get("editorial_views", _editorial_id(user, gym, slug)) is not None


@app.post("/api/gyms/{gym}/problems/{slug}/editorial")
def open_editorial(gym: str, slug: str, user: User = Depends(auth.publisher)):  # also personal API tokens
    """Show the editorial before trying: recorded, so it stays open and the owner's stats can tell."""
    require_gym(user, gym)
    doc = store.get("problems", problem_id(gym, slug))
    if doc is None or not doc.get("editorial"):
        raise HTTPException(404, "This exercise has no editorial")
    store.set("editorial_views", _editorial_id(user, gym, slug), {"uid": user.uid, "gym": gym, "slug": slug, "at": now()})
    return {"editorial": doc["editorial"]}


class JobRequest(BaseModel):
    gym: str = Field(max_length=40)
    slug: str = Field(max_length=80)
    lang: str = Field(default="", max_length=20)
    action: str
    code: str | None = Field(default=None, max_length=200_000)
    options: dict = Field(default_factory=dict)
    # The revision of the saved code this one was written from (see `_revision`). A save or submit
    # whose base is no longer what the board holds is refused with 409. Omitted: overwrite.
    base: str | None = Field(default=None, max_length=64)
    # Run on (gymhub/board/runtargets.py): {"runner": id} or {"pool_machine": name}; omitted: Auto.
    target: dict | None = None


def _revision(code: str) -> str:
    """An opaque name for one version of a workspace file: clients only compare it for equality."""
    return hashlib.sha256(code.encode()).hexdigest()[:16]


def _store_workspace(user: User, request: "JobRequest", workspace: str, problem: dict) -> str:
    """Save the user's code in one transaction that also checks `request.base`. Returns the new revision."""
    def apply(saved: dict | None) -> dict:
        if request.base is not None:
            current = saved["code"] if saved else (problem.get("starters") or {}).get(request.lang, "")
            if _revision(current) != request.base:
                raise HTTPException(
                    409,
                    {"message": f"{request.slug} [{request.lang}] changed on the board since you last pulled it. "
                                "Nothing was saved. Pull it again (or push with --force to overwrite it)."},
                )
        return {"uid": user.uid, "gym": request.gym, "slug": request.slug, "lang": request.lang,
                "code": request.code, "updated_at": now()}

    store.transact("workspaces", workspace, apply)
    return _revision(request.code or "")


RUN_ACTIONS = {"run", "test", "bench"}


def _finished_job(user: User, request: "JobRequest", output: str, revision: str | None = None) -> dict:
    """A job the API completed itself (save, reset, hint for workspace users): nothing to run."""
    job_id = new_id()
    store.set(
        "jobs",
        job_id,
        {
            "gym": request.gym, "slug": request.slug, "problem": problem_id(request.gym, request.slug),
            "lang": request.lang, "action": request.action, "code": None, "options": {},
            "status": "done", "ok": True, "output": output, "created_at": now(), "started_at": now(),
            "finished_at": now(), "created_by": user.email, "uid": user.uid, "isolated": True,
            **jobqueue.expiry(request.action),
        },
    )
    return {"id": job_id, **({"revision": revision} if revision else {})}


@app.post("/api/jobs")
def create_job(request: JobRequest, user: User = Depends(auth.publisher)):  # also personal API tokens (gymhub.cli)
    workers.record_activity()
    if request.action not in ACTIONS:
        raise HTTPException(400, f"Unknown action {request.action!r}")
    if request.action in RUN_ACTIONS and not auth.can_run(user):
        raise HTTPException(
            403,
            "Running code is not enabled for members on this board yet. "
            "An admin can turn it on in Admin settings. You can still edit and save.",
        )
    gym_doc = require_gym(user, request.gym)
    problem = store.get("problems", problem_id(request.gym, request.slug))
    if problem is None:
        raise HTTPException(404, "Unknown exercise")
    if request.action == "run" and request.code is None:
        raise HTTPException(400, "Run needs the code to run")
    if request.action != "hint" and request.lang not in problem.get("langs", []):
        raise HTTPException(400, f"{request.slug} has no language {request.lang!r}")
    owner = not uses_workspace(user, gym_doc)
    # Re-solving from scratch (spaced review): a scratch run that never replaces the saved solution.
    review = request.action == "test" and request.options.get("review") is True and request.code is not None
    requires = routing.requirements(request.gym, gym_doc, request.lang)
    # An uploaded gym's jobs grade with the version the learner is looking at.
    bundle = (gym_doc or {}).get("bundle") if gymversions.is_bundle_gym(gym_doc) else None
    routed = {"gym": request.gym, "slug": request.slug, "lang": request.lang, "action": request.action, "uid": user.uid,
              "isolated": not owner or review, "requires": requires, "bundle": bundle}
    target = request.target if request.action in RUN_ACTIONS and request.target else None
    if target is not None:  # Run on: only where this user could be routed anyway
        refused = runtargets.check(store, {**routed, "target": target})
        if refused:
            raise HTTPException(403, refused)
        routed["target"] = target
    job_id = new_id()
    if request.action in RUN_ACTIONS and not user.is_admin:
        try:
            quotas.check_and_count(store, user.uid, job_id)
        except quotas.QuotaExceeded as exc:
            raise HTTPException(429, str(exc)) from exc

    if not owner:
        workspace = _workspace_id(user.uid, request.gym, request.slug, request.lang)
        revision = None
        if request.action in {"save", "test", "bench"} and request.code is not None and not review:
            revision = _store_workspace(user, request, workspace, problem)
        if request.action == "save":
            return _finished_job(user, request, "Saved to your workspace.\n", revision)
        if request.action == "reset":
            store.delete("workspaces", workspace)
            return _finished_job(user, request, f"Restored the starter for {request.slug} [{request.lang}].\n")
        if request.action == "hint":
            count = request.options.get("count")
            if isinstance(count, int) and count > 0:
                data = progress_for(user, request.gym, gym_doc)
                entry = data["exercises"].setdefault(request.slug, {"hints_seen": 0, "langs": {}})
                entry["hints_seen"] = max(entry.get("hints_seen", 0), count)
                _save_user_progress(user, request.gym, data)
            return _finished_job(user, request, "")

    options = {key: value for key, value in request.options.items() if key in {"runs", "count"} and isinstance(value, int)}
    if request.action == "bench" and request.options.get("profile") is True:
        options["profile"] = True
    if review:
        options["review"] = True
    if request.action == "run" and request.options.get("cases"):
        cases = request.options["cases"]
        if not isinstance(cases, list) or len(cases) > 20 or len(json.dumps(cases)) > 64_000:
            raise HTTPException(400, "Custom cases must be a list of at most 20 cases, 64 KB in total")
        options["cases"] = cases
        options["only_custom"] = bool(request.options.get("only_custom"))
    store.set(
        "jobs",
        job_id,
        {
            "gym": request.gym,
            "slug": request.slug,
            "problem": problem_id(request.gym, request.slug),
            "lang": request.lang,
            "action": request.action,
            "code": request.code,
            "options": options,
            "status": "queued",
            "created_at": now(),
            "created_by": user.email,
            "uid": user.uid,
            # Not the owner, or a review: the runner uses a scratch copy and this user's own progress.
            "isolated": not owner or review,
            "requires": requires,
            "bundle": bundle,
            "tests_hash": problem.get("tests_hash"),
            "output": "",
            **({"target": target} if target else {}),
            **jobqueue.expiry(request.action),
        },
    )
    health.log_created(job_id, {"uid": user.uid, "gym": request.gym, "slug": request.slug, "lang": request.lang,
                                "action": request.action, "isolated": not owner or review})
    reply = {"id": job_id, **({"revision": revision} if not owner and revision else {})}
    if target:
        warning = runtargets.waiting(store, routed)
    else:
        warning = _no_machine_warning(user, request.gym, requires, bool(bundle),
                                      job={"isolated": not owner or review, "bundle": bundle, "action": request.action})
    if warning:
        reply["warning"] = warning
    return reply


def _find_grader(user: User, gym: str, requires: dict, bundle: bool = False, job: dict | None = None,
                 runners: list | None = None) -> tuple[dict | None, set, int, int]:
    """The first online machine this user may use that can run the job (None when there is none), and
    for messages: what the others lack, how many are online, and how many of those grade only their
    owners' jobs. A pool bridge counts with the pool's capabilities for the jobs it grades there
    (gymhub/routing.py). `runners` saves the store read when checking many gyms at once."""
    reasons = set()
    job = {"gym": gym, "uid": user.uid, "bundle": bundle, **(job or {})}
    online, private = 0, 0
    if runners is None:
        runners = store.recent("runners", "heartbeat_at", 50)
    for runner_id, runner in runners:
        if time.time() - runner.get("heartbeat_at", 0) >= RUNNER_ONLINE_SECONDS:
            continue
        online += 1
        # Whose jobs it takes (private unless shared, gymhub/board/workers.py), which gyms may reach it.
        if not runtargets.allowed(store, runner_id, runner, job):
            private += 1
            continue
        if not routing.serves(runner, gym, bundle):
            continue
        if routing.refuses(runner.get("capabilities"), job):
            continue
        missing = routing.unmet(routing.for_job(runner.get("capabilities"), job), requires)
        if missing is None:
            return runner, reasons, online, private
        reasons.add(missing)
    return None, reasons, online, private


def _gradable(user: User, gym_id: str, doc: dict, runners: list) -> bool:
    """Can a machine that is up grade this gym's exercises for this user now? Used to point a new
    learner at an exercise that runs (a machine without a GPU should not start with a GPU kernel).
    A gym whose publishing machine lacks a module it needs (`machine_ready` false, gymhub/runner/catalog.py)
    grades only on the pool, which brings its own image."""
    bundle = doc.get("bundle") if gymversions.is_bundle_gym(doc) else None
    job = {"isolated": uses_workspace(user, doc), "action": "test", "gym_public": gymaccess.public(doc)}
    grader, *_ = _find_grader(user, gym_id, routing.requirements(gym_id, doc, ""), bool(bundle), job, runners)
    if grader is None:
        return False
    full = {"gym": gym_id, "uid": user.uid, "bundle": bundle, **job}
    return doc.get("machine_ready", True) is not False or routing.on_pool(grader.get("capabilities"), full)


def _no_machine_warning(user: User, gym: str, requires: dict, bundle: bool = False, job: dict | None = None) -> str | None:
    """Say so when no online machine this user may use can run the job (it waits in the queue). A
    pool bridge counts with the pool's capabilities for the jobs it grades there (gymhub/routing.py)."""
    grader, reasons, online, private = _find_grader(user, gym, requires, bundle, job)
    if grader is not None:
        return None
    if reasons:
        return f"No online machine can run this yet (needs {', '.join(sorted(reasons))}). It will wait in the queue."
    if online and online == private:
        return ("The machines on this board grade only their owners' jobs, and none is shared with you yet. "
                "The job will wait in the queue until an owner shares a machine (or the pool) with you.")
    return "No machine is online right now. The job will wait in the queue until one connects."


@app.get("/api/run-targets")
def run_targets(gym: str, slug: str, lang: str = "", user: User = Depends(current_user)):
    """The "Run on" choices for an exercise: the machines this user's job could run on now."""
    gym_doc = require_gym(user, gym)
    if store.get("problems", problem_id(gym, slug)) is None:
        raise HTTPException(404, "Unknown exercise")
    if not auth.can_run(user):
        return {"choices": []}
    bundle = (gym_doc or {}).get("bundle") if gymversions.is_bundle_gym(gym_doc) else None
    job = {"gym": gym, "slug": slug, "lang": lang, "action": "test", "uid": user.uid, "isolated": uses_workspace(user, gym_doc),
           "requires": routing.requirements(gym, gym_doc, lang), "bundle": bundle}
    return {"choices": runtargets.choices(store, job)}


def _own_job(job_id: str, user: User) -> dict:
    doc = store.get("jobs", job_id)
    if doc is None or not _visible(user, doc):
        raise HTTPException(404, "No such job")
    return doc


@app.get("/api/jobs/{job_id}")
def job_detail(job_id: str, user: User = Depends(auth.publisher)):  # also personal API tokens (gymhub.cli)
    doc = _own_job(job_id, user)
    doc.pop("code", None)
    return {**doc, "id": job_id}


JOB_FIELDS = ("gym", "slug", "lang", "action", "status", "ok", "created_at", "started_at", "finished_at", "runner", "hardware", "tries",
              "host", "via", "target")
STREAM_SECONDS = 3600


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request, user: User = Depends(current_user)):
    """Server-Sent Events for one job: its fields and its output as it grows, until it finishes.

    Each event is JSON: the job's fields plus either "append" (new output) or "output" (all of it,
    when the output was replaced, e.g. after a retry). A comment line keeps idle connections open.
    """
    _own_job(job_id, user)
    loop = asyncio.get_running_loop()
    updates: asyncio.Queue = asyncio.Queue()
    stop = store.watch_doc("jobs", job_id, lambda doc: loop.call_soon_threadsafe(updates.put_nowait, doc))

    async def stream():
        sent = ""
        started = time.time()
        try:
            while time.time() - started < STREAM_SECONDS:
                if await request.is_disconnected():
                    return
                try:
                    doc = await asyncio.wait_for(updates.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                if doc is None:
                    yield f"data: {json.dumps({'status': 'gone'})}\n\n"
                    return
                event = {key: doc.get(key) for key in JOB_FIELDS} | {"id": job_id}
                output = doc.get("output") or ""
                if output.startswith(sent):
                    event["append"] = output[len(sent):]
                else:
                    event["output"] = output
                sent = output
                yield f"data: {json.dumps(event)}\n\n"
                if doc.get("status") in {"done", "error", "cancelled"}:
                    return
        finally:
            stop()

    headers = {"Cache-Control": "no-store", "X-Accel-Buffering": "no"}
    return StreamingResponse(stream(), media_type="text/event-stream", headers=headers)


@app.get("/api/jobs/{job_id}/code")
def job_code(job_id: str, user: User = Depends(current_user)):
    """The code a submission ran with. Run-mode jobs keep no code."""
    doc = _own_job(job_id, user)
    if doc.get("code") is None:
        raise HTTPException(404, "This job has no stored code (runs are never stored)")
    keys = ("gym", "slug", "lang", "action", "status", "ok", "created_at")
    return {"id": job_id, "code": doc["code"], **{key: doc.get(key) for key in keys}}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str, user: User = Depends(auth.publisher)):  # also personal API tokens (gymhub.cli)
    doc = _own_job(job_id, user)
    if doc.get("status") == "queued":
        end = {"status": "cancelled", "finished_at": now()}
        if store.claim("jobs", job_id, {"status": "queued"}, end):
            health.log_finished(job_id, doc, end)
    else:
        store.set("jobs", job_id, {"cancel": True}, merge=True)
    return {"ok": True}


@app.get("/healthz")
def healthz():
    return {"ok": True}


# The two gymkit modules the browser runs examples with (static/browser-worker.js): comparison and
# the examples runner. Both are public code; no grader is ever served.
BROWSER_PY = {name: Path(__file__).resolve().parents[2] / "gymkit" / name for name in ("compare.py", "examples.py")}


@app.get("/browser/{name}")
def browser_python(name: str):
    if name not in BROWSER_PY:
        raise HTTPException(404, "Not found")
    return PlainTextResponse(BROWSER_PY[name].read_text(), media_type="text/x-python; charset=utf-8",
                             headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def _versioned_index() -> str:
    """index.html with ?v=<content hash> on our assets, so a deploy is never hidden by a browser cache."""
    html = (STATIC / "index.html").read_text()
    for name in ("app.js", "home.js", "gyms.js", "editor.js", "question.js", "notify.js", "browser-run.js", "ide.js", "style.css", "ide.css"):
        version = hashlib.sha256((STATIC / name).read_bytes()).hexdigest()[:12]
        html = html.replace(f"/static/{name}", f"/static/{name}?v={version}")
    return html


INDEX_HTML = _versioned_index()


@app.get("/docs/gym-markdown")
def gym_markdown_guide():
    """The Gym Markdown reference, as plain Markdown for people and agents (gymhub/GYM_MARKDOWN.md)."""
    return PlainTextResponse(gymmd.guide(), media_type="text/markdown; charset=utf-8")


@app.get("/{path:path}")
def spa(path: str):
    del path
    return HTMLResponse(INDEX_HTML, headers={"Cache-Control": "no-cache"})


if os.environ.get("GYMHUB_AUTH") not in {None, "app", "iap", "local", "dev"}:
    raise SystemExit("GYMHUB_AUTH must be app, iap, local or dev")
