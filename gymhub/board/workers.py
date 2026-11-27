"""Workers: machines that run jobs. Token management for admins, and the API workers call.

A worker token looks like `gw_<id>_<secret>`. Only a SHA-256 of the secret is stored, so the board can
check a token but never show it again; revoking deletes it. Workers hold no cloud credentials: everything
they do goes through /v1/worker/*, which only lets them

  - report a heartbeat (machine name, GPU, gyms)
  - publish gym catalogs, the owner's code and progress (only workers owned by an admin)
  - lease a queued job, stream its output, extend the lease, and complete it
  - read and write the progress of the user whose isolated job they are running

workers/{id}   name, owner_uid, owner_email, token_hash, created_at, last_seen, revoked
runners/{id}   heartbeat (what the Compute page shows)
"""

import hashlib
import hmac
import secrets
import threading
import time

import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from gymhub import bundle, gymaccess, health, jobqueue, poolview, routing
from gymhub.board import auth
from gymhub.board.auth import User, current_user, require_admin
from gymhub.store import expires_in, now, problem_id

router = APIRouter()
_cache: dict[str, tuple[float, dict]] = {}
_lock = threading.Lock()
_activity = {"written": 0.0, "read": 0.0, "at": 0.0}

ACTIVE_POLL_MS = 1000
IDLE_POLL_MS = 15_000
ACTIVE_WINDOW = 15 * 60
# Firestore cost per idle poll is what makes extra machines cost money, so these are read rarely:
ACTIVITY_CACHE_SECONDS = 30  # "is anyone using the board?"
SWEEP_EVERY_SECONDS = 60  # requeue expired leases, drop old run-mode jobs (per board instance)
LAST_SEEN_EVERY_SECONDS = 300  # how often a heartbeat also updates workers/{id}.last_seen
_sweep = {"at": 0.0}
_capabilities: dict[str, tuple[float, dict | None]] = {}  # worker id -> (time, capabilities from its heartbeat)
_last_seen: dict[str, float] = {}


def _store():
    from gymhub.board.app import store

    return store


# ---- activity: workers poll fast only while people use the board --------------------------------


def record_activity() -> None:
    """Called on exercise views and job submissions. Written at most once a minute per instance."""
    stamp = time.time()
    if stamp - _activity["written"] < 60:
        return
    _activity["written"] = stamp
    _activity["at"] = stamp
    _store().set("settings", "activity", {"at": stamp})


def poll_interval_ms() -> int:
    stamp = time.time()
    if stamp - _activity["read"] > ACTIVITY_CACHE_SECONDS:
        _activity["read"] = stamp
        _activity["at"] = max(_activity["at"], (_store().get("settings", "activity") or {}).get("at", 0))
    return ACTIVE_POLL_MS if stamp - _activity["at"] < ACTIVE_WINDOW else IDLE_POLL_MS


# ---- tokens ------------------------------------------------------------------------------------


def _hash(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


class Worker(BaseModel):
    id: str
    name: str
    owner_uid: str
    owner_email: str
    owner_is_admin: bool = False
    shared_uids: list[str] = []
    shared_everyone: bool = False


def current_worker(request: Request) -> Worker:
    header = request.headers.get("authorization", "")
    token = header.removeprefix("Bearer ").strip()
    if not token.startswith("gw_") or token.count("_") < 2:
        raise HTTPException(401, "Missing or malformed worker token")
    _prefix, worker_id, secret = token.split("_", 2)
    with _lock:
        cached = _cache.get(worker_id)
    if cached and time.time() - cached[0] < 60:
        doc = cached[1]
    else:
        doc = _store().get("workers", worker_id) or {}
        with _lock:
            _cache[worker_id] = (time.time(), doc)
    if not doc or doc.get("revoked") or not hmac.compare_digest(doc.get("token_hash", ""), _hash(secret)):
        raise HTTPException(401, "This worker token is not valid (revoked or wrong)")
    worker = worker_from_doc(worker_id, doc)
    if worker is None:
        raise HTTPException(403, "The owner of this worker is disabled")
    return worker


def worker_from_doc(worker_id: str, doc: dict) -> Worker | None:
    """The worker a workers/{id} document describes; None when its owner is disabled."""
    owner = auth.load_user(doc["owner_uid"]) if doc.get("owner_uid", "").startswith("g_") else None
    owner_is_admin = (owner or {}).get("role") == "admin" or doc.get("owner_email") in auth.admin_emails() or auth.mode() == "dev"
    if owner is not None and owner.get("status") == "disabled":
        return None
    return Worker(
        id=worker_id, name=doc.get("name", worker_id), owner_uid=doc["owner_uid"], owner_email=doc.get("owner_email", ""),
        owner_is_admin=owner_is_admin, shared_uids=doc.get("shared_uids") or [], shared_everyone=bool(doc.get("shared_everyone")),
    )


def _require_owner_worker(worker: Worker) -> None:
    if not worker.owner_is_admin:
        raise HTTPException(403, "Only workers owned by an admin may publish gyms")


class NewWorker(BaseModel):
    name: str = Field(min_length=1, max_length=60)


@router.post("/api/workers")
def create_worker(body: NewWorker, admin: User = Depends(require_admin)):
    worker_id = secrets.token_hex(6)
    secret = secrets.token_urlsafe(32)
    _store().set(
        "workers", worker_id,
        {"name": body.name.strip(), "owner_uid": admin.uid, "owner_email": admin.email, "token_hash": _hash(secret),
         "created_at": now(), "last_seen": None, "revoked": False},
    )
    # The only time the token is shown.
    return {"id": worker_id, "name": body.name.strip(), "token": f"gw_{worker_id}_{secret}"}


@router.get("/api/workers")
def list_workers(_admin: User = Depends(require_admin)):
    workers = []
    for worker_id, doc in _store().recent("workers", "created_at", 100):
        if doc.get("revoked"):
            continue
        workers.append({key: doc.get(key) for key in ("name", "owner_email", "created_at", "last_seen")} | {"id": worker_id})
    return {"workers": workers}


@router.delete("/api/workers/{worker_id}")
def revoke_worker(worker_id: str, _admin: User = Depends(require_admin)):
    store = _store()
    if store.get("workers", worker_id) is None:
        raise HTTPException(404, "No such worker")
    return _revoke(worker_id)


# ---- the worker API --------------------------------------------------------------------------


class Heartbeat(BaseModel):
    capabilities: dict | None = None
    name: str = Field(default="", max_length=80)
    host: str = Field(default="", max_length=120)
    gpu: str = Field(default="", max_length=120)
    platform: str = Field(default="", max_length=120)
    gyms: list[str] = Field(default_factory=list, max_length=50)
    busy: str | None = None
    repo: str = Field(default="", max_length=400)
    stopping: bool = False
    pool_report: dict | None = None  # a pool bridge's view of its pool, for the Compute page (gymhub/poolview.py)


@router.post("/v1/worker/heartbeat")
def heartbeat(body: Heartbeat, worker: Worker = Depends(current_worker)):
    store = _store()
    fields = body.model_dump(exclude={"stopping", "pool_report"})
    # Only an admin's machine may tell everyone which pool machines grade their jobs.
    report = poolview.clean_report(body.pool_report) if worker.owner_is_admin else None
    if report is not None:
        fields["pool_report"] = report
    at = now()
    # One read per heartbeat (every 30 s) for the online time behind utilization on the Admin page.
    online = health.add_online(store.get("runners", worker.id), at)
    fields.update(name=worker.name, worker_id=worker.id, owner_email=worker.owner_email,
                  heartbeat_at=0 if body.stopping else at, online_slots=online)
    store.set("runners", worker.id, fields)
    _capabilities[worker.id] = (time.time(), body.capabilities)
    if time.time() - _last_seen.get(worker.id, 0) > LAST_SEEN_EVERY_SECONDS:
        store.set("workers", worker.id, {"last_seen": now()}, merge=True)
        _last_seen[worker.id] = time.time()
    return {"worker": worker.id, "poll_ms": poll_interval_ms(), "can_publish": worker.owner_is_admin}


class Catalog(BaseModel):
    gym: str = Field(max_length=40)
    meta: dict
    problems: list[dict] = Field(max_length=500)


@router.post("/v1/worker/catalog")
def publish_catalog(body: Catalog, worker: Worker = Depends(current_worker)):
    _require_owner_worker(worker)
    store = _store()
    if (store.get("gyms", body.gym) or {}).get("source") == "bundle":
        raise HTTPException(409, f"{body.gym!r} is an uploaded gym; a machine's local gym cannot replace it")
    slugs = set()
    for problem in body.problems:
        slug = str(problem.get("slug", ""))
        if not slug or "/" in slug or "__" in slug:
            raise HTTPException(400, f"bad exercise name {slug!r}")
        slugs.add(slug)
        store.set("problems", problem_id(body.gym, slug), {**problem, "gym": body.gym, "updated_at": now()})
    for doc_id, _doc in store.where("problems", "gym", body.gym):
        if doc_id.split("__", 1)[1] not in slugs:
            store.delete("problems", doc_id)
    # merge: keep settings made on the board (visibility, tags)
    store.set("gyms", body.gym, {**body.meta, "id": body.gym, "problems": len(slugs), "runner": worker.name, "updated_at": now()}, merge=True)
    return {"ok": True, "problems": len(slugs)}


class Code(BaseModel):
    gym: str
    slug: str
    current: dict


@router.post("/v1/worker/code")
def publish_code(body: Code, worker: Worker = Depends(current_worker)):
    _require_owner_worker(worker)
    store = _store()
    doc_id = problem_id(body.gym, body.slug)
    if store.get("problems", doc_id) is not None:
        store.set("problems", doc_id, {"current": body.current, "updated_at": now()}, merge=True)
    return {"ok": True}


class Progress(BaseModel):
    gym: str
    data: dict


@router.post("/v1/worker/progress")
def publish_progress(body: Progress, worker: Worker = Depends(current_worker)):
    _require_owner_worker(worker)
    _store().set("progress", body.gym, {"gym": body.gym, "data": body.data, "updated_at": now()})
    return {"ok": True}


class LeaseRequest(BaseModel):
    gyms: list[str] = Field(max_length=50)


@router.post("/v1/worker/lease")
def lease(body: LeaseRequest, worker: Worker = Depends(current_worker)):
    store = _store()
    if time.time() - _sweep["at"] > SWEEP_EVERY_SECONDS:
        _sweep["at"] = time.time()
        jobqueue.sweep(store)
    capabilities = _worker_capabilities(store, worker.id)
    job = jobqueue.lease(store, body.gyms, worker.id, capabilities=capabilities, skip=_lease_rules(store, worker, capabilities),
                         name=worker.name)
    return {"job": job, "poll_ms": poll_interval_ms()}


def _worker_capabilities(store, worker_id: str) -> dict | None:
    """From the worker's latest heartbeat on this instance; read from Firestore only when unknown or stale."""
    cached = _capabilities.get(worker_id)
    if cached and time.time() - cached[0] < 120:
        return cached[1]
    capabilities = (store.get("runners", worker_id) or {}).get("capabilities")
    _capabilities[worker_id] = (time.time(), capabilities)
    return capabilities


def may_run_for(worker: Worker, uid: str | None) -> bool:
    """Whose jobs this machine may take: its owner's, and those of the people its owner shared it with
    (everyone, or by email). Private by default, an admin's machine too: a pool bridge sends only its
    owner's jobs to the owner's pool machines until the owner shares it ("Share the pool" on the
    Compute page, POST /api/my/workers/{id}/sharing)."""
    if not uid:  # a job from before sign-in (no submitter): the board owner's, as before
        return worker.owner_is_admin
    return uid == worker.owner_uid or worker.shared_everyone or uid in worker.shared_uids


def is_shared(worker: Worker) -> bool:
    return worker.shared_everyone or bool(worker.shared_uids)


def _lease_rules(store, worker: Worker, capabilities: dict | None, head_start: bool = True):
    """skip(job): the rules of who may run what on this worker. `head_start`: also leave a fresh job
    to its submitter's own machine (off when only asking whether the worker could take it)."""
    prefer_own = _prefer_own_machine(store, worker) if is_shared(worker) and head_start else None
    gym_docs: dict[str, dict | None] = {}

    def may_hold_gym(job: dict) -> bool:
        """An uploaded gym's bundle (graders, hidden tests) reaches this machine only if the gym is
        public or the machine's owner has a role on it (#117). Local gyms run from files the machine
        already has. A pool bridge passes a gym on only to its own pool machines unless the leased
        job says `gym_public` (gymhub/jobqueue.py sets it; gymhub/runner/pooled.py)."""
        if worker.owner_is_admin or not job.get("bundle"):
            return True
        gym = job.get("gym", "")
        if gym not in gym_docs:
            gym_docs[gym] = store.get("gyms", gym)
        doc = gym_docs[gym]
        if doc is None:
            return False
        return gymaccess.visibility(doc) == "public" or gymaccess.role(worker.owner_uid, False, doc) is not None

    def skip(job: dict) -> bool:
        uid = job.get("uid")
        # The verdict that publishes a gym comes from the machine that validates it, so only an
        # admin's machine does: never a member's own, or one shared with them. Validating runs the
        # uploader's reference code, so another uploader's version waits for an admin machine shared
        # with them (may_run_for).
        if job.get("action") == "validate" and not worker.owner_is_admin:
            return True
        if not may_run_for(worker, uid):
            return True
        if not may_hold_gym(job):
            return True
        # Someone else's code runs only where it is isolated: a pool bridge (gymhub/routing.py) sends
        # it to the pool, whose containers isolate it. (Runners from before #132 report a sandbox.)
        sandboxed = bool((routing.for_job(capabilities, job) or {}).get("sandbox"))
        if job.get("isolated") and uid != worker.owner_uid and not sandboxed:
            return True
        return bool(prefer_own and prefer_own(job))

    return skip


OWN_MACHINE_HEAD_START = 15  # seconds a member's own online machine gets before shared machines take their job


def _prefer_own_machine(store, worker: "Worker"):
    """For shared machines: leave a fresh job to its submitter's own machine if that one is online and able."""
    own_cache: dict[str, list[dict]] = {}

    def own_machines(uid: str) -> list[dict]:
        if uid not in own_cache:
            machines = []
            for worker_id, doc in store.where("workers", "owner_uid", uid):
                runner = store.get("runners", worker_id) or {}
                if not doc.get("revoked") and worker_id != worker.id and time.time() - runner.get("heartbeat_at", 0) < 90:
                    machines.append(runner)
            own_cache[uid] = machines
        return own_cache[uid]

    def skip(job: dict) -> bool:
        if not job.get("uid") or time.time() - job.get("created_at", 0) > OWN_MACHINE_HEAD_START:
            return False
        if job.get("target"):  # the learner chose where it runs ("Run on")
            return False
        if job.get("action") == "validate":  # an own machine may not validate (see _lease_rules): no head start
            return False
        return any(
            routing.serves(runner, job.get("gym", ""), bool(job.get("bundle")))
            and routing.unmet(routing.for_job(runner.get("capabilities"), job), job.get("requires")) is None
            for runner in own_machines(job["uid"])
        )

    return skip


class JobUpdate(BaseModel):
    output: str | None = Field(default=None, max_length=400_000)
    lease_id: str | None = Field(default=None, max_length=80)  # which lease of the job this worker holds
    host: str | None = Field(default=None, max_length=80)  # a pool bridge: the pool machine running it


@router.post("/v1/worker/jobs/{job_id}/update")
def update(job_id: str, body: JobUpdate, worker: Worker = Depends(current_worker)):
    return {"stop": jobqueue.update(_store(), job_id, worker.id, body.output, body.lease_id, host=body.host)}


class JobComplete(BaseModel):
    final: dict
    user_progress: dict | None = None
    lease_id: str | None = Field(default=None, max_length=80)


@router.post("/v1/worker/jobs/{job_id}/bundle")
def job_bundle(job_id: str, worker: Worker = Depends(current_worker)):
    """An uploaded gym's bundle, only to the worker currently running a job that needs it."""
    from gymhub.blobs import BlobMissing, open_blobs
    from gymhub.gymversions import bundle_for_job

    job = _store().get("jobs", job_id)
    if not job or job.get("runner") != worker.id or job.get("status") != "running":
        raise HTTPException(409, "This worker does not hold that job")
    bundle = bundle_for_job(job)
    if bundle is None:
        raise HTTPException(404, "That job has no bundle")
    try:
        data = open_blobs().get(bundle["key"])
    except BlobMissing:
        raise HTTPException(404, "The bundle is missing from storage") from None
    return Response(content=data, media_type="application/gzip")


@router.post("/v1/worker/jobs/{job_id}/complete")
def complete(job_id: str, body: JobComplete, worker: Worker = Depends(current_worker)):
    if not jobqueue.complete(_store(), job_id, worker.id, body.final, body.user_progress, lease_id=body.lease_id):
        return Response(status_code=409)
    return {"ok": True}


# ---- registering a machine: one-time code -> bundle + token -----------------------------------
#
# registrations/{code}   uid, email, name, expires_at, used


REGISTRATION_SECONDS = 10 * 60
INSTALL_SCRIPT = Path(__file__).parent / "install.sh"
_bundle_cache: dict[str, bytes] = {}


def _bundle_bytes() -> bytes:
    """The bundle baked into the image at deploy time, or built from this checkout in development."""
    if "data" not in _bundle_cache:
        baked = Path(os.environ.get("GYMHUB_BUNDLE", "/app/worker-bundle.tar.gz"))
        _bundle_cache["data"] = baked.read_bytes() if baked.is_file() else bundle.build()
    return _bundle_cache["data"]


def _new_code() -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O, 1/I
    raw = "".join(secrets.choice(alphabet) for _ in range(8))
    return f"{raw[:4]}-{raw[4:]}"


class Registration(BaseModel):
    name: str = Field(min_length=1, max_length=40, pattern=r"^[A-Za-z0-9._-]+$")


@router.post("/api/my/workers/registration")
def start_registration(body: Registration, user: User = Depends(current_user)):
    # The machine bundle is the runner's code; adding a machine is running code, so the same rule applies.
    if not auth.can_run(user):
        raise HTTPException(403, "Running code is not enabled for members on this board yet, so you cannot add a machine.")
    code = _new_code()
    _store().set(
        "registrations", code,
        {"uid": user.uid, "email": user.email, "name": body.name.strip(), "expires_at": now() + REGISTRATION_SECONDS,
         "used": False, "created_at": now(), "expire_at": expires_in(86400)},  # expire_at: deleted by Firestore's TTL
    )
    # Behind IAP machines use a second, worker-only service (deploy/gcp.sh): the install command points there.
    return {"code": code, "expires_in": REGISTRATION_SECONDS, "install_url": os.environ.get("GYMHUB_WORKER_URL", "")}


@router.get("/api/my/workers")
def my_workers(user: User = Depends(current_user)):
    store = _store()
    mine = []
    for worker_id, doc in store.where("workers", "owner_uid", user.uid):
        if doc.get("revoked"):
            continue
        runner = store.get("runners", worker_id) or {}
        online = time.time() - runner.get("heartbeat_at", 0) < 90
        recent = sorted(store.where("jobs", "runner", worker_id), key=lambda item: item[1].get("created_at", 0), reverse=True)[:8]
        for_others = [
            {"when": job.get("created_at"), "who": job.get("created_by"), "exercise": f"{job.get('gym')}/{job.get('slug')}",
             "action": job.get("action"), "status": job.get("status"), "ok": job.get("ok"), "host": job.get("host")}
            for _id, job in recent if job.get("uid") != user.uid
        ]
        caps = runner.get("capabilities") or {}
        mine.append({"id": worker_id, "name": doc.get("name"), "created_at": doc.get("created_at"), "last_seen": doc.get("last_seen"),
                     "online": online, "gpu": runner.get("gpu"), "platform": runner.get("platform"), "busy": runner.get("busy"),
                     "sandbox": bool(caps.get("sandbox")), "pool": bool(caps.get("pool")), "capabilities": caps or None,
                     "shared_emails": doc.get("shared_emails") or [],
                     "dispatcher": bool(caps.get("dispatch_only") or (caps.get("pool") or {}).get("dispatch_only")),
                     "shared_everyone": bool(doc.get("shared_everyone")), "jobs_for_others": for_others})
    mine.sort(key=lambda item: item.get("created_at") or 0)
    return {"workers": mine, "runs": "only your jobs", "can_add": can_add_machines()}


def can_add_machines() -> bool:
    """Behind IAP a machine reaches the board only through the worker-only service (GYMHUB_WORKER_URL)."""
    return auth.mode() != "iap" or bool(os.environ.get("GYMHUB_WORKER_URL"))


class Sharing(BaseModel):
    emails: list[str] = Field(default_factory=list, max_length=200)
    everyone: bool = False


@router.post("/api/my/workers/{worker_id}/sharing")
def share_my_worker(worker_id: str, body: Sharing, user: User = Depends(current_user)):
    store = _store()
    doc = store.get("workers", worker_id)
    if doc is None or doc.get("revoked") or doc.get("owner_uid") != user.uid:
        raise HTTPException(404, "No such machine")
    emails = sorted({email.strip().lower() for email in body.emails if email.strip()})
    uids, unknown = [], []
    for email in emails:
        found = [uid for uid, item in store.where("users", "email", email) if item.get("status") == "active"]
        (uids.extend(found) if found else unknown.append(email))
    if unknown:
        raise HTTPException(400, f"Not on this board yet: {', '.join(unknown)}. They need to sign in once first.")
    store.set("workers", worker_id, {"shared_uids": uids, "shared_emails": emails, "shared_everyone": body.everyone}, merge=True)
    with _lock:
        _cache.pop(worker_id, None)
    return {"ok": True}


@router.delete("/api/my/workers/{worker_id}")
def remove_my_worker(worker_id: str, user: User = Depends(current_user)):
    store = _store()
    doc = store.get("workers", worker_id)
    if doc is None or (doc.get("owner_uid") != user.uid and not user.is_admin):
        raise HTTPException(404, "No such machine")
    return _revoke(worker_id)


def _revoke(worker_id: str) -> dict:
    store = _store()
    store.set("workers", worker_id, {"revoked": True, "token_hash": ""}, merge=True)
    store.set("runners", worker_id, {"heartbeat_at": 0}, merge=True)
    with _lock:
        _cache.pop(worker_id, None)
    # The machine can no longer report, so its running jobs end here: a cancelled one is closed,
    # anything else goes back to the queue for another machine (the old lease id stops counting).
    for job_id, job in store.where("jobs", "runner", worker_id):
        if job.get("status") != "running":
            continue
        note = "\n(The machine running this job was removed from the board.)\n"
        if job.get("cancel"):
            update = {"status": "cancelled", "ok": False, "finished_at": now(), "lease_expires_at": None}
        elif job.get("tries", 1) >= jobqueue.MAX_TRIES:  # as sweep does: a job never bounces between machines forever
            jobqueue._end_without_result(store, job_id, job, job.get("lease_expires_at"), "error",
                                         note.rstrip("\n") + f" It had run {jobqueue.MAX_TRIES} times; it was stopped.\n")
            continue
        else:
            update = {"status": "queued", "runner": None, "host": None, "via": None, "lease_id": None, "lease_expires_at": None}
        if not store.claim("jobs", job_id, {"status": "running", "runner": worker_id},
                           {**update, "output": (job.get("output") or "") + note}):
            continue
        if update["status"] == "cancelled":
            health.log_finished(job_id, job, update)
        else:
            health.log_requeued(job_id, job, "machine removed")
    return {"ok": True}


def _valid_registration(code: str) -> dict:
    doc = _store().get("registrations", code)
    if not doc or doc.get("used") or doc.get("expires_at", 0) < now():
        raise HTTPException(403, "This registration code is not valid. Create a new one on the board's Compute page.")
    return doc


@router.get("/install.sh", response_class=PlainTextResponse)
def install_script(request: Request):
    board = os.environ.get("GYMHUB_PUBLIC_URL") or str(request.base_url).rstrip("/")
    if request.headers.get("x-forwarded-proto") == "https":
        board = board.replace("http://", "https://", 1)
    return PlainTextResponse(INSTALL_SCRIPT.read_text().replace("__BOARD__", board), media_type="text/x-shellscript")


@router.get("/v1/register/bundle.tar.gz")
def download_bundle(code: str):
    _valid_registration(code)
    return Response(_bundle_bytes(), media_type="application/gzip", headers={"Content-Disposition": "attachment; filename=gymhub-worker.tar.gz"})


class Register(BaseModel):
    code: str = Field(max_length=20)
    host: str = Field(default="", max_length=120)


@router.post("/v1/register")
def register(body: Register):
    store = _store()
    doc = _valid_registration(body.code)
    if not store.claim("registrations", body.code, {"used": False}, {"used": True, "used_at": now()}):
        raise HTTPException(403, "This registration code was just used.")
    worker_id = secrets.token_hex(6)
    secret = secrets.token_urlsafe(32)
    store.set(
        "workers", worker_id,
        {"name": doc["name"], "owner_uid": doc["uid"], "owner_email": doc["email"], "token_hash": _hash(secret),
         "created_at": now(), "last_seen": None, "revoked": False, "host": body.host},
    )
    return {"worker_id": worker_id, "token": f"gw_{worker_id}_{secret}", "name": doc["name"]}
