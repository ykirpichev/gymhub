"""The pool server (#45 sections 3 and 11), M2: several pools, roles and quotas, chunked artifacts,
ModelRefs, machine profiles and use policies, placement with a network check and cache affinity.

    python -m pool.cli server --data ~/pool-data [--db postgresql://...] [--port 8765] [--public-url URL]

The first start creates the pool "default" and prints a one-time code for its first admin.

Callers, all with bearer tokens:
- members (pm_...): the CLI and SDK. Roles per pool: admin (members, quotas, settings), member
  (submit jobs, add machines), viewer (read only). The first member is also the server admin, who
  may create pools.
- machines (pa_...): agents. Outbound only: heartbeats (with their cache), long-poll leases, logs,
  results. They read only the artifacts of jobs they are running.

The server is also the pool's directory and chunk tracker (pool/peers.py): agents report their LAN
address and peer port (and their iroh sidecar's address), ask who their peers are, report measured
links (islands come from these), announce the chunks they hold, get the holders of the chunks they
need with a ticket per chunk, and meet the other ranks of a gang through the rendezvous. Holders
are offered only for chunks of an artifact the asking machine may read, so the tracker never tells
anyone what else a machine caches.

Uploads are chunked and deduplicated, but only against chunks the uploader has proven to hold:
asking whether the server has some bytes never reveals that someone else uploaded them.
"""

import asyncio
import hashlib
import gzip
import hmac
import io
import ipaddress
import json
import os
import re
import secrets
import shlex
import sys
import tarfile
import threading
import time
import urllib.parse
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

import pool
from pool import actors, coord, installer, metrics, migrations, rounds, trust, artifacts as arts, peers as peering, scheduler, spec as specs, update
from pool.artifacts import ArtifactError, ChunkStore
from pool.db import DB, FINAL, load_job, load_machine, new_id, now, open_db

CODE_DAYS = 7
POLL_MS = 10_000
LEASE_WAIT_MAX = 30
GC_SECONDS = 600
ROLES = ("admin", "member", "viewer")
KINDS = ("data", "code", "result", "model", "checkpoint", "log")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
SPEEDTEST = os.urandom(1 << 20)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def new_code() -> str:
    return "-".join(secrets.token_hex(3) for _ in range(3))  # 9a1b2c-3d4e5f-6a7b8c


def package_tarball() -> bytes:
    """The pool package as served to new machines (standard library only on the agent side)."""
    root = Path(pool.__file__).resolve().parent
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for path in sorted(root.glob("*.py")):
            info = tarfile.TarInfo(f"pool/{path.name}")
            data = path.read_bytes()
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            tar.addfile(info, io.BytesIO(data))
    # gzip with a fixed timestamp: the same files give the same bytes, so the sha256 the installer
    # checks (and `pool machine add` prints) stays put across server restarts.
    return gzip.compress(buffer.getvalue(), mtime=0)


def lan_host(host: str | None) -> bool:
    """A host on the LAN (private, loopback, link-local or CGNAT/Tailscale address, a dotless or .local name)."""
    if not host:
        return False
    host = host.strip("[]").lower().rstrip(".")
    if host == "localhost" or ("." not in host and ":" not in host):
        return True
    if host.endswith((".local", ".lan", ".internal", ".home.arpa", ".localhost")):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_private or address.is_loopback or address.is_link_local or address in ipaddress.ip_network("100.64.0.0/10")


def check_install_url(server: str) -> None:
    """The installer runs as root: refuse to hand out one fetched over plain http from a public address."""
    parts = urllib.parse.urlsplit(server)
    if parts.scheme == "http" and not lan_host(parts.hostname):
        raise HTTPException(400, f"{server} is plain http on a public address, and the installer runs as root: "
                                 "serve the pool over https (--public-url https://...), or use its LAN address")




class LoginBody(BaseModel):
    code: str
    name: str = ""


class JoinBody(BaseModel):
    code: str


class PoolBody(BaseModel):
    name: str


class SettingsBody(BaseModel):
    settings: dict


class MemberBody(BaseModel):
    role: str | None = None
    quota: dict | None = None


class UseBody(BaseModel):
    pool: str


class InviteBody(BaseModel):
    pool: str | None = None
    role: str = "member"


class MachineCodeBody(BaseModel):
    pool: str | None = None
    name: str = ""


class RegisterBody(BaseModel):
    code: str
    name: str
    labels: dict[str, str] = {}
    profile: dict = {}


class PolicyBody(BaseModel):
    who: str | list[str] = "pool"
    hours: str | None = None
    idle_only: bool = False
    owner_first: bool = True


class CacheReport(BaseModel):
    full: bool = False
    add: list[str] = []
    remove: list[str] = []


class HeartbeatBody(BaseModel):
    running: list[str] = []
    profile: dict | None = None
    labels: dict[str, str] | None = None
    cache: CacheReport | None = None
    free: dict | None = None


class LeaseBody(BaseModel):
    free: dict = {}
    wait: float = 25


class ProgressBody(BaseModel):
    job: str
    key: str | None = None
    text: str = ""


class CompleteBody(BaseModel):
    exit_code: int | None = None
    outputs: str | None = None
    error: str | None = None
    preempted: str | None = None


class ArtifactBody(BaseModel):
    manifest: dict
    pool: str | None = None
    kind: str = "data"
    job: str | None = None
    held: bool = False  # a running job's machine keeps the chunks and serves them to peers (tree rounds)


class ModelBody(BaseModel):
    name: str
    artifact: str
    meta: dict
    pool: str | None = None


class DrainBody(BaseModel):
    drain: bool = True


class LinksBody(BaseModel):
    links: list[dict] = []


class ChunksBody(BaseModel):
    add: list[str] = []
    remove: list[str] = []


class PeerSourcesBody(BaseModel):
    artifact: str
    chunks: list[str] = []


class CoordBody(BaseModel):
    """A refs or queues operation (pool/coord.py): op, the pool (members only), and its arguments."""
    op: str = Field(max_length=40)
    pool: str | None = Field(default=None, max_length=64)
    args: dict = Field(default_factory=dict)


SETTINGS_RANGES = {"alpha": (0.0, 100.0), "affinity_wait_s": (0, 3600), "retention_days": (1, 3650), "exclude_after": (1, 1000)}
QUOTA_KEYS = {"max_running_gpus", "max_running_jobs", "gpu_hours_per_day", "storage_gb"}
HOURS_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d-([01]\d|2[0-3]):[0-5]\d$")


def _check_quota(quota: dict) -> dict:
    if set(quota) - QUOTA_KEYS:
        raise HTTPException(400, f"quota keys: {', '.join(sorted(QUOTA_KEYS))}")
    for key, value in quota.items():
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0):
            raise HTTPException(400, f"quota {key} must be a number (or null for no limit)")
    return quota


def public_spec(spec: dict) -> dict:
    """A spec without the model metadata the server copied into it."""
    spec = json.loads(json.dumps(spec))
    spec.get("resources", {}).get("fits", {}).pop("meta", None)
    for item in spec.get("inputs", []):
        item.pop("meta", None)
    return spec


def create_app(data_dir: str | os.PathLike, public_url: str | None = None, max_artifact_mb: int = 512 * 1024,
               db_url: str | None = None, gc_seconds: float | None = GC_SECONDS, poll_ms: int = POLL_MS,
               metrics_token: str | None = None) -> FastAPI:
    data = Path(data_dir)
    db: DB = open_db(db_url or str(data / "pool.db"))
    chunks = ChunkStore(data / "chunks")
    max_artifact = max_artifact_mb * 1024**2
    package = package_tarball()
    package_sha256 = hashlib.sha256(package).hexdigest()
    agent_digest = update.package_digest(Path(pool.__file__).resolve().parent)
    app = FastAPI(title="pool", version=pool.__version__)
    app.state.db = db
    app.state.chunks = chunks
    queued = [0]  # bumped each time jobs are queued, so leases waiting for work look again at once

    async def nap(seconds: float, seen: int) -> None:
        """Sleep up to `seconds`, waking early once new jobs are queued (an in-memory check, no query)."""
        end = time.monotonic() + seconds
        while queued[0] == seen and time.monotonic() < end:
            await asyncio.sleep(0.05)
    app.state.bootstrap_code = None

    # ---- bootstrap: the default pool and its first admin -------------------------------------------
    with db.lock:
        if not db.one("select id from pools where name='default'"):
            db.run("insert into pools (id, name, settings, created_at) values (?, 'default', '{}', ?) on conflict do nothing",
                   new_id("p"), now())
        if not db.one("select id from members limit 1"):
            default = db.one("select id from pools where name='default'")["id"]
            code = new_code()
            db.run("insert into codes (code_hash, kind, pool, role, created_by, created_at, expires_at) "
                   "values (?, 'invite', ?, 'owner', '', ?, ?)", token_hash(code), default, now(), now() + CODE_DAYS * 86400)
            app.state.bootstrap_code = code

    # ---- auth and lookups ------------------------------------------------------------------------
    def bearer(authorization: str | None) -> str:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "missing token: run pool login")
        return authorization[7:].strip()

    def member(authorization: str | None = Header(None)) -> dict:
        token = bearer(authorization)
        found = db.one("select id, name, is_admin, default_pool from members where token_hash=?", token_hash(token)) if token.startswith("pm_") else None
        if not found:
            raise HTTPException(401, "unknown or revoked member token")
        return found

    def machine(request: Request, authorization: str | None = Header(None)) -> dict:
        token = bearer(authorization)
        row = db.one("select * from machines where token_hash=?", token_hash(token)) if token.startswith("pa_") else None
        if not row:
            raise HTTPException(401, "unknown or removed machine")
        db.run("update machines set last_seen=?, public_ip=? where id=?", now(), request.client.host if request.client else None, row["id"])
        return load_machine(row)

    def caller(request: Request, authorization: str | None = Header(None)) -> tuple[str, dict]:
        token = bearer(authorization)
        return ("machine", machine(request, authorization)) if token.startswith("pa_") else ("member", member(authorization))

    def role_in(pool_id: str, who: dict) -> str | None:
        row = db.one("select role from pool_members where pool=? and member=?", pool_id, who["id"])
        if row:
            return row["role"]
        return "admin" if who.get("is_admin") else None

    def resolve_pool(name: str | None, who: dict, need: str = "viewer") -> dict:
        """A pool by name or id (default: the member's default pool) where the member has at least `need`."""
        name = name or who.get("default_pool")
        if not name:
            mine = db.all("select p.id, p.name from pools p join pool_members m on m.pool=p.id where m.member=?", who["id"])
            if len(mine) != 1:
                raise HTTPException(400, "which pool? pass --pool (or run pool use <pool>)")
            name = mine[0]["id"]
        row = db.one("select * from pools where id=? or name=?", name, name)
        role = role_in(row["id"], who) if row else None
        if not row or role is None:
            raise HTTPException(404, f"no pool {name} of yours")
        if ROLES.index(role) > ROLES.index(need):
            raise HTTPException(403, f"you are a {role} of {row['name']}; this needs {need}")
        return {**row, "role": role}

    def use_code(code: str, kind: str) -> dict:
        with db.lock:
            row = db.one("select * from codes where code_hash=? and kind=?", token_hash(code.strip()), kind)
            if not row or row["used_at"] or row["expires_at"] < now():
                raise HTTPException(403, "this code is unknown, used or expired: ask for a new one")
            db.run("update codes set used_at=? where code_hash=?", now(), row["code_hash"])
            return row

    def add_membership(pool_id: str, member_id: str, role: str) -> None:
        db.run("insert into pool_members (pool, member, role, created_at) values (?, ?, ?, ?) on conflict do nothing",
               pool_id, member_id, role, now())

    def names() -> dict[str, str]:
        return {row["id"]: row["name"] for row in db.all("select id, name from members")}

    def pool_names() -> dict[str, str]:
        return {row["id"]: row["name"] for row in db.all("select id, name from pools")}

    def public_job(job: dict, who: dict | None = None) -> dict:
        machine_name = None
        if job.get("machine"):
            found = db.one("select name from machines where id=?", job["machine"])
            machine_name = found["name"] if found else job["machine"]
        out = {
            "id": job["id"], "name": job["name"], "pool": pool_names().get(job["pool"], job["pool"]),
            "owner": names().get(job["owner"], job["owner"]), "priority": job["priority"],
            "status": job["status"], "machine": machine_name, "gpus": job["gpus"], "tries": job["tries"],
            "preemptions": job["preemptions"], "exit_code": job["exit_code"], "error": job["error"],
            "outputs": job["outputs"], "placement": job["placement"], "created_at": job["created_at"],
            "started_at": job["started_at"], "finished_at": job["finished_at"], "spec": public_spec(job["spec"]),
        }
        gang = scheduler.gang_of(job)
        if gang:
            out["gang"] = {**gang, "jobs": [row["job"] for row in db.all("select job from gang_members where gang=? order by rank", gang["id"])]}
        return out

    def own_job(job_id: str, who: dict) -> dict:
        job = load_job(db.one("select * from jobs where id=?", job_id))
        if not job or (job["owner"] != who["id"] and role_in(job["pool"], who) != "admin"):
            raise HTTPException(404, f"no job {job_id}")
        return job

    def member_reads(artifact: str, member_id: str, is_admin: bool = False) -> bool:
        if is_admin:
            return bool(db.one("select 1 as x from artifacts where id=?", artifact))
        if db.one("select 1 as x from artifact_owners where artifact=? and member=?", artifact, member_id):
            return True
        # models and refs are shared with their pool
        if db.one("select 1 as x from models m join pool_members p on p.pool=m.pool where m.artifact=? and p.member=?",
                  artifact, member_id):
            return True
        if db.one("select 1 as x from refs r join pool_members p on p.pool=r.pool where r.artifact=? and p.member=?",
                  artifact, member_id):
            return True
        # contributions to a round are shared with the pool, for whoever reduces it
        return bool(db.one("select 1 as x from contributions c join pool_members p on p.pool=c.pool where c.artifact=? and p.member=?",
                           artifact, member_id))

    def may_read(artifact: str, kind: str, who: dict) -> bool:
        if kind == "member":
            return member_reads(artifact, who["id"], bool(who.get("is_admin")))
        running = db.all("select spec, pool from jobs where machine=? and status='running'", who["id"])
        for row in running:
            if artifact in specs.artifacts_of(json.loads(row["spec"])):
                return True
        # a ref of the pool one of its jobs runs in (pool/coord.py), or a checkpoint of one of its jobs
        pools = sorted({row["pool"] for row in running})
        return (coord.ref_artifact_in_pools(db, artifact, pools) or actors.checkpoint_readable(db, artifact, who["id"])
                or rounds.readable(db, artifact, pools))

    def peer_key(pool_id: str) -> str:
        """The pool's key for agent-to-agent requests (pool/peers.py); made on first use."""
        with db.lock:
            row = db.one("select value from meta where key=?", f"peer_key:{pool_id}")
            if row:
                return row["value"]
            key = secrets.token_hex(32)
            db.run("insert into meta (key, value) values (?, ?) on conflict do nothing", f"peer_key:{pool_id}", key)
            return db.one("select value from meta where key=?", f"peer_key:{pool_id}")["value"]

    def online_peers(pool_id: str, exclude: str | None = None) -> list[dict]:
        out = []
        for row in db.all("select * from machines where pool=? and last_seen > ? order by name",
                          pool_id, now() - scheduler.ONLINE_SECONDS):
            m = load_machine(row)
            peer = (m.get("profile") or {}).get("peer") or {}
            if m["id"] != exclude and peer.get("addr") and peer.get("port"):
                found = {"id": m["id"], "name": m["name"], "addr": peer["addr"], "port": peer["port"]}
                iroh = peer.get("iroh")
                if (isinstance(iroh, dict) and isinstance(iroh.get("id"), str) and isinstance(iroh.get("addr"), dict)
                        and len(json.dumps(iroh)) <= 4096):
                    found["iroh"] = {"id": iroh["id"], "addr": iroh["addr"]}  # its iroh-blobs sidecar (peers.IrohSidecar)
                out.append(found)
        return out

    def storage_used(pool_id: str, member_id: str) -> int:
        return int(db.one("select coalesce(sum(a.size), 0) as n from artifact_owners o join artifacts a on a.id=o.artifact "
                      "where o.member=? and o.pool=?", member_id, pool_id)["n"])

    # ---- public -----------------------------------------------------------------------------------
    @app.get("/healthz")
    def healthz():
        return {"ok": True, "version": pool.__version__, "agent": agent_digest, "package_sha256": package_sha256}

    @app.get("/install.sh", response_class=PlainTextResponse)
    def install(request: Request):
        server = (public_url or str(request.base_url)).rstrip("/")
        if not re.match(r"^https?://[A-Za-z0-9.:\[\]_-]+(/[A-Za-z0-9._/-]*)?$", server):
            raise HTTPException(500, "set --public-url to the address machines use to reach this server")
        check_install_url(server)
        return installer.render(server, package_sha256)

    @app.get("/v1/agent/package.tar.gz")
    def agent_package():
        return Response(package, media_type="application/gzip")

    @app.post("/v1/login")
    def login(body: LoginBody):
        row = use_code(body.code, "invite")
        name = (body.name or "").strip()[:64] or f"member-{secrets.token_hex(2)}"
        if db.one("select 1 as x from members where name=?", name):
            name = f"{name}-{secrets.token_hex(2)}"
        member_id = new_id("m")
        token = f"pm_{member_id}_{secrets.token_urlsafe(24)}"
        owner = row["role"] == "owner"  # the bootstrap code: server admin and admin of the default pool
        db.run("insert into members (id, name, is_admin, token_hash, default_pool, created_at) values (?, ?, ?, ?, ?, ?)",
               member_id, name, int(owner), token_hash(token), row["pool"], now())
        add_membership(row["pool"], member_id, "admin" if owner else row["role"])
        return {"token": token, "member": {"id": member_id, "name": name, "is_admin": owner},
                "pool": pool_names().get(row["pool"])}

    @app.post("/v1/agent/register")
    def register(body: RegisterBody):
        if not NAME_RE.match(body.name):
            raise HTTPException(400, "machine name: letters, digits, '.', '_' and '-'")
        row = use_code(body.code, "machine")
        machine_id = new_id("a")
        token = f"pa_{machine_id}_{secrets.token_urlsafe(24)}"
        name = body.name
        if db.one("select 1 as x from machines where name=?", name):
            name = f"{name}-{machine_id[1:5]}"
        db.run("insert into machines (id, pool, name, owner, token_hash, labels, profile, created_at, last_seen) values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
               machine_id, row["pool"], name, row["created_by"], token_hash(token), json.dumps(body.labels), json.dumps(body.profile), now(), now())
        return {"machine_id": machine_id, "name": name, "token": token, "pool": pool_names().get(row["pool"])}

    # ---- members and pools ------------------------------------------------------------------------
    @app.get("/v1/me")
    def me(who: dict = Depends(member)):
        pools = db.all("select p.name, m.role from pools p join pool_members m on m.pool=p.id where m.member=? order by p.name", who["id"])
        return {"member": {**who, "default_pool": pool_names().get(who["default_pool"])}, "pools": pools}

    @app.post("/v1/join")
    def join(body: JoinBody, who: dict = Depends(member)):
        row = use_code(body.code, "invite")
        add_membership(row["pool"], who["id"], "admin" if row["role"] == "owner" else row["role"])
        return {"pool": pool_names().get(row["pool"])}

    @app.post("/v1/use")
    def use(body: UseBody, who: dict = Depends(member)):
        target = resolve_pool(body.pool, who)
        db.run("update members set default_pool=? where id=?", target["id"], who["id"])
        return {"pool": target["name"]}

    @app.post("/v1/pools")
    def create_pool(body: PoolBody, who: dict = Depends(member)):
        if not who["is_admin"]:
            raise HTTPException(403, "only the server admin creates pools")
        if not NAME_RE.match(body.name) or db.one("select 1 as x from pools where name=?", body.name):
            raise HTTPException(400, "pool names are unique: letters, digits, '.', '_' and '-'")
        pool_id = new_id("p")
        db.run("insert into pools (id, name, settings, created_at) values (?, ?, '{}', ?)", pool_id, body.name, now())
        add_membership(pool_id, who["id"], "admin")
        return {"pool": {"id": pool_id, "name": body.name}}

    @app.get("/v1/pools")
    def list_pools(who: dict = Depends(member)):
        rows = db.all("select p.id, p.name, m.role from pools p join pool_members m on m.pool=p.id where m.member=? order by p.name", who["id"])
        out = []
        for row in rows:
            out.append({**row, "members": db.one("select count(*) as n from pool_members where pool=?", row["id"])["n"],
                        "machines": db.one("select count(*) as n from machines where pool=?", row["id"])["n"],
                        "default": row["id"] == who["default_pool"]})
        return {"pools": out}

    @app.get("/v1/pools/{pool_name}")
    def get_pool(pool_name: str, who: dict = Depends(member)):
        target = resolve_pool(pool_name, who)
        settings = scheduler.pool_settings(db, target["id"])
        used = scheduler.usage(db, target["id"])
        members = []
        for row in db.all("select m.id, m.name, p.role, p.quota from pool_members p join members m on m.id=p.member where p.pool=? order by m.name", target["id"]):
            if target["role"] != "admin" and row["id"] != who["id"]:
                continue
            members.append({"name": row["name"], "role": row["role"], "quota": scheduler.member_quota(db, target["id"], row["id"], settings),
                            "gpu_hours_24h": round(used.get(row["id"], 0.0) / 3600, 3),
                            "storage_gb": round(storage_used(target["id"], row["id"]) / 1024**3, 3)})
        return {"pool": {"name": target["name"], "role": target["role"], "settings": settings, "members": members}}

    @app.patch("/v1/pools/{pool_name}")
    def update_pool(pool_name: str, body: SettingsBody, who: dict = Depends(member)):
        target = resolve_pool(pool_name, who, need="admin")
        saved = json.loads(target["settings"] or "{}")
        for key, value in body.settings.items():
            if key == "quota":
                saved["quota"] = {**(saved.get("quota") or {}), **_check_quota(value or {})}
            elif key == "placement":
                if value not in scheduler.PLACEMENTS:
                    raise HTTPException(400, f"placement is one of {', '.join(scheduler.PLACEMENTS)}")
                saved["placement"] = value
            elif key == "trust":
                if value not in trust.TRUST:
                    raise HTTPException(400, f"trust is one of {', '.join(trust.TRUST)}")
                saved["trust"] = value
            elif key in SETTINGS_RANGES:
                low, high = SETTINGS_RANGES[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
                    raise HTTPException(400, f"{key} must be from {low} to {high}")
                saved[key] = value
            else:
                raise HTTPException(400, f"settings: {', '.join(sorted([*SETTINGS_RANGES, 'placement', 'quota', 'trust']))}")
        db.run("update pools set settings=? where id=?", json.dumps(saved), target["id"])
        return {"settings": scheduler.pool_settings(db, target["id"])}

    @app.put("/v1/pools/{pool_name}/members/{member_name}")
    def update_member(pool_name: str, member_name: str, body: MemberBody, who: dict = Depends(member)):
        target = resolve_pool(pool_name, who, need="admin")
        found = db.one("select m.id, p.quota from members m join pool_members p on p.member=m.id where p.pool=? and m.name=?", target["id"], member_name)
        if not found:
            raise HTTPException(404, f"{member_name} is not in {target['name']}")
        if body.role is not None:
            if body.role not in ROLES:
                raise HTTPException(400, "role is admin, member or viewer")
            db.run("update pool_members set role=? where pool=? and member=?", body.role, target["id"], found["id"])
        if body.quota is not None:
            quota = {**json.loads(found["quota"] or "{}"), **_check_quota(body.quota)}
            db.run("update pool_members set quota=? where pool=? and member=?", json.dumps(quota), target["id"], found["id"])
        return {"ok": True}

    @app.post("/v1/invites")
    def invite(body: InviteBody, who: dict = Depends(member)):
        target = resolve_pool(body.pool, who, need="admin")
        if body.role not in ROLES:
            raise HTTPException(400, "role is admin, member or viewer")
        code = new_code()
        db.run("insert into codes (code_hash, kind, pool, role, created_by, created_at, expires_at) values (?, 'invite', ?, ?, ?, ?, ?)",
               token_hash(code), target["id"], body.role, who["id"], now(), now() + CODE_DAYS * 86400)
        return {"code": code, "pool": target["name"], "expires_at": now() + CODE_DAYS * 86400}

    # ---- machines ---------------------------------------------------------------------------------
    @app.post("/v1/machine-codes")
    def machine_code(body: MachineCodeBody, request: Request, who: dict = Depends(member)):
        target = resolve_pool(body.pool, who, need="member")
        server = (public_url or str(request.base_url)).rstrip("/")
        check_install_url(server)
        code = new_code()
        db.run("insert into codes (code_hash, kind, pool, created_by, name, created_at, expires_at) values (?, 'machine', ?, ?, ?, ?, ?)",
               token_hash(code), target["id"], who["id"], body.name, now(), now() + CODE_DAYS * 86400)
        install = f"curl -fsSL {server}/install.sh | sudo sh -s -- --code {code}" + (f" --name {shlex.quote(body.name)}" if body.name else "")
        return {"code": code, "pool": target["name"], "expires_at": now() + CODE_DAYS * 86400, "install": install,
                "install_no_sudo": install.replace("| sudo sh -s -- ", "| sh -s -- --no-sudo "), "sha256": package_sha256}

    @app.get("/v1/machines")
    def machines(pool: str | None = None, who: dict = Depends(member)):
        target = resolve_pool(pool, who)
        member_names = names()
        out = []
        for row in db.all("select * from machines where pool=? order by name", target["id"]):
            m = load_machine(row)
            running = db.all("select id from jobs where machine=? and status='running'", m["id"])
            cached = db.one("select count(*) as n, coalesce(sum(c.size), 0) as b from chunk_locations l join chunks c on c.sha=l.chunk where l.machine=?", m["id"])
            out.append({
                "id": m["id"], "name": m["name"], "owner": member_names.get(m["owner"], m["owner"]), "labels": m["labels"],
                "mine": m["owner"] == who["id"],  # names can repeat: what resources.machines: own admits
                "online": bool(m["last_seen"] and now() - m["last_seen"] < scheduler.ONLINE_SECONDS),
                "last_seen": m["last_seen"], "drain": bool(m["drain"]), "running": [r["id"] for r in running],
                "profile": m["profile"], "policy": m["policy"], "free": m["free"], "public_ip": m["public_ip"],
                "cached_gb": round(int(cached["b"]) / 1024**3, 2),
                "reputation": trust.standing(db, target["id"], m["id"]),
                "agent": update.agent_version(m["profile"], agent_digest),  # current, outdated or unknown
            })
        return {"machines": out, "pool": target["name"], "agent": agent_digest}

    @app.get("/v1/links")
    def links(pool: str | None = None, who: dict = Depends(member)):
        """Measured machine-to-machine links of a pool, and the islands they make."""
        target = resolve_pool(pool, who)
        machines_here = {row["id"]: row["name"] for row in db.all("select id, name from machines where pool=?", target["id"])}
        found = [{"src": machines_here[r["src"]], "dst": machines_here[r["dst"]], "mbps": r["mbps"], "rtt_ms": r["rtt_ms"],
                  "measured_at": r["measured_at"]}
                 for r in db.all("select * from peer_links order by src, dst") if r["src"] in machines_here and r["dst"] in machines_here]
        return {"pool": target["name"], "links": found, "islands": peering.islands(sorted(machines_here.values()), found),
                "island_rule": {"mbps": peering.ISLAND_MBPS, "rtt_ms": peering.ISLAND_RTT_MS}}

    def owned_machine(machine_id: str, who: dict) -> dict:
        row = db.one("select * from machines where id=? or name=?", machine_id, machine_id)
        if not row or (row["owner"] != who["id"] and role_in(row["pool"], who) != "admin"):
            raise HTTPException(404, f"no machine {machine_id} of yours")
        return row

    @app.post("/v1/machines/{machine_ref}/trust")
    def trust_machine(machine_ref: str, pool: str | None = None, who: dict = Depends(member)):
        """A pool admin clears a machine excluded by failed checks (pool/trust.py)."""
        target = resolve_pool(pool, who, need="admin")
        found = db.one("select id from machines where pool=? and (id=? or name=?)", target["id"], machine_ref, machine_ref)
        if not found:
            raise HTTPException(404, f"no machine {machine_ref}")
        trust.clear(db, target["id"], found["id"])
        return {"machine": trust.standing(db, target["id"], found["id"])}

    @app.post("/v1/machines/{machine_id}/drain")
    def drain(machine_id: str, body: DrainBody, who: dict = Depends(member)):
        row = owned_machine(machine_id, who)
        db.run("update machines set drain=? where id=?", int(body.drain), row["id"])
        return {"ok": True}

    @app.put("/v1/machines/{machine_id}/policy")
    def set_policy(machine_id: str, body: PolicyBody, who: dict = Depends(member)):
        row = owned_machine(machine_id, who)
        if isinstance(body.who, str) and body.who not in ("pool", "owner"):
            raise HTTPException(400, "who is pool, owner, or a list of member names")
        if body.hours is not None and not HOURS_RE.match(body.hours):
            raise HTTPException(400, "hours is HH:MM-HH:MM in the machine's local time, for example 22:00-08:00")
        policy = body.model_dump()
        db.run("update machines set policy=? where id=?", json.dumps(policy), row["id"])
        return {"policy": policy}

    def drop_machine(machine_id: str) -> None:
        db.run("delete from machines where id=?", machine_id)
        db.run("delete from chunk_locations where machine=?", machine_id)
        db.run("update jobs set lease_expires=0 where machine=? and status='running'", machine_id)
        scheduler.expire_leases(db)  # its running jobs go back to the queue (or fail, out of retries)

    @app.delete("/v1/machines/{machine_id}")
    def remove_machine(machine_id: str, who: dict = Depends(member)):
        drop_machine(owned_machine(machine_id, who)["id"])
        return {"ok": True}

    # ---- artifacts --------------------------------------------------------------------------------
    def upload_owner(body_pool: str | None, job_id: str | None, who: tuple) -> tuple[str, str]:
        """(member, pool) an upload belongs to: the member, or for a machine the owner of its job."""
        kind, identity = who
        if kind == "machine":
            row = db.one("select owner, pool from jobs where id=? and machine=? and status='running'", job_id or "", identity["id"])
            if not row:
                raise HTTPException(403, "machines upload only the results of a job they are running")
            return row["owner"], row["pool"]
        return identity["id"], resolve_pool(body_pool, identity, need="member")["id"]

    @app.post("/v1/artifacts")
    def put_artifact(body: ArtifactBody, who: tuple = Depends(caller)):
        """Declare an artifact by its manifest. Returns the chunks still to upload (PUT /v1/chunks/...);
        once there are none, the artifact is stored and belongs to the caller."""
        owner, pool_id = upload_owner(body.pool, body.job, who)
        kind = "result" if who[0] == "machine" else body.kind
        if kind not in KINDS:
            raise HTTPException(400, f"kind is one of {', '.join(KINDS)}")
        try:
            manifest = arts.validate(body.manifest)
        except ArtifactError as exc:
            raise HTTPException(400, str(exc)) from None
        artifact = arts.manifest_id(manifest)
        size = arts.total_size(manifest)
        if size > max_artifact:
            raise HTTPException(413, f"artifact larger than {max_artifact // 1024**3} GB")
        owned = db.one("select 1 as x from artifact_owners where artifact=? and member=?", artifact, owner)
        if not owned:
            quota = scheduler.member_quota(db, pool_id, owner).get("storage_gb")
            if quota is not None and who[0] == "member" and storage_used(pool_id, owner) + size > quota * 1024**3:
                raise HTTPException(413, f"storage quota of {quota} GB in this pool would be exceeded")
        if body.held:
            # Held (#156, the contributions and partials of tree rounds): the machine that made it keeps the
            # chunks and serves them to peers; the server stores none of them and never serves them, since
            # nothing proves the declaring machine had their bytes (no artifact_chunks rows: GET
            # /v1/chunks refuses them, and peer-sources offers only the holder for them).
            if who[0] != "machine":
                raise HTTPException(400, "only a running job's machine holds an artifact")
            if not db.one("select 1 as x from artifact_owners where artifact=? and member=?", artifact, owner):
                # No read without proof: declaring a manifest is not having its bytes. An artifact the server
                # stores is uploaded (chunks proven) instead; one another machine already holds cannot be
                # declared again by someone who does not own it, or the declaration would make them an owner
                # and peer-sources would hand them tickets to fetch it from the real holder.
                if db.one("select 1 as x from artifact_chunks where artifact=?", artifact):
                    raise HTTPException(409, "the server stores this artifact: upload its chunks instead")
                if db.one("select 1 as x from held_artifacts where artifact=? and machine<>?", artifact, who[1]["id"]):
                    raise HTTPException(409, "another machine holds this artifact")
            with db.transaction():
                db.run("insert into artifacts (id, kind, size, files, manifest, created_at) values (?, ?, ?, ?, ?, ?) "
                       "on conflict do nothing", artifact, kind, size, len(manifest["files"]), json.dumps(manifest), now())
                db.run("insert into artifact_owners (artifact, member, pool, created_at) values (?, ?, ?, ?) on conflict do nothing",
                       artifact, owner, pool_id, now())
                db.run("insert into held_artifacts (artifact, machine, pool, created_at) values (?, ?, ?, ?) on conflict do nothing",
                       artifact, who[1]["id"], pool_id, now())
            return {"artifact": artifact, "missing": [], "size": size, "held": True}
        needed = arts.chunks_of(manifest)
        proven = {row["chunk"] for row in db.all("select chunk from chunk_proofs where member=?", owner)} if needed else set()
        missing = [sha for sha in needed if sha not in proven or not chunks.has(sha)]
        if missing:
            return {"artifact": artifact, "missing": missing, "size": size}
        with db.transaction():
            db.run("insert into artifacts (id, kind, size, files, manifest, created_at) values (?, ?, ?, ?, ?, ?) on conflict do nothing",
                   artifact, kind, size, len(manifest["files"]), json.dumps(manifest), now())
            db.many("insert into artifact_chunks (artifact, chunk) values (?, ?) on conflict do nothing", [(artifact, sha) for sha in needed])
            db.run("insert into artifact_owners (artifact, member, pool, created_at) values (?, ?, ?, ?) on conflict do nothing",
                   artifact, owner, pool_id, now())
            db.run("update artifact_owners set created_at=? where artifact=? and member=?", now(), artifact, owner)  # renews retention
        return {"artifact": artifact, "missing": [], "size": size}

    @app.put("/v1/chunks/{sha}")
    async def put_chunk(sha: str, request: Request, job: str | None = None, pool: str | None = None,
                        who: tuple = Depends(caller)):
        owner, _ = upload_owner(pool, job, who)
        try:
            writer = chunks.writer(sha)
        except ArtifactError as exc:
            raise HTTPException(400, str(exc)) from None
        try:
            async for block in request.stream():
                writer.write(block)
            size = writer.commit()
        except ArtifactError as exc:
            writer.abort()
            raise HTTPException(400, str(exc)) from None
        except BaseException:
            writer.abort()
            raise
        db.run("insert into chunks (sha, size, created_at) values (?, ?, ?) on conflict do nothing", sha, size, now())
        db.run("insert into chunk_proofs (chunk, member) values (?, ?) on conflict do nothing", sha, owner)
        return {"chunk": sha, "size": size}

    @app.get("/v1/artifacts/{artifact}")
    def get_artifact(artifact: str, who: tuple = Depends(caller)):
        row = db.one("select * from artifacts where id=?", artifact)
        if not row or not may_read(artifact, *who):
            raise HTTPException(404, "no such artifact")
        return {"artifact": artifact, "kind": row["kind"], "size": row["size"], "manifest": json.loads(row["manifest"])}

    @app.get("/v1/chunks/{sha}")
    def get_chunk(sha: str, artifact: str, who: tuple = Depends(caller)):
        if not db.one("select 1 as x from artifact_chunks where artifact=? and chunk=?", artifact, sha) or not may_read(artifact, *who):
            raise HTTPException(404, "no such chunk")
        try:
            path = chunks.path(sha)
        except ArtifactError as exc:
            raise HTTPException(400, str(exc)) from None
        if not path.is_file():
            raise HTTPException(404, "chunk missing on the server")
        return FileResponse(path, media_type="application/octet-stream")

    # ---- models -----------------------------------------------------------------------------------
    def split_ref(ref: str) -> tuple[str, str | None]:
        name, _, tag = ref.partition("@")
        return name, tag or None

    def find_model(pool_id: str, ref: str) -> dict:
        name, tag = split_ref(ref)
        if tag:
            row = db.one("select * from models where pool=? and name=? and tag=?", pool_id, name, tag)
        else:
            row = db.one("select * from models where pool=? and name=? order by created_at desc limit 1", pool_id, name)
        if not row:
            raise HTTPException(400, f"no model {ref} in this pool (pool model add ...)")
        return {**row, "meta": json.loads(row["meta"])}

    @app.post("/v1/models")
    def add_model(body: ModelBody, who: dict = Depends(member)):
        target = resolve_pool(body.pool, who, need="member")
        name, tag = split_ref(body.name)
        tag = tag or "latest"
        if not NAME_RE.match(name) or not NAME_RE.match(tag):
            raise HTTPException(400, "model names: name@tag, letters, digits, '.', '_' and '-'")
        if not member_reads(body.artifact, who["id"]):
            raise HTTPException(400, "upload the model folder first")
        meta = {k: v for k, v in body.meta.items() if k in ("architecture", "params", "active_params", "weight_bytes", "bytes_by_dtype",
                                                         "layers", "hidden", "experts", "top_k", "quantization")}
        if not isinstance(meta.get("weight_bytes"), int) or meta["weight_bytes"] <= 0:
            raise HTTPException(400, "meta.weight_bytes is required")
        db.run("delete from models where pool=? and name=? and tag=?", target["id"], name, tag)
        db.run("insert into models (pool, name, tag, artifact, meta, created_by, created_at) values (?, ?, ?, ?, ?, ?, ?)",
               target["id"], name, tag, body.artifact, json.dumps(meta), who["id"], now())
        return {"model": f"{name}@{tag}", "pool": target["name"]}

    @app.get("/v1/models")
    def list_models(pool: str | None = None, who: dict = Depends(member)):
        target = resolve_pool(pool, who)
        member_names = names()
        out = []
        for row in db.all("select * from models where pool=? order by name, created_at desc", target["id"]):
            meta = json.loads(row["meta"])
            out.append({"model": f"{row['name']}@{row['tag']}", "artifact": row["artifact"], "meta": meta,
                        "by": member_names.get(row["created_by"]), "created_at": row["created_at"]})
        return {"models": out, "pool": target["name"]}

    @app.delete("/v1/models/{ref}")
    def remove_model(ref: str, pool: str | None = None, who: dict = Depends(member)):
        target = resolve_pool(pool, who, need="member")
        found = find_model(target["id"], ref)
        if found["created_by"] != who["id"] and target["role"] != "admin":
            raise HTTPException(403, "only its creator or a pool admin removes a model")
        db.run("delete from models where pool=? and name=? and tag=?", target["id"], found["name"], found["tag"])
        return {"ok": True}

    # ---- jobs -------------------------------------------------------------------------------------
    def prepare_spec(pool_id: str, member_id: str, raw) -> dict:
        """A submitted spec, checked and with its model inputs resolved: for jobs and for actors."""
        try:
            spec = specs.normalize(raw)
        except specs.SpecError as exc:
            raise HTTPException(400, str(exc)) from None
        for item in spec["inputs"]:
            if item.get("model") and not item.get("artifact"):
                found = find_model(pool_id, item["model"])
                item["artifact"] = found["artifact"]
                item["model"] = f"{found['name']}@{found['tag']}"
                item["meta"] = found["meta"]
        fit = spec["resources"].get("fits")
        if fit:
            source = next(item for item in spec["inputs"] if item["path"] == fit["model"] or (item.get("model") or "").split("@")[0] == fit["model"].split("@")[0])
            if not source.get("meta"):
                raise HTTPException(400, "resources.fits needs a model input (inputs: [{path, model}])")
            fit["meta"] = source["meta"]
        for sha in specs.artifacts_of(spec):
            if not member_reads(sha, member_id):
                raise HTTPException(400, f"artifact {sha[:19]}... is not uploaded (or not yours)")
        return spec

    def create_jobs(pool_id: str, owner: str, spec: dict, name: str | None = None) -> list[str]:
        """Queue a prepared spec: one job, or one per rank of a gang."""
        size = (spec.get("gang") or {}).get("size", 1)
        ids = [new_id("j") for _ in range(size)]
        gang_id = new_id("g") if spec.get("gang") else None
        with db.lock:
            for rank, job_id in enumerate(ids):
                one = json.loads(json.dumps(spec))
                label = name or spec["name"]
                if gang_id:
                    one["gang"] = {"id": gang_id, "size": size, "rank": rank}
                    label = f"{label[:58]}.r{rank}" if size > 1 else label
                db.run("insert into jobs (id, pool, owner, name, priority, spec, status, created_at) values (?, ?, ?, ?, ?, ?, 'queued', ?)",
                       job_id, pool_id, owner, label, spec["priority"], json.dumps(one), now())
                if gang_id:
                    db.run("insert into gang_members (gang, rank, job) values (?, ?, ?)", gang_id, rank, job_id)
        queued[0] += 1
        return ids

    def cancel_queued(job_id: str) -> None:
        with db.lock:
            if db.run("update jobs set status='cancelled', finished_at=?, lease_expires=null where id=? and status='queued'", now(), job_id):
                scheduler.append_log(db, job_id, "\n[pool] removed: its actor scaled down\n")

    def cancel_live(job_id: str, why: str) -> None:
        with db.lock:
            if db.run("update jobs set status='cancelled', finished_at=?, lease_expires=null where id=? and status in ('queued', 'running')",
                      now(), job_id):
                scheduler.append_log(db, job_id, f"\n[pool] cancelled: {why}\n")

    reconciled = {"at": 0.0}

    def reconcile_actors(force: bool = False) -> None:
        """Keep actors at their replica counts; at most once a second unless forced."""
        if not force and time.monotonic() - reconciled["at"] < 1.0:
            return
        reconciled["at"] = time.monotonic()
        with db.lock:
            actors.reconcile(db, create_jobs, cancel_queued)
            rounds.tick(db)

    @app.post("/v1/jobs")
    def submit(body: dict, who: dict = Depends(member)):
        target = resolve_pool(body.get("pool"), who, need="member")
        spec = prepare_spec(target["id"], who["id"], body.get("spec", body))
        if spec.get("gang") and ("id" in spec["gang"] or "rank" in spec["gang"]):
            raise HTTPException(400, "gang: give only size; the server numbers the ranks")
        ids = create_jobs(target["id"], who["id"], spec)
        job = load_job(db.one("select * from jobs where id=?", ids[0]))
        reply = {"job": public_job(job)}
        lines = scheduler.explain(db, job)
        if not any(": can run it" in line or line.endswith(": busy") or ": offline" in line or ": draining" in line for line in lines):
            reply["warning"] = "no machine in the pool can run this job as it stands:\n  " + "\n  ".join(lines)
        elif any(line.startswith("gang: needs") for line in lines):
            reply["warning"] = next(line for line in lines if line.startswith("gang: needs"))
        return reply

    @app.get("/v1/jobs")
    def list_jobs(pool: str | None = None, all: bool = False, limit: int = 50, who: dict = Depends(member)):
        limit = max(1, min(limit, 500))
        if all:
            target = resolve_pool(pool, who, need="admin")
            rows = db.all("select * from jobs where pool=? order by created_at desc limit ?", target["id"], limit)
        elif pool:
            target = resolve_pool(pool, who)
            rows = db.all("select * from jobs where owner=? and pool=? order by created_at desc limit ?", who["id"], target["id"], limit)
        else:
            rows = db.all("select * from jobs where owner=? order by created_at desc limit ?", who["id"], limit)
        return {"jobs": [public_job(load_job(row)) for row in rows]}

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str, who: dict = Depends(member)):
        job = own_job(job_id, who)
        reply = {"job": public_job(job)}
        if job["status"] == "queued":
            reply["why"] = scheduler.explain(db, job)
            # each machine's agent next to why it is (not) running the job: an outdated agent can be the reason
            reply["agents"] = [{"machine": row["name"], **update.agent_version(json.loads(row["profile"] or "{}"), agent_digest)}
                               for row in db.all("select name, profile from machines where pool=? order by name", job["pool"])]
        elif job.get("machine"):
            row = db.one("select name, profile from machines where id=?", job["machine"])
            if row:
                reply["agents"] = [{"machine": row["name"], **update.agent_version(json.loads(row["profile"] or "{}"), agent_digest)}]
        return reply

    @app.post("/v1/jobs/{job_id}/cancel")
    def cancel(job_id: str, who: dict = Depends(member)):
        job = own_job(job_id, who)
        if job["status"] in FINAL:
            return {"job": public_job(job)}
        for target_job in [job] + scheduler.siblings(db, job):  # a gang is cancelled as a whole
            with db.lock:
                done = db.run("update jobs set status='cancelled', finished_at=?, lease_expires=null where id=? and status in ('queued', 'running')",
                              now(), target_job["id"])
            if done:
                scheduler.append_log(db, target_job["id"], f"\n[pool] cancelled by {who['name']}\n")
        return {"job": public_job(own_job(job_id, who))}

    @app.get("/v1/jobs/{job_id}/logs")
    async def logs(job_id: str, after: int = 0, wait: float = 0, who: dict = Depends(member)):
        own_job(job_id, who)
        deadline = time.monotonic() + max(0.0, min(wait, 25.0))
        while True:
            rows = db.all("select id, text from logs where job=? and id>? order by id limit 500", job_id, after)
            job = db.one("select status, machine from jobs where id=?", job_id)
            status = job["status"]
            if rows or status in FINAL or time.monotonic() >= deadline:
                # The machine running it, by name (a board shows "Running on ..."); None while queued.
                machine = db.one("select name from machines where id=?", job["machine"]) if job["machine"] else None
                return {"chunks": rows, "status": status, "next": rows[-1]["id"] if rows else after,
                        "machine": machine["name"] if machine else None}
            await asyncio.sleep(0.5)

    # ---- admin ------------------------------------------------------------------------------------
    @app.post("/v1/admin/gc")
    def run_gc(who: dict = Depends(member)):
        if not who["is_admin"]:
            raise HTTPException(403, "server admins only")
        return collect_garbage(db, chunks)

    @app.get("/metrics")
    def prometheus(authorization: str | None = Header(None)):
        """Prometheus metrics (pool/metrics.py): a server admin's token, or the --metrics-token."""
        token = bearer(authorization)
        if not (metrics_token and hmac.compare_digest(token.encode(), metrics_token.encode())):
            if not member(authorization)["is_admin"]:
                raise HTTPException(403, "server admins (or the --metrics-token) only")
        schema = max(migrations.applied(db), default=None)
        return Response(metrics.render(db, pool.__version__, schema, agent_digest), media_type=metrics.CONTENT_TYPE)

    # ---- agents -----------------------------------------------------------------------------------
    # ---- refs and queues (pool/coord.py) ---------------------------------------------------------
    def coord_call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except coord.CoordError as exc:
            raise HTTPException(exc.status, {"message": exc.message, **exc.extra} if exc.extra else exc.message) from None

    async def waiting(fetch, wait: float, ready):
        """Long-poll: call fetch() until ready(result) or `wait` seconds (at most 30) pass."""
        deadline = time.monotonic() + max(0.0, min(float(wait or 0), 30.0))
        while True:
            result = fetch()
            if ready(result) or time.monotonic() >= deadline:
                return result
            await asyncio.sleep(0.25)

    async def coord_op(op: str, args: dict, pool_id: str, member_id: str, job: str | None = None, machine: str | None = None,
                       may_write: bool = True, role: str | None = None):
        """One operation for a member (job None) or for a running job (through its agent's socket)."""
        a = args or {}
        if op in {"ref.publish", "queue.create", "tasks.add", "tasks.lease", "task.result", "checkpoint.record"} and not may_write:
            raise HTTPException(403, "viewers only read")
        if op == "ref.list":
            return {"refs": coord.ref_list(db, pool_id)}
        if op == "ref.get":
            if a.get("history"):
                return {"versions": coord.ref_history(db, pool_id, a["name"], int(a.get("limit") or 50))}
            after = int(a.get("after") or 0)

            def fetch():
                current = coord.ref_current(db, pool_id, a.get("name") or "")
                return current if a.get("version") is None else coord_call(coord.ref_get, db, pool_id, a["name"], int(a["version"]))
            found = await waiting(fetch, a.get("wait", 0), lambda r: r is not None and r["version"] > after)
            if found is None:
                raise HTTPException(404, f"no ref {a.get('name')}")
            return {"ref": found, "newer": found["version"] > after}
        if op == "ref.publish":
            artifact = a.get("artifact") or ""
            if not member_reads(artifact, member_id):
                raise HTTPException(404, "no such artifact (upload it first)")
            return {"ref": coord_call(coord.ref_publish, db, pool_id, a.get("name") or "", artifact, a.get("parent"),
                                      member_id, bool(a.get("force")), a.get("meta"))}
        if op == "queue.create":
            return {"queue": coord_call(coord.queue_create, db, pool_id, a.get("name") or "", member_id, a.get("ref"),
                                        a.get("max_lag"), a.get("stale") or "requeue", a.get("duplicate_after"), a.get("replicate"))}
        if op == "queue.list":
            return {"queues": coord.queue_list(db, pool_id)}
        if op == "queue.stats":
            return {"queue": coord_call(coord.queue_stats, db, pool_id, a.get("name") or "")}
        if op == "tasks.add":
            return {"ids": coord_call(coord.tasks_add, db, pool_id, a.get("name") or "", a.get("tasks"))}
        if op == "results":
            after = int(a.get("after") or 0)
            return await waiting(lambda: coord_call(coord.results, db, pool_id, a.get("name") or "", after, a.get("limit") or 500),
                                 a.get("wait", 0), lambda r: bool(r["results"]))
        if op.startswith("actor."):
            if op != "actor.list" and op != "actor.status" and not may_write:
                raise HTTPException(403, "viewers only read")
            try:
                if op == "actor.create":
                    spec = prepare_spec(pool_id, member_id, a.get("spec"))
                    actors.create(db, pool_id, a.get("name") or "", member_id, spec, a.get("min"), a.get("max"),
                                           a.get("queue"), a.get("per_replica") or 1)
                    reconcile_actors(force=True)
                    return {"actor": actors.status(db, pool_id, a.get("name"))}
                if op == "actor.scale":
                    actors.scale(db, pool_id, a.get("name") or "", member_id, a.get("min"), a.get("max"))
                    reconcile_actors(force=True)
                    return {"actor": actors.status(db, pool_id, a.get("name"))}
                if op == "actor.stop":
                    for job_id in actors.stop(db, pool_id, a.get("name") or "", member_id):
                        cancel_live(job_id, f"actor {a.get('name')} stopped")
                    return {"actor": actors.status(db, pool_id, a.get("name"))}
                if op == "actor.status":
                    return {"actor": actors.status(db, pool_id, a.get("name") or "")}
                if op == "actor.list":
                    return {"actors": actors.listing(db, pool_id)}
            except actors.ActorError as exc:
                raise HTTPException(exc.status, exc.message) from None
        if op.startswith("round."):
            if op in {"round.open", "round.contribute", "round.claim", "round.finish", "round.partial"} and not may_write:
                raise HTTPException(403, "viewers only read")
            try:
                if op == "round.open":
                    return {"round": rounds.open_round(db, pool_id, a.get("ref") or "", member_id, a.get("quorum") or 1, a.get("min"),
                                                       a.get("deadline_s") or 600, a.get("grace_s", 5), a.get("late") or "drop",
                                                       a.get("decay", 0.5), bool(a.get("repeat")), a.get("mode") or "reduce",
                                                       a.get("fanout") or 2, a.get("bias", 0.5),
                                                       a.get("reducer") or "artifacts", a.get("lighthouse"),
                                                       a.get("reduce") or "flat", a.get("handoff_s", rounds.HANDOFF_S))}
                if op == "round.current":
                    after = a.get("after")

                    def fetch():
                        return rounds.current(db, pool_id, a.get("ref") or "")
                    found = await waiting(fetch, a.get("wait", 0),
                                          lambda r: r is not None and r["status"] == "open" and r["id"] != after)
                    return {"round": found}
                if job is None and op in {"round.contribute", "round.claim", "round.finish", "round.mix", "round.partial"}:
                    raise HTTPException(400, f"{op} is for running jobs (through the agent's socket)")
                if op == "round.contribute":
                    artifact = a.get("artifact") or ""
                    if not member_reads(artifact, member_id):
                        raise HTTPException(404, "no such artifact (upload it first)")
                    return rounds.contribute(db, pool_id, a.get("ref") or "", job, machine, artifact, a.get("weight") or 1.0,
                                             int(a.get("base_version") or 0), a.get("round"))
                if op == "round.mix":
                    return {"mix": await waiting(lambda: rounds.mix(db, pool_id, a.get("round") or "", job), a.get("wait", 0), bool)}
                if op == "round.claim":
                    return {"round": await waiting(lambda: rounds.claim(db, pool_id, a.get("ref") or "", job, machine),
                                                   a.get("wait", 0), bool)}
                if op == "round.partial":
                    artifact = a.get("artifact") or ""
                    if not member_reads(artifact, member_id):
                        raise HTTPException(404, "no such artifact (upload it first)")
                    dropped = a.get("dropped") or []
                    if not isinstance(dropped, list) or len(dropped) > 10_000:
                        raise HTTPException(400, "dropped is a list of contribution ids")
                    return rounds.partial(db, pool_id, a.get("round") or "", job, machine, artifact, dropped)
                if op == "round.finish":
                    artifact = a.get("artifact") or ""
                    if not member_reads(artifact, member_id):
                        raise HTTPException(404, "no such artifact (upload it first)")

                    def publish(ref, art, parent):
                        return coord_call(coord.ref_publish, db, pool_id, ref, art, parent, member_id)
                    return rounds.finish(db, pool_id, a.get("round") or "", job, member_id, artifact, publish)
            except rounds.RoundError as exc:
                raise HTTPException(exc.status, exc.message) from None
        if op == "verify.report":
            # A verifier's finding about a machine of this pool (pool/trust.py): a trainer that recomputed
            # logprobs or rewards, a reducer that found an outlying update. Jobs report; members need admin.
            if job is None and role != "admin":
                raise HTTPException(403, "only jobs and pool admins report on machines")
            found = db.one("select id from machines where pool=? and (id=? or name=?)", pool_id, a.get("machine") or "",
                           a.get("machine") or "")
            if not found:
                raise HTTPException(404, f"no machine {a.get('machine')} in this pool")
            return {"machine": trust.report(db, pool_id, found["id"], bool(a.get("ok")), a.get("kind") or "other",
                                            a.get("detail") or "", job or member_id)}
        if op == "checkpoint.record":
            if job is None:
                raise HTTPException(400, "checkpoint.record is for running jobs")
            artifact = a.get("artifact") or ""
            if not member_reads(artifact, member_id):
                raise HTTPException(404, "no such artifact")
            actors.save_checkpoint(db, job, artifact, a.get("attempt"))
            return {"ok": True}
        if job is None and op in {"tasks.lease", "task.result"}:
            raise HTTPException(400, f"{op} is for running jobs (through the agent's socket)")
        if op == "tasks.lease":
            return {"tasks": await waiting(lambda: coord_call(coord.tasks_lease, db, pool_id, a.get("name") or "", job, machine,
                                                              a.get("n") or 1), a.get("wait", 0), bool)}
        if op == "task.result":
            return coord_call(coord.task_result, db, pool_id, a.get("name") or "", a.get("task") or "", job, machine,
                              a.get("result"), a.get("computed_with"), a.get("failed"))
        raise HTTPException(400, f"unknown operation {op!r}")

    @app.post("/v1/coord")
    async def member_coord(body: CoordBody, who: dict = Depends(member)):
        pool = resolve_pool(body.pool, who)
        role = role_in(pool["id"], who)
        return await coord_op(body.op, body.args, pool["id"], who["id"], may_write=role in ("member", "admin"), role=role)

    @app.post("/v1/agent/jobs/{job_id}/api")
    async def job_coord(job_id: str, body: CoordBody, agent: dict = Depends(machine)):
        """A running job's request, forwarded by its agent from the job's unix socket: as the job's owner, in its pool."""
        row = db.one("select owner, pool, status, machine from jobs where id=?", job_id)
        if not row or row["machine"] != agent["id"] or row["status"] != "running":
            raise HTTPException(409, "this machine is not running that job")
        role = db.one("select role from pool_members where pool=? and member=?", row["pool"], row["owner"])
        return await coord_op(body.op, body.args, row["pool"], row["owner"], job=job_id, machine=agent["id"],
                              may_write=bool(role) and role["role"] in ("member", "admin"))

    @app.get("/v1/agent/speedtest")
    def speedtest_down(bytes: int = 8 << 20, agent: dict = Depends(machine)):
        size = max(1, min(bytes, 64 << 20))
        reps, rest = divmod(size, len(SPEEDTEST))
        return Response(SPEEDTEST * reps + SPEEDTEST[:rest], media_type="application/octet-stream")

    @app.post("/v1/agent/speedtest")
    async def speedtest_up(request: Request, agent: dict = Depends(machine)):
        started, size = time.monotonic(), 0
        async for block in request.stream():
            size += len(block)
            if size > 64 << 20:
                raise HTTPException(413, "at most 64 MB")
        return {"bytes": size, "seconds": time.monotonic() - started}

    @app.get("/v1/agent/me")
    def agent_me(me: dict = Depends(machine)):
        """This machine as the server sees it (pool-agent status)."""
        running = db.all("select id from jobs where machine=? and status='running'", me["id"])
        return {"id": me["id"], "name": me["name"], "pool": pool_names().get(me["pool"]), "drain": bool(me.get("drain")),
                "running": [row["id"] for row in running]}

    @app.delete("/v1/agent/me")
    def agent_unregister(me: dict = Depends(machine)):
        """The machine removes itself (pool-agent uninstall): its token stops working, its jobs requeue."""
        drop_machine(me["id"])
        return {"ok": True, "name": me["name"]}

    @app.post("/v1/agent/heartbeat")
    def heartbeat(body: HeartbeatBody, agent: dict = Depends(machine)):
        if body.profile is not None:
            db.run("update machines set profile=? where id=?", json.dumps(body.profile), agent["id"])
        if body.labels is not None:
            db.run("update machines set labels=? where id=?", json.dumps(body.labels), agent["id"])
        if body.free is not None:
            db.run("update machines set free=?, free_at=? where id=?", json.dumps(body.free), now(), agent["id"])
        if body.cache is not None:
            with db.lock:
                if body.cache.full:
                    db.run("delete from chunk_locations where machine=?", agent["id"])
                good = [sha for sha in body.cache.add if arts.is_hex(sha)]
                db.many("insert into chunk_locations (chunk, machine, seen_at) values (?, ?, ?) on conflict do nothing",
                        [(sha, agent["id"], now()) for sha in good])
                db.many("delete from chunk_locations where chunk=? and machine=?", [(sha, agent["id"]) for sha in body.cache.remove])
        scheduler.extend(db, agent["id"], body.running)
        coord.extend_task_leases(db, agent["id"], body.running)
        scheduler.expire_leases(db)
        coord.expire_task_leases(db)
        stop = []
        for job_id in body.running:
            row = db.one("select status, machine from jobs where id=?", job_id)
            if not row or row["status"] != "running" or row["machine"] != agent["id"]:
                stop.append(job_id)  # cancelled, or given to another machine after a lost lease
        reconcile_actors()
        return {"poll_ms": poll_ms, "cancel": stop, "drain": bool(agent["drain"]), "policy": agent["policy"],
                "peer_key": peer_key(agent["pool"]), "warm": actors.warm_for(db, agent["id"]), "agent": agent_digest}

    # ---- peers: the server as the pool's directory (pool/peers.py) ------------------------------------
    @app.get("/v1/agent/peers")
    def agent_peers(agent: dict = Depends(machine)):
        return {"peers": online_peers(agent["pool"], exclude=agent["id"])}

    @app.post("/v1/agent/links")
    def agent_links(body: LinksBody, agent: dict = Depends(machine)):
        same_pool = {row["id"] for row in db.all("select id from machines where pool=?", agent["pool"])}
        kept = 0
        for link in body.links[:256]:
            dst = link.get("peer")
            try:
                mbps, rtt = float(link["mbps"]), float(link["rtt_ms"])
            except (KeyError, TypeError, ValueError):
                continue
            if dst in same_pool and dst != agent["id"]:
                db.run("insert into peer_links (src, dst, mbps, rtt_ms, measured_at) values (?, ?, ?, ?, ?) "
                       "on conflict (src, dst) do update set mbps=excluded.mbps, rtt_ms=excluded.rtt_ms, measured_at=excluded.measured_at",
                       agent["id"], dst, mbps, rtt, now())
                kept += 1
        return {"kept": kept}

    @app.post("/v1/agent/chunks")
    def agent_chunks(body: ChunksBody, agent: dict = Depends(machine)):
        """The tracker hears at once that a chunk landed on (or left) a machine, so the machines
        fetching the same artifact right now can take it from there. Only chunks the server knows
        are kept; a holder is offered only to machines that may read an artifact with that chunk."""
        good = [sha for sha in body.add[:4096] if arts.is_hex(sha)]
        gone = [sha for sha in body.remove[:4096] if arts.is_hex(sha)]
        known: list[str] = []
        with db.lock:
            if good:
                marks = ",".join("?" * len(good))
                known = [row["sha"] for row in db.all(f"select sha from chunks where sha in ({marks})", *good)]
                db.many("insert into chunk_locations (chunk, machine, seen_at) values (?, ?, ?) on conflict do nothing",
                        [(sha, agent["id"], now()) for sha in known])
            db.many("delete from chunk_locations where chunk=? and machine=?", [(sha, agent["id"]) for sha in gone])
        return {"added": len(known), "removed": len(gone)}

    @app.post("/v1/agent/peer-sources")
    def peer_sources(body: PeerSourcesBody, agent: dict = Depends(machine)):
        """For chunks of an artifact this machine may read: which online peers hold them, best
        measured link first, and a ticket per chunk that those peers accept."""
        if not may_read(body.artifact, "machine", agent):
            raise HTTPException(404, "no such artifact")
        wanted = [sha for sha in body.chunks[:4096] if arts.is_hex(sha)]
        if not wanted:
            return {"sources": {}}
        marks = ",".join("?" * len(wanted))
        belongs = {row["chunk"] for row in db.all(f"select chunk from artifact_chunks where artifact=? and chunk in ({marks})",
                                                  body.artifact, *wanted)}
        peers_by_id = {p["id"]: p for p in online_peers(agent["pool"], exclude=agent["id"])}
        speed = {row["dst"]: row["mbps"] for row in db.all("select dst, mbps from peer_links where src=?", agent["id"])}
        key = peer_key(agent["pool"])
        sources = {}
        for row in db.all(f"select chunk, machine from chunk_locations where chunk in ({marks})", *wanted):
            if row["chunk"] in belongs and row["machine"] in peers_by_id:
                sources.setdefault(row["chunk"], []).append(peers_by_id[row["machine"]])
        holders = [peers_by_id[r["machine"]] for r in db.all("select machine from held_artifacts where artifact=?", body.artifact)
                   if r["machine"] in peers_by_id]
        if holders:  # a held artifact: only the machines that hold it are offered for its chunks
            found = db.one("select manifest from artifacts where id=?", body.artifact)
            listed = set(arts.chunks_of(json.loads(found["manifest"]))) if found else set()
            for sha in wanted:
                if sha in listed:
                    have = sources.setdefault(sha, [])
                    have.extend(h for h in holders if h not in have)
        return {"sources": {sha: {"ticket": peering.ticket(key, agent["id"], sha),
                                  "peers": sorted(found, key=lambda p: -speed.get(p["id"], 0.0))}
                            for sha, found in sources.items()}}

    @app.post("/v1/agent/jobs/{job_id}/rendezvous")
    def gang_rendezvous(job_id: str, agent: dict = Depends(machine)):
        job = load_job(db.one("select * from jobs where id=?", job_id))
        if not job or job["machine"] != agent["id"] or job["status"] != "running":
            return {"stop": True}
        if not scheduler.gang_of(job):
            raise HTTPException(400, "not a gang job")
        scheduler.extend(db, agent["id"], [job_id])
        return scheduler.rendezvous(db, job)

    @app.post("/v1/agent/lease")
    async def lease(body: LeaseBody, agent: dict = Depends(machine)):
        deadline = time.monotonic() + max(0.0, min(body.wait, LEASE_WAIT_MAX))
        db.run("update machines set free=?, free_at=? where id=?", json.dumps(body.free), now(), agent["id"])
        while True:
            seen = queued[0]
            scheduler.expire_leases(db)
            current = load_machine(db.one("select * from machines where id=?", agent["id"]))
            if current is None:
                raise HTTPException(401, "machine removed")
            # A heartbeat during this long-poll may report more room (a job just ended): use the latest.
            reconcile_actors()
            job = scheduler.claim(db, current, current["free"] if current.get("free") is not None else body.free)
            if job:
                job["checkpoint"] = actors.checkpoint_of(db, job["id"])
                db.run("update machines set free_at=null where id=?", agent["id"])  # its free room just changed
                return {"jobs": [job]}
            if time.monotonic() >= deadline:
                return {"jobs": []}
            db.run("update machines set free_at=?, last_seen=? where id=?", now(), now(), agent["id"])
            await nap(1.0, seen)  # jobs requeued elsewhere (expired leases, actors) are seen within a second

    @app.post("/v1/agent/progress")
    def progress(body: ProgressBody, agent: dict = Depends(machine)):
        row = db.one("select status, machine from jobs where id=?", body.job)
        if not row or row["machine"] != agent["id"]:
            return {"stop": True}
        if row["status"] == "running":
            scheduler.extend(db, agent["id"], [body.job])
            if body.text:
                scheduler.append_log(db, body.job, body.text, body.key)
        return {"stop": row["status"] != "running"}

    @app.post("/v1/agent/jobs/{job_id}/complete")
    def complete(job_id: str, body: CompleteBody, agent: dict = Depends(machine)):
        if body.preempted:
            return {"ok": scheduler.requeue_preempted(db, job_id, agent["id"], body.preempted)}
        with db.lock:
            row = db.one("select * from jobs where id=?", job_id)
            if not row or row["machine"] != agent["id"] or row["status"] != "running":
                return {"ok": False}  # cancelled meanwhile, or the lease went to another machine
            if body.outputs and not db.one("select 1 as x from artifact_owners where artifact=? and member=?", body.outputs, row["owner"]):
                raise HTTPException(400, "upload the outputs artifact before completing")
            status = "succeeded" if body.exit_code == 0 and not body.error else "failed"
            error = body.error or (None if body.exit_code == 0 else f"exit code {body.exit_code}")
            db.run("update jobs set status=?, exit_code=?, error=?, outputs=?, finished_at=?, lease_expires=null where id=?",
                   status, body.exit_code, error, body.outputs, now(), job_id)
            job = load_job(row)
            if status == "failed" and scheduler.gang_of(job):
                scheduler.fail_siblings(db, job, f"rank {scheduler.gang_of(job)['rank']} failed ({error})")
        return {"ok": True}

    if gc_seconds:
        def gc_loop():
            while True:
                time.sleep(gc_seconds)
                try:
                    collect_garbage(db, chunks)
                except Exception as exc:  # never take the server down over housekeeping
                    print(f"gc failed: {exc}", flush=True)

        threading.Thread(target=gc_loop, daemon=True, name="gc").start()
    return app


def collect_garbage(db: DB, chunks: ChunkStore) -> dict:
    """Expire unpinned artifacts past their pool's retention, then delete artifacts nobody owns and
    chunks no artifact uses. Artifacts that queued or running jobs or models use are kept."""
    in_use = set()
    for row in db.all("select spec from jobs where status in ('queued', 'running')"):
        in_use.update(specs.artifacts_of(json.loads(row["spec"])))
    in_use.update(row["artifact"] for row in db.all("select artifact from models"))
    expired = 0
    for pool_row in db.all("select id from pools"):
        days = scheduler.pool_settings(db, pool_row["id"])["retention_days"]
        for row in db.all("select artifact, member from artifact_owners where pool=? and pinned=0 and created_at < ?",
                          pool_row["id"], now() - days * 86400):
            if row["artifact"] not in in_use:
                expired += db.run("delete from artifact_owners where artifact=? and member=?", row["artifact"], row["member"])
    orphans = [row["id"] for row in db.all("select id from artifacts a where not exists "
                                           "(select 1 from artifact_owners o where o.artifact=a.id)") if row["id"] not in in_use]
    for artifact in orphans:
        db.run("delete from artifact_chunks where artifact=?", artifact)
        db.run("delete from held_artifacts where artifact=?", artifact)
        db.run("delete from artifacts where id=?", artifact)
    freed = 0
    stale = db.all("select sha from chunks c where created_at < ? and not exists (select 1 from artifact_chunks a where a.chunk=c.sha)",
                   now() - 86400)
    for row in stale:
        freed += chunks.delete(row["sha"])
        db.run("delete from chunks where sha=?", row["sha"])
        db.run("delete from chunk_proofs where chunk=?", row["sha"])
        db.run("delete from chunk_locations where chunk=?", row["sha"])
    return {"expired_owners": expired, "artifacts_deleted": len(orphans), "chunks_deleted": len(stale), "bytes_freed": freed}


def exposure_warning(host: str, default_route: str | None = None) -> str | None:
    """A warning when the server listens where the outside network may reach it: on a public address,
    or on every interface of a machine whose default route leaves from a public one. None on a LAN.
    `default_route` is that address (looked up when not given)."""
    if host in ("", "0.0.0.0", "::"):
        if default_route is None:
            default_route = peering._route_source("192.0.2.1", 9)  # TEST-NET-1: picks the default route
        if default_route is None or peering.private_address(default_route):
            return None
        where = f"every interface, and this machine has the public address {default_route}"
    elif peering.private_address(host) or host == "localhost":
        return None
    else:
        where = f"{host}, which is not a private (LAN) address"
    return (f"WARNING: the pool server listens on {where}. It speaks plain HTTP and is meant for the LAN.\n"
            f"  Bind it to a LAN or loopback address (--host), and if machines elsewhere must reach it, put it\n"
            f"  behind HTTPS: a reverse proxy (Caddy, nginx) or a tunnel such as Tailscale.")


def main(argv=None) -> int:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(prog="pool server")
    parser.add_argument("--data", default=os.environ.get("POOL_DATA", str(Path.home() / ".local/share/pool-server")),
                        help="chunks (and the SQLite database, unless --db)")
    parser.add_argument("--db", default=os.environ.get("POOL_DB"), help="postgresql://... (default: SQLite in --data)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="the address to listen on; keep it on the LAN (a warning is printed for a public one)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--public-url", default=os.environ.get("POOL_PUBLIC_URL"),
                        help="the address machines and members use, if not the one they connect to")
    parser.add_argument("--max-artifact-gb", type=int, default=512)
    parser.add_argument("--metrics-token", default=os.environ.get("POOL_METRICS_TOKEN"),
                        help="a bearer token for Prometheus to scrape /metrics with (server admins' tokens work too)")
    args = parser.parse_args(argv)
    app = create_app(args.data, args.public_url, args.max_artifact_gb * 1024, db_url=args.db, metrics_token=args.metrics_token)
    warning = exposure_warning(args.host)
    if warning:
        print(warning, file=sys.stderr, flush=True)
    url = args.public_url or f"http://{args.host if args.host not in ('0.0.0.0', '::') else 'localhost'}:{args.port}"
    if app.state.bootstrap_code:
        print(f"First start. Make yourself the admin (the code works once, for {CODE_DAYS} days):\n"
              f"  python -m pool.cli login {url} --code {app.state.bootstrap_code} --name <your name>\n", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0
