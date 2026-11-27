"""Pool machines on the board's Compute page (#135, docs/design/worker-v2.md section 6).

Machines that grade board jobs are pool agents (pool/agent.py): the gymhub runner is a dispatcher
that submits pool jobs (gymhub/runner/pooled.py). The board reads the pool server as one pool member
and shows its machines; an admin gets an install command for a new one ("Add a machine").

Settings (environment):
    GYMHUB_POOL_SERVER   the pool server's URL, as the board reaches it
    GYMHUB_POOL_TOKEN    a pool member token (member or admin of the pool); machines added from the
                         board belong to this member, so the dispatcher, submitting as the same
                         member, may send them gyms that are not public (resources.machines: own)
    GYMHUB_POOL          the pool (default: the member's default pool)

Without GYMHUB_POOL_SERVER and GYMHUB_POOL_TOKEN (a board on Cloud Run cannot reach a pool server on a
LAN), the page shows what the dispatcher reports of the pool in its heartbeat (`pool_report`,
gymhub/poolview.py, #162); with neither, only the board's own runners.

    GET  /api/pool/machines         every signed-in user: the pool's machines (name, owner, GPUs, platform,
                                    state, last seen, labels, the gyms each can take), its LAN islands,
                                    and the runner that connects the board to the pool (`bridge`)
    POST /api/pool/machine-codes    admins: a machine code and the install command for it

Adding a machine is for admins, as creating a worker token is (gymhub/board/workers.py). Pool
machines grade only the jobs the bridge takes: its owner's, until the owner shares it with everyone
or by email ("Share the pool" on the Compute page; `grades_for` in the reply says which).
"""

import os
import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from gymhub import poolview
from gymhub.board.auth import User, current_user, require_admin

router = APIRouter()


def pool_client():
    """The pool member the board reads as, or None when no pool is configured. Tests replace this."""
    server = os.environ.get("GYMHUB_POOL_SERVER", "").strip()
    token = os.environ.get("GYMHUB_POOL_TOKEN", "").strip()
    if not server or not token:
        return None
    from pool.client import Client

    return Client(server, token)


def _pool_name() -> str | None:
    return os.environ.get("GYMHUB_POOL", "").strip() or None


machine_view = poolview.machine_view  # tests and older imports


def _store():
    from gymhub.board.app import store

    return store


def _bridge(user: User | None = None) -> tuple[dict | None, dict | None]:
    """The runner that connects the board to the pool (a dispatcher or another pool bridge), as the
    page shows it, and the pool report of its last heartbeat: the freshest runner that has one.
    With `user`, also whose jobs it takes, as that user may know it (`grades_for`)."""
    from gymhub.board.app import RUNNER_ONLINE_SECONDS

    best = None
    for _id, doc in _store().recent("runners", "heartbeat_at", 50):
        capabilities = doc.get("capabilities") or {}
        if not (doc.get("pool_report") or capabilities.get("pool") or capabilities.get("dispatch_only")):
            continue
        if best is None or (doc.get("pool_report") and not best.get("pool_report")):
            best = doc
        if best.get("pool_report"):
            break
    if best is None:
        return None, None
    at = best.get("heartbeat_at") or 0
    capabilities = best.get("capabilities") or {}
    status = {"name": best.get("name"), "heartbeat_at": at or None, "online": time.time() - at < RUNNER_ONLINE_SECONDS,
              "dispatcher": bool(capabilities.get("dispatch_only") or (capabilities.get("pool") or {}).get("dispatch_only")),
              "reaches_pool": bool(capabilities.get("pool") or best.get("pool_report"))}
    if user is not None:
        status["grades_for"] = grades_for(best.get("worker_id"), user)
    report = poolview.clean_report(best.get("pool_report")) or poolview.from_routing(capabilities.get("pool"), at)
    return status, report


def grades_for(worker_id: str | None, user: User) -> dict:
    """Whose jobs the bridge takes (gymhub/board/workers.py may_run_for: its owner's until shared):
    {"who": "owner" | "some" | "everyone", "yours": the user's jobs go there, "mine": the user owns it}.
    Its owner also gets the worker id and the emails it is shared with, to change them ("Share the pool")."""
    doc = (_store().get("workers", worker_id) if worker_id else None) or {}
    if not doc:  # a runner on the store itself (no worker token): the board owner's machine
        return {"who": "everyone" if user.is_admin else "owner", "yours": user.is_admin, "mine": False}
    mine = doc.get("owner_uid") == user.uid
    everyone = bool(doc.get("shared_everyone"))
    found = {"who": "everyone" if everyone else "some" if doc.get("shared_uids") else "owner",
             "yours": mine or everyone or user.uid in (doc.get("shared_uids") or []), "mine": mine}
    if mine:
        found.update(worker=worker_id, emails=doc.get("shared_emails") or [])
    return found


def _gym_needs(user: User) -> list[tuple[str, bool, bool]]:
    """The gyms this user sees: (id, needs a GPU, public)."""
    from gymhub import gymaccess, routing

    found = []
    for gym_id, doc in _store().recent("gyms", "updated_at", 200):
        if gymaccess.listed(user.uid, user.is_admin, doc):
            found.append((gym_id, routing.requirements(gym_id, doc, "")["gpu"], gymaccess.public(doc)))
    return sorted(found)


def _with_gyms(machines: list[dict], user: User) -> list[dict]:
    needs = _gym_needs(user)
    return [{**machine, "gyms": poolview.gyms_for(machine, needs)} for machine in machines]


@router.get("/api/pool/machines")
def pool_machines(user: User = Depends(current_user)):
    bridge, reported = _bridge(user)
    client = pool_client()
    if client is None:
        if reported is None:
            return {"configured": False, "machines": [], "can_add": False, "bridge": bridge}
        # The board cannot reach the pool server (it is on a LAN): show what the dispatcher saw.
        # A report from a bridge that has gone quiet is old news: the page says the states are unknown.
        return {"configured": True, "source": "bridge", "pool": reported["pool"], "reported_at": reported["at"], "can_add": False, "partial": bool(reported.get("partial")),
                "stale": not bridge["online"],
                "machines": _with_gyms(reported["machines"], user), "islands": reported["islands"], "bridge": bridge}
    pool = _pool_name()
    suffix = f"?pool={pool}" if pool else ""
    try:
        reply = client.call("GET", "/v1/machines" + suffix, timeout=15)
    except Exception as exc:
        return {"configured": True, "source": "board", "machines": [], "can_add": user.is_admin, "bridge": bridge,
                "error": f"The pool server did not answer: {exc}"}
    try:
        islands = [{"machines": group} for group in client.call("GET", "/v1/links" + suffix, timeout=15).get("islands") or []]
    except Exception:
        islands = []
    return {"configured": True, "source": "board", "pool": reply.get("pool"), "can_add": user.is_admin, "bridge": bridge,
            "machines": _with_gyms([poolview.machine_view(m) for m in reply.get("machines") or []], user), "islands": islands}


class MachineCode(BaseModel):
    name: str = Field("", max_length=40, pattern=r"^[A-Za-z0-9._-]*$")


@router.post("/api/pool/machine-codes")
def pool_machine_code(body: MachineCode, _admin: User = Depends(require_admin)):
    client = pool_client()
    if client is None:
        raise HTTPException(409, "No pool is configured for this board (GYMHUB_POOL_SERVER, GYMHUB_POOL_TOKEN).")
    try:
        reply = client.call("POST", "/v1/machine-codes", {"pool": _pool_name(), "name": body.name.strip()}, timeout=15)
    except Exception as exc:
        raise HTTPException(502, f"The pool server refused a machine code: {exc}")
    keys = ("code", "pool", "expires_at", "install", "install_no_sudo", "sha256")
    return {key: reply[key] for key in keys if reply.get(key) is not None}
