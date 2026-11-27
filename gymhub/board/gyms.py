"""Gym sharing, Explore, forks and owner stats (gymhub/gymaccess.py has the rules).

    GET  /api/showcase                   anyone, signed out too: public gyms for the landing page (no names)
    GET  /api/explore?q=&tag=            public gyms: search, tags, hardware badge
    GET  /api/gyms/{gym}/settings        owner and editors
    PUT  /api/gyms/{gym}/settings        owner: visibility, editors, people it is shared with, forks, tags
    POST /api/gyms/{gym}/share-code      owner: a new join link (the old one stops working)
    POST /api/gyms/{gym}/join            anyone with the gym's join code
    POST /api/gyms/{gym}/leave           a learner or editor removes themselves (progress is kept)
    POST /api/gyms/{gym}/transfer        owner: hand the gym to another member (the old owner becomes an editor)
    DELETE /api/gyms/{gym}               owner: delete an uploaded gym (learners' progress and history are kept)
    POST /api/gyms/{gym}/fork            copy the live version into a new private gym you own
    GET  /api/gyms/{gym}/stats           owner and editors: per exercise, who started, solved, got stuck
    GET  /api/admin/gym-requests         admins: gyms waiting to become public (when the board reviews them)
    POST /api/admin/gym-requests/{gym}   admins: approve or decline
    POST /api/gyms/{gym}/report          anyone who can open a gym: report it to the board's admins
    GET  /api/admin/gym-reports          admins: reported gyms, most reported first
    POST /api/admin/gym-reports/{gym}    admins: dismiss the reports, hide the gym, or remove it

Moderation (public beta): an admin approves members' gyms before they go public (on by default), and
anyone can report a gym. An admin who hides a gym makes it private and locks it there (`hidden`): only
the owner and editors still see it, and making it shared, unlisted or public again needs an admin.
"""

import hmac
import json
import tempfile
import time
from collections import Counter
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from typing import Any

from gymhub import gymaccess, gymdraft, gymversions, jobqueue, kinds, leaderboard, packages, quotas, tutor
from gymhub.blobs import BlobMissing, open_blobs
from gymhub.board import auth
from gymhub.board.auth import User, can_run, current_user, require_admin
from gymhub.store import new_id, now, problem_id

router = APIRouter()


def _board():
    from gymhub.board import app

    return app


def _emails(uids: list[str]) -> list[str]:
    store = _board().store
    return sorted(filter(None, ((store.get("users", uid) or {}).get("email") for uid in uids)))


def _uids_for(emails: list[str]) -> tuple[list[str], list[str]]:
    """(uids of active board members, emails nobody on the board has yet)."""
    store = _board().store
    uids, unknown, bad = [], [], []
    for email in sorted({email.strip().lower() for email in emails if email.strip()}):
        found = [uid for uid, user in store.where("users", "email", email) if user.get("status") == "active"]
        if found:
            uids.extend(found)
        elif gymaccess.EMAIL.match(email):
            unknown.append(email)
        else:
            bad.append(email)
    if bad:
        raise HTTPException(400, f"Not an email address: {', '.join(bad)}")
    return uids, unknown


SHOWCASE_GYMS = 12
SHOWCASE_SECONDS = 300  # anyone can call it, so the store is read at most once per instance per 5 minutes
_showcase_cache: dict = {}


@router.get("/api/showcase")
def showcase():
    """Signed-out landing page: what this board offers. Public gyms only, and only what Explore
    would show anyone on the board, minus who owns them. No statements, no people."""
    store = _board().store
    cached = _showcase_cache.get(id(store))
    if cached and time.monotonic() - cached[0] < SHOWCASE_SECONDS:
        return cached[1]
    gyms = []
    for gym_id, doc in store.recent("gyms", "updated_at", 200):
        if gymaccess.visibility(doc) != "public":
            continue
        problems = doc.get("problems", 0)
        gyms.append({
            "id": gym_id, "title": str(doc.get("title") or gym_id)[:120],
            "description": str(doc.get("description") or "")[:300],
            "problems": problems if isinstance(problems, int) else 0,
            "gpu": bool((doc.get("requires") or {}).get("gpu")),
            "langs": [str(lang.get("label", lang.get("id"))) for lang in doc.get("langs") or [] if isinstance(lang, dict)][:6],
            "order": doc.get("order") if isinstance(doc.get("order"), (int, float)) else 99,
        })
    gyms.sort(key=lambda gym: (gym["order"], gym["title"].lower()))
    reply = {
        "gyms": [{key: value for key, value in gym.items() if key != "order"} for gym in gyms[:SHOWCASE_GYMS]],
        "total_gyms": len(gyms), "total_exercises": sum(gym["problems"] for gym in gyms),
    }
    _showcase_cache.clear()  # one store per process in production; tests make a new one each time
    _showcase_cache[id(store)] = (time.monotonic(), reply)
    return reply


@router.get("/api/explore")
def explore(q: str = "", tag: str = "", user: User = Depends(current_user)):
    store = _board().store
    query, tag = q.strip().lower()[:80], tag.strip().lower()[:24]
    gyms, tags, names = [], Counter(), {}
    for gym_id, doc in store.recent("gyms", "updated_at", 200):
        if gymaccess.visibility(doc) != "public":
            continue
        doc_tags = doc.get("tags") or []
        tags.update(doc_tags)
        text = " ".join([gym_id, doc.get("title", ""), doc.get("description", ""), *doc_tags]).lower()
        if (query and query not in text) or (tag and tag not in doc_tags):
            continue
        owner = doc.get("owner_uid")
        if owner and owner not in names:
            names[owner] = (store.get("users", owner) or {}).get("name") or "a member"
        current = gymaccess.role(user.uid, user.is_admin, doc)
        gyms.append({
            "id": gym_id, "title": doc.get("title", gym_id), "description": doc.get("description", ""),
            "tags": doc_tags, "gpu": bool((doc.get("requires") or {}).get("gpu")),
            "langs": [lang.get("label", lang.get("id")) for lang in doc.get("langs") or []],
            "problems": doc.get("problems", 0), "by": names.get(owner, "this board"), "version": doc.get("version"),
            "forked_from": doc.get("forked_from"), "can_fork": gymaccess.can_fork(current, doc), "order": doc.get("order") if isinstance(doc.get("order"), (int, float)) else 99,
        })
    gyms.sort(key=lambda gym: (gym["order"], gym["title"].lower()))
    return {"gyms": gyms, "tags": [name for name, _count in tags.most_common(20)]}


def _settings(gym: str, doc: dict, current: str) -> dict:
    reply = {
        "role": current, "visibility": gymaccess.visibility(doc), "allow_fork": bool(doc.get("allow_fork")),
        "tags": doc.get("tags") or [], "editors": _emails(doc.get("editors") or []),
        "learners": _emails(doc.get("learners") or []), "uploaded": doc.get("source") == "bundle",
        **{f"pending_{role}s": emails for role, emails in gymaccess.pending(_board().store, gym).items()},
        "has_draft": _board().store.get("gym_drafts", gym) is not None,
        "tutor": doc.get("tutor", True), "tutor_available": tutor.enabled(), "leaderboard": bool(doc.get("leaderboard")),
        "visibility_requested": doc.get("visibility_requested"),
        "public_needs_review": bool(auth.auth_settings().get("public_gyms_need_review")),
        "hidden": doc.get("hidden"),
    }
    if current == "owner":
        reply["share_code"] = doc.get("share_code")
    return reply


@router.get("/api/gyms/{gym}/settings")
def gym_settings(gym: str, user: User = Depends(current_user)):
    board = _board()
    doc = board.require_gym(user, gym, "editor")
    return _settings(gym, doc, board.gym_role(user, doc))


class GymSettings(BaseModel):
    visibility: str
    allow_fork: bool = False
    tags: list[str] = Field(default_factory=list, max_length=20)
    editor_emails: list[str] = Field(default_factory=list, max_length=50)
    learner_emails: list[str] = Field(default_factory=list, max_length=500)
    tutor: bool = True
    leaderboard: bool = False
    # Narrowing a public gym to shared or unlisted: keep everyone who has practised it (#119).
    keep_learners: bool = True


def _practising(gym: str) -> list[str]:
    """Everyone with progress in this gym."""
    return [item["uid"] for _id, item in _board().store.where("user_progress", "gym", gym)
            if item.get("uid") and (item.get("data") or {}).get("exercises")]


@router.put("/api/gyms/{gym}/settings")
def update_gym_settings(gym: str, body: GymSettings, user: User = Depends(current_user)):
    board = _board()
    doc = board.require_gym(user, gym, "owner")
    if body.visibility not in gymaccess.VISIBILITIES:
        raise HTTPException(400, f"visibility must be one of {', '.join(gymaccess.VISIBILITIES)}")
    owner = doc.get("owner_uid")
    before = gymaccess.visibility(doc)
    target = body.visibility
    fields = {"tags": gymaccess.clean_tags(body.tags), "tutor": body.tutor, "leaderboard": body.leaderboard,
              "visibility_requested": None}
    hidden = doc.get("hidden") and not user.is_admin
    if hidden and target in {"shared", "unlisted"}:
        raise HTTPException(403, "An admin hid this gym after reports, so it stays private. Ask to make it public to have it reviewed again.")
    if user.is_admin and target != "private":
        fields["hidden"] = None  # an admin opening it again lifts the hold
    if target == "public" and before != "public" and not user.is_admin and (hidden or auth.auth_settings().get("public_gyms_need_review")):
        # An admin approves first (#121); until then the gym keeps its current visibility.
        target = before
        fields.update({"visibility_requested": "public", "requested_at": now(), "requested_by": user.email})
    fields["visibility"] = target
    invites: dict[str, str] = {}  # people not on the board yet, listed once they sign in (#118)
    learners, invites_learners = _uids_for(body.learner_emails)
    invites.update(dict.fromkeys(invites_learners, "learner"))
    if doc.get("source") == "bundle":  # local gyms have no editors (versions come from a machine) and no bundle to fork
        editors, invites_editors = _uids_for(body.editor_emails)
        fields["editors"] = [uid for uid in editors if uid != owner]
        invites.update(dict.fromkeys(invites_editors, "editor"))  # editor wins over learner
        fields["allow_fork"] = body.allow_fork
    fields["learners"] = [uid for uid in learners if uid != owner]
    kept = 0
    if body.keep_learners and before == "public" and target in {"shared", "unlisted"}:
        extra = [uid for uid in _practising(gym) if uid != owner and uid not in fields["learners"]
                 and uid not in fields.get("editors", doc.get("editors") or [])]
        fields["learners"].extend(extra)
        kept = len(extra)
    gymaccess.set_pending(board.store, gym, invites, user.email)
    if target in {"shared", "unlisted"} and not doc.get("share_code"):
        fields["share_code"] = gymaccess.new_share_code()
    board.store.set("gyms", gym, fields, merge=True)
    doc = board.store.get("gyms", gym)
    return {**_settings(gym, doc, "owner"), "kept_learners": kept}


@router.post("/api/gyms/{gym}/share-code")
def new_share_code(gym: str, user: User = Depends(current_user)):
    board = _board()
    board.require_gym(user, gym, "owner")
    code = gymaccess.new_share_code()
    board.store.set("gyms", gym, {"share_code": code}, merge=True)
    return {"share_code": code}


class Join(BaseModel):
    code: str = Field(max_length=64)


@router.post("/api/gyms/{gym}/join")
def join_gym(gym: str, body: Join, user: User = Depends(current_user)):
    store = _board().store
    doc = store.get("gyms", gym)
    expected = (doc or {}).get("share_code") or ""
    if not doc or not expected or gymaccess.visibility(doc) == "private" or not hmac.compare_digest(expected, body.code):
        raise HTTPException(404, "This join link is not valid. Ask the gym's owner for a new one.")
    learners = doc.get("learners") or []
    if user.uid not in learners and doc.get("owner_uid") != user.uid:
        store.set("gyms", gym, {"learners": [*learners, user.uid]}, merge=True)
    return {"ok": True, "gym": gym}


@router.post("/api/gyms/{gym}/leave")
def leave_gym(gym: str, user: User = Depends(current_user)):
    """Off this user's Gyms page, and no longer able to open a shared or unlisted gym. Progress stays,
    so joining again resumes where they were."""
    board = _board()
    doc = board.require_gym(user, gym)
    if doc.get("owner_uid") == user.uid:
        raise HTTPException(400, "The owner cannot leave their gym. Transfer it to someone else, or delete it.")

    def drop(current):
        if current is None:
            return None
        for field in ("learners", "editors"):
            current[field] = [uid for uid in current.get(field) or [] if uid != user.uid]
        return current

    board.store.transact("gyms", gym, drop)
    return {"ok": True, "still_open": gymaccess.visibility(doc) == "public"}


class Transfer(BaseModel):
    email: str = Field(max_length=200)


@router.post("/api/gyms/{gym}/transfer")
def transfer_gym(gym: str, body: Transfer, user: User = Depends(current_user)):
    board = _board()
    store = board.store
    doc = board.require_gym(user, gym, "owner")
    if doc.get("source") != "bundle":
        raise HTTPException(400, "Gyms published from a machine's local files belong to the board's admins")
    uids, _unknown = _uids_for([body.email])
    if not uids:
        raise HTTPException(400, f"{body.email.strip().lower()} is not on this board yet. They need to sign in once first.")
    new = uids[0]
    old = doc.get("owner_uid")
    if new == old:
        raise HTTPException(400, "They already own this gym")
    new_doc = store.get("users", new) or {}
    if new_doc.get("role") != "admin" and len(store.where("gyms", "owner_uid", new)) >= board.MAX_GYMS_PER_MEMBER:
        raise HTTPException(400, f"{new_doc.get('email')} already owns {board.MAX_GYMS_PER_MEMBER} gyms, the most a member may have")
    editors = [uid for uid in doc.get("editors") or [] if uid != new]
    if old and old not in editors:
        editors.append(old)  # the old owner keeps working on it
    store.set("gyms", gym, {"owner_uid": new, "owner_email": new_doc.get("email"), "editors": editors,
                            "learners": [uid for uid in doc.get("learners") or [] if uid != new]}, merge=True)
    if store.get("gym_drafts", gym) is not None:
        store.set("gym_drafts", gym, {"owner_uid": new}, merge=True)
    return {"ok": True, "owner": new_doc.get("email")}


class DeleteGym(BaseModel):
    confirm: str = Field(max_length=40)


@router.delete("/api/gyms/{gym}")
def delete_gym(gym: str, body: DeleteGym, user: User = Depends(current_user)):
    """Gone for everyone. Learners keep their progress and submissions; the id stays reserved for its
    owner (gym_tombstones), so nobody else can publish a different gym under it."""
    board = _board()
    store = board.store
    doc = board.require_gym(user, gym, "owner")
    if doc.get("source") != "bundle":
        raise HTTPException(400, "Gyms published from a machine's local files come back from that machine; make it private instead")
    if body.confirm != gym:
        raise HTTPException(400, f"Type the gym's id ({gym}) to confirm")
    remove_gym(store, gym, doc, user.email, reserve_for=doc.get("owner_uid"))
    return {"ok": True}


def remove_gym(store, gym: str, doc: dict, deleted_by: str, reserve_for: str | None) -> None:
    """Delete a gym for everyone (its owner's Delete, or their account deletion: gymhub/accounts.py).

    The tombstone keeps the id reserved for the uid `reserve_for`; None leaves it to admins."""
    for job_id, job in store.where("jobs", "status", "queued"):
        if job.get("gym") == gym:
            store.claim("jobs", job_id, {"status": "queued"}, {"status": "cancelled", "finished_at": now()})
    for job_id, job in store.where("jobs", "status", "running"):
        if job.get("gym") == gym:
            store.set("jobs", job_id, {"cancel": True}, merge=True)
    for version_id, version in store.where("gym_versions", "gym", gym):
        if version.get("status") == "validating":  # would publish the gym again when it finishes
            store.set("gym_versions", version_id, {"status": "failed", "errors": ["the gym was deleted"], "finished_at": now()}, merge=True)
    for collection in ("problems", "answer_keys", "gym_invites"):
        for doc_id, _doc in store.where(collection, "gym", gym):
            store.delete(collection, doc_id)
    for collection in ("gym_drafts", "gym_sources"):
        store.delete(collection, gym)
    store.set("gym_tombstones", gym, {"owner_uid": reserve_for,
                                      "title": doc.get("title", gym), "deleted_at": now(), "deleted_by": deleted_by})
    store.delete("gyms", gym)


class Fork(BaseModel):
    id: str = Field(max_length=40)
    title: str = Field(default="", max_length=120)


@router.post("/api/gyms/{gym}/fork")
def fork_gym(gym: str, body: Fork, user: User = Depends(current_user)):
    board = _board()
    doc = board.require_gym(user, gym)
    if not gymaccess.can_fork(board.gym_role(user, doc), doc):
        raise HTTPException(403, "This gym's owner does not allow forks")
    if not can_run(user):
        raise HTTPException(403, "Creating gyms runs their code on a machine, which is not enabled for members yet.")
    if not packages.GYM_ID.match(body.id):
        raise HTTPException(400, "The new gym's id must be 3-40 characters: lowercase letters, digits and -")
    try:
        data = open_blobs().get(doc["bundle"]["key"])
    except (BlobMissing, KeyError):
        raise HTTPException(409, "This gym's files are missing; it cannot be forked") from None
    with tempfile.TemporaryDirectory(prefix="gym-fork-") as tmp:
        package = packages.unpack(data, Path(tmp) / "pkg")
        manifest = {**package.manifest, "id": body.id, "title": body.title.strip() or f"{package.manifest.get('title', gym)} (fork)"}
        (package.root / "gym.json").write_text(json.dumps(manifest, indent=2) + "\n")
        data = packages.pack(package.root)
        spec = gymdraft.from_package(packages.load(package.root))  # made in the browser editor: editable there too
    try:
        reply = board._start_upload(user, data, {"forked_from": {"gym": gym, "version": doc.get("version")}})
    except gymversions.UploadRejected as exc:
        raise HTTPException(400, str(exc)) from exc
    if spec is not None and board.store.get("gym_drafts", body.id) is None:
        board.store.set("gym_drafts", body.id, {"spec": spec, "updated_at": now(), "updated_by": user.email, "owner_uid": user.uid})
    return reply


@router.get("/api/admin/gym-requests")
def gym_requests(_admin: User = Depends(require_admin)):
    store = _board().store
    rows = []
    for gym_id, doc in store.where("gyms", "visibility_requested", "public"):
        rows.append({"id": gym_id, "title": doc.get("title", gym_id), "description": doc.get("description", ""),
                     "owner": doc.get("owner_email") or doc.get("requested_by"), "problems": doc.get("problems", 0),
                     "visibility": gymaccess.visibility(doc), "requested_at": doc.get("requested_at"),
                     "requested_by": doc.get("requested_by")})
    rows.sort(key=lambda row: row.get("requested_at") or 0)
    return {"requests": rows}


class Decision(BaseModel):
    approve: bool


@router.post("/api/admin/gym-requests/{gym}")
def decide_gym_request(gym: str, body: Decision, admin: User = Depends(require_admin)):
    store = _board().store
    doc = store.get("gyms", gym)
    if not doc or doc.get("visibility_requested") != "public":
        raise HTTPException(404, "No request for this gym")
    fields = {"visibility_requested": None, "reviewed_by": admin.email, "reviewed_at": now()}
    if body.approve:
        fields.update({"visibility": "public", "hidden": None})
    else:
        fields["visibility_declined_at"] = now()
    store.set("gyms", gym, fields, merge=True)
    return {"ok": True, "visibility": "public" if body.approve else gymaccess.visibility(doc)}


# ---- reports and moderation ----------------------------------------------------------------------
#
#   gym_reports/{gym}__{uid}   gym, uid, email, reason, details, created_at, status (open|dismissed|hidden|removed),
#                              resolved_by, resolved_at. One per person and gym: reporting again replaces it.

REPORT_REASONS = {
    "spam": "Spam or advertising",
    "offensive": "Offensive, hateful or harassing",
    "harmful": "Harmful code or instructions (malware, credential theft)",
    "copyright": "Copies someone else's work without permission",
    "personal": "Shares someone's personal information",
    "broken": "Broken: exercises cannot be solved",
    "other": "Something else",
}
MODERATION_ACTIONS = ("dismiss", "hide", "remove")


class Report(BaseModel):
    reason: str = Field(max_length=20)
    details: str = Field(default="", max_length=2000)


@router.get("/api/report-reasons")
def report_reasons(_user: User = Depends(current_user)):
    return {"reasons": [{"id": key, "label": label} for key, label in REPORT_REASONS.items()]}


@router.post("/api/gyms/{gym}/report")
def report_gym(gym: str, body: Report, user: User = Depends(current_user)):
    board = _board()
    doc = board.require_gym(user, gym)
    if doc.get("owner_uid") == user.uid:
        raise HTTPException(400, "This is your gym. Make it private or delete it from Manage this gym.")
    if body.reason not in REPORT_REASONS:
        raise HTTPException(400, f"reason is one of {', '.join(REPORT_REASONS)}")
    if body.reason == "other" and not body.details.strip():
        raise HTTPException(400, "Say what is wrong with it")
    board.store.set("gym_reports", f"{gym}__{user.uid}", {
        "gym": gym, "uid": user.uid, "email": user.email, "reason": body.reason, "details": body.details.strip(),
        "created_at": now(), "status": "open", "version": doc.get("version"),
    })
    return {"ok": True}


def _open_reports(store, gym: str | None = None) -> list[tuple[str, dict]]:
    rows = store.where("gym_reports", "gym", gym) if gym else store.where("gym_reports", "status", "open")
    return [(doc_id, report) for doc_id, report in rows if report.get("status") == "open"]


@router.get("/api/admin/gym-reports")
def gym_reports(_admin: User = Depends(require_admin)):
    store = _board().store
    by_gym: dict[str, list[dict]] = {}
    for _id, report in _open_reports(store):
        by_gym.setdefault(report["gym"], []).append(report)
    rows = []
    for gym_id, reports in by_gym.items():
        doc = store.get("gyms", gym_id)
        if doc is None:
            continue  # its owner deleted it
        reports.sort(key=lambda report: report.get("created_at") or 0)
        rows.append({
            "id": gym_id, "title": doc.get("title", gym_id), "owner": doc.get("owner_email") or "this board",
            "visibility": gymaccess.visibility(doc), "hidden": bool(doc.get("hidden")), "uploaded": doc.get("source") == "bundle",
            "count": len(reports), "reasons": dict(Counter(report.get("reason") for report in reports)),
            "first_at": reports[0].get("created_at"),
            "reports": [{"by": report.get("email"), "reason": report.get("reason"), "details": report.get("details", ""),
                         "at": report.get("created_at")} for report in reports],
        })
    rows.sort(key=lambda row: (-row["count"], row["first_at"] or 0))
    return {"reports": rows, "reasons": REPORT_REASONS}


class Moderation(BaseModel):
    action: str = Field(max_length=20)
    note: str = Field(default="", max_length=500)


@router.post("/api/admin/gym-reports/{gym}")
def moderate_gym(gym: str, body: Moderation, admin: User = Depends(require_admin)):
    """dismiss: the reports are closed, the gym stays as it is. hide: private and locked there until an admin
    opens it again. remove: deleted for everyone, as if its owner had (uploaded gyms only)."""
    store = _board().store
    if body.action not in MODERATION_ACTIONS:
        raise HTTPException(400, f"action is one of {', '.join(MODERATION_ACTIONS)}")
    doc = store.get("gyms", gym)
    if doc is None:
        raise HTTPException(404, f"No gym {gym!r}")
    if body.action == "remove":
        if doc.get("source") != "bundle":
            raise HTTPException(400, "Gyms published from a machine's local files come back from that machine; hide it instead")
        remove_gym(store, gym, doc, admin.email, reserve_for=doc.get("owner_uid"))
    elif body.action == "hide":
        store.set("gyms", gym, {"visibility": "private", "visibility_requested": None,
                                "hidden": {"by": admin.email, "at": now(), "note": body.note.strip()}}, merge=True)
    status = {"dismiss": "dismissed", "hide": "hidden", "remove": "removed"}[body.action]
    for doc_id, _report in _open_reports(store, gym):
        store.set("gym_reports", doc_id, {"status": status, "resolved_by": admin.email, "resolved_at": now()}, merge=True)
    return {"ok": True, "status": status}


@router.get("/api/gyms/{gym}/stats")
def gym_stats(gym: str, user: User = Depends(current_user)):
    """Aggregates only, never who: how many people started and solved each exercise, and where they get stuck."""
    board = _board()
    doc = board.require_gym(user, gym, "editor")
    store = board.store
    progresses = [(item.get("data") or {}) for _id, item in store.where("user_progress", "gym", gym)]
    if doc.get("source") != "bundle":
        progresses.append((store.get("progress", gym) or {}).get("data") or {})  # the owner's local files
    problems = sorted((item for _id, item in store.where("problems", "gym", gym)), key=lambda item: item.get("order", 0))
    rows = []
    for problem in problems:
        slug = problem["slug"]
        started = solved = hints = 0
        attempts_solved, attempts_stuck = [], []
        for progress in progresses:
            entry = (progress.get("exercises") or {}).get(slug)
            if not entry:
                continue
            langs = (entry.get("langs") or {}).values()
            attempts = sum(int(lang.get("attempts") or 0) for lang in langs)
            if not attempts and not entry.get("hints_seen"):
                continue
            started += 1
            hints += int(entry.get("hints_seen") or 0)
            if any("first_solved" in lang for lang in langs):
                solved += 1
                attempts_solved.append(attempts)
            else:
                attempts_stuck.append(attempts)
        rows.append({
            "slug": slug, "title": problem.get("title", slug), "started": started, "solved": solved,
            "stuck": started - solved, "solve_rate": round(solved / started, 2) if started else None,
            "attempts_to_solve": round(sum(attempts_solved) / len(attempts_solved), 1) if attempts_solved else None,
            "attempts_while_stuck": round(sum(attempts_stuck) / len(attempts_stuck), 1) if attempts_stuck else None,
            "hints_per_learner": round(hints / started, 1) if started else None,
        })
    learners = sum(1 for progress in progresses if progress.get("exercises"))
    return {"learners": learners, "problems": rows}


# ---- answer, choice, essay and checklist exercises (gymhub/kinds.py) -----------------------------


class Answer(BaseModel):
    response: Any = None


@router.post("/api/gyms/{gym}/problems/{slug}/answer")
def submit_answer(gym: str, slug: str, body: Answer, user: User = Depends(current_user)):
    """Grade a non-code answer on the board, record progress and history. No machine involved."""
    import time

    board = _board()
    store = board.store
    gym_doc = board.require_gym(user, gym)
    problem = store.get("problems", problem_id(gym, slug))
    if problem is None:
        raise HTTPException(404, "Unknown exercise")
    kind = problem.get("kind", "code")
    if kind == "code":
        raise HTTPException(400, "Code exercises are submitted as jobs")
    if not board.uses_workspace(user, gym_doc):
        raise HTTPException(400, "Answer exercises are supported in uploaded gyms")
    if len(json.dumps(body.response)) > kinds.MAX_RESPONSE + 1000:
        raise HTTPException(413, "That answer is too long")
    ai_grading = not tutor.board_cap_reached(store)
    essay_by_llm = kind == "essay" and not isinstance(body.response, dict) and ai_grading and tutor.enabled()
    if essay_by_llm and not user.is_admin:
        try:
            quotas.check_and_count(store, user.uid)  # an essay costs a model call
        except quotas.QuotaExceeded as exc:
            raise HTTPException(429, str(exc)) from exc
    attempt_counted = not user.is_admin and not essay_by_llm
    if attempt_counted:
        try:
            quotas.begin_attempt(store, user.uid, problem_id(gym, slug))
        except quotas.QuotaExceeded as exc:
            raise HTTPException(429, str(exc)) from exc
    key = (store.get("answer_keys", problem_id(gym, slug)) or {}).get("key")
    try:
        result = kinds.grade(kind, problem, key, body.response, problem.get("statement", ""), ai=ai_grading)
    except kinds.BadResponse as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # the essay grader's API failed: nothing recorded, try again
        raise HTTPException(502, f"The grader is unavailable right now ({type(exc).__name__}). Try again in a minute.") from exc

    if "usd" in result:
        tutor.charge_board(store, result.pop("usd"))
    if attempt_counted:
        quotas.finish_attempt(store, user.uid, problem_id(gym, slug), bool(result["ok"]))
    progress = board.progress_for(user, gym, gym_doc)
    entry = progress["exercises"].setdefault(slug, {"hints_seen": 0, "langs": {}})
    state = entry.setdefault("langs", {}).setdefault(kind, {})
    state["attempts"] = int(state.get("attempts", 0)) + 1
    state["last_attempt"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if "checked" in result:
        state["checked"] = result["checked"]
    if result["ok"] and "first_solved" not in state:
        state["first_solved"] = state["last_attempt"]
    board._save_user_progress(user, gym, progress)

    job_id = new_id()
    job = {
        "gym": gym, "slug": slug, "problem": problem_id(gym, slug), "lang": kind, "action": "answer",
        "code": json.dumps(body.response), "options": {}, "status": "done", "ok": bool(result["ok"]),
        "output": result.get("feedback", ""), "created_at": now(), "started_at": now(), "finished_at": now(),
        "created_by": user.email, "uid": user.uid, "isolated": True, "tests_hash": problem.get("tests_hash"),
        "bundle": gym_doc.get("bundle"),
    }
    store.set("jobs", job_id, job)
    if result["ok"]:
        jobqueue.record_solve(store, {**job, "id": job_id})
    return {**result, "job": job_id, "progress": state}


# ---- the AI tutor (gymhub/tutor.py) --------------------------------------------------------------


class TutorRequest(BaseModel):
    action: str
    lang: str = Field(default="", max_length=20)
    code: str | None = Field(default=None, max_length=200_000)
    job_id: str | None = Field(default=None, max_length=80)


@router.get("/api/my/tutor")
def my_tutor(user: User = Depends(current_user)):
    store = _board().store
    return {"enabled": tutor.enabled(), "budget": None if user.is_admin else tutor.monthly_budget(),
            "usage": tutor.usage(store, user.uid), "paused": tutor.board_cap_reached(store)}


@router.post("/api/gyms/{gym}/problems/{slug}/tutor")
def ask_tutor(gym: str, slug: str, body: TutorRequest, user: User = Depends(current_user)):
    board = _board()
    store = board.store
    gym_doc = board.require_gym(user, gym)
    if not gym_doc.get("tutor", True):
        raise HTTPException(403, "The owner of this gym turned the tutor off")
    problem = store.get("problems", problem_id(gym, slug))
    if problem is None:
        raise HTTPException(404, "Unknown exercise")
    if body.action not in tutor.ACTIONS:
        raise HTTPException(400, f"action is one of {', '.join(sorted(tutor.ACTIONS))}")
    code, output, lang = body.code or "", "", body.lang
    if body.action == "explain":
        job = store.get("jobs", body.job_id or "")
        if not job or not board._visible(user, job) or job.get("problem") != problem_id(gym, slug):
            raise HTTPException(404, "Pick one of your submissions of this exercise")
        if job.get("ok"):
            raise HTTPException(400, "That submission passed; ask for a review instead")
        output, lang = job.get("output") or "", job.get("lang") or lang
        code = job.get("code") or code
    if body.action != "hint" and not code.strip():
        raise HTTPException(400, "There is no code to look at yet")
    prompt = tutor.build_prompt(
        body.action, gym_title=gym_doc.get("title", gym), lang=lang or "", statement=problem.get("statement", ""),
        hints=problem.get("hints") or [], code=code, output=output,
    )
    try:
        return tutor.ask(store, user.uid, user.is_admin, body.action, prompt)
    except tutor.TutorUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except tutor.BudgetExceeded as exc:
        raise HTTPException(429, str(exc)) from exc
    except Exception as exc:  # the API failed; nothing was charged
        raise HTTPException(502, f"The tutor is unavailable right now ({type(exc).__name__}). Try again in a minute.") from exc


# ---- leaderboards (gymhub/leaderboard.py) -----------------------------------------------------------


@router.get("/api/gyms/{gym}/problems/{slug}/leaderboard")
def problem_leaderboard(gym: str, slug: str, lang: str | None = None, hardware: str | None = None,
                        user: User = Depends(current_user)):
    board = _board()
    gym_doc = board.require_gym(user, gym)
    if not gym_doc.get("leaderboard"):
        raise HTTPException(404, "This gym has no leaderboard")
    return leaderboard.standings(board.store, gym, slug, user.uid, lang, hardware)
