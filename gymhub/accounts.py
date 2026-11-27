"""A member's own data: export it all, or delete the account and everything tied to it.

Both are self-service (Your data page, /api/me/export and DELETE /api/me) and cover every collection
that holds a uid or an email (gymhub/store.py lists them). What deletion does, by collection:

    users/{uid}                    deleted (email, name, photo URL, role, sign-in times, settings)
    user_progress, workspaces,     deleted: progress, saved code, submissions with their code and
    jobs, solves, bests, reviews,  output, best times, review queue, leaderboard rows, API tokens,
    leaderboard, api_tokens,       usage counters, AI tutor spend
    usage, tutor_usage,            reports they filed about gyms
    gym_reports
    gyms the member owns           deleted for everyone (as the owner's Delete does, gymhub/board/gyms.py
                                   remove_gym), with their versions, drafts and uploaded bundles; the id
                                   stays reserved, for admins only
    gyms the member was invited to the uid is taken off editors and learners; invites to their email go
    gym_versions they uploaded     to another owner's gym: kept (the gym needs its history), with the
    tombstones of their gyms       uploader's email replaced by "deleted account"
    workers (machines)             revoked and deleted; their runners stop being listed

Kept: board invites an admin created (admins cannot delete themselves here while on the admin list)
and the per-attempt throttle documents, which hold no email and expire within a day by themselves.
"""

from gymhub.blobs import BlobMissing, bundle_key
from gymhub.store import Store, now

DELETED = "deleted account"

# Collections whose documents carry the member's uid in a "uid" field and belong to nobody else.
# passwords: a local account's password hash (GYMHUB_AUTH=local; "hash" is never exported).
# gym_reports: reports this member filed about gyms (moderation, PR #178), with their email.
OWN_BY_UID = ("user_progress", "workspaces", "jobs", "solves", "bests", "reviews", "leaderboard", "api_tokens", "tutor_usage",
              "gym_reports", "editorial_views", "passwords")
SECRET_FIELDS = {"token_hash", "hash"}


def _strip(doc: dict) -> dict:
    return {key: value for key, value in doc.items() if key not in SECRET_FIELDS}


def owned_gyms(store: Store, uid: str) -> list[tuple[str, dict]]:
    return store.where("gyms", "owner_uid", uid)


def export(store: Store, uid: str) -> dict:
    """Everything the board holds about the member, as JSON-ready data (secrets left out)."""
    user = store.get("users", uid) or {}
    email = user.get("email")
    data: dict = {"exported_at": now(), "uid": uid, "account": _strip(user)}
    for collection in OWN_BY_UID:
        data[collection] = {doc_id: _strip(doc) for doc_id, doc in store.where(collection, "uid", uid)}
    data["usage"] = {uid: store.get("usage", uid)} if store.get("usage", uid) else {}
    data["gyms_owned"] = {gym_id: _strip(doc) for gym_id, doc in owned_gyms(store, uid)}
    data["gym_roles"] = {}
    for field, role in (("editors", "editor"), ("learners", "learner")):
        for gym_id, _doc in store.where_contains("gyms", field, uid):
            data["gym_roles"].setdefault(gym_id, []).append(role)
    data["gym_drafts"] = {gym_id: doc for gym_id, doc in store.where("gym_drafts", "owner_uid", uid)}
    data["gym_versions_uploaded"] = {doc_id: doc for doc_id, doc in store.where("gym_versions", "uid", uid)}
    data["gym_invites"] = {doc_id: doc for doc_id, doc in store.where("gym_invites", "email", email)} if email else {}
    data["machines"] = {doc_id: _strip(doc) for doc_id, doc in store.where("workers", "owner_uid", uid)}
    return data


def preview(store: Store, uid: str) -> dict:
    """What deleting the account removes that other people notice, for the confirmation dialog."""
    gyms = [{"id": gym_id, "title": doc.get("title", gym_id), "visibility": doc.get("visibility", "private"),
             "learners": len(doc.get("learners") or []), "editors": len(doc.get("editors") or [])}
            for gym_id, doc in owned_gyms(store, uid)]
    machines = [{"id": doc_id, "name": doc.get("name", doc_id)}
                for doc_id, doc in store.where("workers", "owner_uid", uid) if not doc.get("revoked")]
    return {"gyms": gyms, "machines": machines}


def delete(store: Store, uid: str, blobs=None) -> dict:
    """Delete the member's account and their data. Returns counts, for the board's log."""
    from gymhub.board.gyms import remove_gym

    user = store.get("users", uid) or {}
    email = user.get("email")
    counts: dict[str, int] = {}

    def gone(collection: str, doc_id: str) -> None:
        store.delete(collection, doc_id)
        counts[collection] = counts.get(collection, 0) + 1

    # Their gyms first: removing one cancels its queued jobs and fails versions still validating.
    for gym_id, doc in owned_gyms(store, uid):
        remove_gym(store, gym_id, doc, DELETED, reserve_for=None)
        counts["gyms"] = counts.get("gyms", 0) + 1
        for version_id, version in store.where("gym_versions", "gym", gym_id):
            for key in (bundle_key(gym_id, version.get("sha", "")), f"catalogs/{gym_id}/{version.get('sha', '')}.json"):
                if blobs is not None and version.get("sha"):
                    try:
                        blobs.delete(key)
                    except BlobMissing:
                        pass
            gone("gym_versions", version_id)
    for gym_id, _doc in store.where("gym_drafts", "owner_uid", uid):
        gone("gym_drafts", gym_id)
    for gym_id, _doc in store.where("gym_tombstones", "owner_uid", uid):
        store.set("gym_tombstones", gym_id, {"owner_uid": None, "deleted_by": DELETED}, merge=True)

    # Other people's gyms: off their member lists; their version history stays, without the email.
    for field in ("editors", "learners"):
        for gym_id, _doc in store.where_contains("gyms", field, uid):
            store.transact("gyms", gym_id, lambda doc, field=field: {
                **(doc or {}), field: [member for member in (doc or {}).get(field) or [] if member != uid]})
    for version_id, _version in store.where("gym_versions", "uid", uid):
        store.set("gym_versions", version_id, {"uid": None, "created_by": DELETED}, merge=True)
    if email:
        for invite_id, _invite in store.where("gym_invites", "email", email):
            gone("gym_invites", invite_id)

    # Their machines: revoked first, so a running runner stops at its next call.
    for worker_id, _doc in store.where("workers", "owner_uid", uid):
        store.set("workers", worker_id, {"revoked": True}, merge=True)
        for runner_id, _runner in store.where("runners", "worker_id", worker_id):
            gone("runners", runner_id)
        gone("workers", worker_id)

    for collection in OWN_BY_UID:
        for doc_id, job in store.where(collection, "uid", uid):
            if collection == "jobs" and job.get("status") == "running":
                store.set("jobs", doc_id, {"cancel": True}, merge=True)  # the runner stops it; then it is deleted
            gone(collection, doc_id)
    if store.get("usage", uid) is not None:
        gone("usage", uid)
    gone("users", uid)
    return counts
