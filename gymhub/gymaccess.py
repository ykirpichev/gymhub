"""Who may see and change a gym.

    visibility   private    only the owner and editors
                 shared     plus the people it is shared with (by email, or who used its join link)
                 unlisted   anyone on the board who used its join link (the link carries a code:
                            the gym's id alone opens nothing); not listed in Explore
                 public     everyone on the board; listed in Explore

    roles        owner      everything: settings, sharing, versions, stats       (admins act as owners)
                 editor     upload versions, see stats
                 learner    practise

    gyms/{id}    visibility, owner_uid, editors[uid], learners[uid], share_code, allow_fork, tags[],
                 forked_from
    gym_invites/{email}__{gym}   email, gym, role (learner|editor), invited_by, created_at:
                 people a gym is shared with who have not signed in yet; claimed at their first
                 sign-in (#118). Board access itself still comes from board invites and domains.

Gyms a machine publishes from its local files have no owner_uid: admins own them, and they are
public unless an admin changes that. Uploaded gyms start private.

Where a gym's graders and hidden tests may go (#117): a board machine leases an uploaded gym's job
only if the gym is public or the machine's owner has a role on it (gymhub/board/workers.py). A
runner that grades on the compute pool ships the gym's graders to a pool machine, whose owner the
board does not know; so the board tells it, with each leased job, whether the gym is public
(`gym_public`, from `public` here), and a job of any other gym runs only on the runner's own pool
machines (gymhub/runner/pooled.py, the pool's resources.machines: own).
"""

import re
import secrets
import time

VISIBILITIES = ("private", "shared", "unlisted", "public")
RANK = {"learner": 1, "editor": 2, "owner": 3}
MAX_TAGS = 8
TAG = re.compile(r"^[a-z0-9][a-z0-9 +#.-]{0,23}$")
EMAIL = re.compile(r"^[^@\s,]+@[^@\s,]+\.[^@\s,]+$")


def visibility(doc: dict) -> str:
    value = doc.get("visibility")
    if value in VISIBILITIES:
        return value
    return "private" if doc.get("source") == "bundle" else "public"


def public(doc: dict | None) -> bool:
    """May anyone see this gym's content? False for a gym with no document (fail closed)."""
    return bool(doc) and visibility(doc) == "public"


def role(uid: str, is_admin: bool, doc: dict | None) -> str | None:
    if not doc:
        return None
    if is_admin or (doc.get("owner_uid") and doc.get("owner_uid") == uid):
        return "owner"
    if uid in (doc.get("editors") or []):
        return "editor"
    seen = visibility(doc)
    if seen == "public":
        return "learner"
    if seen in {"shared", "unlisted"} and uid in (doc.get("learners") or []):
        return "learner"
    return None


def at_least(current: str | None, needed: str) -> bool:
    return current is not None and RANK[current] >= RANK[needed]


def listed(uid: str, is_admin: bool, doc: dict) -> bool:
    """Shown on this user's Gyms page: public gyms, and the ones they own, edit or joined."""
    current = role(uid, is_admin, doc)
    if current is None:
        return False
    if visibility(doc) == "public" or at_least(current, "editor"):
        return True
    return uid in (doc.get("learners") or [])


def can_fork(current: str | None, doc: dict) -> bool:
    if doc.get("source") != "bundle" or current is None:
        return False  # local gyms have no bundle to copy
    return at_least(current, "editor") or bool(doc.get("allow_fork"))


def clean_tags(tags) -> list[str]:
    clean = []
    for tag in tags if isinstance(tags, list) else []:
        tag = str(tag).strip().lower()
        if TAG.match(tag) and tag not in clean:
            clean.append(tag)
    return clean[:MAX_TAGS]


def new_share_code() -> str:
    return secrets.token_urlsafe(12)


# ---- invitations for people not on the board yet (#118) -----------------------------------------


def invite_id(email: str, gym: str) -> str:
    return f"{email}__{gym}"


def pending(store, gym: str) -> dict[str, list[str]]:
    """{"learner": [emails], "editor": [emails]} invited to this gym and not signed in yet."""
    found: dict[str, list[str]] = {"learner": [], "editor": []}
    for _id, invite in store.where("gym_invites", "gym", gym):
        found.setdefault(invite.get("role", "learner"), []).append(invite["email"])
    return {role: sorted(emails) for role, emails in found.items()}


def set_pending(store, gym: str, wanted: dict[str, str], invited_by: str) -> None:
    """Make this gym's invitations exactly `wanted` (email -> role)."""
    for doc_id, invite in store.where("gym_invites", "gym", gym):
        if wanted.get(invite.get("email")) != invite.get("role"):
            store.delete("gym_invites", doc_id)
    current = pending(store, gym)
    for email, role in wanted.items():
        if email not in current.get(role, []):
            store.set("gym_invites", invite_id(email, gym),
                      {"email": email, "gym": gym, "role": role, "invited_by": invited_by, "created_at": time.time()})


def claim_invites(store, uid: str, email: str) -> list[str]:
    """A new user signed in: every gym that invited their email now lists them. Returns those gyms."""
    joined = []
    for doc_id, invite in store.where("gym_invites", "email", email.lower()):
        field = "editors" if invite.get("role") == "editor" else "learners"

        def add(doc, field=field):
            if doc is None:
                return None
            if doc.get("owner_uid") != uid and uid not in (doc.get(field) or []):
                doc[field] = [*(doc.get(field) or []), uid]
            return doc

        if store.get("gyms", invite["gym"]) is not None:
            store.transact("gyms", invite["gym"], add)
            joined.append(invite["gym"])
        store.delete("gym_invites", doc_id)
    return joined
