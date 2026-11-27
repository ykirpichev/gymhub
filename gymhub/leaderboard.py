"""Leaderboards per exercise, language and hardware class, fed only by trusted machines.

A gym's owner opts in (gyms/{id}.leaderboard). A benchmark counts only when the machine that ran it is
trusted for that gym ("grading: trusted"): a direct-mode runner (it holds the project's own
credentials), a worker owned by the gym's owner, or one owned by an admin. Members' own machines
never feed a leaderboard, since they could report anything.

    leaderboard/{gym}__{slug}__{lang}__{hardware}__{uid}   board ("gym__slug"), lang, hardware, uid,
                                                           name, ms, at, job       (each user's best)

Basics, not a fortress: the time comes from the trusted machine's own bench output, after the
solution passed the tests on that machine.
"""

import os
import re

from gymhub.store import Store, now

MAX_ROWS = 50


def _admins() -> set[str]:
    return {email.strip().lower() for email in os.environ.get("GYMHUB_ADMIN_EMAILS", "").split(",") if email.strip()}


def trusted(store: Store, runner_id: str | None, gym_doc: dict | None) -> bool:
    if not runner_id:
        return False
    worker = store.get("workers", runner_id)
    if worker is None:
        return True  # a direct-mode runner: only the project's owners have those credentials
    if worker.get("revoked"):
        return False
    if gym_doc and gym_doc.get("owner_uid") and worker.get("owner_uid") == gym_doc["owner_uid"]:
        return True
    return str(worker.get("owner_email", "")).lower() in _admins()


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(text))[:60]


def record(store: Store, job: dict, ms: float) -> bool:
    """Called with a passing bench. True when it entered (or improved) the leaderboard."""
    gym_doc = store.get("gyms", job.get("gym", "")) or {}
    if not gym_doc.get("leaderboard") or not job.get("uid") or not job.get("hardware"):
        return False
    if not trusted(store, job.get("runner"), gym_doc):
        return False
    key = "__".join([job["gym"], job["slug"], job.get("lang", ""), _safe(job["hardware"]), job["uid"]])
    current = store.get("leaderboard", key)
    if current and current.get("ms", float("inf")) <= ms:
        return False
    user = store.get("users", job["uid"]) or {}
    store.set("leaderboard", key, {
        "board": f"{job['gym']}__{job['slug']}", "gym": job["gym"], "slug": job["slug"], "lang": job.get("lang", ""),
        "hardware": job["hardware"], "uid": job["uid"], "name": user.get("name") or str(job.get("created_by", "")).split("@")[0],
        "ms": ms, "at": now(), "job": job.get("id"),
    })
    return True


def standings(store: Store, gym: str, slug: str, uid: str, lang: str | None = None, hardware: str | None = None) -> dict:
    rows = [doc for _id, doc in store.where("leaderboard", "board", f"{gym}__{slug}")]
    langs = sorted({row["lang"] for row in rows})
    hardwares = sorted({row["hardware"] for row in rows})
    lang = lang if lang in langs else (langs[0] if langs else None)
    if hardware not in hardwares:
        mine = [row["hardware"] for row in rows if row["uid"] == uid and row["lang"] == lang]
        hardware = mine[0] if mine else (hardwares[0] if hardwares else None)
    selected = sorted((row for row in rows if row["lang"] == lang and row["hardware"] == hardware), key=lambda row: row["ms"])
    entries = [{"rank": rank, "name": row["name"], "ms": row["ms"], "at": row["at"], "me": row["uid"] == uid}
               for rank, row in enumerate(selected, start=1)]
    mine = next((entry for entry in entries if entry["me"]), None)
    return {"lang": lang, "hardware": hardware, "langs": langs, "hardwares": hardwares,
            "entries": entries[:MAX_ROWS], "total": len(entries), "me": mine}
