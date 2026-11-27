"""Spaced review: solved exercises come back 1, 7 and 30 days later to be solved again from scratch.

    reviews/{uid}__{gym}__{slug}   uid, gym, slug, stage (reviews done), due_at, last_reviewed

The first review is due a day after the exercise was first solved (taken from progress, so solves
made from a terminal count too). A passing "re-solve from scratch" submission (a test job with
options.review) advances the stage; after the last interval the exercise leaves the queue.
"""

import calendar
import time
from datetime import datetime

INTERVALS_DAYS = [1, 7, 30]
DAY = 86400


def _solved_at(entry: dict) -> float | None:
    stamps = []
    for lang in (entry.get("langs") or {}).values():
        text = lang.get("first_solved")
        if not text:
            continue
        try:
            stamps.append(calendar.timegm(datetime.fromisoformat(str(text)[:19]).timetuple()))
        except ValueError:
            continue
    return min(stamps) if stamps else None


def queue(store, uid: str, gyms: list[dict], now: float | None = None, limit: int = 6) -> dict:
    """gyms: [{"id", "title", "progress", "problem_list"}] as the Home API builds them.
    Returns {"due": [...oldest first], "upcoming": n, "next_due_at": ts}."""
    now = now or time.time()
    states = {f"{doc.get('gym')}__{doc.get('slug')}": doc for _id, doc in store.where("reviews", "uid", uid)}
    due, upcoming, next_due = [], 0, None
    for gym in gyms:
        problems = {p.get("slug"): p for p in gym.get("problem_list") or []}
        for slug, entry in (gym.get("progress") or {}).get("exercises", {}).items():
            problem = problems.get(slug)
            if not problem or problem.get("warmup") or (problem.get("kind") or "code") != "code":
                continue
            solved = _solved_at(entry)
            if solved is None:
                continue
            state = states.get(f"{gym['id']}__{slug}") or {}
            stage = int(state.get("stage", 0))
            if stage >= len(INTERVALS_DAYS):
                continue
            due_at = state.get("due_at") or solved + INTERVALS_DAYS[0] * DAY
            if due_at <= now:
                due.append({"gym": gym["id"], "gym_title": gym.get("title", gym["id"]), "slug": slug,
                            "title": problem.get("title", slug), "difficulty": problem.get("difficulty"),
                            "stage": stage, "of": len(INTERVALS_DAYS), "due_at": due_at, "solved_at": solved})
            else:
                upcoming += 1
                next_due = due_at if next_due is None else min(next_due, due_at)
    due.sort(key=lambda item: item["due_at"])
    return {"due": due[:limit], "due_count": len(due), "upcoming": upcoming, "next_due_at": next_due}


def advance(store, uid: str, gym: str, slug: str, now: float | None = None) -> dict:
    """A review passed: schedule the next one, or finish."""
    now = now or time.time()
    key = f"{uid}__{gym}__{slug}"
    state = store.get("reviews", key) or {}
    stage = int(state.get("stage", 0)) + 1
    due_at = now + INTERVALS_DAYS[stage] * DAY if stage < len(INTERVALS_DAYS) else None
    doc = {"uid": uid, "gym": gym, "slug": slug, "stage": stage, "due_at": due_at, "last_reviewed": now}
    store.set("reviews", key, doc)
    return doc
