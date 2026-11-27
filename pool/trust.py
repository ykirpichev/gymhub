"""The checked trust tier (M6, #45 section 9): verification reports, reputation and exclusion.

A pool is `trusted` (the default: nobody is checked) or `checked`. In a checked pool:

- Verifiers report on machines. Built in: the duplicate check of a queue with `replicate` (pool/coord.py:
  a fraction of tasks runs on two machines and must agree, a third breaks a tie). Jobs report their own
  checks through `pool_job.report(machine, ok, kind, detail)`: a trainer that recomputes logprobs or
  rewards or verifies TOPLOC proofs (pool/verify.py, pool_verify in a job), a reducer that finds an update far from the others (pool/reducers.py robust_mean).
- A machine with `exclude_after` failed checks (pool setting, default 2) is excluded: it gets no more
  tasks, its task results and round contributions are refused, and `pool machine ls` shows it.
  An admin clears that with `pool machine trust NAME` after looking into it.

Reputation is per machine and pool: checks, failures, a score (passed / checked, 1.0 with no checks)
and the last few verifications, so an admin can see why.
"""

import json

from pool.db import DB, now

TRUST = ("trusted", "checked")
KINDS = ("duplicate", "logprobs", "toploc", "reward", "update", "other")
EXCLUDE_AFTER = 2


def settings(db: DB, pool: str) -> dict:
    row = db.one("select settings from pools where id=?", pool)
    saved = json.loads(row["settings"]) if row else {}
    return {"trust": saved.get("trust", "trusted"), "exclude_after": int(saved.get("exclude_after", EXCLUDE_AFTER))}


def checked(db: DB, pool: str) -> bool:
    return settings(db, pool)["trust"] == "checked"


def report(db: DB, pool: str, machine: str, ok: bool, kind: str, detail: str = "", reporter: str | None = None) -> dict:
    """Record one verification. In a checked pool, enough failures exclude the machine."""
    if kind not in KINDS:
        kind = "other"
    with db.lock:
        db.run("insert into verifications (pool, machine, kind, ok, detail, reporter, created_at) values (?, ?, ?, ?, ?, ?, ?)",
               pool, machine, kind, int(bool(ok)), str(detail or "")[:500], reporter, now())
        db.run("insert into reputation (pool, machine, checks, failures, updated_at) values (?, ?, 0, 0, ?) "
               "on conflict (pool, machine) do nothing", pool, machine, now())
        db.run("update reputation set checks=checks+1, failures=failures+?, updated_at=? where pool=? and machine=?",
               0 if ok else 1, now(), pool, machine)
        rep = db.one("select * from reputation where pool=? and machine=?", pool, machine)
        conf = settings(db, pool)
        if conf["trust"] == "checked" and not rep["excluded"] and rep["failures"] >= conf["exclude_after"]:
            db.run("update reputation set excluded=1 where pool=? and machine=?", pool, machine)
    return standing(db, pool, machine)


def standing(db: DB, pool: str, machine: str) -> dict:
    rep = db.one("select * from reputation where pool=? and machine=?", pool, machine)
    if not rep:
        return {"machine": machine, "checks": 0, "failures": 0, "score": 1.0, "excluded": False, "recent": []}
    recent = db.all("select kind, ok, detail, created_at from verifications where pool=? and machine=? order by id desc limit 5",
                    pool, machine)
    return {"machine": machine, "checks": rep["checks"], "failures": rep["failures"],
            "score": round(1 - rep["failures"] / rep["checks"], 3) if rep["checks"] else 1.0,
            "excluded": bool(rep["excluded"]), "recent": [{**r, "ok": bool(r["ok"])} for r in recent]}


def excluded(db: DB, pool: str, machine: str | None) -> bool:
    """True when this machine may not take or report work in this (checked) pool."""
    if not machine or not checked(db, pool):
        return False
    row = db.one("select excluded from reputation where pool=? and machine=?", pool, machine)
    return bool(row and row["excluded"])


def clear(db: DB, pool: str, machine: str) -> None:
    """An admin's decision: the machine is trusted again; its counts start over."""
    db.run("update reputation set excluded=0, checks=0, failures=0, updated_at=? where pool=? and machine=?", now(), pool, machine)
