"""Rounds: synchronisation steps with a quorum and a deadline (M5, #45 section 8).

A round belongs to a ref. It opens on the ref's current version (the base); workers train from that
version and contribute an artifact (a delta, a pseudo-gradient, a set of weights) with a weight,
for example the tokens they processed. The round becomes ready:

- once `quorum` contributions are in, plus `grace` seconds for stragglers; or
- at the deadline, if at least `min` contributions are in (otherwise it fails).

A reducer (a job) claims a ready round, gets its contributions, merges them however it likes
(pool/reducers.py has a weighted mean for .npy and .safetensors files) and finishes the round with
the result, which is published as the ref's next version, compare-and-swap on the base version.
With `repeat`, the next round opens on the new version right away.

A contribution for a round that is no longer open is late. `late: drop` refuses it; `late: next`
keeps it for the next round of the same ref, where the reducer sees it marked late and can scale
it (`decay`, given to the reducer with the round); it does not count toward that round's quorum.

Contributions travel through the server as artifacts, so the machines of a round need no direct
link to each other: a round may span islands, which a gang may not.

`mode: gossip` (M7) has no reducer: when the round is ready the server draws a mixing plan over the
measured link graph (pool/gossip.py) and closes it; each worker then averages its contribution with
a few sampled peers' (`mix`) and keeps its own model. The ref does not move.

A `mode: reduce` round with `reducer: torchft` is reduced in-band: its contributors (islands with a
link to each other, however slow) all-reduce with torchft, whose lighthouse address the round
carries, and each contributes only a small record. The round keeps the quorum, the deadline and the
ledger. A gossip round has no reducer, so gossip with torchft is refused.

`reduce: tree` (#156; opt-in, `flat` is the default) reduces an artifact round island first. When the
round is ready the server groups its contributions by island (machines with island-grade links,
pool/peers.py islands) and plans one island stage per island:

- An island with one contribution forwards it: that contribution is the island's partial.
- Otherwise a machine of the island reduces its members' contributions into a partial. The plan
  names the member machine that collects them fastest over the measured links (`collect_cost`);
  that machine may claim the stage at once, any machine of the island after `handoff_s`, and any
  machine at all after twice that (an island with no reducer of its own still gets its partial,
  over the slow link). A partial is the island's weighted mean, in the contributions' own format,
  and the server records its weight: the sum of the included members' weights (late ones times
  `decay`), computed by the server from the contributions, never taken from the reducer. Mean times
  weight is the island's weighted sum, so the weighted mean of the partials equals the flat one.
- Once every island stage is done, a final reducer claims the partials (the plan prefers the
  machine that collects them fastest), merges them and finishes the round as usual; the result is
  the ref's next version, which subscribed jobs fetch over the peer chunk swarm.

Contributions and partials of a tree round are held: their chunks stay on the machine that made them
(held_artifacts) and travel machine to machine, so a member's update stays inside its island and
only one partial per island crosses to the final reducer.

Failures: a reducer whose job stops or whose machine goes offline hands its stage back (to any
machine of its island, or for the final stage to anyone). Island stages not done by the round's
deadline (counted again from when it became ready) fail; a failed partial counts its members as
missing, and the round fails when fewer than `min` contributions are left in done partials.

Trust (pool/trust.py): an island reducer gets only the contributions of machines that are not
excluded and hands back the ones it dropped (pool_reducers checks each member with robust_mean
before forming the partial, and reports the outliers); the final reducer sees every partial's
inputs (`members`), so it or any verifier can recompute a partial (pool_reducers.verify_partial).
A partial from an excluded machine is refused, and a partial whose reducer is excluded before the
final claim is formed again by another machine.
"""

import json
import re

from pool import gossip, trust
from pool import peers as peering
from pool.db import DB, new_id, now
from pool.scheduler import ONLINE_SECONDS

LATE = ("drop", "next")
MODES = ("reduce", "gossip")
REDUCERS = ("artifacts", "torchft")
REDUCE = ("flat", "tree")
HANDOFF_S = 30.0
LIGHTHOUSE = re.compile(r"^https?://[A-Za-z0-9.\-]{1,253}:[0-9]{1,5}$")


class RoundError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def _public(db: DB, row: dict | None) -> dict | None:
    if row is None:
        return None
    count = db.one("select count(*) as n from contributions where round=?", row["id"])["n"]
    out = {"id": row["id"], "ref": row["ref"], "base_version": row["base_version"], "status": row["status"],
           "quorum": row["quorum"], "min": row["min_contrib"], "deadline": row["deadline"], "grace": row["grace"],
           "contributions": count, "settings": json.loads(row["settings"] or "{}"), "opened_at": row["opened_at"],
           "ready_at": row["ready_at"], "closed_at": row["closed_at"], "result_version": row["result_version"],
           "note": row["note"]}
    if out["settings"].get("reduce") == "tree" and row.get("tree_plan"):
        plan = json.loads(row["tree_plan"])
        out["tree"] = {"final": plan.get("final"), "partials": [
            {"island": p["island"], "status": p["status"], "members": len(json.loads(p["members"])), "planned": p["planned"],
             "machine": p["reducer_machine"], "weight": p["weight"], "forwarded": bool(p["forwarded"]), "note": p["note"]}
            for p in db.all("select * from round_partials where round=? order by island", row["id"])]}
    return out


def _current_version(db: DB, pool: str, ref: str) -> int:
    row = db.one("select max(version) as v from refs where pool=? and name=?", pool, ref)
    return int(row["v"] or 0)


def open_round(db: DB, pool: str, ref: str, owner: str, quorum: int, min_contrib: int | None = None,
               deadline_s: float = 600, grace_s: float = 5, late: str = "drop", decay: float = 0.5,
               repeat: bool = False, mode: str = "reduce", fanout: int = gossip.DEFAULT_FANOUT,
               bias: float = gossip.DEFAULT_BIAS, reducer: str = "artifacts", lighthouse: str | None = None,
               reduce: str = "flat", handoff_s: float = HANDOFF_S) -> dict:
    """Open a round on the ref's current version; if one is already open there, return it.

    mode gossip: no reducer; contributors mix with `fanout` sampled peers (see pool/gossip.py).
    reducer (mode reduce only): artifacts (contributions are folders, a reducer job merges them) or
    torchft (the contributors reduce among themselves with torchft, and contribute only a record;
    `lighthouse` is the address of torchft's lighthouse, http://host:port, set by the first job to
    open the round, see pool/reducers.py TorchftRounds).
    reduce (reducer artifacts only): flat (one reducer gets every contribution) or tree (island
    first, see the module docstring); `handoff_s` is how long a tree's planned reducers have to
    claim their stage before other machines may."""
    if not 1 <= int(quorum) <= 10_000:
        raise RoundError(400, "quorum is 1 or more")
    low = int(min_contrib if min_contrib is not None else quorum)
    if not 1 <= low <= int(quorum):
        raise RoundError(400, "min is between 1 and quorum")
    if late not in LATE:
        raise RoundError(400, "late is drop or next")
    if mode not in MODES:
        raise RoundError(400, "mode is reduce or gossip")
    if mode == "gossip" and not (1 <= int(fanout) <= 64 and 0 <= float(bias) <= 4):
        raise RoundError(400, "gossip: fanout from 1 to 64, bias from 0 to 4")
    if not 1 <= float(deadline_s) <= 7 * 24 * 3600 or not 0 <= float(grace_s) <= 3600:
        raise RoundError(400, "deadline_s from 1 s to a week, grace_s from 0 to an hour")
    reducer = reducer or "artifacts"
    if reducer not in REDUCERS:
        raise RoundError(400, "reducer is artifacts or torchft")
    if mode == "gossip" and reducer != "artifacts":
        raise RoundError(400, f"a gossip round has no reducer (workers mix with peers), so reducer {reducer} "
                              "does not apply: use mode reduce for a torchft round")
    if reducer == "torchft" and not LIGHTHOUSE.match(str(lighthouse or "")):
        raise RoundError(400, "a torchft round needs lighthouse: http://host:port")
    reduce = reduce or "flat"
    if reduce not in REDUCE:
        raise RoundError(400, "reduce is flat or tree")
    if reduce == "tree" and (mode != "reduce" or reducer != "artifacts"):
        raise RoundError(400, "reduce tree is for mode reduce with reducer artifacts (gossip and torchft rounds "
                              "reduce their own way)")
    if not 0 <= float(handoff_s) <= 3600:
        raise RoundError(400, "handoff_s from 0 to an hour")
    with db.lock:
        version = _current_version(db, pool, ref)
        if version == 0:
            raise RoundError(404, f"no ref {ref}: publish its first version, then open rounds on it")
        existing = db.one("select * from rounds where pool=? and ref=? and status in ('open', 'ready', 'reducing')", pool, ref)
        if existing:
            return _public(db, existing)
        settings = {"late": late, "decay": float(decay), "repeat": bool(repeat), "deadline_s": float(deadline_s)}
        if mode == "gossip":
            settings.update(mode="gossip", fanout=int(fanout), bias=float(bias))
        else:
            settings["reducer"] = reducer
        if reducer == "torchft":
            settings["lighthouse"] = lighthouse
        if reduce == "tree":
            settings.update(reduce="tree", handoff_s=float(handoff_s))
        round_id = new_id("r")
        db.run("insert into rounds (id, pool, ref, base_version, owner, quorum, min_contrib, deadline, grace, settings, status, "
               "opened_at) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)", round_id, pool, ref, version, owner, int(quorum),
               low, now() + float(deadline_s), float(grace_s), json.dumps(settings), now())
        # late contributions kept for this ref join this round
        db.run("update contributions set round=? where pool=? and ref=? and round is null", round_id, pool, ref)
    return _public(db, db.one("select * from rounds where id=?", round_id))


def current(db: DB, pool: str, ref: str) -> dict | None:
    """The round in progress on a ref (open, ready or being reduced), else the latest one."""
    row = db.one("select * from rounds where pool=? and ref=? and status in ('open', 'ready', 'reducing') "
                 "order by opened_at desc limit 1", pool, ref)
    return _public(db, row or db.one("select * from rounds where pool=? and ref=? order by opened_at desc limit 1", pool, ref))


def contribute(db: DB, pool: str, ref: str, job: str, machine: str | None, artifact: str, weight: float,
               base_version: int, round_id: str | None = None) -> dict:
    """Contribute to the latest round on `base_version`, or to `round_id` when given (gossip rounds
    with repeat follow each other on one version, so their workers name the round)."""
    if not float(weight) > 0:
        raise RoundError(400, "weight must be positive")
    if trust.excluded(db, pool, machine):
        return {"accepted": False, "reason": "this machine is excluded from the pool's work (failed checks)"}
    with db.lock:
        tick(db)
        if round_id:
            row = db.one("select * from rounds where pool=? and ref=? and id=?", pool, ref, round_id)
        else:
            row = db.one("select * from rounds where pool=? and ref=? and base_version=? order by opened_at desc limit 1",
                         pool, ref, int(base_version))
        if row is None:
            raise RoundError(404, f"no round of {ref} on version {base_version}")
        if row["status"] == "open":
            done = db.one("select id from contributions where round=? and job=?", row["id"], job)
            if done:  # a retry of the same job replaces its contribution
                db.run("update contributions set artifact=?, weight=?, created_at=? where id=?", artifact, float(weight), now(), done["id"])
            else:
                db.run("insert into contributions (id, pool, ref, round, job, machine, artifact, weight, base_version, created_at) "
                       "values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", new_id("c"), pool, ref, row["id"], job, machine, artifact,
                       float(weight), int(base_version), now())
            tick(db)
            return {"accepted": True, "round": row["id"]}
        settings = json.loads(row["settings"] or "{}")
        if settings.get("late") == "next":
            db.run("insert into contributions (id, pool, ref, round, job, machine, artifact, weight, base_version, late, created_at) "
                   "values (?, ?, ?, null, ?, ?, ?, ?, ?, 1, ?)", new_id("c"), pool, ref, job, machine, artifact,
                   float(weight), int(base_version), now())
            following = db.one("select id from rounds where pool=? and ref=? and status='open'", pool, ref)
            if following:
                db.run("update contributions set round=? where pool=? and ref=? and round is null", following["id"], pool, ref)
            return {"accepted": True, "late": True, "round": following["id"] if following else None}
        return {"accepted": False, "late": True, "reason": f"round {row['id']} is {row['status']}; this round drops late work"}


def tick(db: DB) -> int:
    """Move open rounds on: ready at quorum plus grace, or at the deadline with at least min; else failed."""
    changed = 0
    with db.lock:
        for row in db.all("select * from rounds where status='open'"):
            # late work carried over from an earlier round rides along but does not count toward quorum or min
            count = db.one("select count(*) as n from contributions where round=? and late=0", row["id"])["n"]
            if count >= row["quorum"] and row["quorum_at"] is None:
                db.run("update rounds set quorum_at=? where id=?", now(), row["id"])
                row["quorum_at"] = now()
            if row["quorum_at"] is not None and now() >= row["quorum_at"] + row["grace"]:
                done = db.run("update rounds set status='ready', ready_at=? where id=? and status='open'", now(), row["id"])
                changed += done
                if done:
                    _gossip_close(db, row)
                    _tree_plan(db, row)
            elif now() >= row["deadline"]:
                if count >= row["min_contrib"]:
                    done = db.run("update rounds set status='ready', ready_at=?, note=? where id=? and status='open'",
                                  now(), f"deadline with {count} of {row['quorum']}", row["id"])
                    changed += done
                    if done:
                        _gossip_close(db, row)
                        _tree_plan(db, row)
                else:
                    changed += db.run("update rounds set status='failed', closed_at=?, note=? where id=? and status='open'",
                                      now(), f"deadline with {count} contributions, fewer than {row['min_contrib']}", row["id"])
        changed += _tree_tick(db)
    return changed


def _gossip_close(db: DB, row: dict) -> None:
    """A gossip round that just became ready: draw its mixing plan and close it (no reducer)."""
    settings = json.loads(row["settings"] or "{}")
    if settings.get("mode") != "gossip":
        return
    items = db.all("select id, machine from contributions where round=?", row["id"])
    machines = sorted({i["machine"] for i in items if i["machine"]})
    links = {}
    if machines:
        marks = ",".join("?" * len(machines))
        for link in db.all(f"select src, dst, mbps from peer_links where src in ({marks}) and dst in ({marks})",
                           *machines, *machines):
            links[(link["src"], link["dst"])] = float(link["mbps"])
    plan = gossip.plan([i["id"] for i in items], {i["id"]: i["machine"] for i in items}, links,
                       settings.get("fanout", gossip.DEFAULT_FANOUT), settings.get("bias", gossip.DEFAULT_BIAS), row["id"])
    for contribution, peers in plan.items():
        for peer, weight in peers.items():
            db.run("insert into gossip_mix (round, contribution, peer, weight) values (?, ?, ?, ?)",
                   row["id"], contribution, peer, weight)
    edges = sum(len(p) - 1 for p in plan.values()) // 2
    db.run("update rounds set status='closed', closed_at=?, note=? where id=?", now(),
           f"gossip: {len(items)} contributions mixed over {edges} links", row["id"])
    if settings.get("repeat"):
        open_round(db, row["pool"], row["ref"], row["owner"], row["quorum"], row["min_contrib"], settings.get("deadline_s", 600),
                   row["grace"], settings.get("late", "drop"), settings.get("decay", 0.5), True, mode="gossip",
                   fanout=settings.get("fanout", gossip.DEFAULT_FANOUT), bias=settings.get("bias", gossip.DEFAULT_BIAS),
                   reducer=settings.get("reducer", "artifacts"), lighthouse=settings.get("lighthouse"))


def mix(db: DB, pool: str, round_id: str, job: str) -> dict | None:
    """For a worker of a gossip round: what to average its contribution with, once the round closed.
    {"round", "items": [{"artifact", "weight", "self", "machine", "contribution_weight"}]}; None while open."""
    with db.lock:
        tick(db)
        row = db.one("select * from rounds where id=? and pool=?", round_id, pool)
        if row is None:
            raise RoundError(404, f"no round {round_id}")
        if json.loads(row["settings"] or "{}").get("mode") != "gossip":
            raise RoundError(400, f"round {round_id} is not a gossip round (claim it with a reducer)")
        if row["status"] == "failed":
            raise RoundError(409, f"round {round_id} failed: {row['note']}")
        if row["status"] != "closed":
            return None
        mine = db.one("select id from contributions where round=? and job=? order by created_at desc limit 1", round_id, job)
        if mine is None:
            raise RoundError(404, f"this job has no contribution in round {round_id}")
        rows = db.all("select g.peer, g.weight, c.artifact, c.machine, c.weight as cw from gossip_mix g "
                      "join contributions c on c.id=g.peer where g.round=? and g.contribution=? order by g.peer",
                      round_id, mine["id"])
        return {"round": round_id, "items": [{"artifact": r["artifact"], "weight": r["weight"], "self": r["peer"] == mine["id"],
                                              "machine": r["machine"], "contribution_weight": r["cw"]} for r in rows]}


def claim(db: DB, pool: str, ref: str, job: str, machine: str | None = None) -> dict | None:
    """A ready round of this ref for a reducer, with its contributions; None if none is ready.
    In a tree round: an island stage this job's machine may take, else the final stage once every
    island's partial is in; "stage" says which ("island" or "final")."""
    with db.lock:
        tick(db)
        row = db.one("select * from rounds where pool=? and ref=? and (status='ready' or (status='reducing' and reducer=?)) "
                     "order by opened_at limit 1", pool, ref, job)
        if row is None:
            return None
        if json.loads(row["settings"] or "{}").get("reduce") == "tree":
            return _tree_claim(db, row, job, machine)
        db.run("update rounds set status='reducing', reducer=? where id=? and status in ('ready', 'reducing')", job, row["id"])
        items = db.all("select artifact, weight, base_version, late, machine, job from contributions where round=? order by created_at",
                       row["id"])
        return {**_public(db, db.one("select * from rounds where id=?", row["id"])),
                "items": [{"artifact": i["artifact"], "weight": i["weight"], "late": bool(i["late"]), "machine": i["machine"],
                           "job": i["job"], "base_version": i["base_version"]} for i in items]}


def finish(db: DB, pool: str, round_id: str, job: str, member: str, artifact: str, publish) -> dict:
    """Publish the reduced artifact as the ref's next version (on the round's base) and close the round.
    `publish(ref, artifact, parent)` is the ref compare-and-swap. With repeat, open the next round."""
    with db.lock:
        row = db.one("select * from rounds where id=? and pool=?", round_id, pool)
        if not row:
            raise RoundError(404, f"no round {round_id}")
        if row["status"] != "reducing" or row["reducer"] != job:
            raise RoundError(409, f"round {round_id} is {row['status']}; claim it first")
        try:
            ref = publish(row["ref"], artifact, row["base_version"])
        except Exception:
            db.run("update rounds set status='failed', closed_at=?, note=? where id=?", now(),
                   "the ref moved on since the round opened (compare-and-swap failed)", round_id)
            raise
        db.run("update rounds set status='closed', closed_at=?, result_version=? where id=?", now(), ref["version"], round_id)
        settings = json.loads(row["settings"] or "{}")
        following = None
        if settings.get("repeat"):
            following = open_round(db, pool, row["ref"], row["owner"], row["quorum"], row["min_contrib"],
                                   settings.get("deadline_s", 600), row["grace"], settings.get("late", "drop"),
                                   settings.get("decay", 0.5), True, mode=settings.get("mode", "reduce"),
                                   fanout=settings.get("fanout", gossip.DEFAULT_FANOUT),
                                   bias=settings.get("bias", gossip.DEFAULT_BIAS),
                                   reducer=settings.get("reducer", "artifacts"), lighthouse=settings.get("lighthouse"),
                                   reduce=settings.get("reduce", "flat"), handoff_s=settings.get("handoff_s", HANDOFF_S))
    return {"ref": ref, "next": following}


# ---- reduce: tree (#156) -------------------------------------------------------------------------------

def _decayed(item: dict, decay: float) -> float:
    return float(item["weight"]) * (decay if item["late"] else 1.0)


def _links(db: DB, machines: list[str]) -> tuple[dict, list[dict]]:
    """Measured links among these machines: {(src, dst): Mbit/s}, and the rows peers.islands takes."""
    if not machines:
        return {}, []
    marks = ",".join("?" * len(machines))
    rows = db.all(f"select src, dst, mbps, rtt_ms from peer_links where src in ({marks}) and dst in ({marks})",
                  *machines, *machines)
    return {(r["src"], r["dst"]): float(r["mbps"]) for r in rows}, rows


def collect_cost(target: str, sources: list, links: dict) -> float:
    """How long `target` takes to fetch one update from each of `sources` (machines; None for one
    the server holds), in seconds per Mbit of update: summed, since the downloads share the target's
    link. Updates already on the target cost nothing; unmeasured links count as gossip.GUESS_MBPS."""
    return sum(1.0 / gossip.link_speed(links, source, target) for source in sources if source != target)


def best_collector(candidates, sources: list, links: dict) -> str | None:
    """The candidate machine that collects `sources` fastest; on a tie the one already holding most."""
    found = sorted({c for c in candidates if c})
    if not found:
        return None
    return min(found, key=lambda c: (collect_cost(c, sources, links), -sources.count(c), c))


def _tree_plan(db: DB, row: dict) -> None:
    """A tree round that just became ready: group its contributions by island and plan the stages."""
    settings = json.loads(row["settings"] or "{}")
    if settings.get("reduce") != "tree":
        return
    decay = float(settings.get("decay", 0.5))
    items = db.all("select id, machine, artifact, weight, late, job from contributions where round=? order by created_at, id",
                   row["id"])
    known = [r["id"] for r in db.all("select id from machines where pool=?", row["pool"])]
    everyone = sorted(set(known) | {i["machine"] for i in items if i["machine"]})
    links, rows = _links(db, everyone)
    island_of = {m: group for group in peering.islands(everyone, rows) for m in group}
    groups: dict = {}
    for item in items:  # a contribution without a machine (a member's upload) is an island of its own
        key = tuple(island_of[item["machine"]]) if item["machine"] else ("", item["id"])
        groups.setdefault(key, []).append(item)
    t = now()
    ordered = sorted(groups.items(), key=lambda kv: min(str(i["machine"] or i["id"]) for i in kv[1]))
    for index, (key, members) in enumerate(ordered):
        machines = [m for m in key if m] if key[0] else []
        ids = json.dumps([m["id"] for m in members])
        if len(members) == 1:  # nothing to reduce: the contribution is the island's partial
            only = members[0]
            db.run("insert into round_partials (round, island, members, machines, planned, status, given, reducer_machine, "
                   "free_at, artifact, weight, forwarded, done_at, note) values (?, ?, ?, ?, ?, 'done', ?, ?, ?, ?, ?, 1, ?, ?)",
                   row["id"], index, ids, json.dumps(machines), only["machine"], ids, only["machine"], t, only["artifact"],
                   _decayed(only, decay), t, "forwarded: one contribution")
            continue
        sources = [m["machine"] for m in members]
        db.run("insert into round_partials (round, island, members, machines, planned, status, free_at) "
               "values (?, ?, ?, ?, ?, 'waiting', ?)", row["id"], index, ids, json.dumps(machines),
               best_collector(sources, sources, links), t)
    db.run("update rounds set tree_plan=? where id=?",
           json.dumps({"islands": len(ordered), "final": None, "final_at": None, "planned_at": t}), row["id"])


def _alive(db: DB, job: str | None, machine: str | None) -> bool:
    """Whether a reducer can still finish its stage: its job runs (on that machine), which is online.
    A job or machine the server does not know is taken at its word."""
    if job:
        found = db.one("select status, machine from jobs where id=?", job)
        if found is not None and (found["status"] != "running" or (machine and found["machine"] != machine)):
            return False
    if machine:
        found = db.one("select last_seen from machines where id=?", machine)
        if found is not None and (not found["last_seen"] or now() - found["last_seen"] > ONLINE_SECONDS):
            return False
    return True


def _included(partial: dict) -> list[str]:
    dropped = set(json.loads(partial["dropped"] or "[]"))
    return [c for c in json.loads(partial["given"] or "[]") if c not in dropped]


def _tree_tick(db: DB) -> int:
    """Move tree rounds on: hand off stages whose reducer is gone, fail island stages at the deadline,
    and once every island stage is over, fail the round (too few contributions covered) or plan the
    final reducer."""
    changed = 0
    for row in db.all("select * from rounds where status in ('ready', 'reducing') and tree_plan is not null"):
        settings = json.loads(row["settings"] or "{}")
        plan = json.loads(row["tree_plan"])
        t = now()
        for p in db.all("select * from round_partials where round=? order by island", row["id"]):
            if p["status"] == "reducing" and not _alive(db, p["reducer"], p["reducer_machine"]):
                changed += db.run("update round_partials set status='waiting', reducer=null, reducer_machine=null, given=null, "
                                  "claimed_at=null, planned=?, free_at=?, note=? where round=? and island=? and status='reducing'",
                                  None if p["planned"] == p["reducer_machine"] else p["planned"], t,
                                  f"handed off: the reducer {p['reducer']} stopped", row["id"], p["island"])
            elif (p["status"] == "done" and row["status"] == "ready" and p["reducer_machine"]
                  and trust.excluded(db, row["pool"], p["reducer_machine"])):
                if p["forwarded"]:
                    changed += db.run("update round_partials set status='failed', note=? where round=? and island=?",
                                      "the contribution's machine is excluded", row["id"], p["island"])
                else:
                    changed += db.run("update round_partials set status='waiting', reducer=null, reducer_machine=null, given=null, "
                                      "artifact=null, weight=null, dropped='[]', done_at=null, planned=null, free_at=?, note=? "
                                      "where round=? and island=?", t, "formed again: its reducer's machine is excluded",
                                      row["id"], p["island"])
        if row["status"] == "ready" and t >= (row["ready_at"] or t) + float(settings.get("deadline_s", 600)):
            changed += db.run("update round_partials set status='failed', note=? where round=? and status in ('waiting', 'reducing')",
                              "no partial by the deadline", row["id"])
        if row["status"] == "reducing" and not _alive(db, row["reducer"], plan.get("machine")):
            plan.update(final=None, final_at=t, machine=None)
            changed += db.run("update rounds set status='ready', reducer=null, tree_plan=?, note=? where id=? and status='reducing'",
                              json.dumps(plan), f"handed off: the final reducer {row['reducer']} stopped", row["id"])
            continue
        if row["status"] != "ready":
            continue
        partials = db.all("select * from round_partials where round=?", row["id"])
        if any(p["status"] in ("waiting", "reducing") for p in partials):
            continue
        done = [p for p in partials if p["status"] == "done"]
        # the members of a done partial count, those its reducer dropped or never got (excluded machines)
        # too, as in a flat round, whose reducer checks what it gets; the members of a failed one are missing
        covered = [c for p in done for c in json.loads(p["members"])]
        counted = 0
        if covered:
            marks = ",".join("?" * len(covered))
            counted = db.one(f"select count(*) as n from contributions where late=0 and id in ({marks})", *covered)["n"]
        if counted < row["min_contrib"] or not done:
            changed += db.run("update rounds set status='failed', closed_at=?, note=? where id=? and status='ready'", t,
                              f"tree: {counted} contributions in the islands' partials, fewer than {row['min_contrib']}", row["id"])
        elif plan.get("final_at") is None:
            holders = [p["reducer_machine"] for p in done]
            links, _ = _links(db, sorted({h for h in holders if h}))
            plan.update(final=best_collector(holders, holders, links), final_at=t)
            changed += db.run("update rounds set tree_plan=? where id=?", json.dumps(plan), row["id"])
    return changed


def _may_take(planned: str | None, island: list | None, since: float, machine: str | None, handoff: float) -> bool:
    """The planned machine at once; a machine of the island (island None: any) after `handoff` s, or
    at once when nothing is planned; any machine at all after twice that."""
    waited = now() - since
    if planned and machine == planned:
        return True
    if island is None or machine in island:
        if planned is None or waited >= handoff:
            return True
    return waited >= 2 * handoff


def _contribution_items(db: DB, ids: list[str]) -> list[dict]:
    if not ids:
        return []
    marks = ",".join("?" * len(ids))
    rows = db.all(f"select id, artifact, weight, base_version, late, machine, job from contributions where id in ({marks}) "
                  "order by created_at, id", *ids)
    return [{"contribution": r["id"], "artifact": r["artifact"], "weight": r["weight"], "late": bool(r["late"]),
             "machine": r["machine"], "job": r["job"], "base_version": r["base_version"]} for r in rows]


def _island_task(db: DB, row: dict, island: int) -> dict:
    p = db.one("select * from round_partials where round=? and island=?", row["id"], island)
    return {**_public(db, db.one("select * from rounds where id=?", row["id"])), "stage": "island", "island": island,
            "items": _contribution_items(db, json.loads(p["given"] or "[]"))}


def _final_task(db: DB, row: dict) -> dict:
    items = []
    for p in db.all("select * from round_partials where round=? and status='done' order by island", row["id"]):
        members = _contribution_items(db, _included(p))
        items.append({"artifact": p["artifact"], "weight": p["weight"], "late": False, "machine": p["reducer_machine"],
                      "job": p["reducer"] or (members[0]["job"] if members else None), "island": p["island"], "partial": True,
                      "forwarded": bool(p["forwarded"]), "members": members})
    return {**_public(db, db.one("select * from rounds where id=?", row["id"])), "stage": "final", "items": items}


def _tree_claim(db: DB, row: dict, job: str, machine: str | None) -> dict | None:
    settings = json.loads(row["settings"] or "{}")
    handoff = float(settings.get("handoff_s", HANDOFF_S))
    partials = db.all("select * from round_partials where round=? order by island", row["id"])
    for p in partials:
        if p["status"] == "reducing" and p["reducer"] == job:
            return _island_task(db, row, p["island"])
    if row["status"] == "reducing":
        return _final_task(db, row)
    for p in partials:
        if p["status"] != "waiting" or not _may_take(p["planned"], json.loads(p["machines"]), p["free_at"], machine, handoff):
            continue
        members = _contribution_items(db, json.loads(p["members"]))
        given = [m["contribution"] for m in members if not trust.excluded(db, row["pool"], m["machine"])]
        if not given:
            db.run("update round_partials set status='failed', note=? where round=? and island=? and status='waiting'",
                   "every member's machine is excluded", row["id"], p["island"])
            continue
        if db.run("update round_partials set status='reducing', reducer=?, reducer_machine=?, claimed_at=?, given=? "
                  "where round=? and island=? and status='waiting'", job, machine, now(), json.dumps(given), row["id"], p["island"]):
            return _island_task(db, row, p["island"])
    tick(db)  # an island stage may have just failed
    row = db.one("select * from rounds where id=?", row["id"])
    plan = json.loads(row["tree_plan"] or "{}")
    if row["status"] != "ready" or plan.get("final_at") is None:
        return None
    if not _may_take(plan.get("final"), None, plan["final_at"], machine, handoff):
        return None
    plan["machine"] = machine
    if not db.run("update rounds set status='reducing', reducer=?, tree_plan=? where id=? and status='ready'",
                  job, json.dumps(plan), row["id"]):
        return None
    return _final_task(db, row)


def partial(db: DB, pool: str, round_id: str, job: str, machine: str | None, artifact: str,
            dropped: list | None = None) -> dict:
    """An island reducer hands in its island's partial: the weighted mean of the members it was given,
    without those it `dropped` (contribution ids, outliers it found). The server sets the partial's
    weight from the contributions it includes."""
    with db.lock:
        tick(db)
        row = db.one("select * from rounds where id=? and pool=?", round_id, pool)
        if row is None:
            raise RoundError(404, f"no round {round_id}")
        settings = json.loads(row["settings"] or "{}")
        if settings.get("reduce") != "tree":
            raise RoundError(400, f"round {round_id} is not a tree round: finish it with its result")
        p = db.one("select * from round_partials where round=? and reducer=? and status='reducing'", round_id, job)
        if p is None:
            raise RoundError(409, f"this job holds no island stage of round {round_id} (claim it first)")
        if trust.excluded(db, pool, machine):
            db.run("update round_partials set status='waiting', reducer=null, reducer_machine=null, given=null, planned=null, "
                   "free_at=?, note=? where round=? and island=?", now(), "handed off: its reducer's machine is excluded",
                   round_id, p["island"])
            raise RoundError(403, "this machine is excluded from the pool's work (failed checks)")
        given = json.loads(p["given"] or "[]")
        dropped = [str(d) for d in (dropped or [])]
        if set(dropped) - set(given):
            raise RoundError(400, f"dropped names contributions this island stage was not given: {sorted(set(dropped) - set(given))}")
        members = _contribution_items(db, given)
        newly = [m["contribution"] for m in members
                 if m["contribution"] not in dropped and trust.excluded(db, pool, m["machine"])]
        if newly:  # a member was excluded since the claim: its update must not be in the partial
            db.run("update round_partials set status='waiting', reducer=null, reducer_machine=null, given=null, free_at=?, "
                   "planned=?, note=? where round=? and island=?", now(), machine, "claim again: a member was excluded",
                   round_id, p["island"])
            return {"accepted": False, "round": round_id, "reason": f"contributions {newly} come from machines excluded since "
                                                                     "the claim: claim the island again"}
        included = [m for m in members if m["contribution"] not in dropped]
        decay = float(settings.get("decay", 0.5))
        weight = sum(_decayed(m, decay) for m in included)
        status = "done" if included else "failed"
        db.run("update round_partials set status=?, artifact=?, weight=?, dropped=?, done_at=?, note=? where round=? and island=?",
               status, artifact if included else None, weight if included else None, json.dumps(dropped), now(),
               f"{len(included)} of {len(given)} contributions" if included else "every member was dropped",
               round_id, p["island"])
        tick(db)
    return {"accepted": True, "round": round_id, "island": p["island"], "weight": weight, "included": len(included),
            "dropped": dropped}


def readable(db: DB, artifact: str, pools: list[str]) -> bool:
    """A contribution of a round in a pool one of the machine's jobs runs in (for reducers)."""
    if not pools:
        return False
    marks = ",".join("?" * len(pools))
    return bool(db.one(f"select 1 as x from contributions where artifact=? and pool in ({marks})", artifact, *pools)
                or db.one(f"select 1 as x from round_partials p join rounds r on r.id=p.round where p.artifact=? "
                          f"and r.pool in ({marks})", artifact, *pools))
