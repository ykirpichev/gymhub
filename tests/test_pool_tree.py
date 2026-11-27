"""#156: island-first reduce trees for rounds (pool/rounds.py reduce tree, pool/reducers.py merge_stage).

The plan groups contributions by island; island stages make partials that carry their islands'
weights, so the tree equals the flat reduction; reducers that stop hand off; a missing partial
counts against min; checks still catch poisoned updates; and on agents, one partial per island
crosses the slow link instead of one update per member."""

import json
import sys
import time

import numpy as np
import pytest

from pool import agent as agent_module
from pool import client as client_module
from pool import coord, reducers, rounds, trust
from pool.client import Client
from tests.test_pool import db_url, job_state, pool_env, small_chunks, start_agent, wait_for, write  # noqa: F401  (fixtures)
from tests.test_pool_coord import op

FAST, SLOW = (10_000.0, 0.2), (100.0, 20.0)  # Mbit/s, ms: island-grade inside, a slow link between


def link(db, a, b, mbps_rtt):
    db.run("insert into peer_links (src, dst, mbps, rtt_ms, measured_at) values (?, ?, ?, ?, ?) "
           "on conflict (src, dst) do update set mbps=excluded.mbps, rtt_ms=excluded.rtt_ms", a, b, *mbps_rtt, time.time())


def islands(db, groups, between=SLOW, inside=FAST):
    everyone = [m for g in groups for m in g]
    for a in everyone:
        for b in everyone:
            if a != b:
                same = any(a in g and b in g for g in groups)
                link(db, a, b, inside if same else between)


def setup(env):
    db = env["db"]
    pool = db.one("select id from pools where name='default'")["id"]
    member = db.one("select id from members")["id"]
    start = env["tmp"] / "start"
    start.mkdir()
    np.save(start / "x.npy", np.zeros(16, dtype=np.float32))
    coord.ref_publish(db, pool, "w", Client.member().upload_folder(start)["artifact"], 0, member)
    return db, pool, member


class Updates:
    """Contribution folders on disk, uploaded; fetch(artifact) is the folder (what pool_job.fetch_artifact gives)."""

    def __init__(self, env):
        self.env, self.folders, self.n = env, {}, 0

    def add(self, array) -> str:
        self.n += 1
        folder = self.env["tmp"] / f"u{self.n}"
        folder.mkdir()
        np.save(folder / "x.npy", np.asarray(array))
        return self.register(folder)

    def register(self, folder) -> str:
        artifact = Client.member().upload_folder(folder)["artifact"]
        self.folders[artifact] = str(folder)
        return artifact

    def fetch(self, artifact):
        return self.folders[artifact]

    def out(self, name):
        return str(self.env["tmp"] / name)


def publish(db, pool, member):
    return lambda ref, a, parent: coord.ref_publish(db, pool, ref, a, parent, member)


def run_tree(db, pool, member, ups, ref="w", machine_of_job=None, report=None, robust=False, verify=0.0, final="final"):
    """Drive a ready tree round to its end the way reduce_round does in jobs: each island stage on its
    planned machine, then the final stage. Returns (result array, final claim)."""
    current = rounds.current(db, pool, ref)
    for p in current["tree"]["partials"]:
        if p["status"] != "waiting":
            continue
        job = f"red-{p['island']}"
        claimed = rounds.claim(db, pool, ref, job, p["planned"])
        assert claimed["stage"] == "island" and claimed["island"] == p["island"]
        out = ups.out(f"partial-{claimed['id']}-{p['island']}")
        merged = reducers.merge_stage(claimed, out, ups.fetch, robust=robust, report=report)
        reply = rounds.partial(db, pool, claimed["id"], job, p["planned"], ups.register(out), merged["dropped"])
        assert reply["accepted"], reply
    plan = rounds.current(db, pool, ref)["tree"]["final"]
    claimed = rounds.claim(db, pool, ref, final, plan)
    assert claimed["stage"] == "final"
    out = ups.out(f"final-{claimed['id']}")
    merged = reducers.merge_stage(claimed, out, ups.fetch, report=report, verify=verify)
    rounds.finish(db, pool, claimed["id"], final, member, ups.register(out), publish(db, pool, member))
    return np.load(f"{out}/x.npy"), claimed, merged


def test_tree_is_opt_in_and_validated(pool_env):
    db, pool, member = setup(pool_env)
    with pytest.raises(rounds.RoundError, match="reduce is flat or tree"):
        rounds.open_round(db, pool, "w", member, 2, reduce="ring")
    with pytest.raises(rounds.RoundError, match="reduce tree is for mode reduce"):
        rounds.open_round(db, pool, "w", member, 2, mode="gossip", reduce="tree")
    with pytest.raises(rounds.RoundError, match="reduce tree is for mode reduce"):
        rounds.open_round(db, pool, "w", member, 2, reducer="torchft", lighthouse="http://10.0.0.5:1", reduce="tree")
    with pytest.raises(Exception, match="handoff_s"):
        op("round.open", ref="w", quorum=2, reduce="tree", handoff_s=-1)
    flat = rounds.open_round(db, pool, "w", member, 1, grace_s=0)
    assert "reduce" not in flat["settings"]  # flat stays the default, and its settings are as before
    with pytest.raises(client_module.PoolError, match="only a running job's machine"):  # members upload their bytes
        Client.member().call("POST", "/v1/artifacts", {"manifest": {"v": 1, "files": []}, "held": True})
    db.run("update rounds set status='closed' where id=?", flat["id"])
    tree = op("round.open", ref="w", quorum=1, grace_s=0, reduce="tree", handoff_s=7, repeat=True)["round"]
    assert tree["settings"]["reduce"] == "tree" and tree["settings"]["handoff_s"] == 7
    ups = Updates(pool_env)
    rounds.contribute(db, pool, "w", "j1", "m1", ups.add(np.ones(16, dtype=np.float32)), 1, 1)
    claimed = rounds.claim(db, pool, "w", "red", "m1")  # one contribution: forwarded, straight to the final stage
    assert claimed["stage"] == "final" and claimed["items"][0]["forwarded"]
    done = rounds.finish(db, pool, claimed["id"], "red", member, claimed["items"][0]["artifact"], publish(db, pool, member))
    assert done["next"]["settings"]["reduce"] == "tree" and done["next"]["settings"]["handoff_s"] == 7


def test_the_plan_groups_by_island_and_picks_the_fastest_collectors(pool_env):
    db, pool, member = setup(pool_env)
    islands(db, [["a1", "a2", "a3"], ["b1", "b2"], ["c1"]])
    for other in ("a1", "a3"):  # a2 has the fastest links in its island: it collects there
        link(db, other, "a2", (40_000.0, 0.1))
    rounds.open_round(db, pool, "w", member, quorum=7, grace_s=0, reduce="tree")
    ups = Updates(pool_env)
    for index, machine in enumerate(["a1", "a2", "a3", "b1", "b2", "c1", None]):
        rounds.contribute(db, pool, "w", f"j{index}", machine, ups.add(np.full(16, index, dtype=np.float32)), 1, 1)
    tree = rounds.current(db, pool, "w")["tree"]
    by_size = sorted(tree["partials"], key=lambda p: -p["members"])
    assert [p["members"] for p in by_size] == [3, 2, 1, 1]
    assert by_size[0]["planned"] == "a2" and by_size[0]["status"] == "waiting"
    assert by_size[1]["planned"] in ("b1", "b2")
    assert all(p["forwarded"] and p["status"] == "done" for p in by_size[2:])  # c1, and a member's upload
    # a2 may take its island at once; a1 (same island) waits out the handoff; b1 takes its own
    assert rounds.claim(db, pool, "w", "ja1", "a1") is None
    assert rounds.claim(db, pool, "w", "ja2", "a2")["island"] == by_size[0]["island"]
    assert rounds.claim(db, pool, "w", "jb", by_size[1]["planned"])["island"] == by_size[1]["island"]
    assert rounds.claim(db, pool, "w", "jc", "c1") is None  # no final before every partial is in


def test_tree_equals_flat_with_weights_and_late_work(pool_env):
    """Partials carry their islands' total weights (late work times decay), so the weighted mean of
    the partials is the flat weighted mean of the members, up to float rounding."""
    db, pool, member = setup(pool_env)
    groups = [["a1", "a2", "a3"], ["b1", "b2", "b3"], ["c1"]]
    islands(db, groups)
    rng = np.random.default_rng(0)
    ups = Updates(pool_env)
    # round 1, flat, reduced at once; a straggler on v1 arrives late and is carried to round 2
    first = rounds.open_round(db, pool, "w", member, quorum=1, grace_s=0, late="next", decay=0.25)
    rounds.contribute(db, pool, "w", "j0", "a1", ups.add(np.zeros(16, dtype=np.float32)), 1, 1)
    claimed = rounds.claim(db, pool, "w", "red")
    rounds.finish(db, pool, claimed["id"], "red", member, claimed["items"][0]["artifact"], publish(db, pool, member))
    del first
    late = rng.normal(size=16).astype(np.float32)
    rounds.contribute(db, pool, "w", "slow", "b3", ups.add(late), 5.0, 1)
    rounds.open_round(db, pool, "w", member, quorum=7, grace_s=0, late="next", decay=0.25, reduce="tree")
    members = [(m, rng.normal(size=16).astype(np.float32) * (i + 1), float(i + 1)) for i, m in
               enumerate(["a1", "a2", "a3", "b1", "b2", "c1", "a2"])]
    for index, (machine, array, weight) in enumerate(members):
        assert rounds.contribute(db, pool, "w", f"j{index}", machine, ups.add(array), weight, 2)["accepted"]
    result, final, _ = run_tree(db, pool, member, ups)
    assert len(final["items"]) == 3  # one partial per island
    weights = [w for _m, _a, w in members] + [5.0 * 0.25]
    arrays = [a.astype(np.float64) for _m, a, _w in members] + [late.astype(np.float64)]
    flat = sum(w * a for w, a in zip(weights, arrays)) / sum(weights)
    assert np.allclose(result, flat, rtol=1e-5, atol=1e-6)
    assert sum(item["weight"] for item in final["items"]) == pytest.approx(sum(weights))
    # the same updates reduced flat by pool_reducers give the same answer
    flat_folders = [(ups.fetch(i["artifact"]), i["weight"], i["late"]) for p in final["items"] for i in p["members"]]
    reducers.weighted_mean(flat_folders, ups.out("flat"), decay=0.25)
    assert np.allclose(np.load(ups.out("flat") + "/x.npy"), result, rtol=1e-5, atol=1e-6)
    assert rounds.current(db, pool, "w")["status"] == "closed"


def test_a_reducer_that_stops_hands_off_within_its_island(pool_env):
    db, pool, member = setup(pool_env)
    islands(db, [["a1", "a2"], ["b1", "b2"]])
    owner = member
    for job, machine in (("ja1", "a1"), ("ja2", "a2"), ("jb1", "b1")):
        db.run("insert into jobs (id, pool, owner, spec, status, machine, created_at) values (?, ?, ?, '{}', 'running', ?, 0)",
               job, pool, owner, machine)
    rounds.open_round(db, pool, "w", member, quorum=4, grace_s=0, reduce="tree", handoff_s=60)
    ups = Updates(pool_env)
    for index, machine in enumerate(["a1", "a2", "b1", "b2"]):
        rounds.contribute(db, pool, "w", f"w{index}", machine, ups.add(np.full(16, index, dtype=np.float32)), 1, 1)
    tree = rounds.current(db, pool, "w")["tree"]
    island_a = next(p for p in tree["partials"] if p["planned"] in ("a1", "a2"))
    planned = island_a["planned"]
    other = "a2" if planned == "a1" else "a1"
    taker = f"j{planned}"
    assert rounds.claim(db, pool, "w", f"j{other}", other) is None  # the planned machine has the handoff time
    assert rounds.claim(db, pool, "w", taker, planned)["island"] == island_a["island"]
    db.run("update jobs set status='failed' where id=?", taker)  # the reducer dies mid-stage
    claimed = rounds.claim(db, pool, "w", f"j{other}", other)  # handed to the other machine of the island, at once
    assert claimed and claimed["stage"] == "island" and claimed["island"] == island_a["island"]
    assert "handed off" in next(p for p in rounds.current(db, pool, "w")["tree"]["partials"]
                                if p["island"] == island_a["island"])["note"]
    # a machine of another island takes it only after twice the handoff
    db.run("update jobs set status='failed' where id=?", f"j{other}")
    assert rounds.claim(db, pool, "w", "jb1", "b1")["island"] != island_a["island"]  # b1 takes its own island first
    rounds.tick(db)
    db.run("update round_partials set free_at=? where island=?", time.time() - 121, island_a["island"])
    stranger = rounds.claim(db, pool, "w", "jx", "x9")
    assert stranger["island"] == island_a["island"]


def test_a_missing_partial_counts_against_min(pool_env):
    db, pool, member = setup(pool_env)
    islands(db, [["a1", "a2"], ["b1", "b2"]])
    ups = Updates(pool_env)

    def round_with(minimum):
        rounds.open_round(db, pool, "w", member, quorum=4, min_contrib=minimum, grace_s=0, deadline_s=60, reduce="tree")
        version = rounds.current(db, pool, "w")["base_version"]
        for index, machine in enumerate(["a1", "a2", "b1", "b2"]):
            rounds.contribute(db, pool, "w", f"w{index}", machine, ups.add(np.full(16, index + 1, dtype=np.float32)), 1, version)
        tree = rounds.current(db, pool, "w")["tree"]
        a = next(p for p in tree["partials"] if p["planned"] in ("a1", "a2"))
        claimed = rounds.claim(db, pool, "w", "ra", a["planned"])
        reducers.merge_stage(claimed, ups.out(f"pa-{claimed['id']}"), ups.fetch)
        rounds.partial(db, pool, claimed["id"], "ra", a["planned"], ups.register(ups.out(f"pa-{claimed['id']}")))
        # island b never hands in its partial: past the deadline it fails, and its members count as missing
        db.run("update rounds set ready_at=? where id=?", time.time() - 61, claimed["id"])
        rounds.tick(db)
        return rounds.current(db, pool, "w"), claimed["id"]

    failed, rid = round_with(3)
    assert failed["status"] == "failed" and "2 contributions" in failed["note"]
    assert any(p["status"] == "failed" and "deadline" in p["note"] for p in failed["tree"]["partials"])
    ok, rid = round_with(2)
    assert ok["status"] == "ready"
    result, final, _ = run_tree(db, pool, member, ups)
    assert len(final["items"]) == 1 and np.allclose(result, 1.5)  # island a alone: (1 + 2) / 2


def test_an_unreachable_member_is_left_out_of_its_partial(pool_env):
    db, pool, member = setup(pool_env)
    islands(db, [["a1", "a2", "a3"]])
    ups = Updates(pool_env)
    rounds.open_round(db, pool, "w", member, quorum=3, grace_s=0, reduce="tree")
    gone = None
    for index, machine in enumerate(["a1", "a2", "a3"]):
        artifact = ups.add(np.full(16, index + 1, dtype=np.float32))
        rounds.contribute(db, pool, "w", f"w{index}", machine, artifact, index + 1.0, 1)
        gone = artifact if machine == "a3" else gone
    planned = rounds.current(db, pool, "w")["tree"]["partials"][0]["planned"]
    claimed = rounds.claim(db, pool, "w", "r", planned)

    def fetch(artifact):
        if artifact == gone:
            raise OSError("its machine is offline")
        return ups.fetch(artifact)

    merged = reducers.merge_stage(claimed, ups.out("p"), fetch)
    assert len(merged["unreachable"]) == 1 and merged["dropped"] == merged["unreachable"]
    reply = rounds.partial(db, pool, claimed["id"], "r", planned, ups.register(ups.out("p")), merged["dropped"])
    assert reply["weight"] == 3.0 and reply["included"] == 2  # the server's weight leaves it out too
    assert np.allclose(np.load(ups.out("p") + "/x.npy"), (1 * 1 + 2 * 2) / 3)


def test_poisoned_updates_are_caught_with_the_tree_on(pool_env):
    """The checked tier still works: the island stage checks each member before forming its partial
    (the poisoned update is left out, reported, and its machine excluded), the result equals the
    flat robust reduction, an excluded machine's update is never handed to an island reducer, and
    a partial that does not match its inputs is caught by the final stage's verification."""
    db, pool, member = setup(pool_env)
    Client.member().call("PATCH", "/v1/pools/default", {"settings": {"trust": "checked", "exclude_after": 1}})
    islands(db, [["a1", "a2", "a3", "evil"], ["b1", "b2", "b3"]])
    rng = np.random.default_rng(1)
    direction = rng.normal(size=16)
    honest = {m: (direction + 0.1 * rng.normal(size=16)).astype(np.float32) for m in ["a1", "a2", "a3", "b1", "b2", "b3"]}
    poison = (-50 * direction).astype(np.float32)
    ups = Updates(pool_env)

    def report(machine, ok, kind, detail):
        trust.report(db, pool, machine, ok, kind, detail)

    rounds.open_round(db, pool, "w", member, quorum=7, grace_s=0, reduce="tree")
    for index, (machine, array) in enumerate([*honest.items(), ("evil", poison)]):
        rounds.contribute(db, pool, "w", f"j{index}", machine, ups.add(array), 1.0, 1)
    result, final, _ = run_tree(db, pool, member, ups, report=report, robust=True)
    assert trust.excluded(db, pool, "evil")
    assert all(i["machine"] != "evil" for p in final["items"] for i in p["members"])
    expected = np.mean([a.astype(np.float64) for a in honest.values()], axis=0)
    assert np.allclose(result, expected, rtol=1e-5, atol=1e-6)
    flat_folders = [(ups.fetch(i["artifact"]), 1.0, False) for p in final["items"] for i in p["members"]]
    flat_folders.append((ups.fetch(next(c["artifact"] for c in db.all("select artifact, machine from contributions")
                                        if c["machine"] == "evil")), 1.0, False))
    _, dropped = reducers.robust_mean(flat_folders, ups.out("flat-robust"))
    assert np.allclose(np.load(ups.out("flat-robust") + "/x.npy"), result, rtol=1e-5, atol=1e-6) and dropped == [6]

    # next round: the excluded machine's update (in the db from before it was excluded) never reaches a reducer
    version = rounds.current(db, pool, "w")["result_version"]
    rounds.open_round(db, pool, "w", member, quorum=4, grace_s=0, reduce="tree")
    db.run("insert into contributions (id, pool, ref, round, job, machine, artifact, weight, base_version, created_at) "
           "select 'cx', pool, ref, id, 'jevil', 'evil', ?, 1, base_version, 0 from rounds where status='open'", ups.add(poison))
    for machine in ("a1", "b1", "b2"):
        rounds.contribute(db, pool, "w", f"n{machine}", machine, ups.add(honest[machine]), 1.0, version)
    tree = rounds.current(db, pool, "w")["tree"]
    island_a = next(p for p in tree["partials"] if p["members"] == 2)
    assert island_a["status"] == "waiting" and island_a["planned"] in ("a1", "evil")
    island_b = next(p for p in tree["partials"] if p["planned"] in ("b1", "b2"))
    # the poisoned machine may not reduce, and a1 gets only its own update to reduce
    with pytest.raises(rounds.RoundError):
        rounds.partial(db, pool, rounds.current(db, pool, "w")["id"], "jevil", "evil", ups.add(poison))
    db.run("update round_partials set planned='a1' where island=? and round=?", island_a["island"], rounds.current(db, pool, "w")["id"])
    claimed = rounds.claim(db, pool, "w", "ra", "a1")
    assert [i["machine"] for i in claimed["items"]] == ["a1"]
    reducers.merge_stage(claimed, ups.out("pa2"), ups.fetch)
    rounds.partial(db, pool, claimed["id"], "ra", "a1", ups.register(ups.out("pa2")))
    # island b's reducer lies: its partial is not the mean of its inputs; verification catches it
    claimed = rounds.claim(db, pool, "w", "rb", island_b["planned"])
    lie = pool_env["tmp"] / "lie"
    lie.mkdir()
    np.save(lie / "x.npy", poison)
    rounds.partial(db, pool, claimed["id"], "rb", island_b["planned"], ups.register(lie))
    final = rounds.claim(db, pool, "w", "fin", "a1")
    merged = reducers.merge_stage(final, ups.out("final2"), ups.fetch, report=report, verify=1.0)
    assert merged["bad_partials"] == [island_b["island"]] and trust.excluded(db, pool, island_b["planned"])
    assert np.allclose(np.load(ups.out("final2") + "/x.npy"), honest["a1"])


# ---- on agents: what crosses the slow link -------------------------------------------------------------

WORKER = r"""
import os, sys
import numpy as np
sys.path.insert(0, os.environ["POOL_LIB"])
import pool_job, pool_reducers

rank, mode = int(sys.argv[1]), sys.argv[2]
current = pool_job.current_round("weights", wait=30)
path, version = pool_job.fetch_ref("weights", current["base_version"])
x = np.load(os.path.join(path, "x.npy"))
out = os.path.join(os.environ["POOL_OUTPUTS"], "update")
os.makedirs(out)
np.save(os.path.join(out, "x.npy"), x + np.random.default_rng(rank).normal(size=x.shape).astype(x.dtype))
print("contributed", pool_job.contribute("weights", out, weight=rank, base_version=version), flush=True)
if mode == "tree":  # every worker may be its island's reducer, or the final one
    while pool_job.ref("weights")["version"] == version:
        done = pool_reducers.reduce_round("weights", wait=2)
        if done:
            print("reduced", done["stage"], done["island"], flush=True)
"""

REDUCER = r"""
import os, sys
sys.path.insert(0, os.environ["POOL_LIB"])
import pool_reducers
while not pool_reducers.reduce_round("weights", wait=10):
    pass
"""

PER_ISLAND = 3
SIZE = 4096  # float32: 16 KiB a contribution, 16 chunks of 1 KiB


@pytest.mark.parametrize("mode", ["flat", "tree"])
def test_one_partial_per_island_crosses_the_slow_link(pool_env, small_chunks, monkeypatch, mode):
    """Two islands of three machines; the server sits in island A. Flat: every contribution goes to the
    server and on to one reducer, so each of island B's updates crosses (and with the server in a
    third place, every update crosses twice). Tree: contributions are held and reduced inside their
    island over the peer swarm, and only one partial (B's to A, or A's to B) crosses."""
    monkeypatch.setattr(agent_module.Agent, "check_peers", lambda self, force=False: None)  # the links are set below
    traffic = []  # (kind, machine, other machine or None, artifact or chunk, bytes)
    clients = {}
    real_peer, real_server, real_upload = (agent_module.Agent.fetch_from_peer, client_module.Client.fetch_chunk,
                                           client_module.Client.upload_manifest)

    def from_peer(self, peer, sha, ticket, dest):
        got = real_peer(self, peer, sha, ticket, dest)
        traffic.append(("peer", self.config["name"], peer["name"], sha, got))
        return got

    def from_server(self, sha, artifact, dest, limiter=None):
        got = real_server(self, sha, artifact, dest, limiter)
        if id(self) in clients:
            traffic.append(("down", clients[id(self)], None, artifact, got))
        return got

    def upload(self, manifest, sources, **kwargs):
        reply = real_upload(self, manifest, sources, **kwargs)
        if id(self) in clients:
            traffic.append(("up", clients[id(self)], None, reply["artifact"], reply["sent_bytes"]))
        return reply

    monkeypatch.setattr(agent_module.Agent, "fetch_from_peer", from_peer)
    monkeypatch.setattr(client_module.Client, "fetch_chunk", from_server)
    monkeypatch.setattr(client_module.Client, "upload_manifest", upload)
    db = pool_env["db"]
    names = [f"box{i}" for i in range(1, 2 * PER_ISLAND + 1)]
    island_of = {name: "A" if index < PER_ISLAND else "B" for index, name in enumerate(names)}
    agents = [start_agent(pool_env, name, labels={"box": name}) for name in names]
    for agent in agents:
        clients[id(agent.client)] = agent.config["name"]
    wait_for(lambda: all(a.peer_key for a in agents) and db.one(
        "select count(*) as n from machines where profile like '%\"port\"%'")["n"] == len(names), timeout=30)
    ids = {r["name"]: r["id"] for r in db.all("select id, name from machines")}
    islands(db, [[ids[n] for n in names if island_of[n] == "A"], [ids[n] for n in names if island_of[n] == "B"]])
    member = Client.member()
    start = pool_env["tmp"] / "w0"
    start.mkdir()
    np.save(start / "x.npy", np.zeros(SIZE, dtype=np.float32))
    op("ref.publish", name="weights", artifact=member.upload_folder(start)["artifact"], parent=0)
    op("round.open", ref="weights", quorum=len(names), grace_s=0, deadline_s=120, reduce=mode)
    code = member.upload_folder(write(pool_env["tmp"] / "code", {"worker.py": WORKER, "reducer.py": REDUCER}))["artifact"]

    def submit(command, box):
        spec = {"image": "x", "command": [sys.executable, *command], "code": code, "resources": {"labels": {"box": box}}}
        return member.call("POST", "/v1/jobs", {"spec": spec})["job"]["id"]

    jobs = [submit(["worker.py", str(rank), mode], name) for rank, name in enumerate(names, start=1)]
    if mode == "flat":
        jobs.append(submit(["reducer.py"], "box1"))
    for job in jobs:
        final = wait_for(lambda: (s := job_state(job))["status"] in ("succeeded", "failed") and s, timeout=180)
        logs = "".join(c["text"] for c in member.call("GET", f"/v1/jobs/{job}/logs")["chunks"])
        assert final["status"] == "succeeded", logs
    ref = op("ref.get", name="weights")["ref"]
    assert ref["version"] == 2
    folder = pool_env["tmp"] / "final"
    member.download_folder(ref["artifact"], folder)
    weights = np.arange(1, len(names) + 1, dtype=np.float64)
    updates = [np.random.default_rng(rank).normal(size=SIZE).astype(np.float32).astype(np.float64)
               for rank in range(1, len(names) + 1)]
    flat = sum(w * u for w, u in zip(weights, updates)) / weights.sum()
    assert np.allclose(np.load(folder / "x.npy"), flat, rtol=1e-5, atol=1e-6)  # tree == flat on the same updates

    # what crossed between the islands: round inputs (contributions and partials), the server in island A
    inputs = {r["artifact"] for r in db.all("select artifact from contributions")}
    inputs |= {r["artifact"] for r in db.all("select artifact from round_partials where artifact is not null")}
    chunk_of = {}
    for artifact in inputs:
        manifest = db.one("select manifest from artifacts where id=?", artifact)["manifest"]
        for sha in json.loads(manifest)["files"][0].get("chunks", []):
            chunk_of[sha] = artifact
    update = SIZE * 4 + 128  # one update's bytes: the .npy file
    crossed = everywhere = 0
    for kind, machine, other, what, size in traffic:
        if (what if kind != "peer" else chunk_of.get(what)) not in inputs:
            continue  # the start version and the result: the same in both modes
        everywhere += size if kind != "peer" or island_of[machine] != island_of[other] else 0  # server elsewhere
        if kind == "peer" and island_of[machine] != island_of[other]:
            crossed += size
        elif kind in ("up", "down") and island_of[machine] != "A":
            crossed += size
    print(f"\n{mode}: {crossed / update:.2f} updates crossed the slow link (server in island A), "
          f"{everywhere / update:.2f} with the server in a third place; {len(traffic)} transfers")
    if mode == "flat":
        assert crossed == pytest.approx(PER_ISLAND * update, rel=0.01)  # each of island B's contributions
        assert everywhere == pytest.approx(4 * PER_ISLAND * update, rel=0.01)  # every update up, and down again
    else:
        assert crossed == pytest.approx(update, rel=0.01)  # one partial
        assert everywhere == pytest.approx(update, rel=0.01)
        assert not any(kind in ("up", "down") and what in inputs for kind, _m, _o, what, _s in traffic)  # never the server
        # held: the server stores none of their chunks and serves none, even to a member who may read them
        held = db.one("select artifact from held_artifacts limit 1")["artifact"]
        sha = next(s for s, a in chunk_of.items() if a == held)
        with pytest.raises(client_module.PoolError) as refused:
            member.call("GET", f"/v1/chunks/{sha}?artifact={held}")
        assert refused.value.status == 404
        partials = db.all("select * from round_partials")
        assert len(partials) == 2 and all(p["status"] == "done" and not p["forwarded"] for p in partials)


def test_a_held_artifact_cannot_be_claimed_by_declaring_its_manifest(pool_env, small_chunks):
    """Declaring a manifest is not having its bytes: a machine of another member that declares an artifact
    another machine already holds gets 409, so it never becomes an owner or gets tickets to fetch it."""
    import os

    from pool.agent import register
    from pool.artifacts import build_manifest
    from tests.test_pool import invite

    def machine_of(member, name):
        code = member.call("POST", "/v1/machine-codes", {})["code"]
        return Client(pool_env["url"], register(pool_env["url"], code, name, {}, [], ["*"])["token"])

    def running(member, machine):
        """A job of `member` that `machine` is running (machines declare artifacts only for those)."""
        code = member.upload_folder(write(pool_env["tmp"] / f"code-{os.urandom(4).hex()}", {"run.sh": b"true"}))["artifact"]
        member.call("POST", "/v1/jobs", {"spec": {"image": "x", "command": ["true"], "code": code}})
        leased = machine.call("POST", "/v1/agent/lease", {"free": {"gpus": [], "cpus": 4, "memory_gb": 8}, "wait": 0})["jobs"]
        assert leased
        return leased[0]["id"]

    def declare(machine, job):
        return machine.call("POST", "/v1/artifacts", {"manifest": manifest, "held": True, "job": job})

    ann, bob = Client.member(), invite(pool_env, name="bob")
    manifest, _sources = build_manifest(write(pool_env["tmp"] / "held", {"x.bin": os.urandom(3000)}))
    anns = machine_of(ann, "anns")
    held = declare(anns, running(ann, anns))
    assert held["held"]
    bobs = machine_of(bob, "bobs")
    with pytest.raises(client_module.PoolError) as exc:
        declare(bobs, running(bob, bobs))
    assert exc.value.status == 409
    db = pool_env["db"]
    bob_id = db.one("select id from members where name='bob'")["id"]
    assert not db.one("select 1 as x from artifact_owners where artifact=? and member=?", held["artifact"], bob_id)
    anns2 = machine_of(ann, "anns2")
    assert declare(anns2, running(ann, anns2))["held"]  # the same member's other machine may hold it too
