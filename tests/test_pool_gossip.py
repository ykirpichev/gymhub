"""M7 gossip rounds (pool/gossip.py, pool/rounds.py mode gossip): the mixing plan, convergence on a
toy problem against central averaging, the round life cycle, and workers gossiping on an agent."""

import re
import sys
import time

import numpy as np
import pytest

from pool import coord, gossip, rounds
from pool.client import Client
from tests.test_pool import db_url, job_state, pool_env, start_agent, wait_for, write  # noqa: F401  (fixtures)
from tests.test_pool_coord import op, upload

# two islands of four machines: 10 Gbit/s inside, 100 Mbit/s between
MACHINES = [f"m{i}" for i in range(8)]
ISLAND = {m: ("a" if i < 4 else "b") for i, m in enumerate(MACHINES)}
LINKS = {(a, b): (10_000.0 if ISLAND[a] == ISLAND[b] else 100.0) for a in MACHINES for b in MACHINES if a != b}


def matrix(plan, nodes):
    return np.array([[plan[a].get(b, 0.0) for b in nodes] for a in nodes])


def test_the_plan_is_symmetric_and_doubly_stochastic():
    nodes = [f"c{i}" for i in range(8)]
    machine_of = dict(zip(nodes, MACHINES))
    for seed in range(20):
        w = matrix(gossip.plan(nodes, machine_of, LINKS, fanout=2, seed=seed), nodes)
        assert np.allclose(w, w.T) and np.allclose(w.sum(0), 1) and np.allclose(w.sum(1), 1) and (w >= 0).all()
        assert ((w > 0).sum(1) >= 3).all()  # itself and at least `fanout` partners
    assert gossip.plan(["solo"], {}, {}) == {"solo": {"solo": 1.0}}
    assert gossip.plan(nodes, machine_of, LINKS, seed="r1") == gossip.plan(nodes, machine_of, LINKS, seed="r1")  # deterministic


def test_bias_prefers_fast_links_but_still_crosses_islands():
    nodes = [f"c{i}" for i in range(8)]
    machine_of = dict(zip(nodes, MACHINES))
    island_of = ISLAND

    def crossing(bias):
        return sum(gossip.traffic(gossip.plan(nodes, machine_of, LINKS, 2, bias, seed), machine_of, island_of)["cross_island"]
                   for seed in range(200))

    uniform, biased = crossing(0.0), crossing(0.5)
    assert biased < 0.5 * uniform and biased > 0  # fewer slow-link downloads, but the islands still meet
    assert gossip.link_speed({}, "x", "x") == gossip.SAME_MACHINE_MBPS and gossip.link_speed({}, "x", "y") == gossip.GUESS_MBPS


def _toy(mode: str, rounds_n: int = 40, steps: int = 5, lr: float = 0.3, seed: int = 0):
    """Logistic regression, one shard per worker (each skewed to one class), local SGD between rounds.
    Returns (loss of the workers' average model on all data, disagreement between workers)."""
    rng = np.random.default_rng(seed)
    n, dim = 8, 5
    truth = rng.normal(size=dim)
    shards = []
    for i in range(n):
        x = rng.normal(size=(200, dim)) + (0.8 if i % 2 else -0.8)  # heterogeneous shards
        y = (x @ truth + 0.3 * rng.normal(size=200) > 0).astype(float)
        shards.append((x, y))
    xs = np.concatenate([s[0] for s in shards])
    ys = np.concatenate([s[1] for s in shards])

    def grad(w, x, y):
        return x.T @ (1 / (1 + np.exp(-x @ w)) - y) / len(y)

    def loss(w):
        p = np.clip(1 / (1 + np.exp(-xs @ w)), 1e-9, 1 - 1e-9)
        return float(-(ys * np.log(p) + (1 - ys) * np.log(1 - p)).mean())

    nodes = [f"c{i}" for i in range(n)]
    machine_of = dict(zip(nodes, MACHINES))
    models = np.zeros((n, dim))
    for r in range(rounds_n):
        for i, (x, y) in enumerate(shards):
            for _ in range(steps):
                models[i] -= lr * grad(models[i], x, y)
        if mode == "central":
            models[:] = models.mean(0)
        else:
            models = matrix(gossip.plan(nodes, machine_of, LINKS, seed=f"{seed}-{r}"), nodes) @ models
    return loss(models.mean(0)), float(np.linalg.norm(models - models.mean(0), axis=1).max())


def test_gossip_converges_like_central_averaging_on_a_toy_problem():
    central, _ = _toy("central")
    mixed, spread = _toy("gossip")
    start = _toy("central", rounds_n=1, steps=1)[0]
    assert mixed < start * 0.8                     # it learns
    assert abs(mixed - central) / central < 0.02   # and lands within 2% of central averaging
    assert spread < 0.1                            # with the workers close to each other


def test_pure_gossip_reaches_consensus_on_the_mean():
    nodes = [f"c{i}" for i in range(8)]
    machine_of = dict(zip(nodes, MACHINES))

    def spread(bias, rounds_n):
        values = np.arange(8, dtype=float)
        for r in range(rounds_n):
            values = matrix(gossip.plan(nodes, machine_of, LINKS, 2, bias, seed=r), nodes) @ values
        assert np.isclose(values.mean(), 3.5)  # the mean never moves
        return np.ptp(values)

    assert spread(0.0, 30) < 1e-6
    # preferring the island's fast links slows agreement between islands (the price of fewer slow downloads)
    assert spread(0.5, 30) > spread(0.0, 30) and spread(0.5, 60) < 1e-3


# ---- the round life cycle on the server -------------------------------------------------------------


def setup(env):
    db = env["db"]
    pool = db.one("select id from pools where name='default'")["id"]
    member = db.one("select id from members")["id"]
    coord.ref_publish(db, pool, "g", upload(env, {"x.npy": b"0"}), 0, member)
    return db, pool, member


def test_a_gossip_round_closes_with_a_plan_and_repeats_on_the_same_version(pool_env):
    db, pool, member = setup(pool_env)
    for (a, b), mbps in LINKS.items():
        db.run("insert into peer_links (src, dst, mbps, rtt_ms, measured_at) values (?, ?, ?, ?, ?)", a, b, mbps, 0.5, time.time())
    r = rounds.open_round(db, pool, "g", member, quorum=4, grace_s=0, mode="gossip", fanout=1, repeat=True)
    assert r["settings"]["mode"] == "gossip"
    for i in range(4):
        reply = rounds.contribute(db, pool, "g", f"j{i}", MACHINES[i * 2], upload(pool_env, {"c.txt": str(i)}), 1, 1, r["id"])
        assert reply["accepted"]
        if i < 3:
            assert rounds.mix(db, pool, r["id"], f"j{i}") is None  # still open
    closed = db.one("select * from rounds where id=?", r["id"])
    assert closed["status"] == "closed" and closed["note"].startswith("gossip:")
    assert rounds.claim(db, pool, "g", "reducer") is None  # nothing for a reducer
    for i in range(4):
        items = rounds.mix(db, pool, r["id"], f"j{i}")["items"]
        assert sum(item["weight"] for item in items) == pytest.approx(1)
        assert sum(item["self"] for item in items) == 1 and len(items) >= 2
    following = rounds.current(db, pool, "g")
    assert following["status"] == "open" and following["id"] != r["id"] and following["base_version"] == 1
    late = rounds.contribute(db, pool, "g", "j9", "m1", upload(pool_env, {"c.txt": "late"}), 1, 1, r["id"])
    assert not late["accepted"] and late["late"]
    with pytest.raises(rounds.RoundError):
        rounds.mix(db, pool, r["id"], "j9")  # no contribution of this job in the round
    with pytest.raises(rounds.RoundError):
        rounds.open_round(db, pool, "other", member, quorum=1, mode="flood")


def test_mix_is_only_for_gossip_rounds(pool_env):
    db, pool, member = setup(pool_env)
    r = rounds.open_round(db, pool, "g", member, quorum=1, grace_s=0)
    rounds.contribute(db, pool, "g", "j1", "m1", upload(pool_env, {"c.txt": "a"}), 1, 1)
    with pytest.raises(rounds.RoundError) as exc:
        rounds.mix(db, pool, r["id"], "j1")
    assert exc.value.status == 400


WORKER = r"""
import os, sys
import numpy as np
sys.path.insert(0, os.environ["POOL_LIB"])
import pool_job, pool_reducers

rank = int(sys.argv[1])
x = np.full(4, 10.0 * rank)
seen = None
for step in range(3):
    current = pool_job.current_round("g", after=seen, wait=30)
    seen = current["id"]
    mine = os.path.join(os.environ["POOL_OUTPUTS"], f"s{step}")
    os.makedirs(mine)
    np.save(os.path.join(mine, "x.npy"), x)
    assert pool_job.contribute("g", mine, weight=1, base_version=current["base_version"], round=seen)["accepted"]
    mix = None
    while mix is None:
        mix = pool_job.mix_round(seen, wait=30)
    folders = [(mine if i["self"] else pool_job.fetch_artifact(i["artifact"]), i["weight"], False) for i in mix["items"]]
    out = os.path.join(os.environ["POOL_OUTPUTS"], f"m{step}")
    pool_reducers.weighted_mean(folders, out)
    x = np.load(os.path.join(out, "x.npy"))
print("final", float(x[0]), flush=True)
"""


def test_three_workers_gossip_three_rounds_without_a_reducer(pool_env):
    member = Client.member()
    start = pool_env["tmp"] / "g0"
    start.mkdir()
    np.save(start / "x.npy", np.zeros(4))
    op("ref.publish", name="g", artifact=member.upload_folder(start)["artifact"], parent=0)
    op("round.open", ref="g", quorum=3, grace_s=0, deadline_s=60, repeat=True, mode="gossip", fanout=1)
    code = member.upload_folder(write(pool_env["tmp"] / "code", {"worker.py": WORKER}))["artifact"]
    start_agent(pool_env, "box1")
    jobs = [member.call("POST", "/v1/jobs", {"spec": {"image": "x", "command": [sys.executable, "worker.py", str(rank)],
                                                      "code": code}})["job"]["id"] for rank in range(3)]
    finals = []
    for job in jobs:
        final = wait_for(lambda: (s := job_state(job))["status"] in ("succeeded", "failed") and s, timeout=120)
        logs = "".join(c["text"] for c in member.call("GET", f"/v1/jobs/{job}/logs")["chunks"])
        assert final["status"] == "succeeded", logs
        finals.append(float(re.search(r"final (\S+)", logs).group(1)))
    assert sum(finals) == pytest.approx(30.0)        # mixing keeps the average (0, 10, 20 -> mean 10)
    assert max(finals) - min(finals) < 20.0           # and pulls the workers together
    assert op("ref.get", name="g")["ref"]["version"] == 1  # the ref never moved
