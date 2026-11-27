"""M7 max-flow placement planner (pool/planner.py) and the scheduler's `placement: maxflow` for gangs."""

import itertools
import json
import random

import pytest

from pool import planner, scheduler
from pool.db import load_job, now
from tests.test_pool import _gang, _peer_machine, add_machine, db, db_url  # noqa: F401  (fixtures)


def test_max_flow_and_min_cut_on_a_textbook_network():
    # CLRS figure 26.6: max flow 23
    edges = {("s", "v1"): 16, ("s", "v2"): 13, ("v1", "v3"): 12, ("v2", "v1"): 4, ("v2", "v4"): 14,
             ("v3", "v2"): 9, ("v3", "t"): 20, ("v4", "v3"): 7, ("v4", "t"): 4}
    value, flows, side = planner.max_flow(edges, "s", "t")
    assert value == pytest.approx(23)
    cut = sum(cap for (u, v), cap in edges.items() if u in side and v not in side)
    assert cut == pytest.approx(23)  # max flow = min cut
    for node in ("v1", "v2", "v3", "v4"):  # conservation
        assert sum(f for (u, v), f in flows.items() if v == node) == pytest.approx(sum(f for (u, v), f in flows.items() if u == node))
    assert all(flows[e] <= edges[e] + 1e-9 for e in edges)


def _net(links, default=1.0):
    return lambda a, b: links.get((a, b), links.get((b, a), default))


def test_a_two_stage_pipeline_goes_to_the_machines_with_the_fast_link():
    rate = {"a": 5.0, "b": 10.0, "c": 20.0}  # c is the fastest, but far away
    bw = _net({("a", "b"): 100.0, ("a", "c"): 1.0, ("b", "c"): 1.0})
    stages = [{"flops": 1.0, "bytes": 10.0}, {"flops": 1.0, "bytes": 0}]
    found = planner.plan(stages, ["a", "b", "c"], rate, bw)
    assert sorted(g[0] for g in found["stages"]) == ["a", "b"] and found["throughput"] == pytest.approx(5.0)
    assert found["bottleneck"] == ["compute of a (stage %d)" % [g[0] for g in found["stages"]].index("a")]
    # with little traffic, compute wins: b and c
    light = [{"flops": 1.0, "bytes": 0.01}, {"flops": 1.0, "bytes": 0}]
    assert sorted(g[0] for g in planner.plan(light, ["a", "b", "c"], rate, bw)["stages"]) == ["b", "c"]
    assert planner.plan(stages, ["a"], rate, bw) is None  # fewer machines than stages


def test_the_min_cut_names_a_slow_link():
    rate = {"a": 100.0, "b": 100.0}
    found = planner.throughput([["a"], ["b"]], [{"flops": 1, "bytes": 1}, {"flops": 1}], rate, _net({("a", "b"): 3.0}))
    assert found["throughput"] == pytest.approx(3.0) and found["bottleneck"] == ["link a -> b (stage 0 -> 1)"]


def test_replicas_add_machines_to_the_slow_stage():
    rate = {"big": 30.0, "s1": 10.0, "s2": 10.0, "s3": 10.0}
    bw = _net({}, default=1e9)
    stages = [{"flops": 1.0, "bytes": 1.0}, {"flops": 3.0, "bytes": 0}]  # stage 1 is three times heavier
    single = planner.plan(stages, list(rate), rate, bw)
    assert single["throughput"] == pytest.approx(10.0)  # big on stage 1 (30/3), one small on stage 0
    many = planner.plan(stages, list(rate), rate, bw, replicas=True)
    assert many["throughput"] > single["throughput"]
    assert sum(rate[m] / 3 for m in many["stages"][1]) == pytest.approx(many["throughput"]) or \
        sum(rate[m] for m in many["stages"][0]) == pytest.approx(many["throughput"])
    # the flow network agrees with the closed form when the links are free: min over stages of summed rate/flops
    expect = min(sum(rate[m] / s["flops"] for m in g) for g, s in zip(many["stages"], stages))
    assert many["throughput"] == pytest.approx(expect)


def test_local_search_finds_the_optimum_on_small_random_instances():
    rng = random.Random(7)
    worse = exact = 0
    for _ in range(30):
        machines = [f"m{i}" for i in range(6)]
        rate = {m: rng.uniform(1, 10) for m in machines}
        links = {(a, b): rng.choice([0.5, 5, 50]) for a, b in itertools.combinations(machines, 2)}
        stages = [{"flops": rng.uniform(0.5, 2), "bytes": rng.uniform(0.1, 2)} for _ in range(3)]
        best = planner.plan(stages, machines, rate, _net(links))
        local = planner.plan(stages, machines, rate, _net(links), limit=0)
        assert best["searched"] == "exhaustive" and local["searched"] == "local"
        assert local["throughput"] <= best["throughput"] + 1e-9
        worse += local["throughput"] < 0.9 * best["throughput"]
        exact += local["throughput"] == pytest.approx(best["throughput"])
    assert worse == 0 and exact >= 27  # restarts plus swaps on the sorted capacities: (nearly) always the optimum


# ---- the scheduler ---------------------------------------------------------------------------------------


def _links(db, pairs):
    for (a, b), mbps in pairs.items():
        for src, dst in ((a, b), (b, a)):
            db.run("insert into peer_links (src, dst, mbps, rtt_ms, measured_at) values (?, ?, ?, ?, ?)", src, dst, mbps, 1.0, now())


def _with_cost(db, ids, cost):
    for job_id in ids:
        spec = json.loads(db.one("select spec from jobs where id=?", job_id)["spec"])
        spec["cost"] = {"flops": 1e12, "dtype": "bf16", "bytes_out": 0, "bulk": True, **cost}
        db.run("update jobs set spec=? where id=?", json.dumps(spec), job_id)


def test_maxflow_placement_puts_a_pipeline_gang_on_the_fast_link(db):
    db.run("update pools set settings=? where id='p1'", json.dumps({"placement": "maxflow"}))
    r0, r1 = _gang(db)
    _with_cost(db, [r0, r1], {"link_bytes": 1e10})
    far, a, b = _peer_machine(db, "far"), _peer_machine(db, "aa"), _peer_machine(db, "bbb")
    _links(db, {("aa", "bbb"): 10_000, ("aa", "far"): 50, ("bbb", "far"): 50})
    # greedy would give rank 0 to whichever machine asks first; the plan keeps `far` out
    assert scheduler.claim(db, far, far["free"]) is None
    first = scheduler.claim(db, a, a["free"])
    second = scheduler.claim(db, b, b["free"])
    assert {first["id"], second["id"]} == {r0, r1}
    assert first["placement"]["plan"]["followed"] and second["placement"]["plan"]["throughput"] > 0
    lines = scheduler.explain(db, load_job(db.one("select * from jobs where id=?", r0)))
    assert any(line.startswith("plan: rank 0 on") and "far" not in line for line in lines)


def test_without_the_setting_or_after_the_wait_any_machine_takes_a_rank(db):
    r0, r1 = _gang(db)
    _with_cost(db, [r0, r1], {"link_bytes": 1e10})
    far, a, b = _peer_machine(db, "far"), _peer_machine(db, "aa"), _peer_machine(db, "bbb")
    _links(db, {("aa", "bbb"): 10_000, ("aa", "far"): 50, ("bbb", "far"): 50})
    assert scheduler.claim(db, far, far["free"])["id"] == r0  # greedy: first come
    db.run("update jobs set status='queued', machine=null where id=?", r0)
    db.run("update pools set settings=? where id='p1'", json.dumps({"placement": "maxflow"}))
    assert scheduler.claim(db, far, far["free"]) is None
    db.run("update jobs set created_at=? where id in (?, ?)", now() - 3600, r0, r1)
    assert scheduler.claim(db, far, far["free"]) is not None  # the plan is a preference, not a wait forever


def test_a_plan_is_redrawn_when_its_machine_goes_away(db):
    db.run("update pools set settings=? where id='p1'", json.dumps({"placement": "maxflow"}))
    r0, r1 = _gang(db)
    _with_cost(db, [r0, r1], {"link_bytes": 1e10})
    _peer_machine(db, "far"), _peer_machine(db, "aa"), _peer_machine(db, "bbb")
    _links(db, {("aa", "bbb"): 10_000, ("aa", "far"): 50, ("bbb", "far"): 50})
    job = load_job(db.one("select * from jobs where id=?", r0))
    settings = scheduler.pool_settings(db, "p1")
    assert sorted(scheduler.gang_plan(db, job, settings)["ranks"]) == ["aa", "bbb"]
    db.run("update machines set last_seen=0 where id='bbb'")
    assert "far" in scheduler.gang_plan(db, job, settings)["ranks"]


def test_placement_is_a_pool_setting_and_link_bytes_a_cost_hint(db):
    from pool import spec as specs

    assert scheduler.pool_settings(db, "p1")["placement"] == "greedy"
    spec = specs.normalize({"image": "x", "command": ["x"], "gang": {"size": 2}, "cost": {"flops": 1e15, "link_bytes": 4e9}})
    assert spec["cost"]["link_bytes"] == 4e9
    with pytest.raises(specs.SpecError):
        specs.normalize({"image": "x", "command": ["x"], "cost": {"flops": 1, "links": 3}})
    add_machine(db, "lonely")
    assert scheduler.bandwidth(db, [{"id": "x", "profile": {"network": {"up_mbps": 80}}},
                                    {"id": "y", "profile": {"network": {"down_mbps": 800}}}])("x", "y") == pytest.approx(80e6 / 8)
