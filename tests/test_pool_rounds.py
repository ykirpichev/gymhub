"""M5 rounds (pool/rounds.py): quorum, deadline, late work, reducers, and a run on an agent."""

import json
import sys
import time

import numpy as np
import pytest

from pool import coord, rounds
from pool.client import Client
from tests.test_pool import db_url, job_state, pool_env, start_agent, wait_for, write  # noqa: F401  (fixtures)
from tests.test_pool_coord import op, upload


def setup(env):
    db = env["db"]
    pool = db.one("select id from pools where name='default'")["id"]
    member = db.one("select id from members")["id"]
    coord.ref_publish(db, pool, "w", upload(env, {"x.npy": b"0"}), 0, member)
    return db, pool, member


def art(env, tag):
    return upload(env, {"c.txt": tag})


def test_quorum_then_grace_makes_a_round_ready(pool_env):
    db, pool, member = setup(pool_env)
    r = rounds.open_round(db, pool, "w", member, quorum=2, grace_s=0, deadline_s=60)
    assert r["base_version"] == 1 and rounds.open_round(db, pool, "w", member, 2)["id"] == r["id"]  # one at a time
    assert rounds.contribute(db, pool, "w", "j1", "m1", art(pool_env, "a"), 1, 1)["accepted"]
    assert rounds.current(db, pool, "w")["status"] == "open"
    rounds.contribute(db, pool, "w", "j2", "m2", art(pool_env, "b"), 3, 1)
    assert rounds.current(db, pool, "w")["status"] == "ready"
    claimed = rounds.claim(db, pool, "w", "reducer")
    assert sorted(i["weight"] for i in claimed["items"]) == [1.0, 3.0]


def test_deadline_with_and_without_the_minimum(pool_env):
    db, pool, member = setup(pool_env)
    r = rounds.open_round(db, pool, "w", member, quorum=3, min_contrib=1, deadline_s=60)
    rounds.contribute(db, pool, "w", "j1", "m1", art(pool_env, "a"), 1, 1)
    db.run("update rounds set deadline=? where id=?", time.time() - 1, r["id"])
    rounds.tick(db)
    assert rounds.current(db, pool, "w")["status"] == "ready" and "deadline" in rounds.current(db, pool, "w")["note"]
    db.run("update rounds set status='closed' where id=?", r["id"])
    empty = rounds.open_round(db, pool, "w", member, quorum=2, deadline_s=60)
    db.run("update rounds set deadline=? where id=?", time.time() - 1, empty["id"])
    rounds.tick(db)
    assert db.one("select status from rounds where id=?", empty["id"])["status"] == "failed"


def test_late_work_is_dropped_or_carried_to_the_next_round(pool_env):
    db, pool, member = setup(pool_env)
    publish = lambda ref, a, parent: coord.ref_publish(db, pool, ref, a, parent, member)  # noqa: E731
    first = rounds.open_round(db, pool, "w", member, quorum=1, grace_s=0, late="next", decay=0.25, repeat=True)
    rounds.contribute(db, pool, "w", "j1", "m1", art(pool_env, "a"), 1, 1)
    claimed = rounds.claim(db, pool, "w", "red")
    done = rounds.finish(db, pool, claimed["id"], "red", member, art(pool_env, "merged"), publish)
    assert done["ref"]["version"] == 2 and done["next"]["base_version"] == 2  # repeat: the next round is open
    late = rounds.contribute(db, pool, "w", "j2", "m2", art(pool_env, "slow"), 1, 1)  # still on version 1
    assert late["accepted"] and late["late"] and late["round"] == done["next"]["id"]
    rounds.contribute(db, pool, "w", "j1", "m1", art(pool_env, "b"), 1, 2)
    items = rounds.claim(db, pool, "w", "red")["items"]
    assert sorted(i["late"] for i in items) == [False, True]
    del first
    # drop: refused
    coord.ref_publish(db, pool, "d", art(pool_env, "x"), 0, member)
    r = rounds.open_round(db, pool, "d", member, quorum=1, grace_s=0, late="drop")
    rounds.contribute(db, pool, "d", "j1", "m1", art(pool_env, "y"), 1, 1)
    rounds.claim(db, pool, "d", "red")
    rounds.finish(db, pool, r["id"], "red", member, art(pool_env, "z"), lambda ref, a, parent: coord.ref_publish(db, pool, ref, a, parent, member))
    assert not rounds.contribute(db, pool, "d", "j2", "m2", art(pool_env, "q"), 1, 1)["accepted"]


def test_finish_fails_when_the_ref_moved(pool_env):
    db, pool, member = setup(pool_env)
    r = rounds.open_round(db, pool, "w", member, quorum=1, grace_s=0)
    rounds.contribute(db, pool, "w", "j1", "m1", art(pool_env, "a"), 1, 1)
    rounds.claim(db, pool, "w", "red")
    coord.ref_publish(db, pool, "w", art(pool_env, "other"), 1, member)  # someone else published v2
    with pytest.raises(coord.CoordError):
        rounds.finish(db, pool, r["id"], "red", member, art(pool_env, "m"),
                      lambda ref, a, parent: coord.ref_publish(db, pool, ref, a, parent, member))
    assert db.one("select status from rounds where id=?", r["id"])["status"] == "failed"


WORKER = r"""
import os, sys
import numpy as np
sys.path.insert(0, os.environ["POOL_LIB"])
import pool_job

rank = int(sys.argv[1])
seen = None
for _ in range(3):
    current = pool_job.current_round("weights", after=seen, wait=30)
    seen = current["id"]
    path, version = pool_job.fetch_ref("weights", current["base_version"])
    x = np.load(os.path.join(path, "x.npy"))
    out = os.path.join(os.environ["POOL_OUTPUTS"], f"r{version}")
    os.makedirs(out)
    np.save(os.path.join(out, "x.npy"), x + rank)         # "training": worker k moves the weights by k
    print("contributed", pool_job.contribute("weights", out, weight=2 * rank - 1, base_version=version), flush=True)
"""

REDUCER = r"""
import os, sys
sys.path.insert(0, os.environ["POOL_LIB"])
import pool_job, pool_reducers

for _ in range(3):
    claimed = pool_job.claim_round("weights", wait=30)
    folders = [(pool_job.fetch_artifact(i["artifact"]), i["weight"], i["late"]) for i in claimed["items"]]
    out = os.path.join(os.environ["POOL_OUTPUTS"], claimed["id"])
    pool_reducers.weighted_mean(folders, out, decay=claimed["settings"]["decay"])
    print("published", pool_job.finish_round(claimed["id"], out)["ref"]["version"], flush=True)
"""


def test_two_workers_and_a_reducer_run_three_rounds(pool_env):
    member = Client.member()
    start = pool_env["tmp"] / "w0"
    start.mkdir()
    np.save(start / "x.npy", np.zeros(4))
    op("ref.publish", name="weights", artifact=member.upload_folder(start)["artifact"], parent=0)
    op("round.open", ref="weights", quorum=2, grace_s=0, deadline_s=60, repeat=True)
    code = member.upload_folder(write(pool_env["tmp"] / "code", {"worker.py": WORKER, "reducer.py": REDUCER}))["artifact"]
    start_agent(pool_env, "box1")
    jobs = [member.call("POST", "/v1/jobs", {"spec": {"image": "x", "command": [sys.executable, *cmd], "code": code}})["job"]["id"]
            for cmd in (["worker.py", "1"], ["worker.py", "2"], ["reducer.py"])]
    for job in jobs:
        final = wait_for(lambda: (s := job_state(job))["status"] in ("succeeded", "failed") and s, timeout=120)
        logs = "".join(c["text"] for c in member.call("GET", f"/v1/jobs/{job}/logs")["chunks"])
        assert final["status"] == "succeeded", logs
    ref = op("ref.get", name="weights")["ref"]
    assert ref["version"] == 4
    folder = pool_env["tmp"] / "final"
    member.download_folder(ref["artifact"], folder)
    # each round: worker 1 adds 1 (weight 1), worker 2 adds 2 (weight 3): +1.75 per round
    assert np.allclose(np.load(folder / "x.npy"), 3 * 1.75)


def test_torchft_rounds_carry_the_lighthouse_and_the_first_opener_wins(pool_env):
    db, pool, member = setup(pool_env)
    with pytest.raises(rounds.RoundError):
        rounds.open_round(db, pool, "w", member, 2, reducer="torchft")  # no lighthouse
    with pytest.raises(rounds.RoundError):
        rounds.open_round(db, pool, "w", member, 2, reducer="torchft", lighthouse="http://a b:1; rm -rf /")
    with pytest.raises(rounds.RoundError):
        rounds.open_round(db, pool, "w", member, 2, reducer="gossip")
    first = rounds.open_round(db, pool, "w", member, 1, grace_s=0, repeat=True, reducer="torchft", lighthouse="http://10.0.0.5:29510")
    again = rounds.open_round(db, pool, "w", member, 1, reducer="torchft", lighthouse="http://10.0.0.6:29510")
    assert again["id"] == first["id"] and again["settings"]["lighthouse"] == "http://10.0.0.5:29510"
    rounds.contribute(db, pool, "w", "j1", "m1", art(pool_env, "a"), 1, 1)
    claimed = rounds.claim(db, pool, "w", "j1")
    done = rounds.finish(db, pool, claimed["id"], "j1", member, art(pool_env, "r"),
                         lambda ref, a, parent: coord.ref_publish(db, pool, ref, a, parent, member))
    assert done["next"]["settings"]["reducer"] == "torchft" and done["next"]["settings"]["lighthouse"] == "http://10.0.0.5:29510"
    assert done["next"]["settings"].get("mode", "reduce") == "reduce"


def test_gossip_with_torchft_is_refused_and_repeats_keep_mode_and_reducer(pool_env):
    """A gossip round has no reducer, so reducer torchft makes no sense there; a repeating round
    opens its successor with the same mode and reducer."""
    db, pool, member = setup(pool_env)
    with pytest.raises(rounds.RoundError, match="gossip round has no reducer"):
        rounds.open_round(db, pool, "w", member, 2, mode="gossip", reducer="torchft", lighthouse="http://10.0.0.5:29510")
    with pytest.raises(Exception, match="gossip round has no reducer"):
        op("round.open", ref="w", quorum=2, mode="gossip", reducer="torchft", lighthouse="http://10.0.0.5:29510")
    assert rounds.current(db, pool, "w") is None  # nothing was opened
    # gossip with the default reducer is fine, and its repeat stays a gossip round with its fanout
    g = rounds.open_round(db, pool, "w", member, 1, grace_s=0, repeat=True, mode="gossip", fanout=3, bias=1.0)
    assert g["settings"]["mode"] == "gossip" and "reducer" not in g["settings"]
    rounds.contribute(db, pool, "w", "j1", "m1", art(pool_env, "a"), 1, 1, g["id"])
    following = rounds.current(db, pool, "w")
    assert following["id"] != g["id"] and following["status"] == "open"
    assert following["settings"]["mode"] == "gossip" and following["settings"]["fanout"] == 3
    assert following["settings"]["bias"] == 1.0 and "lighthouse" not in following["settings"]


ISLAND = r"""
import json, os, sys
sys.path.insert(0, os.environ["POOL_LIB"])
import pool_reducers

class Manager:  # what TorchftRounds reads from a torchft.Manager
    def __init__(self, rank): self.step, self.rank = 0, rank
    def current_step(self): return self.step
    def participating_rank(self): return self.rank
    def num_participants(self): return 2

island = int(sys.argv[1])
rounds = pool_reducers.TorchftRounds("ft/model", quorum=2, deadline_s=60, grace_s=0, host="127.0.0.1",
                                     port=29600, replica_id=f"island{island}", start_lighthouse=False)
print("lighthouse", rounds.lighthouse, "hosting", rounds.hosting, os.environ["TORCHFT_LIGHTHOUSE"], flush=True)
manager = Manager(island)
for outer in range(3):
    manager.step += 1                     # torchft committed an outer step
    assert rounds.synced(manager) and not rounds.synced(manager)
    rounds.record(manager, weight=100 * (island + 1), meta={"loss": 1.0 / (outer + 1)})
rounds.close()
assert not rounds.errors, rounds.errors
"""


def test_torchft_rounds_record_each_outer_step_of_two_islands(pool_env):
    """Two islands use pool_reducers.TorchftRounds (with a stand-in for torchft's Manager): the round
    hands both the same lighthouse, and the leader closes one pool round per outer step."""
    member = Client.member()
    code = member.upload_folder(write(pool_env["tmp"] / "code", {"island.py": ISLAND}))["artifact"]
    start_agent(pool_env, "box1")
    jobs = [member.call("POST", "/v1/jobs", {"spec": {"image": "x", "command": [sys.executable, "island.py", str(i)],
                                                      "code": code}})["job"]["id"] for i in (0, 1)]
    logs = {}
    for job in jobs:
        final = wait_for(lambda: (s := job_state(job))["status"] in ("succeeded", "failed") and s, timeout=120)
        logs[job] = "".join(c["text"] for c in member.call("GET", f"/v1/jobs/{job}/logs")["chunks"])
        assert final["status"] == "succeeded", logs[job]
    lighthouses = {line.split()[1] for text in logs.values() for line in text.splitlines() if line.startswith("lighthouse")}
    assert lighthouses == {"http://127.0.0.1:29600"}
    assert sum("hosting True" in text for text in logs.values()) == 2  # same host here, so both think so
    ref = op("ref.get", name="ft/model")["ref"]
    assert ref["version"] == 4  # the start, then one version per outer step
    folder = pool_env["tmp"] / "ledger"
    member.download_folder(ref["artifact"], folder)
    record = json.loads((folder / "round.json").read_text())
    assert record["base_version"] == 3 and sorted(c["weight"] for c in record["contributions"]) == [100.0, 200.0]
    assert op("round.current", ref="ft/model")["round"]["settings"]["reducer"] == "torchft"
