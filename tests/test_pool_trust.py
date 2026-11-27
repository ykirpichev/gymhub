"""M6, the checked tier (pool/trust.py): duplicate checks, reports, exclusion, robust reduction."""

import os
import sys

import numpy as np

from pool import coord, reducers, trust
from pool.client import Client
from tests.test_pool import db_url, job_state, pool_env, start_agent, wait_for, write  # noqa: F401  (fixtures)
from tests.test_pool_coord import op


def ids(env):
    db = env["db"]
    return db, db.one("select id from pools where name='default'")["id"]


def check_pool(env, exclude_after=2):
    Client.member().call("PATCH", "/v1/pools/default", {"settings": {"trust": "checked", "exclude_after": exclude_after}})


def test_reports_exclude_only_in_a_checked_pool(pool_env):
    db, pool = ids(pool_env)
    for _ in range(3):
        trust.report(db, pool, "m1", False, "reward")
    assert not trust.excluded(db, pool, "m1")  # trusted pool: recorded, nobody excluded
    check_pool(pool_env)
    trust.report(db, pool, "m2", True, "reward")
    trust.report(db, pool, "m2", False, "logprobs", "far off")
    assert not trust.excluded(db, pool, "m2")
    standing = trust.report(db, pool, "m2", False, "logprobs", "far off again")
    assert standing["excluded"] and standing["score"] == round(1 - 2 / 3, 3) and standing["recent"][0]["detail"] == "far off again"
    trust.clear(db, pool, "m2")
    assert not trust.excluded(db, pool, "m2")


def test_replicated_tasks_need_two_machines_to_agree(pool_env):
    db, pool = ids(pool_env)
    check_pool(pool_env)
    coord.queue_create(db, pool, "q", "owner", replicate=1.0)
    (task,) = coord.tasks_add(db, pool, "q", [{"inputs": 3}])
    coord.tasks_lease(db, pool, "q", "jA", "honest1")
    assert coord.task_result(db, pool, "q", task, "jA", "honest1", 9)["pending"]
    assert coord.tasks_lease(db, pool, "q", "jA2", "honest1") == []  # never the same machine twice
    coord.tasks_lease(db, pool, "q", "jE", "evil")
    coord.task_result(db, pool, "q", task, "jE", "evil", 10)          # disagrees: a third machine decides
    assert coord.results(db, pool, "q")["results"] == []
    coord.tasks_lease(db, pool, "q", "jB", "honest2")
    assert coord.task_result(db, pool, "q", task, "jB", "honest2", 9)["accepted"]
    assert [r["result"] for r in coord.results(db, pool, "q")["results"]] == [9]
    assert trust.standing(db, pool, "evil")["failures"] == 1 and trust.standing(db, pool, "honest1")["failures"] == 0
    # one more lie and the machine is out: no tasks, no results
    (other,) = coord.tasks_add(db, pool, "q", [{"inputs": 4}])
    coord.tasks_lease(db, pool, "q", "jA", "honest1")
    coord.task_result(db, pool, "q", other, "jA", "honest1", 16)
    coord.tasks_lease(db, pool, "q", "jE", "evil")
    coord.task_result(db, pool, "q", other, "jE", "evil", 17)
    coord.tasks_lease(db, pool, "q", "jB", "honest2")
    coord.task_result(db, pool, "q", other, "jB", "honest2", 16)
    assert trust.excluded(db, pool, "evil")
    coord.tasks_add(db, pool, "q", [{"inputs": 5}])
    assert coord.tasks_lease(db, pool, "q", "jE", "evil") == []


def test_robust_mean_drops_a_poisoned_update(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    np.save(base / "w.npy", np.zeros(8))
    folders = []
    for index, delta in enumerate([np.full(8, 1.0), np.full(8, 1.1), np.full(8, 0.9), np.full(8, -50.0)]):
        folder = tmp_path / f"c{index}"
        folder.mkdir()
        np.save(folder / "w.npy", delta)
        folders.append((str(folder), 1.0, False))
    out, dropped = reducers.robust_mean(folders, str(tmp_path / "out"), base=str(base))
    assert dropped == [3]
    assert np.allclose(np.load(os.path.join(out, "w.npy")), 1.0)


WORKER = r"""
import os, sys
sys.path.insert(0, os.environ["POOL_LIB"])
import pool_job
import pool_verify  # shipped next to pool_job (the M6 verifiers); imports only the standard library

assert pool_verify.judge([(0, 1.0, 1.0)])["ok"]
lie =os.environ.get("LIE") == "1"
while True:
    tasks = pool_job.lease("squares", n=1, wait=3)
    if not tasks:
        break
    for task in tasks:
        x = task["inputs"]
        pool_job.result("squares", task["id"], x * x + (1 if lie else 0))
"""


def test_a_lying_machine_is_caught_and_excluded(pool_env):
    check_pool(pool_env)
    member = Client.member()
    for name in ("honest1", "honest2", "liar"):
        start_agent(pool_env, name, labels={"role": name})
    op("queue.create", name="squares", replicate=1.0)
    op("tasks.add", name="squares", tasks=[{"inputs": n} for n in range(6)])
    code = member.upload_folder(write(pool_env["tmp"] / "code", {"worker.py": WORKER}))["artifact"]

    def submit(name):
        spec = {"image": "x", "command": [sys.executable, "worker.py"], "code": code, "resources": {"labels": {"role": name}},
                "env": {"LIE": "1" if name == "liar" else "0"}}
        return member.call("POST", "/v1/jobs", {"spec": spec})["job"]["id"]

    # A machine is only checked on the tasks it takes. If the honest pair started first they could decide
    # every task before the liar leased one (it would vote once or not at all, and never reach
    # exclude_after), so the liar goes first: it votes on all six (each stays pending) and its worker exits
    # once nothing is left that it has not voted on. Then the honest machines disagree with it and decide.
    liar = submit("liar")
    wait_for(lambda: job_state(liar)["status"] in ("succeeded", "failed"), timeout=60)
    assert op("results", name="squares")["results"] == []  # one machine's word decides nothing
    jobs = [liar] + [submit(name) for name in ("honest1", "honest2")]
    wait_for(lambda: op("queue.stats", name="squares")["queue"]["counts"]["done"] == 6, timeout=90)
    assert sorted(r["result"] for r in op("results", name="squares")["results"]) == [0, 1, 4, 9, 16, 25]  # no lie got through
    machines = {m["name"]: m for m in member.call("GET", "/v1/machines")["machines"]}
    assert machines["liar"]["reputation"]["excluded"] and not machines["honest1"]["reputation"]["excluded"]
    for job in jobs:
        wait_for(lambda: job_state(job)["status"] in ("succeeded", "failed"), timeout=60)
