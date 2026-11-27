"""M3 coordination (pool/coord.py, pool/jobsocket.py, pool/job.py): refs, queues, and a job that uses them."""

import json
import sys
import time
from pathlib import Path

import pytest

from pool import coord
from pool.client import Client, PoolError
from tests.test_pool import db_url, job_state, pool_env, start_agent, wait_for, write  # noqa: F401  (fixtures)

REPO = Path(__file__).resolve().parents[1]


def op(operation, **args):
    return Client.member().call("POST", "/v1/coord", {"op": operation, "args": args})


def upload(env, files) -> str:
    folder = write(env["tmp"] / f"up-{time.time_ns()}", files)
    return Client.member().upload_folder(folder)["artifact"]


def test_refs_are_compare_and_swap(pool_env):
    first = upload(pool_env, {"w.txt": "v1"})
    assert op("ref.publish", name="grpo/policy", artifact=first, parent=0)["ref"]["version"] == 1
    with pytest.raises(PoolError) as exc:  # someone else published first: parent 0 is not current
        op("ref.publish", name="grpo/policy", artifact=first, parent=0)
    assert exc.value.status == 409 and exc.value.detail["current"] == 1
    second = upload(pool_env, {"w.txt": "v2"})
    assert op("ref.publish", name="grpo/policy", artifact=second, parent=1)["ref"]["version"] == 2
    assert op("ref.publish", name="grpo/policy", artifact=first, parent=0, force=True)["ref"]["version"] == 3
    current = op("ref.get", name="grpo/policy")["ref"]
    assert current["version"] == 3 and current["artifact"] == first
    assert [v["version"] for v in op("ref.get", name="grpo/policy", history=True)["versions"]] == [3, 2, 1]
    assert op("ref.get", name="grpo/policy", version=2)["ref"]["artifact"] == second
    with pytest.raises(PoolError) as exc:
        op("ref.publish", name="x", artifact="sha256:" + "0" * 64, parent=0)
    assert exc.value.status == 404  # an artifact the member cannot read
    with pytest.raises(PoolError):
        op("ref.publish", name="bad//name", artifact=first, parent=0)


def test_waiting_for_a_newer_version(pool_env):
    artifact = upload(pool_env, {"a": "1"})
    op("ref.publish", name="r", artifact=artifact, parent=0)
    started = time.monotonic()
    reply = op("ref.get", name="r", after=1, wait=1)
    assert not reply["newer"] and time.monotonic() - started >= 0.9
    op("ref.publish", name="r", artifact=artifact, parent=1)
    assert op("ref.get", name="r", after=1, wait=5)["ref"]["version"] == 2


def test_tasks_are_idempotent_by_key_and_leased_only_by_jobs(pool_env):
    op("queue.create", name="work")
    ids = op("tasks.add", name="work", tasks=[{"inputs": {"n": 1}, "key": "a"}, {"inputs": {"n": 2}, "key": "b"}])["ids"]
    again = op("tasks.add", name="work", tasks=[{"inputs": {"n": 1}, "key": "a"}])["ids"]
    assert again == ids[:1]
    assert op("queue.stats", name="work")["queue"]["counts"]["queued"] == 2
    with pytest.raises(PoolError) as exc:
        op("tasks.lease", name="work")
    assert exc.value.status == 400
    with pytest.raises(PoolError) as exc:
        op("tasks.add", name="work", tasks=[{"inputs": "x" * 70_000}])
    assert exc.value.status == 413


def test_duplicates_first_result_wins_and_lapsed_leases_come_back(pool_env):
    db, pool = pool_env["db"], db_pool(pool_env)
    coord.queue_create(db, pool, "q", "m1", duplicate_after=1)
    (task,) = coord.tasks_add(db, pool, "q", [{"inputs": 1}])
    assert [t["id"] for t in coord.tasks_lease(db, pool, "q", "jobA", "machineA")] == [task]
    assert coord.tasks_lease(db, pool, "q", "jobB", "machineB") == []  # not slow yet
    db.run("update tasks set started_at=? where id=?", time.time() - 5, task)
    copy = coord.tasks_lease(db, pool, "q", "jobB", "machineB")
    assert copy and copy[0]["duplicate"]
    assert coord.task_result(db, pool, "q", task, "jobB", "machineB", {"r": "b"})["accepted"]
    assert not coord.task_result(db, pool, "q", task, "jobA", "machineA", {"r": "a"})["accepted"]
    assert [r["result"] for r in coord.results(db, pool, "q")["results"]] == [{"r": "b"}]
    # a lease that is not renewed lapses, and the task is queued again
    (second,) = coord.tasks_add(db, pool, "q", [{"inputs": 2}])
    coord.tasks_lease(db, pool, "q", "jobA", "machineA")
    db.run("update task_leases set expires=? where task=?", time.time() - 1, second)
    assert coord.expire_task_leases(db) == 1
    assert coord.queue_stats(db, pool, "q")["counts"]["queued"] == 1


def test_stale_results_are_requeued_beyond_max_lag(pool_env):
    db, pool = pool_env["db"], db_pool(pool_env)
    artifact = upload(pool_env, {"p": "x"})
    for version in range(3):  # the policy is at version 3
        coord.ref_publish(db, pool, "policy", artifact, version, "m1")
    coord.queue_create(db, pool, "rollouts", "m1", ref="policy", max_lag=1)
    (task,) = coord.tasks_add(db, pool, "rollouts", [{"inputs": "p"}])
    coord.tasks_lease(db, pool, "rollouts", "j", "m")
    stale = coord.task_result(db, pool, "rollouts", task, "j", "m", {"reward": 1}, computed_with=1)
    assert not stale["accepted"] and stale["current"] == 3
    stats = coord.queue_stats(db, pool, "rollouts")
    assert stats["counts"]["queued"] == 1 and stats["stale_results"] == 1
    coord.tasks_lease(db, pool, "rollouts", "j", "m")
    assert coord.task_result(db, pool, "rollouts", task, "j", "m", {"reward": 1}, computed_with=2)["accepted"]


def db_pool(env) -> str:
    return env["db"].one("select id from pools where name='default'")["id"]


WORKER = r"""
import json, os, sys
sys.path.insert(0, os.environ["POOL_LIB"])
import pool_job

path, version = pool_job.fetch_ref("policy")
scale = int(open(os.path.join(path, "scale.txt")).read())
done = 0
while True:
    tasks = pool_job.lease("numbers", n=2, wait=2)
    if not tasks:
        break
    for task in tasks:
        reply = pool_job.result("numbers", task["id"], {"y": task["inputs"]["x"] * scale}, computed_with=version)
        done += reply["accepted"]
out = os.path.join(os.environ["POOL_OUTPUTS"], "next")
os.makedirs(out)
open(os.path.join(out, "scale.txt"), "w").write(str(scale + 1))
new = pool_job.publish_ref("policy", "next", parent=version)
try:
    pool_job.publish_ref("policy", "next", parent=version)
except pool_job.PoolJobError as exc:
    print("second publish refused", exc.status)
print("done", done, "published", new["version"])
"""


def test_a_job_uses_refs_and_queues_through_the_agent_socket(pool_env):
    member = Client.member()
    policy = upload(pool_env, {"scale.txt": "3"})
    op("ref.publish", name="policy", artifact=policy, parent=0)
    op("queue.create", name="numbers", ref="policy", max_lag=0)
    op("tasks.add", name="numbers", tasks=[{"inputs": {"x": x}, "key": f"x{x}"} for x in range(5)])
    code = member.upload_folder(write(pool_env["tmp"] / "code", {"worker.py": WORKER}))["artifact"]
    start_agent(pool_env, "box1")
    job = member.call("POST", "/v1/jobs", {"spec": {"image": "x", "command": [sys.executable, "worker.py"], "code": code}})["job"]
    final = wait_for(lambda: (s := job_state(job["id"]))["status"] in ("succeeded", "failed") and s, timeout=60)
    logs = "".join(chunk["text"] for chunk in member.call("GET", f"/v1/jobs/{job['id']}/logs")["chunks"])
    assert final["status"] == "succeeded", logs
    assert "done 5 published 2" in logs and "second publish refused 409" in logs
    got = sorted(r["result"]["y"] for r in op("results", name="numbers")["results"])
    assert got == [0, 3, 6, 9, 12]
    assert all(r["computed_with"] == 1 for r in op("results", name="numbers")["results"])
    newest = op("ref.get", name="policy")["ref"]
    assert newest["version"] == 2
    folder = pool_env["tmp"] / "check"
    member.download_folder(newest["artifact"], folder)  # a job's published ref is readable by the pool
    assert (folder / "scale.txt").read_text() == "4"


def test_the_cli(pool_env, capsys, monkeypatch):
    from pool import cli

    folder = write(pool_env["tmp"] / "pol", {"w.txt": "1"})
    assert cli.main(["ref", "publish", "policy", str(folder)]) == 0
    write(folder, {"w.txt": "2"})
    assert cli.main(["ref", "publish", "policy", str(folder)]) == 0  # on top of the current version
    with pytest.raises(SystemExit, match="is at version 2"):
        cli.main(["ref", "publish", "policy", str(folder), "--parent", "1"])
    assert cli.main(["ref", "publish", "policy", str(folder), "--parent", "1", "--force"]) == 0
    assert cli.main(["ref", "log", "policy"]) == 0
    assert cli.main(["ref", "fetch", "policy", str(pool_env["tmp"] / "got"), "--version", "1"]) == 0
    assert (pool_env["tmp"] / "got" / "w.txt").read_text() == "1"
    assert cli.main(["queue", "create", "jobs", "--ref", "policy", "--max-lag", "1"]) == 0
    lines = pool_env["tmp"] / "tasks.jsonl"
    lines.write_text('{"inputs": {"x": 1}, "key": "one"}\n{"x": 2}\n')
    assert cli.main(["queue", "add", "jobs", str(lines)]) == 0
    assert cli.main(["queue", "ls"]) == 0
    assert cli.main(["queue", "results", "jobs"]) == 0
    out = capsys.readouterr().out
    assert "policy v3" in out and "v3     parent v2" in out and "2 tasks in jobs" in out and "versions behind policy" in out
    assert op("queue.stats", name="jobs")["queue"]["counts"]["queued"] == 2
