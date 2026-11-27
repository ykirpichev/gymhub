"""M3 actors, checkpoints and ref subscriptions (pool/actors.py), on a real agent with the process runtime."""

import sys
import time

from pool import actors
from pool.client import Client
from tests.test_pool import db_url, job_state, pool_env, start_agent, wait_for, write  # noqa: F401  (fixtures)
from tests.test_pool_coord import op, upload


def code(env, files) -> str:
    return Client.member().upload_folder(write(env["tmp"] / f"code-{time.time_ns()}", files))["artifact"]


def spec(artifact: str, script: str = "main.py", **extra) -> dict:
    return {"image": "x", "command": [sys.executable, script], "code": artifact, "timeout_minutes": 10, **extra}


SLEEPER = "import time\ntime.sleep(300)\n"


def test_an_actor_keeps_its_replicas_and_replaces_a_lost_one(pool_env):
    start_agent(pool_env, "box1")
    artifact = code(pool_env, {"main.py": SLEEPER})
    actor = op("actor.create", name="servers", spec=spec(artifact), min=2)["actor"]
    assert actor["desired"] == 2 and len(actor["replicas"]) == 2
    status = wait_for(lambda: (a := op("actor.status", name="servers")["actor"])["running"] == 2 and a, timeout=30)
    lost = status["replicas"][0]["job"]
    Client.member().call("POST", f"/v1/jobs/{lost}/cancel")
    replaced = wait_for(lambda: (a := op("actor.status", name="servers")["actor"])["running"] == 2
                        and lost not in {r["job"] for r in a["replicas"]} and a, timeout=30)
    assert {r["replica"] for r in replaced["replicas"]} == {0, 1}  # the free replica number is reused
    assert op("actor.scale", name="servers", min=1, max=1)["actor"]["desired"] == 1
    stopped = op("actor.stop", name="servers")["actor"]
    assert stopped["status"] == "stopped"
    wait_for(lambda: op("actor.status", name="servers")["actor"]["running"] == 0, timeout=30)
    assert op("actor.list")["actors"][0]["name"] == "servers"


WORKER = r"""
import os, sys
sys.path.insert(0, os.environ["POOL_LIB"])
import pool_job

while True:
    tasks = pool_job.lease("squares", n=1, wait=1)
    if not tasks:
        break
    for task in tasks:
        pool_job.result("squares", task["id"], task["inputs"] ** 2)
"""


def test_an_actor_scales_with_its_queue_and_drains_it(pool_env):
    start_agent(pool_env, "box1")
    op("queue.create", name="squares")
    op("tasks.add", name="squares", tasks=[{"inputs": n} for n in range(6)])
    artifact = code(pool_env, {"main.py": WORKER})
    actor = op("actor.create", name="workers", spec=spec(artifact), min=0, max=3, queue="squares", per_replica=2)["actor"]
    assert actor["desired"] == 3
    wait_for(lambda: op("queue.stats", name="squares")["queue"]["counts"]["done"] == 6, timeout=60)
    assert sorted(r["result"] for r in op("results", name="squares")["results"]) == [0, 1, 4, 9, 16, 25]
    final = wait_for(lambda: (a := op("actor.status", name="workers")["actor"])["running"] + a["queued"] == 0 and a, timeout=30)
    assert final["desired"] == 0  # an empty queue needs no replicas (min 0)


def test_a_crash_looping_actor_stops_relaunching(pool_env, monkeypatch):
    monkeypatch.setattr(actors, "CRASH_LIMIT", 2)  # each crash is a real launch: fewer of them, same path
    start_agent(pool_env, "box1")
    artifact = code(pool_env, {"main.py": "raise SystemExit(3)\n"})
    op("actor.create", name="broken", spec=spec(artifact), min=1)
    failing = wait_for(lambda: (a := op("actor.status", name="broken")["actor"])["status"] == "failing" and a, timeout=60)
    assert "failed within" in failing["note"]
    jobs = pool_env["db"].one("select count(*) as n from actor_jobs where actor='broken'")["n"]
    time.sleep(2)
    assert pool_env["db"].one("select count(*) as n from actor_jobs where actor='broken'")["n"] == jobs  # no more launches


RESUMABLE = r"""
import os, sys, time
sys.path.insert(0, os.environ["POOL_LIB"])
import pool_job

state = os.path.join(os.environ["POOL_CHECKPOINT"], "step.txt")
if pool_job.resumed():
    print("resumed at step", open(state).read())
    raise SystemExit(0)
open(state, "w").write("7")
pool_job.save_checkpoint()
print("checkpoint saved", flush=True)
time.sleep(300)
"""


def test_a_requeued_job_resumes_from_its_checkpoint(pool_env):
    start_agent(pool_env, "box1")
    member = Client.member()
    job = member.call("POST", "/v1/jobs", {"spec": spec(code(pool_env, {"main.py": RESUMABLE}), retries=2)})["job"]
    db = pool_env["db"]
    wait_for(lambda: db.one("select 1 as x from job_checkpoints where job=?", job["id"]), timeout=120)
    # its machine stops reporting, and the lease runs out: expire_leases puts the job back in the queue.
    # (Done directly: this agent keeps renewing the lease.) The agent then drops its copy and leases it again.
    db.run("update jobs set status='queued', machine=null, gpus=null, lease_expires=null where id=?", job["id"])
    logs = lambda: "".join(c["text"] for c in member.call("GET", f"/v1/jobs/{job['id']}/logs")["chunks"])  # noqa: E731
    try:
        final = wait_for(lambda: (s := job_state(job["id"]))["status"] in ("succeeded", "failed") and s, timeout=120)
    except AssertionError:
        raise AssertionError(f"job did not finish: {job_state(job['id'])}\n{logs()}") from None
    logs = logs()
    assert final["status"] == "succeeded", logs
    assert "resuming from the checkpoint" in logs and "resumed at step 7" in logs


def test_a_subscribed_ref_is_fetched_before_the_job_asks(pool_env):
    agent = start_agent(pool_env, "box1")
    first = upload(pool_env, {"w.bin": "1" * 100})
    op("ref.publish", name="policy", artifact=first, parent=0)
    job = Client.member().call("POST", "/v1/jobs", {"spec": spec(code(pool_env, {"main.py": SLEEPER}), subscribe=["policy"])})["job"]
    wait_for(lambda: job_state(job["id"])["status"] == "running", timeout=30)
    second = upload(pool_env, {"w.bin": "2" * 100})
    op("ref.publish", name="policy", artifact=second, parent=1)
    key = second.removeprefix("sha256:")
    wait_for(lambda: (agent.cache.trees / key).is_dir(), timeout=30)
    Client.member().call("POST", f"/v1/jobs/{job['id']}/cancel")


def test_the_actor_cli(pool_env, capsys):
    from pool import cli

    folder = write(pool_env["tmp"] / "svc", {"main.py": SLEEPER})
    spec_file = folder / "actor.json"
    spec_file.write_text('{"image": "x", "command": ["python", "main.py"], "code": "."}')
    assert cli.main(["actor", "create", "svc", str(spec_file), "--min", "0", "--max", "2"]) == 0
    assert cli.main(["actor", "scale", "svc", "--max", "3"]) == 0
    assert cli.main(["actor", "ls"]) == 0
    assert cli.main(["actor", "stop", "svc"]) == 0
    out = capsys.readouterr().out
    assert "actor svc: 0 replicas wanted" in out and "0-3" in out and "stopped" in out
