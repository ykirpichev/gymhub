"""Several small jobs on one GPU (resources.share_gpu): rollers next to a trainer on a single GPU."""

import pytest

from pool import spec as specs
from pool.agent import Agent
from pool.client import Client
from pool.agent import register
from tests.test_pool import db_url, pool_env  # noqa: F401  (fixtures)

GPU = {"index": 0, "name": "NVIDIA GB10", "memory_gb": 100.0}


def machine(env):
    member = Client.member()
    code = member.call("POST", "/v1/machine-codes", {})["code"]
    config = register(env["url"], code, "gpu-box", {}, [0], ["*"])
    client = Client(env["url"], config["token"])
    client.call("POST", "/v1/agent/heartbeat", {"running": [], "profile": {"gpus": [GPU]}, "free": None})
    return member, client


def job(member, gb, share=True):
    resources = {"gpus": 1, "gpu_memory_gb": gb, "memory_gb": 1, "cpus": 1, **({"share_gpu": True} if share else {})}
    return member.call("POST", "/v1/jobs", {"spec": {"image": "x", "command": ["true"], "resources": resources}})["job"]["id"]


def lease(client, gpus, shared):
    free = {"gpus": gpus, "shared_gpus": shared, "cpus": 16, "memory_gb": 64}
    return client.call("POST", "/v1/agent/lease", {"free": free, "wait": 0})["jobs"]


def test_shared_jobs_pack_onto_one_gpu_by_memory(pool_env):
    member, client = machine(pool_env)
    first, second, third = job(member, 40), job(member, 40), job(member, 40)
    assert lease(client, [0], {})[0]["id"] == first          # a free GPU
    assert lease(client, [], {"0": 60})[0]["gpus"] == [0]    # 60 GB left: the second fits next to it
    assert lease(client, [], {"0": 20}) == []                # the third does not
    assert pool_env["db"].one("select status from jobs where id=?", third)["status"] == "queued"
    assert pool_env["db"].one("select gpus from jobs where id=?", second)["gpus"] == "[0]"


def test_an_exclusive_job_never_lands_on_a_shared_gpu(pool_env):
    member, client = machine(pool_env)
    job(member, 10, share=False)
    assert lease(client, [], {"0": 90}) == []  # the GPU has shared jobs on it: not free for a whole-GPU job


def test_the_agent_reports_what_is_left_on_shared_gpus(tmp_path):
    agent = Agent.__new__(Agent)
    agent.lock = __import__("threading").RLock()
    agent.profile = {"gpus": [GPU, {"index": 1, "memory_gb": 100.0}]}
    agent.cpus, agent.memory_gb = 20, 100

    class Run:
        def __init__(self, gpus, gb, share):
            self.job = {"gpus": gpus, "spec": {"resources": {"cpus": 1, "memory_gb": 2, "gpu_memory_gb": gb, "share_gpu": share, "gpus": 1}}}

    agent.running = {"a": Run([0], 30, True), "b": Run([0], 25, True), "c": Run([1], 0, False)}
    free = agent.free()
    assert free["gpus"] == [] and free["shared_gpus"] == {"0": 45.0}


def test_share_gpu_needs_one_gpu_and_its_memory():
    with pytest.raises(specs.SpecError):
        specs.normalize({"image": "x", "command": ["true"], "resources": {"gpus": 1, "share_gpu": True}})
    with pytest.raises(specs.SpecError):
        specs.normalize({"image": "x", "command": ["true"], "resources": {"gpus": 2, "gpu_memory_gb": 4, "share_gpu": True}})
    assert specs.normalize({"image": "x", "command": ["true"],
                            "resources": {"gpus": 1, "gpu_memory_gb": 4, "share_gpu": True}})["resources"]["share_gpu"]
