"""An actor replica: keeps vLLM loaded and turns queue tasks (batches of prompts) into token counts.

It leases as many tasks as fit in the engine's TARGET requests in flight, and leases again as soon
as a task's worth of room opens, so the engine stays as full as in the standalone run. Each result
reports when its batch started and finished, for the driver's
throughput figure. It exits after IDLE seconds without work (an actor with min 0 then scales to 0).
"""

import os
import socket
import sys
import time

sys.path.insert(0, os.environ["POOL_LIB"])
import pool_job  # noqa: E402

import common  # noqa: E402

QUEUE = os.environ.get("QUEUE", "prompts")
TARGET = int(os.environ.get("TARGET", "256"))
IDLE = float(os.environ.get("IDLE", "60"))

engine = common.engine(os.path.join(os.environ.get("POOL_INPUTS", "/inputs"), "base"))
feeder = common.Feeder(engine, TARGET)
open_tasks: dict[str, dict] = {}  # task id -> {left, tokens, started}
last_work = time.time()
host = socket.gethostname()


BATCH = int(os.environ.get("BATCH", "64"))  # prompts per task, to size the lease


def more():
    global last_work
    room = TARGET - engine.get_num_unfinished_requests()
    if room < BATCH and open_tasks:
        return None  # full: lease again once a task's worth of room opens
    tasks = pool_job.lease(QUEUE, n=max(1, room // BATCH), wait=0 if open_tasks else 5)
    batch = []
    for task in tasks:
        prompts = common.prompts(task["inputs"]["count"], task["inputs"]["seed"])
        open_tasks[task["id"]] = {"left": len(prompts), "tokens": 0, "started": time.time()}
        batch += [(f"{task['id']}:{i}", text) for i, text in enumerate(prompts)]
    if batch:
        last_work = time.time()
    return batch


def done(request_id, output):
    task_id = request_id.rsplit(":", 1)[0]
    state = open_tasks[task_id]
    state["left"] -= 1
    state["tokens"] += len(output.outputs[0].token_ids)
    if state["left"] == 0:
        del open_tasks[task_id]
        pool_job.result(QUEUE, task_id, {"tokens": state["tokens"], "started": state["started"],
                                         "finished": time.time(), "machine": host})


feeder.run(more, done, lambda: not open_tasks and time.time() - last_work > IDLE)
print(f"idle for {IDLE:.0f} s, stopping; generated {feeder.generated} tokens")
