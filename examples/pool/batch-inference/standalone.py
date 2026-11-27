"""The baseline: the same engine and prompts, run directly (not through the pool).

    docker run --rm --gpus all --ipc=host -v $MODEL:/model -v $PWD:/work -w /work nvcr.io/nvidia/vllm:26.04-py3 \
        python standalone.py /model --tasks 32 --batch 64
"""

import argparse
import json
import time

import common

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("--tasks", type=int, default=32, help="the same tasks the pool run gets (make_tasks.py)")
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--target", type=int, default=256, help="requests kept in flight")
args = parser.parse_args()

loaded = time.time()
engine = common.engine(args.model)
start = time.time()
todo = [(f"{seed}:{i}", text) for seed in range(args.tasks) for i, text in enumerate(common.prompts(args.batch, seed))]
feeder = common.Feeder(engine, args.target)


def more():
    room = args.target - engine.get_num_unfinished_requests()
    batch, todo[:] = todo[:room], todo[room:]
    return batch


feeder.run(more, lambda request_id, output: None, lambda: not todo)
seconds = time.time() - start
print(json.dumps({"mode": "standalone", "model": args.model, "prompts": args.tasks * args.batch, "tokens": feeder.generated,
                  "seconds": round(seconds, 2), "tokens_per_s": round(feeder.generated / seconds, 1),
                  "load_seconds": round(start - loaded, 1)}))
