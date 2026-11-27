"""Tasks for the queue, one JSON line each: a batch of `--batch` prompts made from a seed (common.prompts)."""
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--tasks", type=int, default=32)
parser.add_argument("--batch", type=int, default=64)
args = parser.parse_args()
for seed in range(args.tasks):
    print(json.dumps({"inputs": {"count": args.batch, "seed": seed}, "key": f"batch-{seed}"}))
