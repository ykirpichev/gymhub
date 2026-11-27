"""Idle share and reward of one run's metrics.jsonl (trainer.py)."""
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
steady = rows[1:]  # the first step waits for every roller to load the model
wait = sum(r["wait_s"] for r in steady)
train = sum(r["train_s"] for r in steady)
third = max(1, len(rows) // 3)
print(json.dumps({
    "steps": len(rows), "idle_share": round(wait / (wait + train), 3),
    "reward_first_third": round(sum(r["reward"] for r in rows[:third]) / third, 3),
    "reward_last_third": round(sum(r["reward"] for r in rows[-third:]) / third, 3),
    "lag_mean": round(sum(r["lag_mean"] for r in rows) / len(rows), 2),
    "longest_step_s": max(round(r["wait_s"] + r["train_s"], 1) for r in steady),
    "minutes": round((rows[-1]["at"] - rows[0]["at"]) / 60, 1),
    "rejected": rows[-1].get("rejected", 0),
}))
