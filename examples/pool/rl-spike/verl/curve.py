"""Pull per-step metrics out of verl's console log into metrics.jsonl and a small text chart."""
import json
import re
import sys
from pathlib import Path

log, out = Path(sys.argv[1]).read_text(), Path(sys.argv[2])
steps = []
for line in log.splitlines():
    if "step:" not in line or "critic/score/mean" not in line:
        continue
    metrics = dict(re.findall(r"([\w/@\-.]+):(?:np\.float\d+\()?(-?[\d.eE+-]+|nan)", line))
    steps.append({k: float(v) for k, v in metrics.items() if v != "nan"})
val = [float(v) for v in re.findall(r"val-aux/arith/reward/mean@1:(?:np\.float64\()?(-?[\d.]+)", log)]
(out / "metrics.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps))
lines = [f"step {int(s['step']):>3}  reward {s['critic/score/mean']:.3f}  " + "#" * int(40 * s["critic/score/mean"]) for s in steps]
summary = "\n".join(lines) + f"\nvalidation reward (before, then every 10 steps): {val}\n"
(out / "reward_curve.txt").write_text(summary)
print(summary)
