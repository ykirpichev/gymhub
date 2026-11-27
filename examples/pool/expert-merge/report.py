"""The per-domain held-out loss of every model of a Demo C2 run: python3 report.py out/*/*.json"""
import json
import sys

order = {"anchor": 0, "expert": 1, "merge": 2, "dense": 3}
rows = sorted((json.load(open(path)) for path in sys.argv[1:]), key=lambda r: (order[r["mode"]], r.get("domain", "")))
print(f"{'model':<14} {'params':>10} {'code':>8} {'legal':>8} {'mean':>8}")
for r in rows:
    name = r["mode"] + ("-" + r["domain"] if r.get("domain") else "")
    print(f"{name:<14} {r['params']:>10,} {r['eval']['code']:>8.4f} {r['eval']['legal']:>8.4f} {r['eval']['mean']:>8.4f}")
    if r.get("before_router"):
        b = r["before_router"]
        print(f"{'  (even mix)':<14} {'':>10} {b['code']:>8.4f} {b['legal']:>8.4f} {b['mean']:>8.4f}")
json.dump(rows, open("results/demo-c2.json", "w"), indent=2)
