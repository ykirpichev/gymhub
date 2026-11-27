"""Summarise metrics.json files from train.py: python3 report.py results/*.json"""

import json
import sys

rows = [json.load(open(path)) for path in sys.argv[1:]]
single = next((r for r in rows if r["mode"] == "single"), None)
print(f"{'run':<26} {'tokens':>10} {'val loss':>9} {'vs single':>9} {'util':>6} {'step s':>7} {'outer syncs, s (median)':>24} {'train s':>8}")
for r in rows:
    if r["mode"] == "bench":
        print(f"bench {r['model']}: {r['params'] / 1e9:.2f}B params, {r['step_s']} s/step at {r['batch']}x{r['seq']}, "
              f"{r['tokens_per_s']} tok/s, {r['peak_gb']} GB, {r['sync_bytes_bf16'] / 1e9:.1f} GB per bf16 sync")
        continue
    name = "single island" if r["mode"] == "single" else f"island {r['island']} ({r['link'] or 'none'})"
    gap = f"{100 * (r['final_val_loss'] / single['final_val_loss'] - 1):+.1f}%" if single and r is not single else ""
    syncs = sorted(r.get("sync_s") or [])
    sync = f"{syncs[len(syncs) // 2]:.1f} ({len(syncs)}x)" if syncs else "-"
    util = f"{100 * r['utilisation']:.0f}%" if r.get("utilisation") is not None else "-"
    print(f"{name:<26} {r['tokens_total']:>10,} {r['final_val_loss']:>9.4f} {gap:>9} {util:>6} {r['step_s']:>7.3f} {sync:>24} {r['train_s']:>8.0f}")
