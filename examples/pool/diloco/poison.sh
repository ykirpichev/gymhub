#!/usr/bin/env bash
# Poisoned DiLoCo updates and the checked tier's defence (M6, issue #90).
#
# A gang of WORLD ranks on WORLD machines of a checked pool (pool pool set default trust=checked). The rank
# on the machine EVIL sends a poisoned outer update every round. Each attack runs twice: without the
# defence (the updates are all-reduced as usual) and with --robust (update sanity: outliers left out,
# every machine reported, excluded machines ignored). Rounds are a fixed number of steps, so runs compare
# at equal work. Results (final val loss) go to poison-results/summary.json.
#   ./poison.sh                          # clean, scale, flip, random; each without and with the defence
#   ATTACKS="scale" ROUNDS=6 ./poison.sh
set -euo pipefail
WORLD="${WORLD:-4}"; EVIL="${EVIL:-evil}"; ROUNDS="${ROUNDS:-8}"; STEPS="${STEPS:-150}"
ATTACKS="${ATTACKS:-none scale flip random}"
IMAGE="${IMAGE:-nvcr.io/nvidia/vllm:26.04-py3}"
OUT="${OUT:-poison-results}"
mkdir -p "$OUT"
[ -f corpus/corpus.txt ] || python3 make_corpus.py corpus/
for attack in $ATTACKS; do
  for defence in off on; do
    name="$attack-$defence"
    flags=(--rounds "$ROUNDS" --round-steps "$STEPS" --poison "$attack" --poison-machine "$EVIL")
    [ "$defence" = on ] && flags+=(--robust)
    echo "== $name"
    pool run --gang "$WORLD" --network peers --gpus 1 --mem 6 --image "$IMAGE" --data corpus=./corpus \
        --name "diloco-$name" --to "$OUT/$name" -- python3 diloco.py "${flags[@]}" | grep -E "^(round [0-9]+:.*val|final)" || true
    pool machine trust "$EVIL" > /dev/null 2>&1 || true  # clear the exclusion before the next run
  done
done
python3 - "$OUT" <<'PY'
import glob, json, os, sys
rows = {}
for path in sorted(glob.glob(os.path.join(sys.argv[1], "*", "*", "metrics.json"))):  # <run>/<rank 0 job>/
    name = os.path.basename(os.path.dirname(os.path.dirname(path)))
    m = json.load(open(path))
    left = [h.get("checks", {}).get("left_out", []) for h in m["history"][1:]]
    rows[name] = {"final_val_loss": m["final_val_loss"], "best": min(h["val_loss"] for h in m["history"]),
                  "rounds_with_a_rank_left_out": sum(1 for x in left if x), "excluded": m.get("excluded", []),
                  "curve": [h["val_loss"] for h in m["history"]]}
    print(f"{name:14} final {m['final_val_loss']:.4f}  left out in {rows[name]['rounds_with_a_rank_left_out']} rounds  "
          f"excluded {rows[name]['excluded']}")
json.dump(rows, open(os.path.join(sys.argv[1], "summary.json"), "w"), indent=2)
PY
