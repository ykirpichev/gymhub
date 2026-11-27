#!/usr/bin/env bash
# Demo C1 in small on one machine: two islands (two pool jobs on the same GPU) train OLMoE-small with
# torchft DiLoCo over a shaped link, and a single-island baseline sees the same tokens.
#   ./demo.sh single              the baseline
#   ./demo.sh islands 100mbit     two islands, the link between them shaped to 100 Mbit/s each way
#   ./demo.sh islands 1gbit
#   ./demo.sh islands none        two islands, loopback unshaped
# Extra arguments go to train.py (--rounds, --inner-steps, --fragments, --sync-delay ...).
# Needs a pool whose machine allows peer networking (pool agent run --allow-peer-network), the
# image (docker build -t pool/diloco-torchft:0.3.1 .) and the data (python3 make_data.py data).
set -euo pipefail
cd "$(dirname "$0")"
MODE="$1"; shift
IMAGE="${IMAGE:-pool/diloco-torchft:0.3.1}"
DATA="${DATA:-data}"
LINK=none
if [ "$MODE" = islands ]; then LINK="$1"; shift; fi
RUN="c1-$(date +%H%M%S)"
if [ "$MODE" = single ]; then TAG="${TAG:-single}"; else TAG="${TAG:-$LINK}"; fi  # the results file names
mkdir -p results

spec() {  # spec <name> <gang: 0|1> <args...>: a job spec on stdout
  python3 - "$@" <<'PY'
import json, os, sys
name, gang, *args = sys.argv[1:]
spec = {"name": name, "image": os.environ["IMAGE"], "code": ".", "command": ["python", "train.py", *args],
        "resources": {"gpus": 1, "share_gpu": True, "gpu_memory_gb": 24, "cpus": 8, "memory_gb": 24, "shm_gb": 4}
                      if os.environ.get("GPUS", "1") != "0" else {"cpus": 4, "memory_gb": 8},  # GPUS=0: a CPU smoke test
        "inputs": [{"dir": os.environ["DATA"], "path": "data"}], "timeout_minutes": 90}
if gang == "1":  # an island: a gang (of one rank here), allowed to reach the other islands
    spec.update({"gang": {"size": 1}, "network": "peers"})
print(json.dumps(spec))
PY
}
submit() { spec "$@" > "spec-$1.json"; pool submit "spec-$1.json" 2>&1 | grep -o "j[0-9a-f]\{8\}" | head -1; }
finish() {
  until pool show "$1" | grep -qE "succeeded|failed|cancelled"; do sleep 5; done
  pool get "$1" --to "out/$1" > /dev/null
  pool show "$1" | grep -q succeeded || { pool logs "$1" | tail -30 >&2; return 1; }
}
export IMAGE DATA

if [ "$MODE" = single ]; then
  job=$(submit "$RUN-single" 0 --mode single "$@")
  finish "$job"
  cp "out/$job/metrics.json" "results/$TAG.json"
  python3 report.py "results/$TAG.json"
  exit 0
fi

if [ "$LINK" != none ]; then ./shape.sh on "$LINK"; trap './shape.sh off' EXIT; fi
a=$(submit "$RUN-island0" 1 --mode island --island 0 --bind 127.0.0.2 --ref "c1/$RUN" --link "$LINK" "$@")
b=$(submit "$RUN-island1" 1 --mode island --island 1 --bind 127.0.0.3 --ref "c1/$RUN" --link "$LINK" "$@")
echo "islands: $a $b, rounds on ref c1/$RUN"
finish "$a"; finish "$b"
cp "out/$a/metrics.json" "results/islands-$TAG-0.json"
cp "out/$b/metrics.json" "results/islands-$TAG-1.json"
pool ref log "c1/$RUN" --limit 3 || true
python3 report.py results/islands-"$TAG"-*.json
rm -f spec-*.json
