#!/usr/bin/env bash
# Demo C2 in small on the pool: an anchor, two experts trained on separate machines, a merged MoE,
# and a dense baseline of the same size.   ./demo.sh [steps] [router steps]
# Needs a pool with two machines labelled role=a and role=b (pool agent run --label role=a); a GPU
# image with PyTorch (IMAGE, default the NGC vLLM image).
set -euo pipefail
STEPS="${1:-3000}"; ROUTER="${2:-600}"; IMAGE="${IMAGE:-nvcr.io/nvidia/vllm:26.04-py3}"
A="${ROLE_A:-a}"; B="${ROLE_B:-b}"  # the labels of the two expert machines
[ -d data ] || python3 make_corpus.py data
submit() {  # submit <name> <role or -> <command> [input path=artifact ...]; prints the job id
  python3 - "$@" > "spec-$1.json" <<'PY'
import json, sys
name, role, command, *inputs = sys.argv[1:]
spec = {"name": name, "image": "IMAGE", "code": ".", "command": ["python", "train.py", *command.split()],
        "resources": {"gpus": 1, "cpus": 4, "memory_gb": 16}, "timeout_minutes": 120,
        "inputs": [{"dir": "data", "path": "data"}] + [{"path": i.split("=")[0], "artifact": i.split("=", 1)[1]} for i in inputs]}
if role != "-":
    spec["resources"]["labels"] = {"role": role}
print(json.dumps(spec))
PY
  sed -i "s#\"IMAGE\"#\"$IMAGE\"#" "spec-$1.json"
  pool submit "spec-$1.json" 2>&1 | grep -o "job j[0-9a-f]*" | cut -d' ' -f2
}
finish() {  # finish <job>: wait, print its report, echo its outputs artifact
  until pool show "$1" | grep -qE "succeeded|failed|cancelled"; do sleep 5; done
  pool show "$1" | grep -q succeeded || { pool logs "$1" | tail -20 >&2; exit 1; }
  pool show "$1" | sed -n 's/.*outputs \(sha256:[0-9a-f]*\).*/\1/p'
}
ANCHOR_JOB=$(submit anchor - "anchor --steps $STEPS --data /inputs/data")
ANCHOR=$(finish "$ANCHOR_JOB"); echo "anchor $ANCHOR_JOB -> $ANCHOR"
CODE_JOB=$(submit expert-code "$A" "expert --domain code --steps $STEPS --anchor /inputs/anchor --data /inputs/data" "anchor=$ANCHOR")
LEGAL_JOB=$(submit expert-legal "$B" "expert --domain legal --steps $STEPS --anchor /inputs/anchor --data /inputs/data" "anchor=$ANCHOR")
DENSE_JOB=$(submit dense - "dense --steps $STEPS --router-steps $ROUTER --data /inputs/data")
CODE=$(finish "$CODE_JOB"); LEGAL=$(finish "$LEGAL_JOB")
echo "experts: code $CODE_JOB $(pool show "$CODE_JOB" | grep -o 'on [a-z0-9-]*' | head -1), legal $LEGAL_JOB $(pool show "$LEGAL_JOB" | grep -o 'on [a-z0-9-]*' | head -1)"
MERGE_JOB=$(submit merge - "merge --router-steps $ROUTER --anchor /inputs/anchor --experts /inputs/code /inputs/legal --data /inputs/data" \
            "anchor=$ANCHOR" "code=$CODE" "legal=$LEGAL")
finish "$MERGE_JOB" > /dev/null; finish "$DENSE_JOB" > /dev/null
mkdir -p results
for job in "$ANCHOR_JOB" "$CODE_JOB" "$LEGAL_JOB" "$MERGE_JOB" "$DENSE_JOB"; do pool get "$job" --to "out/$job" > /dev/null; done
python3 report.py out/*/*.json
rm -f spec-*.json
