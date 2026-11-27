#!/usr/bin/env bash
# Demo A on one machine: the same batch through vLLM directly, then through the pool (queue + actor).
#   ./demo.sh <model name> <model folder> [tasks] [batch]
# Needs a pool login (pool login ...) and a GPU agent in the pool; writes results-<name>.json here.
set -euo pipefail
NAME="$1"; MODEL="$2"; TASKS="${3:-32}"; BATCH="${4:-64}"
HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE=nvcr.io/nvidia/vllm:26.04-py3
SAFE="${NAME//[@\/]/-}"

echo "== standalone"
if [ -z "${SKIP_STANDALONE:-}" ]; then
# A Hugging Face snapshot is symlinks into blob stores (per model, or shared at the hub level with
# newer huggingface_hub): mount the whole hub folder.
REAL="$(cd "$MODEL" && pwd -P)"; HUB="$(cd "$REAL/../../.." && pwd -P)"; INSIDE="/hf/${REAL#"$HUB"/}"
docker run --rm --gpus all --ipc=host -e HF_HUB_OFFLINE=1 -v "$HUB":/hf:ro -v "$HERE":/work -w /work "$IMAGE" \
  python standalone.py "$INSIDE" --tasks "$TASKS" --batch "$BATCH" | grep "^{\"mode\"" | tee "standalone-$SAFE.json"
fi
BASE=$(python3 -c "import json; print(json.load(open('standalone-$SAFE.json'))['tokens_per_s'])")

echo "== through the pool"
pool model add "$NAME" "$MODEL" || true          # once; a second time costs no upload
QUEUE="prompts-$SAFE-$(date +%s)"
pool queue create "$QUEUE"
python3 make_tasks.py --tasks "$TASKS" --batch "$BATCH" | pool queue add "$QUEUE"
sed -e "s/MODEL_NAME/$NAME/" -e "s/\"QUEUE\": \"prompts\"/\"QUEUE\": \"$QUEUE\"/" actor.json > "actor-$SAFE.json"
pool actor create "rollers-$SAFE" "actor-$SAFE.json" --min 0 --max 1 --queue "$QUEUE" --per-replica 100000
until [ "$(pool queue stats "$QUEUE" | python3 -c 'import json,sys; c=json.load(sys.stdin)["counts"]; print(c["queued"]+c["running"])')" = 0 ]; do sleep 10; done
pool queue results "$QUEUE" > "results-$SAFE.jsonl"
python3 report.py "results-$SAFE.jsonl" --standalone "$BASE" | tee "pool-$SAFE.json"
pool actor stop "rollers-$SAFE"
