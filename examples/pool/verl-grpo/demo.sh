#!/usr/bin/env bash
# Recipe B on verl: verl's fully-async GRPO trainer as a pool job, vLLM rollers as a pool actor.
#   ./demo.sh <run name> <max lag> <steps ahead> [steps] [kill]
#   ./demo.sh sync  0 0 30          # the synchronous baseline: on-policy rollouts only, lock-step
#   ./demo.sh async 1 1 30 kill     # async: rollouts up to one version old; a third of the rollers killed midway
# ROLLERS (default 3) sets the actor's size; "kill" cancels ROLLERS/3 of them (at least one) at step STEPS/2.
# Needs the model in the pool once: pool model add qwen0.5b@instruct <Qwen2.5-0.5B-Instruct folder>
set -euo pipefail
POOL="${POOL:-pool}"
RUN="$1"; LAG="$2"; AHEAD="$3"; STEPS="${4:-30}"; KILL="${5:-}"; ROLLERS="${ROLLERS:-3}"
STAMP="$RUN-$(date +%s)"; QUEUE="rollouts-$STAMP"; POLICY="grpo/policy-$STAMP"
fill() { sed -e "s#QUEUE_NAME#$QUEUE#" -e "s#POLICY_NAME#$POLICY#g" -e "s#\"STEPS\": \"30\"#\"STEPS\": \"$STEPS\"#" \
             -e "s#\"AHEAD\": \"1\"#\"AHEAD\": \"$AHEAD\"#" -e "s#\"MAX_LAG\": \"1\"#\"MAX_LAG\": \"$LAG\"#" "$1" > "$2"; }
fill rollers.json "rollers-$RUN.json"
fill trainer.json "trainer-$RUN.json"
$POOL queue create "$QUEUE" --ref "$POLICY" --max-lag "$LAG"
$POOL actor create "rollers-$STAMP" "rollers-$RUN.json" --min "$ROLLERS" --max "$ROLLERS"
trap '$POOL actor stop "rollers-$STAMP" > /dev/null 2>&1 || true' EXIT   # also when the trainer fails
JOB=$($POOL submit "trainer-$RUN.json" 2>&1 | grep -o "job j[0-9a-f]*" | cut -d' ' -f2)
echo "trainer $JOB"
if [ -n "$KILL" ]; then
  until [ "$($POOL ref log "$POLICY" 2>/dev/null | head -1 | sed -n 's/^v\([0-9]*\).*/\1/p')" -ge $((STEPS / 2)) ] 2>/dev/null; do sleep 5; done
  COUNT=$(( ROLLERS / 3 > 0 ? ROLLERS / 3 : 1 ))
  for VICTIM in $($POOL actor status "rollers-$STAMP" | python3 -c "
import json, sys
running = [r['job'] for r in json.load(sys.stdin)['replicas'] if r['status'] == 'running']
print(' '.join(running[:$COUNT]))"); do
    $POOL cancel "$VICTIM" > /dev/null
    echo "killed roller $VICTIM at $(date +%s) (policy v$((STEPS / 2)))"
  done
fi
until $POOL show "$JOB" | grep -qE "succeeded|failed|cancelled"; do sleep 10; done
$POOL show "$JOB" | head -5
$POOL actor stop "rollers-$STAMP"
$POOL get "$JOB" --to "out-$RUN"
python3 summary.py "out-$RUN/metrics.jsonl"
