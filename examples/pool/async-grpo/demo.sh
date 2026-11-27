#!/usr/bin/env bash
# Demo B on the pool: async GRPO with a LoRA policy, a trainer and rollers sharing one GPU (or spread
# over every GPU machine of the pool).
#   ./demo.sh <run name> <max lag> <steps ahead> [steps] [kill]
#   ./demo.sh sync  0 0 30          # the synchronous baseline: on-policy rollouts only, lock-step
#   ./demo.sh async 1 1 30 kill     # async: rollouts up to one version old; one roller of three killed midway
# Needs the model in the pool once: pool model add qwen0.5b@instruct <Qwen2.5-0.5B-Instruct folder>
# The checked tier (M6): FAKE_ROLLERS=N adds N dishonest rollers (FAKE=1 by default, FAKE=fp8 for one that
# serves the policy in fp8); TOPLOC=1 makes every roller attach TOPLOC proofs and the trainer check them
# (image pool/rl-spike:verl-toploc, Dockerfile.toploc).
set -euo pipefail
RUN="$1"; LAG="$2"; AHEAD="$3"; STEPS="${4:-30}"; KILL="${5:-}"; ROLLERS="${ROLLERS:-3}"
STAMP="$RUN-$(date +%s)"; QUEUE="rollouts-$STAMP"; POLICY="grpo/policy-$STAMP"
fill() { sed -e "s#QUEUE_NAME#$QUEUE#" -e "s#POLICY_NAME#$POLICY#g" -e "s#\"STEPS\": \"30\"#\"STEPS\": \"$STEPS\"#" \
             -e "s#\"AHEAD\": \"1\"#\"AHEAD\": \"$AHEAD\"#" -e "s#\"MAX_LAG\": \"1\"#\"MAX_LAG\": \"$LAG\"#" "$1" > "$2"; }
fill rollers.json "rollers-$RUN.json"
fill trainer.json "trainer-$RUN.json"
if [ -n "${FAKE_ROLLERS:-}" ]; then
  # The checked-tier test (M6): honest work on machines labelled role=honest, plus an actor of fake rollers
  # (every reward 1.0, made-up logprobs) on a machine labelled role=evil. Run it in a checked pool.
  python3 - "$RUN" "${FAKE:-1}" "${TOPLOC:-}" <<'PY'
import json, sys
run, fake, toploc = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
for name in (f"rollers-{run}.json", f"trainer-{run}.json"):
    spec = json.load(open(name)); spec["resources"]["labels"] = {"role": "honest"}
    if toploc:
        spec["image"] = "pool/rl-spike:verl-toploc"; spec["env"]["TOPLOC"] = "1"
    json.dump(spec, open(name, "w"), indent=2)
spec = json.load(open(f"rollers-{run}.json")); spec["resources"]["labels"] = {"role": "evil"}; spec["env"]["FAKE"] = fake
spec["name"] = "fake-roller"; json.dump(spec, open(f"fakes-{run}.json", "w"), indent=2)
PY
fi
pool queue create "$QUEUE" --ref "$POLICY" --max-lag "$LAG"
pool actor create "rollers-$STAMP" "rollers-$RUN.json" --min "$ROLLERS" --max "$ROLLERS"
if [ -n "${FAKE_ROLLERS:-}" ]; then pool actor create "fakes-$STAMP" "fakes-$RUN.json" --min "$FAKE_ROLLERS" --max "$FAKE_ROLLERS"; fi
JOB=$(pool submit "trainer-$RUN.json" 2>&1 | grep -o "job j[0-9a-f]*" | cut -d' ' -f2)
echo "trainer $JOB"
if [ -n "$KILL" ]; then
  until [ "$(pool ref log "$POLICY" 2>/dev/null | head -1 | sed -n 's/^v\([0-9]*\).*/\1/p')" -ge $((STEPS / 2)) ] 2>/dev/null; do sleep 5; done
  VICTIM=$(pool actor status "rollers-$STAMP" | python3 -c "import json,sys; print(next(r['job'] for r in json.load(sys.stdin)['replicas'] if r['status']=='running'))")
  pool cancel "$VICTIM" > /dev/null
  echo "killed roller $VICTIM at $(date +%s) (policy v$((STEPS / 2)))"
fi
until pool show "$JOB" | grep -qE "succeeded|failed|cancelled"; do sleep 10; done
pool show "$JOB" | head -5
pool get "$JOB" --to "out-$RUN"
pool actor stop "rollers-$STAMP"
if [ -n "${FAKE_ROLLERS:-}" ]; then pool actor stop "fakes-$STAMP"; pool machine ls; fi
python3 summary.py "out-$RUN/metrics.jsonl"
