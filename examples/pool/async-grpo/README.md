# Demo B: async GRPO on the pool (issue #87, M3)

A LoRA policy on Qwen2.5-0.5B-Instruct learns synthetic arithmetic ("What is 51 * 8 + 12?", answer after `####`) with GRPO. The work is split the way M3 describes, using only pool primitives:

| Piece | Pool primitive | File |
|---|---|---|
| the policy (the LoRA adapter) | a **ref**, `grpo/policy-<run>`, one version per training step | `trainer.py` publishes |
| rollout requests (2 prompts each) | a **queue** tied to the ref with `max_lag` | `trainer.py` adds tasks |
| rollers (vLLM with LoRA, 8 samples per prompt) | an **actor** with 3 replicas, **subscribed** to the ref | `roller.py` |
| the trainer (PEFT, one GRPO step per 8 tasks) | a **job** | `trainer.py` |

All of it runs in the pool's sandbox (no network). On one DGX Spark the trainer and the three rollers share the GPU (`resources.share_gpu`, 24 GB for the trainer and 12 GB per roller). On a pool with more machines, the same actor spreads its replicas over every machine that fits one.

How a step goes:
1. The trainer keeps the queue topped up: tasks for the current step, plus `AHEAD` more steps.
2. Rollers lease tasks, sample with the newest policy version they have, and report completions, their sampling logprobs and rewards, tagged with that version. The agent pre-fetches each new version as soon as it is published (the actor subscribes to the ref), so switching adapters is a local copy.
3. The queue accepts a result only if its version is at most `max_lag` behind the ref; anything older goes back to the queue.
4. The trainer takes the first 8 accepted tasks (128 samples), computes group-normalized advantages and does one step with importance weights against the sampling logprobs (clipped as in PPO), which corrects for rollouts made with a slightly older policy. It then publishes the adapter as the next version.

`max_lag 0` with `AHEAD 0` is the synchronous baseline: rollers only ever sample with the current policy, and the trainer waits for them every step.

## Run it

```
docker build -t pool/rl-spike:verl -f ../rl-spike/Dockerfile.verl ../rl-spike   # vLLM + PEFT
pool model add qwen0.5b@instruct ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/<rev>
./demo.sh sync  0 0 30          # synchronous baseline
./demo.sh async 1 1 30 kill     # async; one roller is cancelled at step 15 (ROLLERS=4 for four rollers)
```

Each run prints a summary (`summary.py`) and leaves the trainer's per-step metrics in `out-<run>/metrics.jsonl`: reward, the policy lag of the rollouts used, time spent waiting for rollouts (idle) and training.

## Results on the DGX Spark (GB10)

30 steps of 8 tasks × 2 prompts × 8 samples: 3,840 samples per run, the same for every run. Trainer and rollers share the one GB10. Reward is the mean training reward (1 = right answer after `####`, 0.1 = format only).

| Run | Trainer idle, steps 2–20 | Trainer idle, steps 2–30 | Reward, steps 1–10 | 11–20 | 21–30 | Minutes |
|---|---|---|---|---|---|---|
| synchronous baseline (`max_lag 0`, lock-step) | 71% | 75% | 0.63 | 0.95 | 0.97 | 7.0 |
| async, 3 rollers, one killed at step 15 | 21% | 28% | 0.61 | 0.94 | 0.96 | 6.2 |
| async, 4 rollers, one killed at step 15 | **16%** | 28% | 0.60 | 0.94 | **0.98** | 7.0 |
| async, 3 rollers, lag checked only on arrival | 2% | 7% | 0.29 | 0.74 | 0.92 | 7.5 |

What this shows:
- **Killing a roller does not stall training.** The actor replaces it (its replica number is reused), and no step took longer than 28 s, the same as the longest steps before the kill.
- **Reward matches the synchronous baseline at equal samples** once rollouts are kept within one policy version when they are *used*, not only when they arrive. The queue's `max_lag` checks a result on arrival; the trainer drops any that the policy has since moved past (17 and 39 tasks over the two runs, out of 240+). Without that second check (last row) the trainer is almost never idle, but it trains on rollouts 2–2.5 versions old and learns noticeably slower.
- **Idle time:** with 4 rollers the trainer waits 16% of the time while it is learning (steps 2–20). Over the whole run it is 28%: once the task is solved, most groups score 1 on every sample, their advantages are 0, and a training step takes 1–2 s while rollouts still take about 5 s.
- **On one GPU, async does not finish sooner.** The rollers and the trainer take turns on the same GPU, so wall time stays about 7 minutes. The point of the async layout is that rollers can run anywhere: on a pool with more machines the same actor spreads over them, and the trainer's GPU is left to the trainer. This run had only the Spark (the pool's other machine belongs to another, live pool), so the multi-machine version of this demo is still to be run.

Along the way: three vLLM engines starting at the same moment on one GPU made one of them fail its memory profiling (it counted the others' allocations as its own), and the actor restarted it. Rollers now stagger their start by up to 20 s.

Per-step metrics: `results/*.jsonl` (reward, policy lag, wait and train seconds, stale rollouts dropped).

## With a dishonest machine (the checked tier, M6)

The trainer checks every rollout before it trains on it: it recomputes each completion's reward, and for one sample per prompt it recomputes the logprobs under the policy version the rollout claims and compares them with the ones reported (cheap because a GRPO trainer computes those logprobs anyway). Rollouts that went stale while waiting are dropped without a report. It reports each machine's result with `pool_job.report`; in a `checked` pool, a machine with 2 failed checks is excluded.

```
pool pool set default trust=checked
FAKE_ROLLERS=1 ROLLERS=4 ./demo.sh checked 1 1 30   # 4 honest rollers (role=honest) and 1 fake one (role=evil)
```

The fake roller claims every answer is right and invents its logprobs. On the Spark, with a second agent registered as the machine `spark-evil`:

| | |
|---|---|
| fake rollouts rejected | 2 (both by the reward check) |
| fake rollouts trained on | **0**: every training batch came from the honest machine |
| `spark-evil` after its second failed check | excluded: it got no more tasks |
| reward, steps 1–10 / 21–30 | 0.59 / 0.97, as in the honest runs |

Per-step metrics: `results/checked-with-a-fake-roller.jsonl`.

### TOPLOC proofs

Lying about rewards is easy to catch. A subtler machine does real work with a cheaper model: the right rewards for its own completions and the logprobs of the model it really ran. With `TOPLOC=1` every roller attaches [TOPLOC](https://arxiv.org/abs/2501.16007) proofs to every completion: for the prompt and for each 32 generated tokens, the 128 largest values of the policy's last hidden states, encoded as a polynomial (about 260 bytes each). The trainer checks them in the same forward pass that recomputes the logprobs, under the policy version the rollout claims. It keeps the LoRA weights of the last `MAX_LAG + 1` versions for that, so a rollout cannot dodge the checks by claiming to be one version old. The helpers are `pool_verify` (`pool/verify.py`, shipped next to `pool_job` in every job). The image needs the `toploc` package (`Dockerfile.toploc`, built from source against the image's torch: the published wheels fail to import with NGC's torch).

```
docker build -t pool/rl-spike:verl-toploc -f Dockerfile.toploc .
TOPLOC=1 FAKE=base FAKE_ROLLERS=1 ROLLERS=2 ./demo.sh checked-base 1 1 12   # a roller that skips the LoRA
TOPLOC=1 FAKE=fp8  FAKE_ROLLERS=1 ROLLERS=2 ./demo.sh checked-fp8  1 1 12   # a roller that serves fp8
```

On the roller side, `pool_verify.VllmStates` hooks the model's final norm in the vLLM engine and runs one prefill of prompt and completion per sample, with a fresh cache salt so no prefix is skipped (about 25 ms per sample). That needs `enforce_eager=True` (hooks do not run inside CUDA graphs) and the engine in the same process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`). The trainer side uses transformers, so honest proofs cross two implementations.

Thresholds (`pool/verify.py`): a chunk fails with more than 38 of 128 exponents different, or mantissa errors above 10 (mean) or 8 (median). Calibrated on the GB10 with Qwen2.5-0.5B and a LoRA, vLLM proofs checked with transformers: honest worst case 19 / 7.7 / 6; fp8 served for bf16: 122–128 exponents different in every chunk.

Results (on the Spark; private pool with agents `honest-a`, `honest-b`, `honest-c` and `evil` on the one GB10; 2 honest rollers, 1 fake, 12 steps):

**Roller that serves fp8** (`FAKE=fp8`): vLLM's fp8 weight quantization, everything else honest.

| | |
|---|---|
| its rollouts | 2 sent, **2 rejected, by TOPLOC and by the logprob check** (mean gap 7.2 and 7.3), 0 trained on |
| `evil` | excluded after its second failed check; no honest rollout was rejected |
| reward, steps 1–4 / 9–12 | 0.34 / 0.96 |

fp8 on this GPU changes the model far more than a careful quantization would (completions run to the token limit), so both checks catch it. Per-step metrics: `results/checked-toploc-fp8-roller.jsonl`.

**Roller that skips the adapter** (`FAKE=base`): it serves the base model, ignores policy updates, and is otherwise honest.

| | |
|---|---|
| steps 1–4 | its rollouts passed and were trained on. The LoRA starts at zero, so the policy *was* the base model, and these rollouts were valid |
| from step 5 | 4 of its rollouts rejected: **all 4 by TOPLOC**, only 1 of them by the logprob check (mean gap 0.70 against a tolerance of 0.5; the other three stayed under it) |
| `evil` | excluded after its second failed check; no honest rollout was rejected |
| reward, steps 1–4 / 9–12 | 0.25 / 0.95 |

TOPLOC catches the model swap as soon as the policy has moved away from the base model. The logprob check, which only looks at the sampled tokens' probabilities, mostly does not. Per-step metrics: `results/checked-toploc-base-model-roller.jsonl`.

For contrast, the reward liar (`FAKE=1`) with proofs on: 2 rollouts sent, both rejected by the reward check alone. Its proofs pass, because it did run the real model, and its invented logprobs (-0.01 per token) stay within 0.5 of a confident policy's.

What TOPLOC does not prove is which tokens were sampled: a machine could pick tokens with any model and then run the right model over them to make the proof. The logprob check covers the sampling, the reward check covers the outcome, and the trainer runs all three.
