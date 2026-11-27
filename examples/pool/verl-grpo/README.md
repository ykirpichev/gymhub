# Recipe B on verl (issue #87, M3)

Demo B (`../async-grpo`) trains with its own small GRPO loop. This example runs the same async GRPO
recipe with **verl's own trainer**: verl 0.8's fully-async PPO trainer
(`verl.experimental.fully_async_policy.FullyAsyncTrainer`) does the training, and pool primitives
replace the parts of verl that need one Ray cluster spanning trainer and rollout GPUs.

verl was picked over prime-rl because spike S1 (`../rl-spike`) found it the better fit for the Spark:
it installs on NVIDIA's arm64 vLLM image without its dependencies and ran in the sandbox unchanged.
prime-rl needs its own 47 GB CUDA stack and four fixes to run offline. prime-rl's rollouts also go
to vLLM servers over HTTP, and a pool job cannot reach another job over the network.

| verl piece | replaced by | file |
|---|---|---|
| `FullyAsyncTrainer`: FSDP actor with LoRA, GRPO advantages, PPO-clip loss | kept as is, in a pool **job** | `pool_verl.py` (`PoolTrainer`) |
| `MessageQueue`, a Ray actor that holds rollouts | a pool **queue** tied to the policy ref with `max_lag` | `PoolTrainer._get_samples_from_queue` |
| checkpoint engine (NCCL or IPC weight push to vLLM) | a pool **ref**: each step's LoRA adapter is published as its next version | `PoolTrainer._fit_update_weights`, `PoolActorWorker.export_lora` |
| `FullyAsyncRollouter`, vLLM servers in the Ray cluster | a pool **actor** of vLLM rollers **subscribed** to the ref | `roller.py` (Demo B's, unchanged) |

What verl still computes on each step: GRPO advantages grouped by prompt (`compute_grpo_outcome_advantage`),
and the PPO-clip loss in verl's *bypass mode* (`algorithm.rollout_correction.bypass_mode=True`). In bypass mode
the old log-probs are the ones the roller sampled with, so the ratio corrects for rollouts made with a slightly
older policy. The update runs on verl's FSDP engine with LoRA r=16, through verl's own `update_actor`, and the
adapter comes out through `get_per_tensor_param`, the same call verl's weight sync uses. `PoolTrainer` changes
only where samples come from, where weights go, and turns off the validation and checkpoints that verl runs
through its rollouter. Inside the trainer job verl still runs a one-node Ray cluster on loopback, as in S1.

The task, reward and sizes are Demo B's: synthetic arithmetic on Qwen2.5-0.5B-Instruct, 30 steps of
8 tasks × 2 prompts × 8 samples (128 samples, one optimizer step, lr 5e-5, no KL), 3,840 samples per run.
The trainer recomputes every reward from the completion text; it does not use the roller's.

## Run it

```
docker build -t pool/rl-spike:verl -f ../rl-spike/Dockerfile.verl ../rl-spike   # verl 0.8 + vLLM 0.19, arm64 or x86
pool model add qwen0.5b@instruct ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/<rev>
./demo.sh sync  0 0 30          # synchronous baseline: max_lag 0, no tasks queued ahead
./demo.sh async 1 1 30 kill     # async: rollouts up to one version old; a third of the rollers cancelled at step 15
```

`ROLLERS` sets the actor's size (default 3). Each run prints `summary.py`'s summary and leaves per-step
metrics in `out-<run>/metrics.jsonl`. These include Demo B's fields (reward, policy lag, seconds waiting for
rollouts and training) and some of verl's own: `actor/pg_loss`, `actor/pg_clipfrac`, `actor/ppo_kl` (between
the trained policy and the rollout policy) and `timing_s/update_actor`.

## Results on the DGX Spark (GB10)

These ran through a pool server and one Docker agent on the Spark. The trainer and three rollers share its
one GPU (`share_gpu`: 24 GB for the trainer, 12 GB for each roller). Each run is 30 steps and 3,840 samples.

| Run | Trainer idle, steps 2–20 | steps 2–30 | Reward, steps 1–10 | 11–20 | 21–30 | Minutes |
|---|---|---|---|---|---|---|
| synchronous baseline (`max_lag 0`, lock-step) | 70% | 69% | 0.76 | 0.98 | 0.97 | 6.9 |
| async, 3 rollers, one (a third) killed at step 15 | **8%** | **6%** | 0.60 | 0.96 | 0.95 | 7.6 |

For comparison, Demo B's own trainer under the same conditions: 71% and 21% idle (steps 2–20), and rewards in the last third of 0.97 and 0.96.

What this shows:
- **verl trains from the pool's rollouts.** On-policy, verl's `actor/ppo_kl` between its policy and the
  rollers' sampling log-probs stays near 0.0003 throughout the synchronous run. So the LoRA adapter that
  verl exports, publishes as a ref version and vLLM loads is the policy verl is training. Async, rollouts are
  one version old (mean lag 0.94), and the KL is 0.001.
- **Reward matches the synchronous baseline at equal samples** once training gets going: 0.95–0.96 against
  0.97–0.98 over steps 11–30. Async starts slower over steps 1–10 (0.60 vs 0.76), because each step trains on
  rollouts from the policy before. Also, the first eight or so steps had only two rollers: one roller's vLLM
  failed to start three times (see below).
- **Killing a third of the rollers does not stall training.** One of three rollers was cancelled after step 14.
  The actor started a replacement, and the steps after the kill took 13–18 s, against 14–20 s before it. The
  trainer did not wait more than 3.6 s for rollouts at any step after the kill.
- **The trainer is almost never idle async** (6%, against 69% in sync). verl's update is slower than Demo B's
  trainer while the rollers share its GPU (13 s against 4 s for `update_actor` alone), so three rollers
  outpace it. The other side of that: 54 of 312 finished tasks went stale before the trainer could use them,
  and were dropped by the use-time lag check.
- **On one GPU, async does not finish sooner** (7.6 against 6.9 minutes), as in Demo B. Trainer and rollers
  take turns on the same GPU. The layout pays off when the rollers' actor spreads over other machines.

An earlier pair of runs gave the same picture: sync 69% idle and reward 0.68 / 0.96 (first / last third), against async 4% idle and 0.60 / 0.96. In those runs a stray idle roller from a failed test run was also holding GPU memory.

Per-step metrics, with verl's own: `results/sync.jsonl`, `results/async-3-rollers-kill.jsonl`.

## What it took, and what to watch

- **verl's fully-async trainer runs without its rollouter.** `FullyAsyncTrainer` is a Ray actor class. The
  subclass derives from `FullyAsyncTrainer.__ray_actor_class__` and is wrapped again with `ray.remote(num_cpus=1)`,
  since verl's own wrapper asks for 10 CPUs. It needs `actor_rollout_ref.hybrid_engine=False`, and
  `checkpoint_engine.backend=naive` so the actor worker does not set up NCCL for a rollouter that is not there.
- **Exporting the adapter** needs one worker method (`PoolActorWorker.export_lora`, registered `ONE_TO_ALL`).
  It writes `get_per_tensor_param(base_sync_done=True)`, which gives PEFT-named LoRA tensors, plus the PEFT config.
  That folder is what vLLM's `LoRARequest` loads. Exporting and publishing a version takes 0.5 s alone and
  1.7 s while the rollers are running.
- **Hydra:** verl's fully-async config folder is not a Python package, so `pkg://` does not find it, and
  `pool_verl.py` passes its file path. The job's working directory is read-only, so `hydra.run.dir=/tmp/hydra`, as in S1.
- **Ray workers need the job's paths:** `run.sh` puts `/work` and `$POOL_LIB` on `PYTHONPATH`, because the
  trainer actor (a Ray worker process) imports `pool_job` and `common`. The job's socket is inherited, so
  `pool_job` works from inside Ray actors unchanged.
- **vLLM startup next to verl:** while verl's trainer starts (Ray, FSDP, model load), a roller's vLLM sometimes
  counts the trainer's memory as its own during profiling and fails with "No available memory for the cache
  blocks". This happened one to three times per run. The actor replaces the replica, which starts once the
  trainer settles. Demo B had the same failure among the rollers. Here the trainer is the bigger neighbour.
