# Spike S1: GRPO inside a pool job (issue #45)

Goal: one pool job on the DGX Spark that runs a small GRPO training run inside the pool sandbox (`--network none`, read-only root, no capabilities, unprivileged uid). The run uses a single-node Ray, vLLM rollouts and LoRA, and rewards should rise. Along the way, it records what the sandbox has to allow.

## verl: works

Files:
- `Dockerfile.verl` builds verl 0.8.0 on `nvcr.io/nvidia/vllm:26.04-py3` (arm64, CUDA 13.2, torch 2.12, vLLM 0.19, transformers 4.57). It pins the image's torch, vLLM and transformers with a constraints file, then installs verl with `--no-deps`. verl 0.9 needs transformers 5, but vLLM 0.19 needs transformers below 5. The build takes under a minute on top of the base image, and the image is 19.6 GB, almost all of it the base.
- `verl/`: the job's code.
  - `make_data.py`: synthetic arithmetic ("What is 37 * 6 + 145?"); nothing is downloaded.
  - `reward.py`: 1 for the right number after `####`, 0.1 for the format alone.
  - `run.sh`: GRPO with Qwen2.5-0.5B-Instruct, LoRA r=16, 8 rollouts per prompt, 32 prompts per step.
  - `curve.py`: writes the metrics to `/outputs`.

### Run it through the pool

```
docker build -t pool/rl-spike:verl -f Dockerfile.verl .
pool model add qwen2.5-0.5b@instruct ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/<rev>
cd verl
pool run --name grpo-verl --image pool/rl-spike:verl --gpus 1 --cpus 16 --mem 64 \
    --shm-gb 8 --tmp-gb 16 --pids 8192 --model base=qwen2.5-0.5b@instruct --env STEPS=20 -- bash run.sh
```

### Results on the Spark (GB10), through a local pool server and one Docker agent

The job ran 20 steps in 7.3 minutes, at about 18 s per step. The 953 MB ModelRef was fetched in 0.9 s.

| step | 1 | 2 | 3 | 4 | 5 | 10 | 15 | 20 |
|---|---|---|---|---|---|---|---|---|
| train reward | 0.181 | 0.373 | 0.624 | 0.805 | 0.900 | 0.958 | 0.989 | 0.975 |

Validation reward, on 128 held-out prompts with greedy decoding: 0.00 before training, 0.972 after steps 10 and 20. Most of the gain is the model learning the `####` answer format; the arithmetic itself is easy for it. The task is deliberately easy, so the run shows the mechanics in a few minutes.

## What the sandbox must allow

| Item | Needed? | Why / what we saw |
|---|---|---|
| `--network none` | **Kept** | Ray (GCS, raylets, dashboard), the vLLM HTTP server and torch.distributed (world size 1) all bind to loopback, which a `--network none` container still has. |
| Capabilities, `--ipc=host`, `--privileged` | **No** | It runs with every capability dropped, `no-new-privileges` and a private IPC namespace. |
| Unprivileged uid, read-only root | **Kept** | It needs a writable `HOME` (`/tmp`, already set) and a writable run folder: Hydra writes `outputs/` into the working directory, which is read-only, so `run.sh` passes `hydra.run.dir=/tmp/hydra`. |
| `/dev/shm` (default 4 GB) | Works; more is better | With 4 GB, Ray puts its object store in `/tmp/ray` and warns that this is slower. `resources.shm_gb` (new, up to 256) raises it. |
| `/tmp` (default 8 GB) | Works at 8 GB for this size | It holds the Ray session, the object store fallback, vLLM and Triton compile caches, and the parquet data. `resources.tmp_gb` (new, up to 512) raises it for bigger models. |
| PIDs (default 4,096) | Works, but close | Peak 2,976 tasks (threads count): Ray workers, vLLM, dataloaders. Larger runs will hit the limit, so `resources.pids` (new, up to 65,536). |
| Memory | 64 GB was plenty | Peak 24.3 GiB with vLLM at `gpu_memory_utilization=0.25`. On the GB10's unified memory, the GPU allocations count against the container's memory limit, so size `--mem` for both. |
| Network at run time | **No** | Everything is in the image or the inputs: `HF_HUB_OFFLINE=1`, the model is a pool ModelRef at `/inputs/base`, and the data is generated. |
| Network at build time | Yes | pip downloads verl and its dependencies; the base image comes from NGC. |

`/dev/shm` and `/tmp` are memory-backed, so they count against `memory_gb` when used. The spec refuses `shm_gb` larger than `memory_gb`.

## prime-rl: works too, with more effort

Files:
- `Dockerfile.prime-rl` builds prime-rl from its lock file, pinned to one commit, on Ubuntu 24.04 with torch 2.14 cu130, vLLM 0.34 and transformers 5.20, all arm64 wheels from PyPI. The build takes about 13 minutes and the image is 46.6 GB. It is a separate CUDA stack: prime-rl will not install over the NGC vLLM image.
- `prime-rl/`:
  - `rl.toml`: prime-rl's own reverse-text example, shrunk to 15 steps, 64 rollouts per step and 8 per prompt, on PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT.
  - `run.sh`: runs `rl @ rl.toml`.

What it took to run in the sandbox:
1. **Submodules:** they use SSH URLs, so the build rewrites them to HTTPS. All of them are public.
2. **vllm-router:** the inference launcher needs it, but it ships in the optional `disagg` extra. The build installs the wheel.
3. **One shared GPU:** `rl` assigns one GPU to inference and one to the trainer. `CUDA_VISIBLE_DEVICES=0,0` gives both the same GB10.
4. **Weight hand-over:** NCCL cannot hold two ranks on one GPU, so the trainer hands weights to vLLM through files (`[weight_broadcast] type = "filesystem"`).
5. **Run-time package installs:** the verifiers harness installs its program's packages with `uv sync --script` *at run time* and forces `UV_OFFLINE=false`. The image pre-downloads those wheels to `/opt/wheels`. A `uv` wrapper first on PATH adds `--no-index --find-links /opt/wheels`, because uv ignores `UV_NO_INDEX` for script syncs. The job then needs no network.
6. **Model and dataset:** both are downloaded on the host and passed in as pool inputs: `--model base=...` and `--data data=...`, the latter for `PrimeIntellect/Reverse-Text-RL`.

Results on the Spark, run directly with the pool's exact docker flags. The run is asynchronous: rollouts used policies up to 5 steps stale.

| | policy v0 | v5 | v10 | v15 |
|---|---|---|---|---|
| eval reward (64 prompts, LCS) | 0.063 | 0.622 | 0.825 | 0.835 |

Through the pool (`pool run --image pool/rl-spike:prime-rl --gpus 1 --mem 64 --shm-gb 8 --tmp-gb 16 --pids 8192 --model base=qwen3-0.6b-reverse@sft --data data=<dataset dir> -- bash run.sh`), the job succeeded in 6.1 min. Eval reward went 0.161 → 0.629 → 0.826 → 0.846 at policy v0, v5, v10 and v15, and train reward 0.24 → 0.72. It fetched 2.23 GB of inputs in 2.1 s. The returned outputs were 2.9 GB, because the filesystem weight hand-over writes into `PRL_OUTPUT_DIR`. Point that at `/tmp` and copy only the logs, if the weights aren't wanted.

Direct run: train reward went from 0.30 at step 1 to 0.71 at step 15. The 15 steps took about 6 minutes after a 3-minute start. Peak memory was 21 GiB, with about 460 PIDs.

## verl or prime-rl for Demo B?

- **verl** drops into the NGC vLLM image (`--no-deps`, about 30 s on top of it) and ran unchanged inside the sandbox. It wants a writable Hydra folder and up to about 3,000 PIDs. It runs on Ray, and its rollouts are synchronous on-policy by default.
- **prime-rl** is asynchronous by design: orchestrator, trainer and inference are separate processes, with configurable staleness and file-based weight hand-over. That is the shape Demo B needs across slow links. It costs its own 47 GB CUDA stack, and needed four fixes to run offline in the sandbox, all recorded above.
