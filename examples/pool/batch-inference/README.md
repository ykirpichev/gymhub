# Demo A: batch inference through the pool (issue #87, M3)

The same batch of prompts goes through vLLM twice on the same machine:

1. **Standalone:** `docker run --gpus all --ipc=host` with vLLM, every prompt fed through one engine (`standalone.py`).
2. **Through the pool:** the prompts become queue tasks of 64 prompts each (`make_tasks.py`). An actor (`roller.py`, one replica here) keeps vLLM loaded inside the pool's sandbox (no network, read-only root, no capabilities), leases tasks through the agent's socket and reports token counts.

Both runs use the same engine settings (`common.py`: max model length 4096, 80% of GPU memory, 256 sequences in flight), the same prompts (made from a seed per task) and a fixed output length (256 tokens, `ignore_eos`), so they generate exactly the same number of tokens. Throughput excludes model loading in both: standalone times the generation; the pool run times from the first task's lease to the last result.

The roller leases the next task while the engine still works on the current one, so the GPU does not drain at task boundaries. A replica that finds no work for `IDLE` seconds exits, and an actor with `--min 0` then scales to zero.

## Run it

```
pool login <server> --code <code> --name me          # a pool with a GPU agent (pool machine add)
./demo.sh gpt-oss-20b@mxfp4 ~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/<rev> 32 64
./demo.sh qwen3-30b-a3b@fp8 ~/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/<rev> 32 64
```

`demo.sh` runs the standalone baseline, uploads the model as a pool ModelRef (once), fills a queue, starts the actor, waits for the queue to drain and prints tokens/s, the ratio to standalone and the cost per 1M tokens (`report.py`).

**Cost per 1M tokens** = (power × electricity price + hardware price spread over its life) / tokens per hour. The defaults are 140 W for the whole machine (`nvidia-smi` shows the GPU's own draw, measured below), $0.15/kWh and a $3,999 machine over 3 years; pass `--watts`, `--price-kwh`, `--hardware-usd` and `--years` to `report.py` for your own numbers.

## Results on the DGX Spark (GB10), vLLM 0.19 (`nvcr.io/nvidia/vllm:26.04-py3`)

32 tasks of 64 prompts, 256 output tokens each: 524,288 generated tokens per run. One machine, one actor replica.

| Model | Standalone | Through the pool | Pool / standalone | GPU power while generating | Cost per 1M tokens |
|---|---|---|---|---|---|
| gpt-oss-20b (MXFP4) | 2,948 tokens/s | 2,942 tokens/s | **0.998** | about 66 W (max 80 W) | $0.016 |
| Qwen3-30B-A3B (FP8) | 1,942 tokens/s | 2,001 tokens/s | **1.03** | about 50–70 W | $0.024 |

Both clear the 90% bar: once the engine is full, the queue costs nothing measurable. The pooled Qwen run edging out standalone is run-to-run variation (the timing windows start at slightly different moments). Cost uses 140 W for the whole machine, $0.15/kWh and $3,999 over 3 years: $0.173 an hour, almost all of it the hardware. At the measured GPU power alone (66 W) the figures barely move ($0.015 and $0.023), because electricity is about a tenth of the hourly cost.

Not included: the model's first load, once per replica and machine (gpt-oss-20b 111 s and Qwen3-30B-A3B 249 s here, including CUDA graph capture), and the first fetch of the ModelRef into the agent's cache.

The first try reached only 68% of standalone: the roller leased one task at a time and kept about 128 requests in flight instead of 256. Leasing as many tasks as the engine has room for fixed it.

Raw numbers: `results/` (the standalone JSON, the queue's results, and the report).
