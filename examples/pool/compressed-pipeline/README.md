# C3 in small: a pipeline across machines with compressed activations (M7, #91)

Recipe C3 of #45 is pipeline parallelism across machines joined by slow links: each machine holds
consecutive layers, and the activations (forward) and their gradients (backward) cross the link
every step. Here a byte-level transformer (4 blocks of width 256, 3.36M parameters) is cut in two:
rank 0 holds the embedding and two blocks, rank 1 two blocks, the head and the loss. Both messages
can be compressed:

| `--compress` | wire format per value | notes |
|---|---|---|
| `none` | fp32 | baseline |
| `bf16` | 2 bytes | |
| `int8` | 1 byte + one fp32 scale per token | per-token absmax |
| `topk:F` | 4 bytes for the kept fraction F (bf16 value + int16 index) | error feedback on the gradients only |

`--compress-grad` picks a different one for the gradients.

```bash
pool run --gang 2 --network peers --gpus 1 --image nvcr.io/nvidia/pytorch:25.01-py3 \
    --link-bytes 2.5e9 -- python3 pipeline.py --compress int8     # two machines, a pool gang
python3 pipeline.py --compress int8                             # one machine, two processes
sh compare.sh                                                   # every mode, into results.jsonl
```

With `pool pool set <pool> placement=maxflow`, `--link-bytes` (bytes per run sent from rank 0 to
rank 1: 600 steps x 4.2 MB of fp32 activations) lets the planner put the two ranks on the pair of
machines with the fastest link. The numbers below are from one machine; a two-machine gang run is still to do.

## Measured on the DGX Spark (2026-11-24)

Two processes on the Spark's GPU over loopback (gloo), 600 steps of batch 16 x 256 bytes, AdamW,
held-out loss on 20 batches of Python source the model did not train on. Seconds per step at
100 Mbit/s and 1 Gbit/s are the measured compute plus bytes / link rate (no overlap). Raw: `results.jsonl`.

| activations / gradients | held-out loss | MB per step | vs fp32 | s/step loopback | s/step @ 1 Gbit/s | s/step @ 100 Mbit/s |
|---|---|---|---|---|---|---|
| none / none | 1.984 | 8.39 | - | 0.025 | 0.082 | 0.686 |
| bf16 / bf16 | 1.968 | 4.19 | 2.0x less | 0.026 | 0.050 | 0.352 |
| int8 / int8 | 1.985 | 2.13 | 3.9x less | 0.028 | 0.035 | 0.188 |
| int8 / topk:0.1 | 4.342 | 1.49 | 5.6x less | 0.029 | 0.031 | 0.138 |
| topk:0.25 / topk:0.25 | 3.676 | 2.10 | 4.0x less | 0.031 | 0.037 | 0.188 |
| topk:0.1 / topk:0.1 | 4.028 | 0.85 | 9.8x less | 0.030 | 0.027 | 0.088 |

- **int8 is free here**: the same loss as fp32 (1.985 vs 1.984) with 3.9x less traffic, which at
  100 Mbit/s turns 0.69 s per step into 0.19 s (3.6x faster). bf16 halves the traffic, also at no cost.
- **top-k is not**: keeping 25% or 10% of each token's values loses most of what the model learns
  (3.7-4.0 against 2.0), even though it is the smallest message. int8 activations with top-k
  gradients is worse still (4.34): error feedback on activation gradients carries one batch's
  leftover into the next batch's samples, which is noise, not delayed signal (it is sound for
  weight gradients, where DiLoCo-style methods use it).
- The first version of this script stored the int8 scale in fp16, and gradients (~1e-6 / 127)
  underflowed to zero: loss 3.47. The scale is fp32 now; worth remembering for any gradient codec.
- Compute is ~16-20 ms a step; the codecs add 1-4 ms on the GPU.

## Design notes (what a full C3 needs)

- **Overlap.** This sends one batch at a time and waits (no micro-batches), so the times above add
  compute and transfer. A GPipe or 1F1B schedule with m micro-batches hides the link when the
  transfer of one micro-batch takes less than its compute; compression is what makes that true on
  100 Mbit/s-1 Gbit/s home links.
- **Where compression hurts.** Quantising the residual stream is close to free (int8); dropping
  most of it (top-k) is not, for activations or for their gradients. Below int8, the next things to
  try are int4 with per-group scales, and a learned bottleneck at the cut (a low-rank projection
  trained with the model, as in SWARM parallelism), rather than sparsification.
- **Placement.** Stages should sit on the fastest links: the max-flow planner in `pool/planner.py`
  does this for gangs (`placement: maxflow`), and with `replicas` it can give a slow stage more
  machines (SWARM-style), which needs a router between stage replicas that this example does not have.
