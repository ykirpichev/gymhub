# Hierarchical DiLoCo of OLMoE with torchft (Demo C1 in small)

Issue #89 (M5), recipe C1 from #45. Islands are pool jobs. Inside an island the ranks would sync
every step. Between islands, **torchft**'s DiLoCo syncs every `H` steps over a slow link. The pool
does not reduce anything itself here: a round with `reducer: torchft` (`pool_reducers.TorchftRounds`)
carries the lighthouse's address, the quorum and the deadline, and records every outer step.

- **Islands.** Each island is a gang job with `network: peers`, so the islands can reach each other.
  Gloo and torchft's manager bind the island's own address (`--bind`).
- **Rendezvous through the pool.** The first island to open the round on `--ref` starts
  `torchft_lighthouse` and writes its address into the round. The other islands read it from the
  round. Each island gets a `torchft.Manager` with `min_replica_size` set to the round's min and the
  round's deadline as its quorum timeout.
- **Reduction by torchft.** `torchft.local_sgd.DiLoCo` all-reduces the pseudo-gradients directly
  between the islands, then takes the outer SGD step (lr 0.7).
- **Ledger in the pool.** After each committed outer step, every island contributes a small record
  (its tokens as the weight). The island with participating rank 0 closes the round. The record
  becomes the ref's next version (`pool ref log c1/<run>` shows one version per outer step), ready
  for the checked tier's reports (M6). This bookkeeping runs in a background thread, so training
  never waits for it.

## The setup, and what is scaled down

Only one machine was available: a DGX Spark (GB10, one GPU). The Jetson was not reachable.

- **Two islands on one GPU.** Two pool jobs share the GPU (`share_gpu`). Each island is one rank,
  because there is only one GPU, so the step-by-step sync inside an island is not exercised here.
- **The outer link.** The islands bind 127.0.0.2 and 127.0.0.3. `shape.sh` puts a tc netem qdisc on
  `lo` that shapes only traffic from or to those two addresses: the given rate each way plus 5 ms one
  way. `linktest.py` checks it: 97 Mbit/s through a 100 Mbit link, while other loopback traffic is
  unshaped.
- **The model.** The OLMoE-1B-7B architecture (`OlmoeForCausalLM`: 64 experts, 8 active, its
  tokenizer and vocabulary of 50,304), scaled down to fit two islands and a baseline on one GB10 in
  minutes: 8 layers, width 512, expert width 256. That is 262M parameters, about 85M of them active.
- **The data.** WikiText-103, the first 12M tokens, tokenized with OLMoE's tokenizer
  (`make_data.py`).
- **Training.** Each island runs 10 outer steps of H = 50 inner steps (AdamW, lr 1e-3, 50 warm-up
  steps) at 8 × 256 tokens per step. The pseudo-gradients are all-reduced in fp32, 1,046 MB per sync.
- **The single-island baseline.** The same 500 steps at 16 × 256 tokens per step. That equals the
  two islands' tokens, and it is what one island of both GPUs syncing every step would do.

```bash
docker build -t pool/diloco-torchft:0.3.1 .        # torchft 0.3.1 built from source on the NGC vLLM image
python3 make_data.py data                           # or inside that image
./demo.sh single                                    # the baseline
./demo.sh islands none                              # two islands, loopback unshaped
./demo.sh islands 1gbit                             # two islands, 1 Gbit/s each way between them
./demo.sh islands 100mbit
python3 report.py results/*.json
```

## Measured (2026-11-24, DGX Spark, one run each)

| Run | Tokens | Val loss | vs single island | Utilisation | Outer sync, median |
|---|---|---|---|---|---|
| single island, 16 × 256 per step | 2.05M | **5.741** | – | 100% | – |
| 2 islands, loopback unshaped | 2.05M | 6.005 | +4.6% | 96–97% | 1.2–1.5 s |
| 2 islands, 1 Gbit/s link | 2.05M | 6.005 | +4.6% | **57%** | 25.3 s (10 syncs) |
| 2 islands, 100 Mbit/s link | – | (6.005) | – | **not measured** (see below) | – |

- **Utilisation** is 1 minus the time an island spends blocked in outer syncs, divided by its
  training time (evaluation excluded). Both islands run on one GPU, so a step takes 0.67 s instead
  of 0.18 s on its own. Each outer step is therefore about 34 s of compute.
- **The link does not change the result.** The loss is identical with and without shaping (6.0051
  in both runs), because the islands are deterministic. The link only changes how long the syncs
  take. A 100 Mbit run would therefore reach the same loss.
- **Gloo does not fill the link.** At 1 Gbit/s, an fp32 sync of 1,046 MB took 25 s, an effective
  rate of 335 Mbit/s. That is a third of the link (5 ms one-way delay, one TCP stream per pair).
- **100 Mbit/s (not measured).** A 100 Mbit/s sync needs at least 84 s on the wire. That puts
  utilisation at 29% or below at H = 50. The run was stopped before it started: shaping changes a
  qdisc on the host's shared `lo`, and the machine is shared with other agents' jobs, so it waits
  for approval.
- **Loss compared with the single island.** The gap is +4.6% at 2M tokens. It was +3.8% at 1M tokens
  and +2.5% at 0.2M tokens. This is a short run from random weights, with no outer momentum and
  only 10 outer steps. DiLoCo papers report the gap closing over longer runs with outer Nesterov
  momentum. It is "within a few %" only at the loose end.

### Projection for the full OLMoE-1B-7B

The full model was measured on the GB10 with random weights, in bf16, forward and backward, no
optimizer (`train.py --mode bench --model full`): 6.92B parameters, **0.835 s per 8 × 256-token
step**, 27.8 GB peak. It does not fit twice on one GB10 with AdamW state, which is why the runs
above use the scaled-down model.

A bf16 pseudo-gradient is 13.8 GB per sync, which takes at least 111 s on the wire at 1 Gbit/s and
1,107 s at 100 Mbit/s. With H = 500 (418 s of compute per outer step), utilisation comes to about:

| Link | Bound by the wire | Gloo at the measured 1/3 of the link |
|---|---|---|
| 1 Gbit/s | 79% | 56% |
| 100 Mbit/s | 27% | 11% |

These figures are arithmetic from the measured step time, not runs. Streaming DiLoCo (`--fragments`,
`--sync-delay`) overlaps a fragment's sync with inner steps. torchft's fp8 quantised all-reduce
(`--quantize`) would cut the bytes by 4×, but it crashed on the GB10 (exit 139 in both islands), so
it is not in the table.

## Files

- `train.py`: the islands (`--mode island`), the baseline (`--mode single`) and the step-time
  benchmark (`--mode bench`).
- `demo.sh`: submits the pool jobs, shapes the link and collects `results/`.
- `shape.sh`, `linktest.py`: the tc link between the two loopback addresses, and its check.
- `Dockerfile`: torchft 0.3.1. It is sdist-only on PyPI, so the Rust core is built from source with
  `protobuf-compiler` and `libprotobuf-dev`.
- `results/*.json`: the metrics of the runs above.
