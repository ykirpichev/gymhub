# DiLoCo on a DGX Spark and a Jetson over Wi‑Fi

A pool gang job (issue #45, recipe C1 in small). Two unequal machines train one model together over a
home Wi‑Fi link. They find each other through the pool server and then exchange updates directly.

- **Discovery.** Each agent reports its LAN address to the server. `pool run --gang 2` places one
  rank per machine. Each rank waits on its machine until both have one (the rendezvous), then starts
  with `RANK`, `WORLD_SIZE`, `MASTER_ADDR` and `MASTER_PORT`.
- **Communication.** `torch.distributed` over gloo, directly between the two containers
  (`--network peers`, allowed by each machine's owner). The pool server is not in the data path.
- **Algorithm.** Each rank trains its own copy with AdamW for a fixed 30 s of wall time (the inner
  loop). Then the ranks average how far each moved from the shared weights (the pseudo-gradient),
  weighted by the tokens each processed, and take an outer SGD step. Rounds are timed, not counted,
  so the Spark never waits for the Jetson: it does about 5× the steps, and its update weighs about 5× as much.
- **Traffic.** One fp16 message of the model's size (6.7 MB) per rank per round.

## Run it

```bash
python3 make_corpus.py corpus/          # ~12 MB of the Python standard library's source, as bytes
pool run --gang 2 --network peers --gpus 1 --mem 4 \
    --image nvcr.io/nvidia/vllm:26.04-py3 --image-for gpu=tegra=<a Jetson image with PyTorch> \
    --data corpus=./corpus -- python3 diloco.py --rounds 10 --round-seconds 30
```

Without a gang it trains alone (`python3 diloco.py`), and `--outer` keeps the outer loop with one worker.

## Measured (2026-11-24)

Setup: a 3.4M-parameter byte-level transformer, 4 layers with width 256, trained on 11.8 MB of Python source.
- Spark: DGX Spark (GB10), torch 2.12, in the NGC vLLM 26.04 image.
- Jetson: Jetson Orin 8 GB, torch 2.6, in an L4T r36.4 image.
- Link: both machines on home Wi‑Fi. The agents measured 150 Mbit/s with a 12 ms round trip, and gloo
  all-reduce ran at about 55 Mbit/s.

Each run is 10 rounds of 30 s, the same wall time. Validation loss is on held-out bytes (lower is better):

| Run | Machines | Steps per round | Final val loss | Best |
|---|---|---|---|---|
| plain AdamW | Spark | 2,470 | 0.9705 | 0.9657 |
| DiLoCo, 1 worker (control) | Spark | 2,462 | 0.9324 | 0.9324 |
| DiLoCo, 2 workers | Spark + Jetson | 2,465 + 514 | **0.9251** | 0.9225 |

- **Sync cost:** 1.0–3.2 s per 30 s round (3–10%), mostly the Wi‑Fi transfer.
- **Shares:** the Jetson processed 17% of the tokens, and its update was weighted to match.
- **Data path:** the corpus reached the Jetson from the Spark's agent (peer to peer, 0.5 s), not from the server.
- **Mixed torch versions:** the gloo ranks ran torch 2.12 and torch 2.6 and still worked together.

How to read this: these are single runs. Most of the gain over plain AdamW (0.9705 → 0.9324) comes
from the outer step itself: with a small corpus seen 17 times, the outer step acts like weight
averaging and keeps the model from drifting. The Jetson's 17% of the tokens adds a smaller gain on
top (0.9324 → 0.9251, about 0.8%). That is in line with how much it contributes, but one run each is
not enough to call it significant. What this example shows is the mechanics: two unequal machines
found each other through the pool, synced over Wi‑Fi at 3–10% overhead, and trained one model
together without the fast machine waiting for the slow one.

## Outer momentum

DiLoCo uses Nesterov momentum 0.9 for the outer step, but it starts from a warmed-up model and runs
hundreds of rounds. From random weights and only a few rounds, momentum overshoots. A one-worker
sweep on the Spark (8 rounds of 15 s, five runs sharing the GPU) gave these final val losses:

| Outer lr / momentum | Final val loss |
|---|---|
| plain AdamW (no outer step) | 1.190 |
| 0.7 / 0.0 | 1.193 |
| 0.5 / 0.8 | 1.267 |
| 1.0 / 0.5 | 1.345 |
| 0.7 / 0.9 | 1.357 |

So the default here is lr 0.7 with momentum 0. Pass `--outer-momentum 0.9` for long runs from a warm start.

## A poisoned participant (the checked tier, M6)

In a pool wider than one team, one rank may send a bad outer update. `--robust` turns on update sanity.
Every rank all-gathers the updates instead of all-reducing them, and every rank leaves out the same ones,
so they stay in step:

- **Outliers.** `pool_reducers.outliers` compares each update with the element-wise median of the others.
  It drops the update when its norm is more than 3× the median's, or when its cosine with the median is below 0.1.
- **Reputation.** Rank 0 reports every machine's update to the pool (`pool_job.report`, kind `update`).
  In a checked pool, a machine with 2 failed checks is excluded. From then on every rank leaves that
  machine's rank out, whatever its update looks like. `pool machine trust <name>` clears the exclusion.

`--poison scale|flip|random --poison-machine NAME` makes the rank on machine NAME send a poisoned update:
scaled by 10, sign-flipped, or replaced by noise of the same norm. `poison.sh` runs each attack twice,
without and with the defence:

```bash
pool pool set default trust=checked
./poison.sh          # a gang of 4 on machines of a checked pool, one of them named "evil"
```

### Measured (2026-11-24)

Setup: a private pool with four agents on the one DGX Spark (`honest-a`, `honest-b`, `honest-c`, `evil`),
one rank each. Each run is 8 rounds of 150 inner steps per rank. Rounds are a fixed number of steps here,
so runs compare at equal work. Same model and corpus as above. Final validation loss, lower is better:

| Attack by `evil` | Without the defence | With `--robust` | Rounds `evil` was left out | `evil` excluded |
|---|---|---|---|---|
| none (4 honest ranks) | 1.3358 | 1.3357 | 0 of 8 | no |
| scale ×10 | **2.9421** (never below 2.87 after round 1) | 1.3412 | 8 of 8 | after round 2 |
| sign flip | 1.6537 | 1.3412 | 8 of 8 | after round 2 |
| random, same norm | 1.4334 | 1.3412 | 8 of 8 | after round 2 |

- **What caught each attack:** the scaled update was caught by its norm (10.4× the others' median; its
  cosine was a normal 0.91). The flipped update was caught by its cosine (−0.91), and the noise also by
  its cosine (0.00).
- **No false positives:** honest updates had norms 0.9–1.2× the median and cosines 0.66–0.94 in every
  run. With four honest ranks nothing was left out, and the loss matches the plain all-reduce.
- **With the defence, all three attacks end at the same 1.3412.** From round 1 on, the run is the three
  honest ranks alone. That is 0.005 above four honest ranks: the cost of losing a quarter of the updates.
- **Cost:** with an all-gather, every rank receives all `world` updates instead of one all-reduced
  update. Here that took 0.8–1.0 s per round, against 0.02 s for the all-reduce. On a slow link, the
  sync moves the gang size times the traffic of the plain sync.

Per-round losses and checks: `results/poisoned-updates.json`.

Limits:
- A median needs an honest majority. Three updates are the minimum, and with exactly three the reference
  median includes the suspect update.
- A rank that lies only a little passes: an update 2× too large, or slightly rotated.
- The token counts that weight the updates are taken on trust.
