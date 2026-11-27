# Demo C2 in small: experts trained apart, then merged (issue #91, M7)

FlexOlmo-style expert merging on the pool. Each domain's expert is trained on its own machine, from a shared anchor, without seeing the other domain's data; the experts are then merged into a mixture-of-experts model with a learned router, and compared with a dense model of the same size trained on all the data.

| Step | Pool job | What it trains |
|---|---|---|
| anchor | any machine | a 3.4M-parameter byte-level transformer on both domains |
| expert-code | machine with `role=a` | from the anchor, only the FFNs, on Python source |
| expert-legal | machine with `role=b` | from the anchor, only the FFNs, on license and copyright texts |
| merge | any machine | an MoE: shared attention and embeddings from the anchor, three FFN experts per block (anchor, code, legal), top-2 routing; only the router is trained, on both domains |
| dense | any machine | the baseline: one FFN three times as wide (the same parameter count as the MoE), trained on both domains for as many steps as the whole pipeline (anchor + 2 experts + router) |

Models move between the jobs as job outputs passed on as inputs, so the machines need no link to each other, and an expert's training data never leaves its machine. The corpora come from any Linux machine (`make_corpus.py`: the Python standard library's source, and `/usr/share/doc/*/copyright`, deduplicated); nothing is downloaded.

## Run it

```
./demo.sh 3000 600        # steps per anchor and expert, router steps; ROLE_A/ROLE_B pick the two machines
```

## Results on the DGX Spark (GB10)

Held-out loss per byte (nats, lower is better), 3,000 steps per stage, 600 router steps:

| Model | Params | Code | Legal | Mean |
|---|---|---|---|---|
| anchor | 3.36M | 1.417 | 1.075 | 1.246 |
| expert: code | 3.36M | 1.212 | 1.591 | 1.401 |
| expert: legal | 3.36M | 1.843 | 0.839 | 1.341 |
| merged, router untrained (even mix) | 7.56M | 1.266 | 1.164 | 1.215 |
| **merged MoE** | 7.56M | 1.260 | 0.898 | **1.079** |
| dense baseline | 7.56M | 1.144 | 0.738 | **0.941** |

- Each expert gets better on its own domain and worse on the other: specialisation works.
- The merged model is better than the anchor and than either expert overall, and close to each expert on its domain (code 1.26 vs 1.21, legal 0.90 vs 0.84). Training only the router is what does it: the even mix of the same experts reaches 1.215.
- The dense model of the same size, trained on all the data for the same number of steps, is still better (0.941). FlexOlmo reports the same direction at scale: merging gives up some quality in exchange for training each expert on data that stays where it is, and for being able to add or remove an expert later.
- The two experts trained on two pool machines: two agents of the same DGX Spark (`spark-demo` with role=a, `spark-evil` with role=b), since no other machine was available to this pool. Nothing in the recipe depends on that.

Raw numbers: `results/demo-c2.json`.

Not done here (the rest of #91): C3 compressed pipelines, gossip rounds, the max-flow placement planner, a SkyPilot machine source.
