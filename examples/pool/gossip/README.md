# Gossip rounds (M7, #91)

A round opened with `mode: gossip` has no reducer. When it is ready, the server draws a mixing plan
over the pool's measured link graph and closes the round; each worker then averages its own
contribution with a few sampled peers' and keeps training from that. The ref is only the shared
starting point and never moves. See `pool/gossip.py` for the plan (bandwidth-biased peer sampling,
Metropolis-Hastings weights: symmetric and doubly stochastic, so the average of the models is kept).

```bash
pool round open weights --quorum 8 --repeat --mode gossip --fanout 2 --bias 0.5
```

A worker, per round:

```python
current = pool_job.current_round("weights", after=seen, wait=30); seen = current["id"]
pool_job.contribute("weights", mine, weight=tokens, base_version=current["base_version"], round=seen)
mix = pool_job.mix_round(seen, wait=30)              # None until the round closed
folders = [(mine if i["self"] else pool_job.fetch_artifact(i["artifact"]), i["weight"], False) for i in mix["items"]]
pool_reducers.weighted_mean(folders, merged)        # weights already add up to 1
```

## Toy comparison with central averaging

`python toy.py` (numpy only): 8 workers on two islands of 4 (10 Gbit/s inside, 100 Mbit/s between),
heterogeneous logistic-regression shards, 5 local SGD steps per round. Loss of the workers' average
model on all the data, and in brackets the largest distance of a worker from that average:

```
mode                           round 5            round 10            round 20            round 40  downloads/rd  cross/rd
central                0.3574 (0.000)     0.3021 (0.000)     0.2664 (0.000)     0.2466 (0.000)          16.0       8.0
gossip bias=0.0        0.3575 (0.057)     0.3021 (0.038)     0.2664 (0.051)     0.2466 (0.047)          27.0      14.9
gossip bias=0.5        0.3574 (0.055)     0.3021 (0.099)     0.2663 (0.072)     0.2466 (0.058)          24.3       4.8
gossip bias=1.0        0.3574 (0.049)     0.3021 (0.059)     0.2663 (0.083)     0.2465 (0.074)          22.1       0.8
```

- The average model follows central averaging to 4 decimals; the workers stay within ~0.05-0.08 of it.
- With fanout 2 a worker fetches ~3 peers, so gossip moves more models in all than a reducer (24 vs 16
  per round), but with `bias` 0.5 it sends 40% fewer of them over the slow link between islands (4.8 vs
  8), and none goes through a single machine.
- The price of the bias: on pure averaging (tests/test_pool_gossip.py) the workers agree to 1e-6 in
  30 rounds with bias 0, but need ~60 rounds to agree to 1e-3 with bias 0.5. Bias 1.0 nearly stops
  crossing between islands, and the islands then drift apart.

Contribution weights (tokens) are passed to workers but not used by the plan; weighted (push-sum)
gossip is left for later.
