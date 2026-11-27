"""Gossip rounds vs central averaging on a toy problem (M7, #91). numpy only, no pool needed.

Eight workers on two islands of four machines (10 Gbit/s inside an island, 100 Mbit/s between),
each with a skewed shard of a logistic-regression problem, run local SGD between rounds. After each
round the models are either averaged centrally (a reducer: what `mode: reduce` rounds do) or mixed
with the plan pool/gossip.py draws for `mode: gossip` rounds. Printed per mode:

  - loss of the workers' average model on all the data, after 5, 10, 20 and 40 rounds
  - the largest distance of a worker from that average (0 for central averaging)
  - model downloads per round, and how many of them cross the slow link between islands

    python examples/pool/gossip/toy.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from pool import gossip  # noqa: E402

N, DIM, ROUNDS, STEPS, LR = 8, 5, 40, 5, 0.3
MACHINES = [f"m{i}" for i in range(N)]
ISLAND = {m: ("a" if i < N // 2 else "b") for i, m in enumerate(MACHINES)}
LINKS = {(a, b): (10_000.0 if ISLAND[a] == ISLAND[b] else 100.0) for a in MACHINES for b in MACHINES if a != b}
NODES = [f"c{i}" for i in range(N)]
MACHINE_OF = dict(zip(NODES, MACHINES))


def data(seed=0):
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=DIM)
    shards = []
    for i in range(N):
        x = rng.normal(size=(200, DIM)) + (0.8 if i % 2 else -0.8)
        y = (x @ truth + 0.3 * rng.normal(size=200) > 0).astype(float)
        shards.append((x, y))
    return shards


def run(mode: str, bias: float = 0.5, seed: int = 0):
    shards = data(seed)
    xs = np.concatenate([s[0] for s in shards])
    ys = np.concatenate([s[1] for s in shards])

    def loss(w):
        p = np.clip(1 / (1 + np.exp(-xs @ w)), 1e-9, 1 - 1e-9)
        return float(-(ys * np.log(p) + (1 - ys) * np.log(1 - p)).mean())

    models = np.zeros((N, DIM))
    trace, downloads, crossing = {}, 0, 0
    for r in range(ROUNDS):
        for i, (x, y) in enumerate(shards):
            for _ in range(STEPS):
                models[i] -= LR * x.T @ (1 / (1 + np.exp(-x @ models[i])) - y) / len(y)
        if mode == "central":
            models[:] = models.mean(0)
            # reducer in island a fetches N contributions, then every worker fetches the result
            downloads += 2 * N
            crossing += 2 * sum(ISLAND[m] != "a" for m in MACHINES)
        else:
            plan = gossip.plan(NODES, MACHINE_OF, LINKS, 2, bias, seed=f"{seed}-{r}")
            models = np.array([[plan[a].get(b, 0.0) for b in NODES] for a in NODES]) @ models
            moved = gossip.traffic(plan, MACHINE_OF, ISLAND)
            downloads += moved["fetched"]
            crossing += moved["cross_island"]
        if r + 1 in (5, 10, 20, 40):
            trace[r + 1] = (loss(models.mean(0)), float(np.linalg.norm(models - models.mean(0), axis=1).max()))
    return trace, downloads / ROUNDS, crossing / ROUNDS


def main():
    print(f"{N} workers, 2 islands, {STEPS} local steps per round; loss of the average model (max distance from it)")
    rows = [("central", run("central"))] + [(f"gossip bias={b}", run("gossip", b)) for b in (0.0, 0.5, 1.0)]
    print(f"{'mode':<18}" + "".join(f"{'round ' + str(k):>20}" for k in (5, 10, 20, 40)) + f"{'downloads/rd':>14}{'cross/rd':>10}")
    for name, (trace, down, cross) in rows:
        print(f"{name:<18}" + "".join(f"{trace[k][0]:>11.4f} ({trace[k][1]:.3f})" for k in (5, 10, 20, 40))
              + f"{down:>14.1f}{cross:>10.1f}")


if __name__ == "__main__":
    main()
