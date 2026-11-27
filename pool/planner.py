"""A max-flow placement planner (M7, #45 section 6.2 "a Helix-style max-flow planner"). Standard library only.

The greedy scheduler places one job at a time on the machine that asks. For a pipeline (a gang whose
rank r sends its activations to rank r+1, or stages that can be replicated) what matters is the
throughput of the whole chain, which depends on which machines sit next to each other. The planner:

  - models a placement as a flow network (Helix, Mei et al. 2024, in small):
        source -> stage 0 machines -> stage 1 machines -> ... -> sink
    each machine m of stage s is a node split in two with capacity rate(m) / flops(s) (runs per
    second its compute allows), and each pair (a of stage s, b of stage s+1) is an edge with capacity
    bandwidth(a, b) / bytes(s) (runs per second the link allows). The max flow is the throughput
    of the placement; the min cut names its bottleneck (a machine's compute or a link).
  - searches placements: with one machine per stage (a gang: ranks are stages) every ordered choice
    of machines when there are few, else a greedy start improved by swaps; with `replicas`, each
    machine joins the stage where it adds the most flow (or none), then single moves while they help.

`flops(s)` and `bytes(s)` may be per step or for the whole job: only ratios matter. Units: rates in
FLOP/s, bandwidths in bytes/s, throughput in runs (steps, or whole jobs) per second.
"""

import itertools
import random
from collections import deque

BIG = 1e30  # "unbounded" capacity; finite, so residual arithmetic stays exact enough
EPS = 1e-12
EXHAUSTIVE_LIMIT = 20_000


# ---- max flow and min cut ----------------------------------------------------------------------------


def max_flow(edges: dict, source, sink) -> tuple[float, dict, set]:
    """Edmonds-Karp. `edges` maps (u, v) to a capacity. Returns (value, flow per edge, the source
    side of a minimum cut)."""
    residual: dict = {}
    for (u, v), cap in edges.items():
        residual.setdefault(u, {})
        residual.setdefault(v, {})
        residual[u][v] = residual[u].get(v, 0.0) + float(cap)
        residual[v].setdefault(u, 0.0)
    residual.setdefault(source, {})
    residual.setdefault(sink, {})
    value = 0.0
    while True:
        parent = {source: None}
        queue = deque([source])
        while queue and sink not in parent:
            u = queue.popleft()
            for v, cap in residual[u].items():
                if cap > EPS and v not in parent:
                    parent[v] = u
                    queue.append(v)
        if sink not in parent:
            break
        path, v = [], sink
        while parent[v] is not None:
            path.append((parent[v], v))
            v = parent[v]
        push = min(residual[u][v] for u, v in path)
        for u, v in path:
            residual[u][v] -= push
            residual[v][u] += push
        value += push
        if value >= BIG / 2:
            raise ValueError("unbounded flow: every edge from source to sink is unlimited")
    flows = {(u, v): max(0.0, float(cap) - residual[u][v]) for (u, v), cap in edges.items()}
    return value, flows, set(parent)


# ---- pipelines ---------------------------------------------------------------------------------------


def _node_cap(rate: float, flops: float) -> float:
    return BIG if not flops else (rate / flops if rate > 0 else 0.0)


def _link_cap(bandwidth: float, size: float) -> float:
    return BIG if not size else (bandwidth / size if bandwidth > 0 else 0.0)


def pipeline_edges(groups: list[list[str]], stages: list[dict], rate: dict, bandwidth) -> dict:
    """The flow network of a placement: `groups[s]` are the machines of stage s; `stages[s]` is
    {"flops", "bytes"} (bytes stage s sends on to stage s+1); `rate[m]` FLOP/s; bandwidth(a, b) bytes/s."""
    edges = {}
    for s, group in enumerate(groups):
        for m in group:
            edges[((m, s, "in"), (m, s, "out"))] = _node_cap(rate.get(m, 0.0), stages[s].get("flops", 0))
            if s == 0:
                edges[("source", (m, 0, "in"))] = BIG
            if s == len(groups) - 1:
                edges[((m, s, "out"), "sink")] = BIG
            else:
                for b in groups[s + 1]:
                    edges[((m, s, "out"), (b, s + 1, "in"))] = _link_cap(bandwidth(m, b), stages[s].get("bytes", 0))
    return edges


def throughput(groups: list[list[str]], stages: list[dict], rate: dict, bandwidth) -> dict:
    """Throughput (runs/s) of a placement and its bottleneck: the edges of a minimum cut."""
    if len(groups) != len(stages) or any(not g for g in groups):
        return {"throughput": 0.0, "bottleneck": ["a stage has no machine"]}
    edges = pipeline_edges(groups, stages, rate, bandwidth)
    try:
        value, _flows, side = max_flow(edges, "source", "sink")
    except ValueError:
        return {"throughput": BIG, "bottleneck": []}
    cut = []
    for (u, v), cap in edges.items():
        if u in side and v not in side and cap < BIG:
            if u[0] == v[0]:
                cut.append(f"compute of {u[0]} (stage {u[1]})")
            else:
                cut.append(f"link {u[0]} -> {v[0]} (stage {u[1]} -> {v[1]})")
    return {"throughput": value, "bottleneck": sorted(cut)}


def _capacities(order: tuple, stages: list[dict], rate: dict, bandwidth) -> list[float]:
    """With one machine per stage: the capacities along the chain (its max flow is the smallest)."""
    caps = []
    for s, m in enumerate(order):
        caps.append(_node_cap(rate.get(m, 0.0), stages[s].get("flops", 0)))
        if s + 1 < len(order):
            caps.append(_link_cap(bandwidth(m, order[s + 1]), stages[s].get("bytes", 0)))
    return caps


def _chain(order: tuple, stages: list[dict], rate: dict, bandwidth) -> float:
    return min(_capacities(order, stages, rate, bandwidth))


def plan(stages: list[dict], machines: list[str], rate: dict, bandwidth, replicas: bool = False,
         limit: int = EXHAUSTIVE_LIMIT) -> dict | None:
    """The best placement found: {"stages": [[machine, ...] per stage], "throughput", "bottleneck",
    "searched"}; None when there are fewer machines than stages."""
    machines = sorted(dict.fromkeys(machines))
    n = len(stages)
    if n == 0 or len(machines) < n:
        return None
    count = 1
    for k in range(len(machines) - n + 1, len(machines) + 1):
        count *= k
    if count <= limit:  # every ordered choice of n machines: the optimum for one machine per stage
        best = max(itertools.permutations(machines, n), key=lambda o: (_chain(o, stages, rate, bandwidth), [-machines.index(m) for m in o]))
        groups, searched = [[m] for m in best], "exhaustive"
    else:
        groups, searched = _improve_chain(stages, machines, rate, bandwidth), "local"
    if replicas:
        groups = _add_replicas(groups, stages, machines, rate, bandwidth)
        searched += "+replicas"
    found = throughput(groups, stages, rate, bandwidth)
    return {"stages": groups, "throughput": found["throughput"], "bottleneck": found["bottleneck"], "searched": searched}


def _improve_chain(stages, machines, rate, bandwidth, restarts: int = 16) -> list[list[str]]:
    """Local search for one machine per stage: from a greedy start (heaviest stage on the fastest
    machine) and from `restarts` random orders, swap two stages or bring in an unused machine while
    the chain gets better. "Better" compares the sorted capacities, smallest first, so a move that
    widens the second-narrowest point counts too: a plain min is flat and stops the search early."""
    by_speed = sorted(machines, key=lambda m: -rate.get(m, 0.0))
    heavy = sorted(range(len(stages)), key=lambda s: -stages[s].get("flops", 0))
    greedy = [None] * len(stages)
    for s, m in zip(heavy, by_speed):
        greedy[s] = m
    rng = random.Random(0)
    starts = [greedy] + [rng.sample(machines, len(stages)) for _ in range(restarts)]

    def score(order):
        return sorted(_capacities(tuple(order), stages, rate, bandwidth))

    best, best_score = None, None
    for order in starts:
        current = score(order)
        improved = True
        while improved:
            improved = False
            for i in range(len(order)):
                for m in machines:
                    trial = list(order)
                    if m in trial:  # swap two stages
                        j = trial.index(m)
                        trial[i], trial[j] = trial[j], trial[i]
                    else:           # bring in an unused machine
                        trial[i] = m
                    value = score(trial)
                    if value > current:
                        order, current, improved = trial, value, True
        if best_score is None or current > best_score:
            best, best_score = order, current
    return [[m] for m in best]


def _add_replicas(groups, stages, machines, rate, bandwidth) -> list[list[str]]:
    """Each unused machine joins the stage where it adds the most flow; then single moves while they help."""
    groups = [list(g) for g in groups]

    def value(gs):
        return throughput(gs, stages, rate, bandwidth)["throughput"]

    def stage_capacity(gs, s):
        return sum(_node_cap(rate.get(m, 0.0), stages[s].get("flops", 0)) for m in gs[s])

    current = value(groups)
    for m in sorted((m for m in machines if not any(m in g for g in groups)), key=lambda m: -rate.get(m, 0.0)):
        options = []
        for s in range(len(groups)):
            trial = [list(g) for g in groups]
            trial[s].append(m)
            # on a tie, the stage with the least compute: two such additions may lift the bottleneck
            options.append((value(trial), -stage_capacity(groups, s), s))
        gain, _, s = max(options)
        if gain >= current * (1 - 1e-9):
            groups[s].append(m)
            current = max(current, gain)
    for _ in range(4 * len(machines)):  # move one machine to another stage while it helps
        best = None
        for s, group in enumerate(groups):
            if len(group) < 2:
                continue
            for m in group:
                for t in range(len(groups)):
                    if t == s:
                        continue
                    trial = [list(g) for g in groups]
                    trial[s].remove(m)
                    trial[t].append(m)
                    v = value(trial)
                    if v > current * (1 + 1e-9) and (best is None or v > best[0]):
                        best = (v, trial)
        if best is None:
            break
        current, groups = best
    for s in range(len(groups)):  # drop replicas the throughput does not need
        for m in sorted(groups[s], key=lambda m: rate.get(m, 0.0)):
            if len(groups[s]) > 1:
                trial = [list(g) for g in groups]
                trial[s].remove(m)
                if value(trial) >= current * (1 - 1e-9):
                    groups = trial
    return [sorted(g) for g in groups]
