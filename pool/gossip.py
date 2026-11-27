"""Gossip rounds (M7, #45 section 8): averaging with sampled peers instead of a central reducer.

A round opened with `mode: gossip` collects contributions like any round, but nobody claims it.
When it becomes ready, the server draws a mixing plan and closes the round at once: each
contributor gets a short list of the other contributions to average its own with, and weights for
them. Every worker mixes on its own machine (pool_job.mix_round) and goes on training from its own
mixed model, so there is no single merged model and no reducer that downloads every contribution.
The ref is the shared starting point and does not move; with `repeat`, the next gossip round opens
on the same version.

The plan, for contributions on machines m_1..m_n:

  1. each contributor samples `fanout` partners without replacement, with probability proportional
     to (measured Mbit/s of the link between their machines) ** `bias`. bias 0 is uniform; the
     default 0.5 prefers fast links (the same island) but still crosses slow ones now and then,
     which keeps the graph connected from round to round. Unmeasured links count as GUESS_MBPS, two
     contributions on one machine as SAME_MACHINE_MBPS.
  2. the sampled pairs become undirected edges, and the weights are Metropolis-Hastings ones:
         w_ij = 1 / (1 + max(deg_i, deg_j))   for each edge,   w_ii = 1 - sum_j w_ij
     The matrix is symmetric and doubly stochastic, so mixing keeps the average of the models (the
     quantity central averaging computes) and the workers' disagreement shrinks every round in
     which the sampled graph is connected (NoLoCo, Factored-Gossip-DiLoCo; Boyd et al., 2006).

Contribution weights (tokens) are not used by the plan: they are given to the workers, and push-sum
weighted gossip is left for later. Standard library only.
"""

import random

GUESS_MBPS = 100.0
SAME_MACHINE_MBPS = 1e6
DEFAULT_FANOUT = 2
DEFAULT_BIAS = 0.5


def link_speed(links: dict, a: str | None, b: str | None) -> float:
    """Mbit/s between two machines: the measured link (either direction; the faster when both are)."""
    if a is not None and a == b:
        return SAME_MACHINE_MBPS
    found = [links[k] for k in ((a, b), (b, a)) if k in links]
    return max(found) if found else GUESS_MBPS


def sample_edges(nodes: list[str], machine_of: dict, links: dict, fanout: int = DEFAULT_FANOUT,
                 bias: float = DEFAULT_BIAS, seed: str | int = 0) -> set[tuple[str, str]]:
    """Undirected edges (a, b) with a < b: each node picks `fanout` partners, faster links likelier."""
    rng = random.Random(str(seed))
    edges: set[tuple[str, str]] = set()
    ordered = sorted(nodes)
    for node in ordered:
        others = [o for o in ordered if o != node]
        weights = [link_speed(links, machine_of.get(node), machine_of.get(o)) ** bias for o in others]
        for _ in range(min(fanout, len(others))):
            pick = rng.choices(range(len(others)), weights=weights)[0]
            other = others.pop(pick)
            weights.pop(pick)
            edges.add((min(node, other), max(node, other)))
    return edges


def metropolis(nodes: list[str], edges: set[tuple[str, str]]) -> dict[str, dict[str, float]]:
    """Symmetric, doubly stochastic mixing weights on a graph: {node: {node or peer: weight}}."""
    degree = {n: 0 for n in nodes}
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    mix: dict[str, dict[str, float]] = {n: {} for n in nodes}
    for a, b in edges:
        w = 1.0 / (1 + max(degree[a], degree[b]))
        mix[a][b] = w
        mix[b][a] = w
    for n in nodes:
        mix[n][n] = 1.0 - sum(mix[n].values())
    return mix


def plan(nodes: list[str], machine_of: dict, links: dict, fanout: int = DEFAULT_FANOUT, bias: float = DEFAULT_BIAS,
         seed: str | int = 0) -> dict[str, dict[str, float]]:
    """The mixing plan of one gossip round: {contribution: {contribution: weight}}, self included."""
    return metropolis(sorted(nodes), sample_edges(nodes, machine_of, links, fanout, bias, seed))


def traffic(mix: dict[str, dict[str, float]], machine_of: dict, island_of: dict | None = None) -> dict:
    """Downloads a plan causes: contributions fetched in all, and how many cross islands."""
    fetched = crossing = 0
    for node, row in mix.items():
        for peer in row:
            if peer == node:
                continue
            fetched += 1
            if island_of is not None and island_of.get(machine_of.get(node)) != island_of.get(machine_of.get(peer)):
                crossing += 1
    return {"fetched": fetched, "cross_island": crossing}
