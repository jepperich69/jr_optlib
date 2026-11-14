"""Route-free entropy-regularized assignment on a directed state graph.

The model is the recursive-logit / randomized-shortest-path KL projection.
For destination d, desirability z satisfies

    z_i = sum_(i,j) q_ij exp(-c_ij / tau) z_j,   z_d = 1.

Local probabilities follow from the same terms divided by z_i. Expected edge
flows then follow from the fundamental flow equation. Paths are never listed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class RecursiveLogitResult:
    edge_flows: np.ndarray
    policy: np.ndarray
    desirability: np.ndarray
    value: np.ndarray
    od_cost: np.ndarray
    spectral_radius: np.ndarray


def _as_index(x, node_to_idx):
    return int(x) if isinstance(x, (int, np.integer)) else node_to_idx[x]


def recursive_logit_assignment(
    nodes: Iterable,
    tails: np.ndarray,
    heads: np.ndarray,
    costs: np.ndarray,
    origins: Iterable,
    destinations: Iterable,
    demand: np.ndarray,
    tau: float,
    prior: np.ndarray | None = None,
) -> RecursiveLogitResult:
    """Assign OD demand without enumerating routes.

    ``prior`` contains reference action weights q_ij. If omitted, each state's
    outgoing edges receive equal probability. Explicit weights may sum to less
    than one after support pruning; preserving them avoids changing utilities
    when alternatives are removed. The arrays in the result have one row per
    OD pair and one column per edge or node.
    """
    nodes = list(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    if len(node_to_idx) != len(nodes):
        raise ValueError("nodes must be unique")
    tails = np.asarray(tails, dtype=int)
    heads = np.asarray(heads, dtype=int)
    costs = np.asarray(costs, dtype=float)
    demand = np.asarray(demand, dtype=float)
    origins = np.array([_as_index(x, node_to_idx) for x in origins], dtype=int)
    destinations = np.array([_as_index(x, node_to_idx) for x in destinations], dtype=int)
    n, m, n_od = len(nodes), len(costs), len(demand)
    if not (len(tails) == len(heads) == m):
        raise ValueError("tails, heads, and costs must have equal length")
    if not (len(origins) == len(destinations) == n_od):
        raise ValueError("origins, destinations, and demand must have equal length")
    if tau <= 0 or np.any(costs < 0) or np.any(demand < 0):
        raise ValueError("tau must be positive; costs and demand nonnegative")
    if np.any(tails < 0) or np.any(tails >= n) or np.any(heads < 0) or np.any(heads >= n):
        raise ValueError("edge endpoint outside nodes")

    if prior is None:
        outdegree = np.bincount(tails, minlength=n)
        prior = 1.0 / outdegree[tails]
    else:
        prior = np.asarray(prior, dtype=float)
        if prior.shape != (m,) or np.any(prior < 0):
            raise ValueError("prior must be a nonnegative vector with one value per edge")
        row_sum = np.bincount(tails, weights=prior, minlength=n)
        used = np.unique(tails)
        if np.any(row_sum[used] <= 0) or np.any(row_sum[used] > 1.0 + 1e-12):
            raise ValueError("prior row sums must lie in (0, 1]")

    kernel_edge = prior * np.exp(-costs / tau)
    edge_flows = np.zeros((n_od, m))
    policy = np.zeros((n_od, m))
    desirability = np.zeros((n_od, n))
    value = np.full((n_od, n), np.inf)
    od_cost = np.full(n_od, np.inf)
    radii = np.zeros(n_od)

    # Reverse reachability prevents singular blocks that cannot reach d.
    incoming = [[] for _ in range(n)]
    for e, (i, j) in enumerate(zip(tails, heads)):
        incoming[j].append((i, e))

    for r, (origin, dest, mass) in enumerate(zip(origins, destinations, demand)):
        if origin == dest:
            desirability[r, dest] = 1.0
            value[r, dest] = 0.0
            od_cost[r] = 0.0
            continue
        reachable = {dest}
        stack = [dest]
        while stack:
            j = stack.pop()
            for i, _ in incoming[j]:
                if i not in reachable:
                    reachable.add(i)
                    stack.append(i)
        if origin not in reachable:
            raise ValueError(f"origin {origin} cannot reach destination {dest}")
        transient = np.array(sorted(reachable - {dest}), dtype=int)
        pos = {node: k for k, node in enumerate(transient)}
        W = np.zeros((len(transient), len(transient)))
        b = np.zeros(len(transient))
        for e, (i, j) in enumerate(zip(tails, heads)):
            if i not in pos or j not in reachable:
                continue
            if j == dest:
                b[pos[i]] += kernel_edge[e]
            elif j in pos:
                W[pos[i], pos[j]] += kernel_edge[e]
        radius = float(np.max(np.abs(np.linalg.eigvals(W)))) if len(W) else 0.0
        radii[r] = radius
        if radius >= 1.0 - 1e-12:
            raise ValueError(f"improper recursive-logit kernel: spectral radius {radius:.6g}")
        z_t = np.linalg.solve(np.eye(len(W)) - W, b)
        z = np.zeros(n)
        z[dest] = 1.0
        z[transient] = z_t
        if z[origin] <= 0:
            raise ValueError("nonpositive origin desirability")
        desirability[r] = z
        positive = z > 0
        value[r, positive] = -tau * np.log(z[positive])
        od_cost[r] = value[r, origin]

        valid = (z[tails] > 0) & (z[heads] > 0) & (tails != dest)
        p = np.zeros(m)
        p[valid] = kernel_edge[valid] * z[heads[valid]] / z[tails[valid]]
        policy[r] = p

        P = np.zeros_like(W)
        for e, (i, j) in enumerate(zip(tails, heads)):
            if i in pos and j in pos:
                P[pos[i], pos[j]] += p[e]
        injection = np.zeros(len(transient))
        injection[pos[origin]] = mass
        visits = np.linalg.solve(np.eye(len(P)) - P.T, injection)
        for e, i in enumerate(tails):
            if i in pos:
                edge_flows[r, e] = visits[pos[i]] * p[e]

    return RecursiveLogitResult(edge_flows, policy, desirability, value, od_cost, radii)
