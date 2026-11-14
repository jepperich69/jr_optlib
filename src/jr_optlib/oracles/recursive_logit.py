"""Certificates for route-free recursive-logit assignment."""

from __future__ import annotations

import numpy as np

from jr_optlib.oracles.core import OracleResult


def certify_recursive_logit(result, tails, heads, costs, origins, destinations,
                            demand, tau, prior=None, tol=1e-9):
    tails = np.asarray(tails, dtype=int)
    heads = np.asarray(heads, dtype=int)
    costs = np.asarray(costs, dtype=float)
    demand = np.asarray(demand, dtype=float)
    origins = np.asarray(origins, dtype=int)
    destinations = np.asarray(destinations, dtype=int)
    n = result.desirability.shape[1]
    if prior is None:
        prior = 1.0 / np.bincount(tails, minlength=n)[tails]
    prior = np.asarray(prior, dtype=float)
    kernel = prior * np.exp(-costs / tau)

    bellman = 0.0
    policy_residual = 0.0
    conservation = 0.0
    for r, (origin, dest, mass) in enumerate(zip(origins, destinations, demand)):
        z = result.desirability[r]
        rhs = np.bincount(tails, weights=kernel * z[heads], minlength=n)
        active = (z > 0)
        active[dest] = False
        if np.any(active):
            bellman = max(bellman, float(np.max(np.abs(z[active] - rhs[active]))))
        rows = np.bincount(tails, weights=result.policy[r], minlength=n)
        if np.any(active):
            policy_residual = max(
                policy_residual, float(np.max(np.abs(rows[active] - 1.0)))
            )
        outflow = np.bincount(tails, weights=result.edge_flows[r], minlength=n)
        inflow = np.bincount(heads, weights=result.edge_flows[r], minlength=n)
        balance = outflow - inflow
        target = np.zeros(n)
        target[origin] += mass
        target[dest] -= mass
        conservation = max(conservation, float(np.max(np.abs(balance - target))))

    nonnegative = float(max(0.0, -np.min(result.edge_flows), -np.min(result.policy)))
    results = [
        OracleResult("recursive_logit_bellman", bellman <= tol, bellman, tol, True,
                     "desirability satisfies the linear soft-Bellman equation"),
        OracleResult("recursive_logit_policy", policy_residual <= tol,
                     policy_residual, tol, True,
                     "local action probabilities sum to one on active states"),
        OracleResult("recursive_logit_conservation", conservation <= tol,
                     conservation, tol, True,
                     "expected edge flows conserve every OD demand"),
        OracleResult("recursive_logit_nonnegative", nonnegative <= tol,
                     nonnegative, tol, True,
                     "policy and flows are nonnegative"),
    ]
    return results, all(x.passed for x in results)
