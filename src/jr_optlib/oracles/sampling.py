# -*- coding: utf-8 -*-
"""Oracles for verifying sampling and MCMC algorithms."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Hashable, Iterable, List, Sequence, Tuple

import numpy as np

from jr_optlib.oracles.core import OracleResult


def certify_detailed_balance(
    feasible_states: Iterable[Hashable],
    energy_fn: Callable[[Hashable], float],
    run_chain_fn: Callable[[Hashable, float, int], Sequence[Hashable]],
    tau: float,
    n_steps: int,
    tol_tv: float = 0.03,
) -> Tuple[List[OracleResult], bool]:
    """Verify that an MCMC chain converges to the exact Boltzmann distribution.
    
    Args:
        feasible_states: An exhaustive iterable of all states in the small test space.
        energy_fn: Function mapping a state to its energy (cost).
        run_chain_fn: Function `(init_state, tau, n_steps) -> list of visited states`.
        tau: The temperature parameter.
        n_steps: The number of MCMC steps to run.
        tol_tv: The maximum allowed Total Variation (TV) distance.
        
    Returns:
        (results, passed)
    """
    state_list = list(feasible_states)
    if not state_list:
        res = OracleResult("detailed_balance", False, float("inf"), tol_tv, False, "State space is empty")
        return [res], False

    # Compute exact Boltzmann distribution
    e_arr = np.array([energy_fn(s) for s in state_list])
    
    # Check for inf/NaN
    if not np.isfinite(e_arr).all():
        res = OracleResult("detailed_balance", False, float("inf"), tol_tv, False, "Some states have non-finite energy")
        return [res], False
        
    w = np.exp(-(e_arr - e_arr.min()) / tau)
    analytic_probs = w / w.sum()
    analytic_map = {s: float(p) for s, p in zip(state_list, analytic_probs)}

    # Run the chain
    init_state = state_list[0]
    visited = run_chain_fn(init_state, tau, n_steps)
    
    # Compute empirical distribution
    counts: Dict[Hashable, int] = {}
    for s in visited:
        counts[s] = counts.get(s, 0) + 1
        
    total = sum(counts.values())
    if total == 0:
        res = OracleResult("detailed_balance", False, float("inf"), tol_tv, False, "Chain returned 0 states")
        return [res], False
        
    empirical_map = {s: counts.get(s, 0) / total for s in state_list}

    # Compare TV distance
    tv = 0.5 * sum(abs(empirical_map[s] - analytic_map[s]) for s in state_list)
    reached = sum(1 for s in state_list if counts.get(s, 0) > 0)
    
    # Are all states reached?
    all_reached = (reached == len(state_list))
    passed = (tv <= tol_tv) and all_reached
    
    detail = (f"tv={tv:.4f}, reached={reached}/{len(state_list)}, "
              f"steps={n_steps}, tau={tau:.2f}")

    res = OracleResult(
        name="detailed_balance",
        passed=passed,
        residual=tv,
        tol=tol_tv,
        certifies=passed,
        detail=detail,
    )

    return [res], passed
