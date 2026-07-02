# -*- coding: utf-8 -*-
"""Oracles for finite-horizon dynamic-programming primitives.

Two independent checks:

  * ``certify_transition_contraction`` validates the stochastic core --
    ``contract_transitions`` must equal the expectation of the value tensor
    over the product grid, computed here by an independent brute-force sum.

  * ``certify_dp_vs_brute_force`` validates the Bellman recursion end to end --
    on a deterministic-exogenous instance every (endogenous state, exogenous
    grid point) defines an independent finite-horizon shortest-path problem,
    which we solve by exhaustive path enumeration and compare to the value
    returned by backward induction. Because enumeration is exact, a match
    *certifies* the DP optimum.
"""

from __future__ import annotations

from itertools import product as iprod
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

from jr_optlib.oracles.core import OracleResult
from jr_optlib.optimization.dp import contract_transitions, backward_induction_solver


def _independent_expectation(V: np.ndarray, P_list: Sequence[np.ndarray]) -> np.ndarray:
    """E[idx] = sum_{idx'} prod_j P_j[idx_j, idx'_j] * V[idx'].

    Independent of ``contract_transitions`` (plain nested summation), so it is a
    valid cross-check of the tensor contraction.
    """
    n_dim = len(P_list)
    n_g = V.shape[0]
    out = np.zeros_like(V)
    for idx in iprod(range(n_g), repeat=n_dim):
        total = 0.0
        for idxp in iprod(range(n_g), repeat=n_dim):
            w = 1.0
            for j in range(n_dim):
                w *= P_list[j][idx[j], idxp[j]]
            total += w * V[idxp]
        out[idx] = total
    return out


def certify_transition_contraction(
    V: np.ndarray,
    P_list: Sequence[np.ndarray],
    tol: float = 1e-9,
) -> Tuple[List[OracleResult], bool]:
    """Check that the tensor contraction equals the independent expectation."""
    ref = _independent_expectation(V, P_list)
    got = contract_transitions(V, P_list)
    r = float(np.max(np.abs(ref - got))) if ref.size else 0.0
    res = OracleResult(
        "dp_transition_contraction", r <= tol, r, tol, False,
        "tensor contraction equals independent expectation over the product grid",
    )
    return [res], res.passed


def certify_dp_vs_brute_force(
    T: int,
    states: Sequence[Any],
    state_idx: Dict[Any, int],
    exog_grid: np.ndarray,
    n_assets: int,
    terminal_value_fn: Callable[[Any], float],
    feasible_next_fn: Callable[[Any], List[Any]],
    epoch_cost_fn: Callable[[Any, Any, np.ndarray, int], float],
    P_trans: Dict[int, List[np.ndarray]],
    tol: float = 1e-9,
) -> Tuple[List[OracleResult], bool]:
    """Compare backward induction to exhaustive path enumeration.

    Requires the exogenous transitions in ``P_trans`` to be identity matrices
    (deterministic exogenous state); each exogenous grid point is then an
    independent deterministic shortest-path problem the oracle solves by
    enumeration. Raises ``ValueError`` if the transitions are not identity.
    """
    for t, P_list in P_trans.items():
        for P in P_list:
            if not np.allclose(P, np.eye(P.shape[0])):
                raise ValueError(
                    "certify_dp_vs_brute_force requires identity (deterministic) "
                    "exogenous transitions",
                )

    V, _policies, _n, _wall = backward_induction_solver(
        T, list(states), state_idx, exog_grid, n_assets,
        terminal_value_fn, feasible_next_fn, epoch_cost_fn, P_trans,
    )

    n_g = len(exog_grid)
    max_val_res = 0.0

    for pi, x0 in enumerate(states):
        for idxs in iprod(range(n_g), repeat=n_assets):
            theta = np.array([exog_grid[i] for i in idxs])

            def best(x_prev: Any, t: int) -> float:
                # Value of being at x_prev about to decide at epoch t.
                if t > T:
                    return float(terminal_value_fn(x_prev))
                b = np.inf
                for x_curr in feasible_next_fn(x_prev):
                    v = float(epoch_cost_fn(x_curr, x_prev, theta, t)) + best(x_curr, t + 1)
                    if v < b:
                        b = v
                return b

            bf_val = best(x0, 1)
            dp_val = float(V[(pi,) + idxs])
            max_val_res = max(max_val_res, abs(bf_val - dp_val))

    passed = max_val_res <= tol
    res = OracleResult(
        "dp_vs_brute_force_value", passed, float(max_val_res), tol, True,
        "backward-induction value equals exhaustive enumeration (certified optimal)",
    )
    return [res], passed
