# -*- coding: utf-8 -*-
"""Lagrangian dual ascent drivers."""

import numpy as np
from typing import Callable, Tuple, Any

def subgradient_dual_ascent(
    init_lam: np.ndarray,
    inner_solve_fn: Callable[[np.ndarray], Any],
    gap_fn: Callable[[Any], np.ndarray],
    alpha: float,
    max_iter: int,
    tol: float = 1e-3,
    lam_min: float = 0.0,
) -> Tuple[np.ndarray, Any, np.ndarray, bool, int]:
    """Generic subgradient dual ascent for soft-constrained optimization.
    
    The driver iteratively updates Lagrangian multipliers (lam) to satisfy
    a set of inequality constraints via a generic inner solver.
    
    Args:
        init_lam: Initial multipliers.
        inner_solve_fn: Function mapping current multipliers to a primal solution `x`.
        gap_fn: Function mapping `x` to a vector of constraint violations
            (gaps = actual - target). Positive means violation.
        alpha: Step size (learning rate).
        max_iter: Maximum number of ascent iterations.
        tol: Tolerance for feasibility. If all gaps <= tol, it stops.
        lam_min: Minimum value for multipliers (usually 0.0 for inequalities).
        
    Returns:
        (final_lam, final_x, final_gap, is_feasible, num_iters)
    """
    lam = np.array(init_lam, dtype=float)
    x = None
    gap = None
    feasible = False
    
    for k in range(max_iter):
        x = inner_solve_fn(lam)
        gap = gap_fn(x)
        
        if np.all(gap <= tol):
            feasible = True
            return lam, x, gap, feasible, k
            
        lam = np.maximum(lam_min, lam + alpha * gap)
        
    return lam, x, gap, feasible, max_iter - 1
