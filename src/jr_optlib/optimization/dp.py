# -*- coding: utf-8 -*-
"""Dynamic Programming primitives."""

import time
from typing import Callable, Dict, List, Tuple, Any
import numpy as np
from itertools import product as iprod

def contract_transitions(V_xi: np.ndarray, P_list: List[np.ndarray]) -> np.ndarray:
    """
    Contract a value tensor sequentially along each independent exogenous state transition.
    
    Args:
        V_xi: Tensor of shape (n_g, n_g, ..., n_g) representing values for each exogenous state dimension.
        P_list: List of transition matrices, one for each dimension. Each is of shape (n_g, n_g).
        
    Returns:
        np.ndarray: The contracted tensor representing expected future values.
    """
    n_dim = len(P_list)
    temp = V_xi.copy()
    for j in range(n_dim):
        # tensordot along the j-th dimension of temp and the second dimension of P_list[j]
        # P_list[j] is (n_g, n_g). We want sum_k P(current -> k) * V(..., k, ...)
        temp = np.tensordot(P_list[j], temp, axes=(1, j))
        temp = np.moveaxis(temp, 0, j)
    return temp

def backward_induction_solver(
    T: int,
    states: List[Any],
    state_idx: Dict[Any, int],
    exog_grid: np.ndarray,
    n_assets: int,
    terminal_value_fn: Callable[[Any], float],
    feasible_next_fn: Callable[[Any], List[Any]],
    epoch_cost_fn: Callable[[Any, Any, np.ndarray, int], float],
    P_trans: Dict[int, List[np.ndarray]]
) -> Tuple[np.ndarray, List[Dict[Tuple, Any]], int, float]:
    """
    Solve a finite-horizon DP using exact backward induction over a tensor grid.
    
    Args:
        T: Number of epochs.
        states: List of discrete endogenous states (e.g. portfolios).
        state_idx: Mapping from state to its integer index.
        exog_grid: 1D array of exogenous state grid points (assumes same grid for all n_assets dimensions).
        n_assets: Number of exogenous state dimensions.
        terminal_value_fn: Function mapping endogenous state to terminal penalty/value.
        feasible_next_fn: Function mapping current state to list of feasible next states.
        epoch_cost_fn: Function (next_state, current_state, exog_state_vector, epoch_t) -> float.
        P_trans: Dict mapping epoch t (1..T) to a list of n_assets transition matrices.
        
    Returns:
        (V_initial, all_policies, solve_count, wall_clock_time)
        V_initial: Tensor of shape (len(states), len(exog_grid), ..., len(exog_grid))
        all_policies: List (length T) of dicts mapping state tuples to optimal next states.
    """
    n_p = len(states)
    n_g = len(exog_grid)
    
    # Terminal condition
    V = np.zeros([n_p] + [n_g] * n_assets)
    for pi, x in enumerate(states):
        V[pi] = terminal_value_fn(x)

    all_policies = []
    solve_count = 0
    start_time = time.time()

    for t in range(T, 0, -1):
        V_new = np.zeros_like(V)
        policy_t = {}
        P_list = P_trans[t]
        
        # Pre-contract V for all portfolio states
        EV = np.zeros_like(V)
        for pi in range(n_p):
            EV[pi] = contract_transitions(V[pi], P_list)

        for pi, x_prev in enumerate(states):
            nexts = feasible_next_fn(x_prev)
            
            for idxs in iprod(range(n_g), repeat=n_assets):
                theta_p = np.array([exog_grid[idx] for idx in idxs])
                
                best_val = np.inf
                best_x = None
                
                for x_curr in nexts:
                    xi = state_idx[x_curr]
                    ec = epoch_cost_fn(x_curr, x_prev, theta_p, t)
                    ev = EV[xi][idxs]
                    val = ec + ev
                    solve_count += 1
                    
                    if val < best_val:
                        best_val = val
                        best_x = x_curr
                
                V_new[(pi,) + idxs] = best_val
                policy_t[(pi,) + idxs] = best_x
                
        V = V_new
        all_policies.insert(0, policy_t)

    wall_clock = time.time() - start_time
    return V, all_policies, solve_count, wall_clock
