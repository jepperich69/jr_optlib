# -*- coding: utf-8 -*-
"""Exact Metropolis-Hastings for Set-Cover problems."""

from __future__ import annotations

import random
from typing import Callable, List, Optional, Tuple, Sequence

import numpy as np

from jr_optlib.sampling.mcmc import metropolis_hastings, MCMCStats


def exact_setcover_propose(
    x: Tuple[int, ...],
    rng: random.Random,
    A: np.ndarray
) -> Tuple[Optional[Tuple[int, ...]], float]:
    """An exact, detailed-balance-preserving proposer for Set Cover.
    
    Proposes a state by either flipping a single variable or swapping a 0 and 1.
    If the candidate state is infeasible (does not cover all rows), it is rejected
    (returns None). The proposal probability is symmetric, so the log Hastings
    ratio is 0.0.
    """
    n = len(x)
    x_list = list(x)
    
    if rng.random() < 0.5:
        # Flip move
        j = rng.randint(0, n - 1)
        x_list[j] = 1 - x_list[j]
    else:
        # Swap move
        ones = [i for i, v in enumerate(x_list) if v == 1]
        zeros = [i for i, v in enumerate(x_list) if v == 0]
        if not ones or not zeros:
            return None, 0.0
        j1 = rng.choice(ones)
        j0 = rng.choice(zeros)
        x_list[j1] = 0
        x_list[j0] = 1
        
    cand = tuple(x_list)
    
    # Check feasibility: A * cand >= 1
    # We can do this efficiently using numpy
    x_arr = np.array(cand, dtype=int)
    if bool(np.all(A @ x_arr >= 1)):
        return cand, 0.0
        
    return None, 0.0


def mh_exact_setcover(
    A: np.ndarray,
    c: np.ndarray,
    init_state: np.ndarray,
    tau: float,
    n_steps: int,
    seed: int = 12345,
    record_every: int = 1,
    burn_schedule: Optional[Sequence[float]] = None,
    burn_steps_per_temp: int = 400,
) -> Tuple[List[np.ndarray], MCMCStats]:
    """Run an exact Metropolis-Hastings chain for Set Cover.
    
    Args:
        A: Cover matrix (m x n)
        c: Cost vector (n)
        init_state: Initial feasible state as a 1D numpy array
        tau: Temperature
        n_steps: Total MCMC steps
        seed: Random seed
        record_every: Thinning interval
        burn_schedule: Optional sequence of temperatures for burn-in.
        burn_steps_per_temp: Steps per temperature if burn_schedule is provided.
        
    Returns:
        (samples, stats) where samples is a list of numpy arrays.
    """
    def energy_fn(x_tuple: Tuple[int, ...]) -> float:
        return float(np.dot(c, np.array(x_tuple, dtype=int)))
        
    def propose_fn(x_tuple: Tuple[int, ...], rng: random.Random):
        return exact_setcover_propose(x_tuple, rng, A)
        
    init_tuple = tuple(init_state.tolist())
    
    if burn_schedule:
        from jr_optlib.sampling.mcmc import ladder_burn_in
        best_state, _, _ = ladder_burn_in(
            init_state=init_tuple,
            energy_fn=energy_fn,
            propose_fn=propose_fn,
            burn_schedule=burn_schedule,
            steps_per_temp=burn_steps_per_temp,
            seed=seed,
        )
        init_tuple = best_state
    
    samples_tuples, stats = metropolis_hastings(
        init_state=init_tuple,
        energy_fn=energy_fn,
        propose_fn=propose_fn,
        tau=tau,
        n_steps=n_steps,
        seed=seed,
        record_every=record_every,
    )
    
    samples_np = [np.array(s, dtype=int) for s in samples_tuples]
    return samples_np, stats
