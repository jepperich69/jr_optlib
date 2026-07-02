# -*- coding: utf-8 -*-
"""Metropolis-Hastings sampling primitives."""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Generic, Hashable, Iterator, List, Optional, Tuple, TypeVar

T = TypeVar("T")

class MCMCStats:
    def __init__(self):
        self.n_proposed: int = 0
        self.n_accepted: int = 0
        
    @property
    def acceptance_rate(self) -> float:
        return self.n_accepted / max(1, self.n_proposed)


def metropolis_hastings(
    init_state: T,
    energy_fn: Callable[[T], float],
    propose_fn: Callable[[T, random.Random], Tuple[Optional[T], float]],
    tau: float,
    n_steps: int,
    seed: int = 12345,
    record_every: int = 1,
) -> Tuple[List[T], MCMCStats]:
    """Run a generic Metropolis-Hastings chain.
    
    Args:
        init_state: The starting state.
        energy_fn: Function mapping state to its energy/cost.
        propose_fn: Function mapping (current_state, rng) -> (candidate_state, log_q_ratio).
            `log_q_ratio` is log( q(current | candidate) / q(candidate | current) ).
            If candidate_state is None, the move is treated as a null move (rejected).
        tau: Temperature parameter.
        n_steps: Number of MCMC steps.
        seed: Random seed.
        record_every: Retain a sample every N steps.
        
    Returns:
        (samples, stats)
    """
    rng = random.Random(seed)
    stats = MCMCStats()
    samples = []
    
    current = init_state
    current_energy = energy_fn(current)
    
    # Pre-record the initial state if needed, or wait until after step 1?
    # Usually we record after each step (or thin).
    
    for step in range(n_steps):
        candidate, log_q_ratio = propose_fn(current, rng)
        stats.n_proposed += 1
        
        if candidate is not None:
            cand_energy = energy_fn(candidate)
            dE = cand_energy - current_energy
            
            # Acceptance prob = min(1, (q(x|x')/q(x'|x)) * exp(-dE / tau))
            # log(prob) = log_q_ratio - dE/tau
            log_alpha = log_q_ratio - dE / max(tau, 1e-12)
            
            if log_alpha >= 0.0 or rng.random() < math.exp(log_alpha):
                current = candidate
                current_energy = cand_energy
                stats.n_accepted += 1
                
        if (step + 1) % record_every == 0:
            samples.append(current)
            
    return samples, stats


def simulated_annealing(
    init_state: T,
    energy_fn: Callable[[T], float],
    propose_fn: Callable[[T, random.Random], Optional[T]],
    init_temp: float,
    cooling_rate: float,
    n_steps: int,
    seed: int = 12345,
    on_accept: Optional[Callable[[T, float, int, bool], None]] = None,
) -> Tuple[T, float, MCMCStats]:
    """Run classical geometric Simulated Annealing optimization.
    
    Args:
        init_state: The starting state.
        energy_fn: Function mapping state to its energy/cost.
        propose_fn: Function mapping (current_state, rng) -> candidate_state.
            If candidate_state is None, the move is treated as rejected.
        init_temp: Starting temperature.
        cooling_rate: Decay rate per step (temp *= 1.0 - cooling_rate).
        n_steps: Total number of optimization steps.
        seed: Random seed.
        on_accept: Optional callback called when a move is accepted: (state, energy, step, is_best).
        
    Returns:
        (best_state, best_energy, stats)
    """
    rng = random.Random(seed)
    stats = MCMCStats()
    
    current = init_state
    current_energy = energy_fn(current)
    
    best = current
    best_energy = current_energy
    
    temp = float(init_temp)
    
    for step in range(n_steps):
        candidate = propose_fn(current, rng)
        stats.n_proposed += 1
        
        if candidate is not None:
            cand_energy = energy_fn(candidate)
            dE = cand_energy - current_energy
            
            if dE <= 0.0 or rng.random() < math.exp(-dE / max(temp, 1e-9)):
                current = candidate
                current_energy = cand_energy
                stats.n_accepted += 1
                
                is_best = False
                if current_energy < best_energy:
                    best = current
                    best_energy = current_energy
                    is_best = True
                    
                if on_accept is not None:
                    on_accept(current, current_energy, step, is_best)
                    
        temp = max(1e-6, temp * (1.0 - cooling_rate))
        
    return best, best_energy, stats


def ladder_burn_in(
    init_state: T,
    energy_fn: Callable[[T], float],
    propose_fn: Callable[[T, random.Random], Tuple[Optional[T], float]],
    burn_schedule: Sequence[float],
    steps_per_temp: int,
    seed: int = 12345,
) -> Tuple[T, float, MCMCStats]:
    """Run a temperature-ladder burn-in to warm up an MCMC chain.
    
    Args:
        init_state: The starting state.
        energy_fn: Function mapping state to its energy/cost.
        propose_fn: Function mapping (current_state, rng) -> (candidate_state, log_q_ratio).
            `log_q_ratio` is log( q(current | candidate) / q(candidate | current) ).
            If candidate_state is None, the move is treated as a null move (rejected).
        burn_schedule: Sequence of temperatures to visit (e.g., from hot to cold).
        steps_per_temp: Number of MCMC steps to take at each temperature in the ladder.
        seed: Random seed.
        
    Returns:
        (best_state, best_energy, stats) - Note: returns the BEST state found during burn-in,
        which is standard practice for seeding the subsequent sampling phase.
    """
    rng = random.Random(seed)
    stats = MCMCStats()
    
    current = init_state
    current_energy = energy_fn(current)
    
    best = current
    best_energy = current_energy
    
    for tau in burn_schedule:
        for _ in range(steps_per_temp):
            candidate, log_q_ratio = propose_fn(current, rng)
            stats.n_proposed += 1
            
            if candidate is not None:
                cand_energy = energy_fn(candidate)
                dE = cand_energy - current_energy
                
                log_alpha = log_q_ratio - dE / max(tau, 1e-12)
                
                if log_alpha >= 0.0 or rng.random() < math.exp(log_alpha):
                    current = candidate
                    current_energy = cand_energy
                    stats.n_accepted += 1
                    
                    if current_energy < best_energy:
                        best = current
                        best_energy = current_energy
                        
    return best, best_energy, stats
