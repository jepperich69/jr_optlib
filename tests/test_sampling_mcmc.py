# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from itertools import product

import numpy as np

from jr_optlib.sampling.mcmc import metropolis_hastings, ladder_burn_in
from jr_optlib.oracles.sampling import certify_detailed_balance


def test_detailed_balance_setcover():
    """Verify detailed balance on the 5x5 Set-Cover instance from Pub_PMIP_AOR."""
    A = np.array([
        [1, 0, 0, 0, 1],
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
    ], dtype=int)
    c = np.ones(5, dtype=float)
    n = 5

    def energy(x_tuple):
        return float(np.dot(c, x_tuple))

    def feasible(x_tuple):
        x = np.array(x_tuple, dtype=int)
        return bool(np.all(A @ x >= 1))

    feasible_states = []
    for bits in product([0, 1], repeat=n):
        if feasible(bits):
            feasible_states.append(bits)

    def propose_exact(x_tuple, rng):
        x = list(x_tuple)
        if rng.random() < 0.5:
            # Flip
            j = rng.randint(0, n - 1)
            x[j] = 1 - x[j]
        else:
            # Swap
            ones = [i for i, v in enumerate(x) if v == 1]
            zeros = [i for i, v in enumerate(x) if v == 0]
            if not ones or not zeros:
                return None, 0.0
            j1 = rng.choice(ones)
            j0 = rng.choice(zeros)
            x[j1] = 0
            x[j0] = 1
            
        cand = tuple(x)
        if feasible(cand):
            return cand, 0.0  # symmetric proposal
        return None, 0.0

    def run_chain(init_state, tau, n_steps):
        samples, stats = metropolis_hastings(
            init_state=init_state,
            energy_fn=energy,
            propose_fn=propose_exact,
            tau=tau,
            n_steps=n_steps,
            seed=42,
        )
        return samples

    # The 5x5 instance has 11 feasible covers.
    assert len(feasible_states) == 11

    # Run detailed balance test oracle
    results, passed = certify_detailed_balance(
        feasible_states=feasible_states,
        energy_fn=energy,
        run_chain_fn=run_chain,
        tau=0.5,
        n_steps=100_000,
        tol_tv=0.03,
    )
    
    print(results[0].detail)
    assert passed

def test_ladder_burn_in_1d_random_walk():
    """Verify that ladder_burn_in correctly traverses a temperature schedule and finds a better state."""
    # 1D energy landscape: f(x) = x^2. Minimum is at x = 0.
    def energy(x):
        return float(x * x)
        
    def propose(x, rng):
        # Move by -1 or +1
        cand = x + rng.choice([-1, 1])
        return cand, 0.0
        
    init_state = 10
    burn_schedule = [10.0, 5.0, 1.0, 0.1]
    
    best_state, best_energy, stats = ladder_burn_in(
        init_state=init_state,
        energy_fn=energy,
        propose_fn=propose,
        burn_schedule=burn_schedule,
        steps_per_temp=100,
        seed=42
    )
    
    assert best_energy < 100.0  # it should have descended from init_state=10 (energy 100)
    assert best_state ** 2 == best_energy
    assert stats.n_proposed == len(burn_schedule) * 100
    assert stats.n_accepted > 0
