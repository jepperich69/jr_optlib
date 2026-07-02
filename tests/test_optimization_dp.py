# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np

from jr_optlib.optimization.dp import contract_transitions
from jr_optlib.oracles.dp import (
    certify_transition_contraction,
    certify_dp_vs_brute_force,
)


def _row_stochastic(n, rng):
    P = rng.random((n, n))
    return P / P.sum(axis=1, keepdims=True)


def test_transition_contraction_matches_independent_expectation():
    rng = np.random.default_rng(3)
    n_g = 3
    V = rng.normal(size=(n_g, n_g))
    P_list = [_row_stochastic(n_g, rng), _row_stochastic(n_g, rng)]
    results, passed = certify_transition_contraction(V, P_list)
    for r in results:
        print(r)
    assert passed


def test_transition_contraction_identity_is_noop():
    rng = np.random.default_rng(4)
    n_g = 4
    V = rng.normal(size=(n_g, n_g))
    P_list = [np.eye(n_g), np.eye(n_g)]
    assert np.allclose(contract_transitions(V, P_list), V)


def _small_dp_instance():
    states = [0, 1, 2]
    state_idx = {s: i for i, s in enumerate(states)}
    exog_grid = np.array([1.0, 2.0])
    n_assets = 1
    T = 3

    def terminal_value_fn(x):
        return float(x)

    def feasible_next_fn(x):
        return states  # fully connected

    def epoch_cost_fn(x_curr, x_prev, theta, t):
        return abs(x_curr - x_prev) * float(theta[0]) + 0.1 * t * x_curr

    P_trans = {t: [np.eye(len(exog_grid))] for t in range(1, T + 1)}
    return dict(
        T=T, states=states, state_idx=state_idx, exog_grid=exog_grid,
        n_assets=n_assets, terminal_value_fn=terminal_value_fn,
        feasible_next_fn=feasible_next_fn, epoch_cost_fn=epoch_cost_fn,
        P_trans=P_trans,
    )


def test_backward_induction_matches_brute_force():
    inst = _small_dp_instance()
    results, passed = certify_dp_vs_brute_force(**inst)
    for r in results:
        print(r)
    assert passed
    assert results[0].certifies  # exhaustive enumeration certifies the optimum
