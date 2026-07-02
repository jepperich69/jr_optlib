# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np

from jr_optlib.oracles.rl import certify_q_learning_vs_dp


def _small_mdp():
    """Deterministic 3-state MDP; an action names the next state."""
    states = [0, 1, 2]
    T = 3

    def action_space_fn(state, t):
        return states

    def env_step_fn(state, action, t):
        cost = abs(action - state) * 1.0 + 0.2 * t + 0.5 * action
        return action, cost

    def get_terminal_value_fn(x):
        return 0.3 * x

    return dict(
        T=T, initial_state=0, action_space_fn=action_space_fn,
        env_step_fn=env_step_fn, get_terminal_value_fn=get_terminal_value_fn,
    )


def test_q_learning_converges_to_dp_optimum():
    inst = _small_mdp()
    results, passed = certify_q_learning_vs_dp(
        n_episodes=30000, eps=0.3, alpha=0.2, seed=7, tol=1e-2, **inst
    )
    for r in results:
        print(r)
    assert passed
