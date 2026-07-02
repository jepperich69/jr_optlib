# -*- coding: utf-8 -*-
"""Oracle for tabular Q-learning.

On a small deterministic MDP, risk-neutral tabular Q-learning
("loaded_rewards" mode) must converge to the exact dynamic-programming optimum.
The oracle runs many episodes, reads the learned optimal cost-to-go from the
initial state, and compares it against an independent exact backward recursion
over the (enumerable) state space. A gap above tolerance is a defect in the
learning update.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from jr_optlib.sampling.rl import run_q_learning_episode
from jr_optlib.oracles.core import OracleResult


def certify_q_learning_vs_dp(
    T: int,
    initial_state: Any,
    action_space_fn: Callable[[Any, int], list],
    env_step_fn: Callable[[Any, Any, int], Tuple[Any, float]],
    get_terminal_value_fn: Callable[[Any], float],
    n_episodes: int = 20000,
    eps: float = 0.3,
    alpha: float = 0.2,
    seed: int = 0,
    tol: float = 1e-2,
) -> Tuple[List[OracleResult], bool]:
    """Run Q-learning on a deterministic MDP and compare to exact DP.

    ``env_step_fn`` must be deterministic for the comparison to be exact.
    Mirrors the learner's terminal convention: at epoch ``T`` the continuation
    value is ``get_terminal_value_fn(action)``.
    """
    # --- Q-learning (uses the global numpy RNG, so pin and restore it) ---
    rng_state = np.random.get_state()
    np.random.seed(seed)
    Q: Dict[int, Dict[Any, np.ndarray]] = {t: {} for t in range(1, T + 2)}
    U: Dict[int, Dict[Any, np.ndarray]] = {t: {} for t in range(1, T + 2)}
    for _ in range(n_episodes):
        run_q_learning_episode(
            T, initial_state, action_space_fn, env_step_fn,
            get_terminal_value_fn, Q, U,
            eta=0.0, mode="loaded_rewards", eps=eps, alpha=alpha,
        )
    np.random.set_state(rng_state)
    learned = float(np.min(Q[1][initial_state]))

    # --- Independent exact DP over the reachable state space ---
    memo: Dict[Tuple[Any, int], float] = {}

    def exact(state: Any, t: int) -> float:
        key = (state, t)
        if key in memo:
            return memo[key]
        best = np.inf
        for a in action_space_fn(state, t):
            next_state, cost = env_step_fn(state, a, t)
            if t == T:
                v_next = float(get_terminal_value_fn(a))
            else:
                v_next = exact(next_state, t + 1)
            best = min(best, float(cost) + v_next)
        memo[key] = best
        return best

    exact_val = exact(initial_state, 1)
    r = abs(learned - exact_val)
    res = OracleResult(
        "q_learning_vs_dp", r <= tol, float(r), tol, False,
        f"Q-learning optimum {learned:.4f} vs exact DP {exact_val:.4f}",
    )
    return [res], res.passed
