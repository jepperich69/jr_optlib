# -*- coding: utf-8 -*-
"""Reinforcement Learning primitives."""

import numpy as np
from typing import Callable, Dict, Tuple, Any

def run_q_learning_episode(
    T: int,
    initial_state: Any,
    action_space_fn: Callable[[Any, int], list],
    env_step_fn: Callable[[Any, Any, int], Tuple[Any, float]],
    get_terminal_value_fn: Callable[[Any], float],
    Q: Dict[int, Dict[Any, np.ndarray]],
    U: Dict[int, Dict[Any, np.ndarray]],
    eta: float,
    mode: str,
    eps: float,
    alpha: float
) -> None:
    """
    Run a single episodic Q-learning or Risk-Sensitive Q-learning step.
    
    Args:
        T: Horizon.
        initial_state: Starting state tuple/object.
        action_space_fn: Function (state, t) -> list of available actions.
        env_step_fn: Function (state, action, t) -> (next_state, random_cost).
        get_terminal_value_fn: Function (state) -> terminal cost/penalty.
        Q: Q-table (nested dict: Q[t][state] = array_of_action_values).
        U: Exponential U-table for risk-sensitive mode (U[t][state] = array).
        eta: Risk aversion parameter.
        mode: "loaded_rewards" (risk-neutral DP/Q-Learning) or "risk_sensitive" (exponential Q-Learning).
        eps: Epsilon for epsilon-greedy exploration.
        alpha: Learning rate.
    """
    curr_state = initial_state
    
    def ensure_table_entry(t_idx, state_key):
        if state_key not in Q[t_idx]:
            actions = action_space_fn(state_key, t_idx)
            n_actions = len(actions)
            # Initialize with terminal value approximation
            Q_init = np.array([get_terminal_value_fn(a) for a in actions], dtype=float)
            Q[t_idx][state_key] = Q_init
            if mode == "risk_sensitive":
                U[t_idx][state_key] = np.exp(eta * Q_init)

    for t in range(1, T + 1):
        ensure_table_entry(t, curr_state)
        n_actions = len(action_space_fn(curr_state, t))
        
        # Epsilon-greedy
        if np.random.rand() < eps:
            a_idx = np.random.randint(n_actions)
        else:
            a_idx = int(np.argmin(Q[t][curr_state]))
            
        action = action_space_fn(curr_state, t)[a_idx]
        
        # Step environment
        next_state, cost = env_step_fn(curr_state, action, t)
        
        # Update tables
        if t == T:
            v_next = get_terminal_value_fn(action)
        else:
            ensure_table_entry(t + 1, next_state)
            v_next = Q[t + 1][next_state].min()
            
        if mode == "loaded_rewards":
            # cost is interpreted as the loaded deterministic cost
            target = cost + v_next
            Q[t][curr_state][a_idx] += alpha * (target - Q[t][curr_state][a_idx])
            
        elif mode == "risk_sensitive":
            # cost is the sampled random cost
            u_target = np.exp(eta * (cost + v_next))
            U[t][curr_state][a_idx] += alpha * (u_target - U[t][curr_state][a_idx])
            # Clip U to avoid numerical overflow/underflow
            U[t][curr_state][a_idx] = max(1e-10, U[t][curr_state][a_idx])
            Q[t][curr_state][a_idx] = np.log(U[t][curr_state][a_idx]) / max(eta, 1e-12)
            
        curr_state = next_state
