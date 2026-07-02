# -*- coding: utf-8 -*-
"""Discrete choice modeling primitives."""

import numpy as np
from typing import Optional, List, Tuple

def compute_mnl_probabilities(utilities: np.ndarray, availability: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute Multinomial Logit (MNL) probabilities.
    
    Args:
        utilities: Array of shape (..., n_alts) containing alternative utilities.
        availability: Optional boolean array of the same shape indicating available alternatives.
            If provided, unavailable alternatives receive probability 0.
            
    Returns:
        np.ndarray: Probabilities array of shape (..., n_alts), summing to 1 over the last axis.
    """
    # Use max trick for numerical stability
    U = np.copy(utilities)
    if availability is not None:
        U = np.where(availability, U, -np.inf)
        
    U_max = np.max(U, axis=-1, keepdims=True)
    
    # If all utilities are -inf (no available alternatives), handle safely
    safe_U_max = np.where(np.isinf(U_max), 0.0, U_max)
    
    exp_U = np.exp(U - safe_U_max)
    if availability is not None:
        exp_U = np.where(availability, exp_U, 0.0)
        
    sum_exp_U = np.sum(exp_U, axis=-1, keepdims=True)
    
    # Avoid division by zero
    probs = np.divide(exp_U, sum_exp_U, out=np.zeros_like(exp_U), where=(sum_exp_U > 0))
    return probs

def compute_logsum(utilities: np.ndarray, availability: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute the logsum (expected maximum utility) of a choice set.
    
    Args:
        utilities: Array of shape (..., n_alts).
        availability: Optional boolean array.
        
    Returns:
        np.ndarray: Logsum of shape (...).
    """
    U = np.copy(utilities)
    if availability is not None:
        U = np.where(availability, U, -np.inf)
        
    U_max = np.max(U, axis=-1, keepdims=True)
    safe_U_max = np.where(np.isinf(U_max), 0.0, U_max)
    
    exp_U = np.exp(U - safe_U_max)
    if availability is not None:
        exp_U = np.where(availability, exp_U, 0.0)
        
    sum_exp_U = np.sum(exp_U, axis=-1, keepdims=True)
    
    logsum = np.squeeze(safe_U_max + np.log(np.where(sum_exp_U > 0, sum_exp_U, 1.0)), axis=-1)
    # Restore -inf for cases where no alternatives are available
    no_avail = np.all(~availability, axis=-1) if availability is not None else np.zeros_like(logsum, dtype=bool)
    return np.where(no_avail, -np.inf, logsum)

def compute_nested_logit_probabilities(
    utilities: np.ndarray,
    nests: List[List[int]],
    theta: np.ndarray,
    availability: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute probabilities for a two-level Nested Logit model.
    
    Args:
        utilities: Array of shape (..., n_alts).
        nests: List of lists, where each sublist contains the indices of alternatives in that nest.
        theta: Array of shape (n_nests,) containing the inclusive value (logsum) scale parameter for each nest.
            (0 < theta <= 1; theta=1 recovers MNL).
        availability: Optional boolean array of shape (..., n_alts).
        
    Returns:
        (probs, nest_probs, conditional_probs)
        probs: Marginal probability of each alternative (..., n_alts).
        nest_probs: Probability of choosing each nest (..., n_nests).
        conditional_probs: Probability of each alternative given its nest (..., n_alts).
    """
    n_alts = utilities.shape[-1]
    n_nests = len(nests)
    
    conditional_probs = np.zeros_like(utilities)
    nest_logsums = np.zeros(utilities.shape[:-1] + (n_nests,))
    nest_avail = np.zeros(utilities.shape[:-1] + (n_nests,), dtype=bool)
    
    for k, nest_indices in enumerate(nests):
        U_nest = utilities[..., nest_indices]
        avail_nest = availability[..., nest_indices] if availability is not None else None
        
        # Utilities scaled by 1/theta
        scaled_U = U_nest / theta[k]
        
        # Conditional probability within the nest
        cond_p = compute_mnl_probabilities(scaled_U, avail_nest)
        for i, alt_idx in enumerate(nest_indices):
            conditional_probs[..., alt_idx] = cond_p[..., i]
            
        # Nest logsum = theta * ln(sum(exp(U/theta)))
        logsum_k = compute_logsum(scaled_U, avail_nest)
        nest_logsums[..., k] = theta[k] * logsum_k
        
        if availability is not None:
            nest_avail[..., k] = np.any(avail_nest, axis=-1)
        else:
            nest_avail[..., k] = True

    # Upper level probabilities (nest choice)
    nest_probs = compute_mnl_probabilities(nest_logsums, nest_avail)
    
    # Marginal probabilities
    probs = np.zeros_like(utilities)
    for k, nest_indices in enumerate(nests):
        for alt_idx in nest_indices:
            probs[..., alt_idx] = nest_probs[..., k] * conditional_probs[..., alt_idx]
            
    return probs, nest_probs, conditional_probs
