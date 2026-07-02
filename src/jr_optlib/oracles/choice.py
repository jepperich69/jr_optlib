# -*- coding: utf-8 -*-
"""Oracles for discrete-choice primitives.

The MNL and nested-logit forms are closed form, so they are verified by the
structural identities they must satisfy rather than by a separate solver:

  * normalization -- probabilities sum to 1 over the available alternatives;
  * non-negativity;
  * translation invariance -- adding a constant to every utility leaves the
    MNL probabilities unchanged (a defining property of the logit);
  * availability exclusion -- unavailable alternatives receive probability 0;
  * McFadden consistency -- a nested logit with every inclusive-value (theta)
    equal to 1 collapses exactly to the flat MNL.

A violation of any of these is a provable defect in the implementation.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from jr_optlib.oracles.core import OracleResult
from jr_optlib.optimization.choice import (
    compute_mnl_probabilities,
    compute_nested_logit_probabilities,
)


def certify_mnl(
    utilities: np.ndarray,
    availability: Optional[np.ndarray] = None,
    tol: float = 1e-9,
) -> Tuple[List[OracleResult], bool]:
    """Verify MNL probabilities against the identities they must satisfy.

    Args:
        utilities: Array of shape (..., n_alts).
        availability: Optional boolean array of the same shape.
        tol: Tolerance for the exact identities (normalization, non-negativity).

    Returns:
        (results, passed)
    """
    utilities = np.asarray(utilities, dtype=float)
    probs = compute_mnl_probabilities(utilities, availability)
    results: List[OracleResult] = []

    # 1. Normalization: rows sum to 1, except rows with no available alternative.
    row_sum = probs.sum(axis=-1)
    if availability is not None:
        avail = np.asarray(availability, dtype=bool)
        none_avail = np.all(~avail, axis=-1)
        expected = np.where(none_avail, 0.0, 1.0)
    else:
        expected = np.ones_like(row_sum)
    r_sum = float(np.max(np.abs(row_sum - expected))) if row_sum.size else 0.0
    results.append(OracleResult(
        "mnl_normalization", r_sum <= tol, r_sum, tol, False,
        "probabilities sum to 1 over available alternatives",
    ))

    # 2. Non-negativity.
    r_neg = float(max(0.0, -np.min(probs))) if probs.size else 0.0
    results.append(OracleResult(
        "mnl_nonneg", r_neg <= tol, r_neg, tol, False, "probabilities >= 0",
    ))

    # 3. Translation invariance: adding a constant to every utility must not
    #    change any probability. Uses a looser tolerance for float exp/log.
    tol_inv = 1e-7
    probs_shift = compute_mnl_probabilities(utilities + 4.2, availability)
    r_inv = float(np.max(np.abs(probs - probs_shift))) if probs.size else 0.0
    results.append(OracleResult(
        "mnl_translation_invariance", r_inv <= tol_inv, r_inv, tol_inv, False,
        "adding a constant to all utilities leaves probabilities unchanged",
    ))

    # 4. Availability exclusion.
    if availability is not None:
        avail = np.asarray(availability, dtype=bool)
        masked = probs[~avail]
        r_av = float(np.max(np.abs(masked))) if masked.size else 0.0
        results.append(OracleResult(
            "mnl_availability_zero", r_av <= tol, r_av, tol, False,
            "unavailable alternatives receive probability 0",
        ))

    passed = all(r.passed for r in results)
    return results, passed


def certify_nested_logit(
    utilities: np.ndarray,
    nests: Sequence[Sequence[int]],
    availability: Optional[np.ndarray] = None,
    tol: float = 1e-7,
) -> Tuple[List[OracleResult], bool]:
    """Verify a two-level nested logit.

    The central check is McFadden consistency: with every inclusive-value scale
    theta_k = 1, the nested logit must reproduce the flat MNL exactly. We also
    check that the marginal probabilities form a valid distribution for a
    generic theta in (0, 1].

    ``nests`` must partition all ``n_alts`` alternatives for the normalization
    check to hold.

    Returns:
        (results, passed)
    """
    utilities = np.asarray(utilities, dtype=float)
    n_nests = len(nests)
    results: List[OracleResult] = []

    # 1. McFadden consistency: theta = 1 recovers MNL.
    theta_one = np.ones(n_nests)
    probs_nl, _, _ = compute_nested_logit_probabilities(
        utilities, [list(n) for n in nests], theta_one, availability,
    )
    probs_mnl = compute_mnl_probabilities(utilities, availability)
    r_cons = float(np.max(np.abs(probs_nl - probs_mnl))) if probs_nl.size else 0.0
    results.append(OracleResult(
        "nested_logit_theta1_equals_mnl", r_cons <= tol, r_cons, tol, False,
        "inclusive-value=1 collapses nested logit to MNL (McFadden consistency)",
    ))

    # 2. Normalization for a generic theta in (0, 1].
    rng = np.random.default_rng(0)
    theta = rng.uniform(0.3, 1.0, size=n_nests)
    probs_g, _, _ = compute_nested_logit_probabilities(
        utilities, [list(n) for n in nests], theta, availability,
    )
    row_sum = probs_g.sum(axis=-1)
    if availability is not None:
        avail = np.asarray(availability, dtype=bool)
        none_avail = np.all(~avail, axis=-1)
        expected = np.where(none_avail, 0.0, 1.0)
    else:
        expected = np.ones_like(row_sum)
    r_norm = float(np.max(np.abs(row_sum - expected))) if row_sum.size else 0.0
    results.append(OracleResult(
        "nested_logit_normalization", r_norm <= tol, r_norm, tol, False,
        "marginal probabilities sum to 1",
    ))

    # 3. Non-negativity.
    r_neg = float(max(0.0, -np.min(probs_g))) if probs_g.size else 0.0
    results.append(OracleResult(
        "nested_logit_nonneg", r_neg <= tol, r_neg, tol, False, "probabilities >= 0",
    ))

    passed = all(r.passed for r in results)
    return results, passed
