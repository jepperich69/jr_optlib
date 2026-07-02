# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np

from jr_optlib.optimization.choice import (
    compute_mnl_probabilities,
    compute_logsum,
    compute_nested_logit_probabilities,
)
from jr_optlib.oracles.choice import certify_mnl, certify_nested_logit


def test_mnl_known_value():
    # utilities [0, ln 2] -> probabilities [1/3, 2/3].
    p = compute_mnl_probabilities(np.array([0.0, np.log(2.0)]))
    assert np.allclose(p, [1.0 / 3.0, 2.0 / 3.0])


def test_mnl_oracle_batched():
    rng = np.random.default_rng(1)
    U = rng.normal(size=(7, 5))
    results, passed = certify_mnl(U)
    for r in results:
        print(r)
    assert passed


def test_mnl_oracle_with_availability():
    U = np.array([[0.0, 5.0, 1.0], [2.0, 2.0, 2.0]])
    avail = np.array([[True, False, True], [True, True, False]])
    results, passed = certify_mnl(U, availability=avail)
    assert passed
    # Unavailable alternatives get exactly zero probability.
    p = compute_mnl_probabilities(U, avail)
    assert p[0, 1] == 0.0 and p[1, 2] == 0.0


def test_logsum_known_value():
    # logsum of [0, 0] is ln 2.
    assert np.isclose(compute_logsum(np.array([0.0, 0.0])), np.log(2.0))


def test_nested_logit_reduces_to_mnl():
    U = np.array([0.0, np.log(2.0), 1.0, -0.5])
    nests = [[0, 1], [2, 3]]
    results, passed = certify_nested_logit(U, nests)
    for r in results:
        print(r)
    assert passed


def test_nested_logit_batched_and_available():
    rng = np.random.default_rng(2)
    U = rng.normal(size=(4, 6))
    avail = rng.random(size=(4, 6)) > 0.2
    # Guarantee at least one available alternative per row.
    avail[:, 0] = True
    nests = [[0, 1, 2], [3, 4, 5]]
    results, passed = certify_nested_logit(U, nests, availability=avail)
    assert passed


def test_nested_logit_theta_between_zero_and_one_differs_from_mnl():
    # A sanity check that theta != 1 genuinely bends away from MNL, so the
    # theta=1 consistency oracle is not trivially satisfied.
    U = np.array([0.0, 2.0, 1.0, 3.0])
    nests = [[0, 1], [2, 3]]
    p_mnl = compute_mnl_probabilities(U)
    p_nl, _, _ = compute_nested_logit_probabilities(U, nests, np.array([0.4, 0.4]))
    assert not np.allclose(p_mnl, p_nl, atol=1e-3)
