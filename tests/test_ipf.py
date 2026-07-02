# -*- coding: utf-8 -*-
"""Oracle-backed tests for jr_optlib.transport.ipf_2d.

These tests do not assert against a hand-computed expected matrix. They assert
that the *oracles* certify the output -- the same checks the robustness harness
runs on live paper results. A regression in ipf_2d would break a certifying
oracle here.
"""
import numpy as np
import pytest

from jr_optlib.transport import ipf_2d, make_contingency2d
from jr_optlib.oracles import certify_ipf, summarize
from jr_optlib.oracles.core import Verdict, gap_certificate, differential, metamorphic


@pytest.mark.parametrize("seed", [1, 7, 42])
@pytest.mark.parametrize("shape", [(3, 4), (10, 10), (25, 8)])
def test_ipf_certified(seed, shape):
    R, C = shape
    inst = make_contingency2d(R, C, N=5000, seed=seed)
    X, _ = ipf_2d(inst.row_marg, inst.col_marg, tol=1e-12)
    results, certified = certify_ipf(X, inst.row_marg, inst.col_marg, tol=1e-6)
    assert certified, "\n".join(str(r) for r in results)
    assert summarize(results) is Verdict.CERTIFIED


def test_oracle_catches_corrupted_solution():
    """A perturbed matrix must fail at least one oracle -- the point of it all."""
    inst = make_contingency2d(6, 6, N=3000, seed=3)
    X, _ = ipf_2d(inst.row_marg, inst.col_marg, tol=1e-12)
    X_bad = X.copy()
    X_bad[0, 0] += 5.0
    X_bad[0, 1] -= 5.0  # keep row 0 sum, but break the odds-ratio structure
    results, certified = certify_ipf(X_bad, inst.row_marg, inst.col_marg, tol=1e-6)
    assert not certified
    assert summarize(results) is Verdict.FAIL


def test_seed_preserves_odds_ratios():
    """With a non-uniform seed, IPF must preserve the seed's odds ratios."""
    rng = np.random.default_rng(11)
    q = rng.random((5, 7)) + 0.1
    row = np.array([100.0, 200.0, 150.0, 50.0, 300.0])
    col = np.full(7, row.sum() / 7)
    X, _ = ipf_2d(row, col, q=q, tol=1e-12)
    results, certified = certify_ipf(X, row, col, q=q, tol=1e-6)
    assert certified, "\n".join(str(r) for r in results)


def test_structural_zeros_preserved():
    rng = np.random.default_rng(5)
    q = rng.random((4, 4)) + 0.1
    q[0, 0] = 0.0
    q[3, 2] = 0.0
    row = np.array([90.0, 110.0, 100.0, 100.0])
    col = np.full(4, row.sum() / 4)
    X, _ = ipf_2d(row, col, q=q, tol=1e-12)
    assert X[0, 0] < 1e-6 and X[3, 2] < 1e-6
    results, certified = certify_ipf(X, row, col, q=q, tol=1e-5)
    assert certified, "\n".join(str(r) for r in results)


def test_metamorphic_seed_scale_invariance():
    """Scaling the whole seed by a positive constant leaves the fit unchanged."""
    inst = make_contingency2d(6, 5, N=2000, seed=9)
    q = np.ones((6, 5))
    X1, _ = ipf_2d(inst.row_marg, inst.col_marg, q=q, tol=1e-12)
    X2, _ = ipf_2d(inst.row_marg, inst.col_marg, q=3.7 * q, tol=1e-12)
    assert np.abs(X1 - X2).max() < 1e-6


# --- generic oracle unit tests -------------------------------------------

def test_gap_certificate():
    assert gap_certificate(100.0, 100.0).passed
    assert gap_certificate(100.0, 100.0).certifies
    assert not gap_certificate(100.0, 90.0).passed


def test_differential():
    assert differential(42.0, 42.0).passed
    assert not differential(42.0, 40.0).passed


def test_metamorphic_relation():
    assert metamorphic(10.0, 10.0, "==").passed
    assert metamorphic(10.0, 8.0, "<=").passed        # tightening did not raise
    assert not metamorphic(10.0, 12.0, "<=").passed
