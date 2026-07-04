# -*- coding: utf-8 -*-
"""Oracle tests for the entropy-projection curvature identity.

Certifies H = [A diag(x) A^T]^{-1} = grad^2_b Phi(b) for the KL value function
(Pub_PopInt_Part2, Proposition 2). The finite-difference oracle is cross-checked
here against a fully independent dense Hessian of Phi built from the *original*
seed x0 (not x(b)), so the test does not lean on the same reasoning as the
oracle.
"""

from __future__ import annotations

import numpy as np

from jr_optlib.oracles import (
    Verdict,
    certify_entropic_projection,
    certify_margin_curvature,
    summarize,
)
from jr_optlib.population import entropic_projection, margin_information_matrix


def two_factor_model(n_a=4, n_c=3, seed=1):
    """A small two-factor contingency table with two overlapping margins.

    Cells are (a, c) on an n_a x n_c grid. Margins: the a-totals, the c-totals,
    and the grand total (deliberately redundant, so A has dependent rows and the
    information matrix is singular -- exactly the PopInt setting).
    """
    rng = np.random.default_rng(seed)
    K = n_a * n_c
    x0 = rng.gamma(2.0, 1.0, size=K) + 0.1  # strictly positive seed

    rows = []
    # a-margins
    for a in range(n_a):
        r = np.zeros(K)
        for c in range(n_c):
            r[a * n_c + c] = 1.0
        rows.append(r)
    # c-margins
    for c in range(n_c):
        r = np.zeros(K)
        for a in range(n_a):
            r[a * n_c + c] = 1.0
        rows.append(r)
    # grand total (redundant)
    rows.append(np.ones(K))
    A = np.array(rows)
    return x0, A


def phi_dense(x0, A, b, **kw):
    """Value function Phi(b) = D_KL(x(b)||x0) via an independent re-solve."""
    x, res = entropic_projection(x0, A, b, **kw)
    assert res <= 1e-9, f"projection did not converge: res={res}"
    mask = x > 0
    return float(np.sum(x0 - x) + np.sum(x[mask] * np.log(x[mask] / x0[mask])))


def test_curvature_oracle_certifies_true_identity():
    x0, A = two_factor_model()
    # Fit to some feasible interior margins b (perturb the seed's own margins).
    b = A @ x0
    x, res = entropic_projection(x0, A, b, tol=1e-13)
    assert res <= 1e-10

    results, summary = certify_margin_curvature(x, A, n_directions=40, h=0.2, seed=7)
    for r in results:
        assert r.passed, str(r)
    assert summarize(results) is Verdict.CERTIFIED
    # The finite-difference agreement should be tight.
    assert summary["max_rel_err"] < 1e-2
    assert summary["max_pinv_residual"] < 1e-8
    assert summary["min_eig_H"] > -1e-9


def test_finite_diff_matches_independent_dense_hessian():
    """Cross-check: d^T H d equals a dense finite-diff Hessian of Phi from x0."""
    x0, A = two_factor_model(seed=3)
    b = A @ x0
    x, res = entropic_projection(x0, A, b, tol=1e-13)
    assert res <= 1e-10

    M = margin_information_matrix(x, A)
    H = np.linalg.pinv(M, rcond=1e-12)

    rng = np.random.default_rng(0)
    K = A.shape[1]
    h = 0.1
    for _ in range(10):
        i, j = rng.choice(K, size=2, replace=False)
        d = A[:, j] - A[:, i]
        if not np.any(d):
            continue
        q_cf = float(d @ H @ d)
        # Independent central 2nd difference of Phi (referenced to the seed x0).
        fp = phi_dense(x0, A, b + h * d, tol=1e-13)
        fm = phi_dense(x0, A, b - h * d, tol=1e-13)
        f0 = phi_dense(x0, A, b, tol=1e-13)
        q_fd = (fp - 2.0 * f0 + fm) / (h * h)
        assert abs(q_fd - q_cf) <= 1e-2 * max(abs(q_cf), 1e-12), (q_cf, q_fd)


def test_entropic_projection_certified():
    x0, A = two_factor_model(seed=5)
    # Project onto shifted margins that stay feasible.
    b = A @ x0
    d = A[:, 0] - A[:, 1]
    x, res = entropic_projection(x0, A, b + 0.5 * d, tol=1e-13)
    assert res <= 1e-10
    r = certify_entropic_projection(x0, A, x, b + 0.5 * d, tol=1e-8)
    assert r.passed, str(r)
    assert r.verdict is Verdict.CERTIFIED


def test_finite_diff_rejects_a_wrong_curvature_formula():
    """The finite-difference identity discriminates the correct formula.

    A common wrong price drops the fitted mass and uses pinv(A A^T) instead of
    pinv(A diag(x) A^T). Along feasible directions its quadratic form must
    disagree sharply with the true value-function curvature, so the oracle's
    finite-difference check is not vacuous.
    """
    x0, A = two_factor_model(seed=4)
    b = A @ x0
    x, res = entropic_projection(x0, A, b, tol=1e-13)
    assert res <= 1e-10

    H_true = np.linalg.pinv(margin_information_matrix(x, A), rcond=1e-12)
    H_wrong = np.linalg.pinv(A @ A.T, rcond=1e-12)  # forgot diag(x)

    rng = np.random.default_rng(0)
    h = 0.1
    worst_true, worst_wrong = 0.0, 0.0
    for _ in range(20):
        i, j = rng.choice(A.shape[1], size=2, replace=False)
        d = A[:, j] - A[:, i]
        if not np.any(d):
            continue
        xp, _ = entropic_projection(x, A, b + h * d, tol=1e-13)
        xm, _ = entropic_projection(x, A, b - h * d, tol=1e-13)
        q_fd = (_kl_local(xp, x) + _kl_local(xm, x)) / (h * h)
        worst_true = max(worst_true, abs(q_fd - d @ H_true @ d) / max(abs(d @ H_true @ d), 1e-12))
        if d @ H_wrong @ d > 1e-12:
            worst_wrong = max(worst_wrong, abs(q_fd - d @ H_wrong @ d) / (d @ H_wrong @ d))

    assert worst_true < 1e-2          # correct formula tracks the curvature
    assert worst_wrong > 0.1          # wrong formula is caught


def _kl_local(p, q):
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    mask = p > 0
    return float(np.sum(q - p) + np.sum(p[mask] * np.log(p[mask] / q[mask])))
