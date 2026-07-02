# -*- coding: utf-8 -*-
"""
Oracles for IPF / matrix-scaling / 2D transport.

The IPF fixed point is the unique I-projection of the seed ``q`` onto the set
of matrices with the prescribed row/column sums. Two independent, cheap
properties *together certify* it (Sinkhorn 1967, matrix scaling uniqueness):

  (i)  marginal_residual == 0 : X has the required row and column sums.
  (ii) ipf_scaling_form == 0  : X = diag(u) @ q @ diag(v), equivalently the
       log-interaction (two-way ANOVA residual) of X/q vanishes -- X preserves
       every odds ratio of the seed.

A matrix satisfying both is unique, so passing both certifies correctness
without ever re-running IPF. ``ipf_reference`` adds a differential cross-check
against an independently coded log-space raking.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from jr_optlib.oracles.core import OracleResult, differential


def marginal_residual(X: np.ndarray, row_marg: np.ndarray, col_marg: np.ndarray,
                      tol: float = 1e-6) -> OracleResult:
    """Necessary property: fitted marginals equal the targets.

    Independent recomputation of X's row/column sums -- does not use anything
    from the IPF run. Necessary but not alone sufficient (many matrices share
    the same marginals), so it does not certify optimality on its own.
    """
    rr = np.abs(X.sum(axis=1) - np.asarray(row_marg, float)).max()
    cc = np.abs(X.sum(axis=0) - np.asarray(col_marg, float)).max()
    res = float(max(rr, cc))
    return OracleResult(
        name="marginal_residual", passed=res <= tol, residual=res, tol=tol,
        certifies=False,
        detail=f"row_err={rr:.3e} col_err={cc:.3e}",
    )


def ipf_scaling_form(X: np.ndarray, q: Optional[np.ndarray] = None,
                     tol: float = 1e-6) -> OracleResult:
    """Certificate property: X is a diagonal scaling of the seed q.

    IPF only ever multiplies rows and columns of q, so X_ij / q_ij must
    factorize as u_i * v_j. Equivalently log(X/q) is additively separable and
    its two-way interaction residual is zero. This is checked independently of
    how X was produced. Combined with a zero marginal residual it *certifies*
    the IPF solution is correct (uniqueness of matrix scaling).
    """
    X = np.asarray(X, float)
    if q is None:
        q = np.ones_like(X)
    q = np.asarray(q, float)

    # Seed zeros. ipf_2d treats them as near-zeros (clamped), so require the
    # corresponding cells of X to be negligible relative to the table scale
    # ("leak"), and run the interaction test only on the positive-seed support.
    scale = max(float(np.abs(X).max()), 1e-12)
    seed_zero = q <= 0
    leak = float(X[seed_zero].max()) / scale if seed_zero.any() else 0.0

    mask = (q > 0) & (X > 0)
    Lobs = np.log(X[mask] / q[mask])

    # Fit the additive model log(X/q)_ij = a_i + b_j by least squares on the
    # observed support and measure the residual. The residual is exactly the
    # log odds-ratio interaction; it is zero iff X is a diagonal scaling of q.
    # A least-squares fit (rather than row/col means) stays exact when the
    # support is incomplete, e.g. structural zeros in the seed.
    ii, jj = np.where(mask)
    R, C = X.shape
    n = ii.size
    if n == 0:
        inter_res = 0.0
    else:
        # params: a_0..a_{R-1}, b_1..b_{C-1} (b_0 absorbed into a for identifiability)
        D = np.zeros((n, R + C - 1))
        D[np.arange(n), ii] = 1.0
        nz = jj > 0
        D[np.arange(n)[nz], R + (jj[nz] - 1)] = 1.0
        coef, *_ = np.linalg.lstsq(D, Lobs, rcond=None)
        inter_res = float(np.abs(Lobs - D @ coef).max())
    res = max(inter_res, leak)
    return OracleResult(
        name="ipf_scaling_form", passed=res <= tol, residual=res, tol=tol,
        certifies=False,  # certifies only jointly with marginal_residual
        detail=f"max log-interaction (odds-ratio) residual={inter_res:.3e}"
               + (f", seed-zero leak={leak:.3e}" if seed_zero.any() else ""),
    )


def _reference_ipf_logspace(row_marg: np.ndarray, col_marg: np.ndarray,
                            q: Optional[np.ndarray] = None,
                            iters: int = 5000, tol: float = 1e-12) -> np.ndarray:
    """Independent IPF in log-space (raking), coded differently from ipf_2d.

    Uses additive updates and scipy.special.logsumexp instead of the
    multiplicative sweep, so agreement is a genuine differential test, not a
    re-run of the same arithmetic.
    """
    from scipy.special import logsumexp

    row_marg = np.asarray(row_marg, float)
    col_marg = np.asarray(col_marg, float)
    R, C = len(row_marg), len(col_marg)
    if q is None:
        logX = np.zeros((R, C))
    else:
        q = np.asarray(q, float)
        with np.errstate(divide="ignore"):
            logX = np.where(q > 0, np.log(np.where(q > 0, q, 1.0)), -np.inf)

    lr = np.log(row_marg)
    lc = np.log(col_marg)
    for _ in range(iters):
        prev = logX
        logX = logX + (lr - logsumexp(logX, axis=1))[:, None]
        logX = logX + (lc - logsumexp(logX, axis=0))[None, :]
        finite = np.isfinite(logX) & np.isfinite(prev)
        with np.errstate(invalid="ignore"):
            diff = logX - prev
        delta = np.abs(diff[finite]).max() if finite.any() else 0.0
        if delta <= tol:
            break
    return np.exp(logX)


def ipf_reference(X: np.ndarray, row_marg: np.ndarray, col_marg: np.ndarray,
                  q: Optional[np.ndarray] = None,
                  tol: float = 1e-6) -> OracleResult:
    """Differential oracle: candidate X vs an independent log-space raking."""
    Xref = _reference_ipf_logspace(row_marg, col_marg, q)
    denom = max(float(np.abs(Xref).max()), 1e-12)
    res = float(np.abs(np.asarray(X, float) - Xref).max()) / denom
    return OracleResult(
        name="ipf_reference", passed=res <= tol, residual=res, tol=tol,
        certifies=False,
        detail=f"max rel elementwise diff vs log-space reference={res:.3e}",
    )


def certify_sinkhorn(X: np.ndarray, a: np.ndarray, b: np.ndarray, C: np.ndarray,
                     tau: float = 0.05, tol: float = 1e-6) -> Tuple[list, bool]:
    """Certify an entropic-OT (Sinkhorn) plan. Returns (results, certified).

    The entropic-OT solution is the unique diagonal scaling of the Gibbs kernel
    K = exp(-C/tau) with marginals a, b. So matching marginals
    (marginal_residual) AND scaling form relative to K (ipf_scaling_form with
    q=K) together certify X is the correct plan -- without re-running Sinkhorn.
    """
    K = np.exp(-np.asarray(C, float) / max(tau, 1e-12))
    r_marg = marginal_residual(X, a, b, tol=tol)
    r_form = ipf_scaling_form(X, q=K, tol=tol)
    results = [r_marg, r_form]
    certified = r_marg.passed and r_form.passed
    if certified:
        r_form.certifies = True
    return results, certified


def certify_ipf(X: np.ndarray, row_marg: np.ndarray, col_marg: np.ndarray,
                q: Optional[np.ndarray] = None,
                tol: float = 1e-6) -> Tuple[list, bool]:
    """Run the full IPF oracle suite. Returns (results, certified).

    ``certified`` is True iff the two certifying properties both hold: matching
    marginals AND diagonal-scaling form. When both pass, X is provably the
    correct IPF solution.
    """
    r_marg = marginal_residual(X, row_marg, col_marg, tol=tol)
    r_form = ipf_scaling_form(X, q, tol=tol)
    r_ref = ipf_reference(X, row_marg, col_marg, q, tol=tol)
    results = [r_marg, r_form, r_ref]
    certified = r_marg.passed and r_form.passed
    if certified:
        # Promote the pair to a certificate in the coverage map.
        r_form.certifies = True
    return results, certified
