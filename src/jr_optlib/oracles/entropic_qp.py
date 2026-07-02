# -*- coding: utf-8 -*-
"""Oracles for the entropic-risk assignment primitives.

Two independent checks:

* :func:`certify_entropic_risk_mc` -- the closed-form entropic risk is exact for
  Gaussian costs, and a Monte Carlo estimate must converge to it. A differential
  oracle (type 3): closed form vs sampling.
* :func:`certify_entropic_assignment` -- feasibility + objective recomputation
  (type 1) plus, for small instances, an exhaustive brute force over all
  permutation matrices (type 5) that certifies the exact binary optimum.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np

from jr_optlib.oracles.core import OracleResult
from jr_optlib.optimization.entropic_qp import rho_eta_formula, rho_eta_mc


def certify_entropic_risk_mc(x_vec, c_bar, Sigma_c, eta, claimed=None,
                             S=200_000, seed=0, rel_tol=5e-3):
    """Check an entropic-risk value against a Monte Carlo estimate for fixed ``x``.

    ``claimed`` is the value under test; if ``None`` it defaults to the closed
    form :func:`rho_eta_formula`, so the oracle then certifies that the formula
    matches the sampled risk. The MC estimator of ``(1/eta) log E[exp(eta z)]``
    converges at ``O(1/sqrt(S))`` and is slow for large ``eta * std(z)``, so this
    is a convergence check (``certifies`` stays False) and should be run in a
    mild regime (small ``eta`` or many samples).
    """
    if claimed is None:
        claimed = rho_eta_formula(x_vec, c_bar, Sigma_c, eta)
    approx = rho_eta_mc(x_vec, c_bar, Sigma_c, eta, S=S, seed=seed)
    denom = max(abs(claimed), 1e-12)
    rel = abs(claimed - approx) / denom
    return OracleResult(
        name="entropic_risk_value_vs_mc",
        passed=rel <= rel_tol,
        residual=rel,
        tol=rel_tol,
        certifies=False,
        detail=f"claimed={claimed:.6g} mc(S={S})={approx:.6g} rel_diff={rel:.3e}",
    )


def _permutation_min(c_bar, Sigma_c, eta):
    """Exact minimum entropic risk over all permutation matrices (small n)."""
    n = c_bar.shape[0]
    best_obj = np.inf
    best_x = None
    for perm in permutations(range(n)):
        x = np.zeros((n, n))
        for i, j in enumerate(perm):
            x[i, j] = 1.0
        xv = x.flatten()
        obj = rho_eta_formula(xv, c_bar, Sigma_c, eta)
        if obj < best_obj:
            best_obj = obj
            best_x = xv
    return float(best_obj), best_x


def _is_doubly_stochastic(x, n, tol):
    M = x.reshape(n, n)
    return (M >= -tol).all() and \
        np.allclose(M.sum(axis=1), 1.0, atol=tol) and \
        np.allclose(M.sum(axis=0), 1.0, atol=tol)


def certify_entropic_assignment(result, c_bar, Sigma_c, eta, binary=True,
                                brute_force=True, max_brute_n=8, tol=1e-6):
    """Certify an entropic-risk assignment solution.

    Runs, in order:
      1. feasibility -- ``x`` doubly stochastic (and 0/1 if ``binary``);
      2. objective recomputation -- reported ``obj`` equals
         :func:`rho_eta_formula` evaluated at ``x`` (catches obj/x mismatches);
      3. optimality (small ``n`` only) -- brute force over all ``n!`` permutation
         matrices. For a binary solver, equality certifies the exact optimum;
         for the continuous relaxation, the objective must not exceed the best
         permutation (a valid lower-bound check).

    Returns ``(results, ok_all)``.
    """
    c_bar = np.asarray(c_bar, dtype=float)
    n = c_bar.shape[0]
    x = np.asarray(result["x_opt"], dtype=float)
    reported = float(result["obj"])
    results = []

    # 1. feasibility
    feas = _is_doubly_stochastic(x, n, tol)
    if binary:
        feas = feas and np.all(np.isclose(x, np.round(x), atol=tol))
    row_gap = float(np.abs(x.reshape(n, n).sum(axis=1) - 1.0).max())
    col_gap = float(np.abs(x.reshape(n, n).sum(axis=0) - 1.0).max())
    results.append(OracleResult(
        name="assignment_feasible", passed=bool(feas),
        residual=max(row_gap, col_gap), tol=tol, certifies=False,
        detail=f"row_gap={row_gap:.2e} col_gap={col_gap:.2e} binary={binary}",
    ))

    # 2. objective recomputation
    recomputed = rho_eta_formula(x, c_bar, Sigma_c, eta)
    obj_gap = abs(recomputed - reported)
    results.append(OracleResult(
        name="entropic_risk_recompute", passed=obj_gap <= max(tol, 1e-9 * max(1.0, abs(recomputed))),
        residual=obj_gap, tol=tol, certifies=False,
        detail=f"reported={reported:.9g} recomputed={recomputed:.9g} gap={obj_gap:.2e}",
    ))

    # 3. exhaustive optimality (small n)
    if brute_force and n <= max_brute_n:
        best_obj, _ = _permutation_min(c_bar, Sigma_c, eta)
        if binary:
            gap = abs(reported - best_obj)
            results.append(OracleResult(
                name="assignment_bruteforce_optimum",
                passed=gap <= tol, residual=gap, tol=tol, certifies=gap <= tol,
                detail=f"solver={reported:.9g} brute_min={best_obj:.9g} gap={gap:.2e} over {n}! perms",
            ))
        else:
            # continuous relaxation: optimum must not exceed the best vertex
            over = max(0.0, reported - best_obj - tol)
            results.append(OracleResult(
                name="assignment_relaxation_bound",
                passed=over <= 0.0, residual=over, tol=tol, certifies=False,
                detail=f"relaxed={reported:.9g} <= best_perm={best_obj:.9g}? gap_over={over:.2e}",
            ))

    ok_all = all(r.passed for r in results)
    return results, ok_all
