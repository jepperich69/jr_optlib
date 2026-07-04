# -*- coding: utf-8 -*-
"""
Curvature of the entropy (KL I-projection) value function.

For a seed table ``x0``, an aggregation matrix ``A`` (0/1 rows selecting margin
cells), and a target margin vector ``b``, the entropy projection is

    x(b) = argmin_{x >= 0} D_KL(x || x0)   s.t.   A x = b.

Its minimizer has the Furness / exponential-family form ``x = x0 * exp(A^T λ)``,
and the value function ``Φ(b) = D_KL(x(b) || x0)`` has two closed-form
derivatives (see Pub_PopInt_Part2 propositions 1-2):

    ∇_b Φ(b)   = λ                              (balancing factors / shadow prices)
    ∇²_b Φ(b)  = [ A diag(x(b)) A^T ]^{-1}       (inverse margin information matrix)

``margin_information_matrix`` builds ``M = A diag(x) A^T`` (the covariance of the
margin counts under the fitted table -- the Fisher information of the margins);
its Moore-Penrose inverse is the curvature ``H``. ``entropic_projection`` solves
the projection for a general ``A`` by dual Newton, generalising the 2D
``transport.ipf_2d`` to arbitrary margin structures.

The identity ``H = ∇²_b Φ`` is certified independently by
``jr_optlib.oracles.population.certify_margin_curvature``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["margin_information_matrix", "entropic_projection"]

_EXP_CLIP = 700.0  # keep exp(A^T λ) in range


def margin_information_matrix(x: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Return the margin information (covariance) matrix ``M = A diag(x) A^T``.

    ``x`` is the fitted fractional table (length ``K``), ``A`` is ``(m, K)``.
    ``M`` is ``(m, m)``, symmetric positive semidefinite. Its Moore-Penrose
    inverse is the KL curvature of the value function at the margins ``A x``.
    """
    x = np.asarray(x, dtype=float)
    A = np.asarray(A, dtype=float)
    return (A * x[None, :]) @ A.T


def entropic_projection(
    x0: np.ndarray,
    A: np.ndarray,
    b_target: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-12,
    ridge: float = 0.0,
    precond: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """KL I-projection of seed ``x0`` onto ``{x >= 0 : A x = b_target}``.

    Solves ``min D_KL(x||x0) s.t. A x = b_target`` for a general 0/1 aggregation
    matrix ``A`` by dual Newton on ``x(λ) = x0 * exp(A^T λ)``. The Newton system
    Jacobian is exactly the margin information matrix ``A diag(x) A^T``, so each
    step is ``λ <- λ - M^+ (A x - b_target)``. Structural zeros of ``x0`` are
    preserved.

    Parameters
    ----------
    precond : optional fixed ``(m, m)`` preconditioner used in place of the
        per-iteration ``M^+``. Passing ``M(x0)^+`` turns the solver into a fast
        fixed quasi-Newton (used by the curvature oracle, which has already
        formed that pseudo-inverse); the returned table is the same unique
        projection either way and is gated by the residual ``res``.

    Returns
    -------
    (x, res) : the projected table and the final margin residual
        ``max |A x - b_target|``. Callers should treat ``res > tol`` as a
        non-converged solve.
    """
    x0 = np.asarray(x0, dtype=float)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b_target, dtype=float)

    support = x0 > 0
    x = x0.copy()
    lam = np.zeros(A.shape[0])
    res = float("inf")

    for _ in range(max_iter):
        g = A @ x - b
        res = float(np.max(np.abs(g))) if g.size else 0.0
        if res <= tol:
            break
        if precond is not None:
            step = precond @ g
        else:
            M = (A * x[None, :]) @ A.T
            if ridge:
                M = M + ridge * np.eye(M.shape[0])
            step = np.linalg.pinv(M, rcond=1e-12) @ g
        lam = lam - step
        z = np.clip(A.T @ lam, -_EXP_CLIP, _EXP_CLIP)
        x = np.where(support, x0 * np.exp(z), 0.0)

    return x, res
