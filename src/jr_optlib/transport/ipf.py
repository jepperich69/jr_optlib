# -*- coding: utf-8 -*-
"""
Iterative Proportional Fitting (IPF / raking / Furness) for 2D contingency tables.

VETTED FUNCTION -- registry id: transport.ipf_2d
Extracted verbatim (numerics-preserving) from
  Pub_MIPEntropy_MPC/code/mip_hybrid/apps/population_transport.py
so a migrated paper reproduces bit-for-bit. See registry/functions.yaml.

Validated by (registry/SCHEMA.md -> oracles):
  - oracle.marginal_residual  : output marginals match the targets (defining property)
  - oracle.ipf_scaling_form   : X = diag(u) q diag(v), i.e. X/q is rank-1
                                (matrix-scaling / Sinkhorn uniqueness certificate)
  - oracle.ipf_reference      : agrees with an independently written log-space raking
Together the first two *certify* correctness: the I-projection onto the
row/col-marginal set is the unique matrix of scaling form with matching
marginals (Sinkhorn 1967). The third is a differential cross-check.
"""

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = ["ipf_2d", "make_contingency2d", "Contingency2D"]


@dataclass
class Contingency2D:
    row_marg: np.ndarray  # (R,)
    col_marg: np.ndarray  # (C,)
    R: int
    C: int


def make_contingency2d(R: int, C: int, N: int, seed: int = 1,
                       skew: float = 1.2) -> Contingency2D:
    """Build synthetic 2D marginals with mild skew; total N.

    Deterministic given ``seed``. Guarantees ``row.sum() == col.sum() == N``,
    which is the feasibility precondition for IPF to converge to matching
    marginals.
    """
    rng = np.random.default_rng(seed)
    r = rng.pareto(skew, size=R) + 0.5
    c = rng.pareto(skew, size=C) + 0.5
    r = r / r.sum(); c = c / c.sum()
    row = np.round(N * r).astype(int)
    diff = N - int(row.sum())
    if diff != 0:
        row[np.argmax(row)] += diff
    col = np.round(N * c).astype(int)
    diff = N - int(col.sum())
    if diff != 0:
        col[np.argmax(col)] += diff
    assert row.sum() == col.sum() == N
    return Contingency2D(row, col, R, C)


def ipf_2d(row_marg: np.ndarray, col_marg: np.ndarray,
           q: Optional[np.ndarray] = None,
           iters: int = 1000, tol: float = 1e-9) -> Tuple[np.ndarray, float]:
    """
    Classic 2D IPF / raking: KL (I-) projection of seed ``q`` onto the set of
    matrices with the given row and column sums.

    Parameters
    ----------
    row_marg, col_marg : 1-D arrays. Must satisfy ``row_marg.sum() == col_marg.sum()``.
    q : optional seed (R, C). Defaults to the all-ones matrix (max-entropy target).
        Zeros in the seed are structural and preserved (clamped to 1e-12 on entry).
    iters, tol : max sweeps and marginal-residual stopping tolerance.

    Returns
    -------
    (X, elapsed) : the fitted table (R, C) and wall-clock seconds.

    Note
    ----
    The ``(X, elapsed)`` return is preserved from the original paper code so a
    migrated paper reproduces bit-for-bit. Use ``ipf_2d(...)[0]`` for the table.
    """
    t0 = time.time()
    R, C = len(row_marg), len(col_marg)
    if q is None:
        X = np.ones((R, C), dtype=float)
    else:
        X = q.copy().astype(float)
        X[X <= 0] = 1e-12
    # scale to total
    X *= (row_marg.sum() / X.sum())
    for _ in range(iters):
        # rows
        rs = X.sum(axis=1)
        alpha = np.divide(row_marg, np.maximum(rs, 1e-16))
        X = (alpha[:, None]) * X
        # cols
        cs = X.sum(axis=0)
        beta = np.divide(col_marg, np.maximum(cs, 1e-16))
        X = X * (beta[None, :])
        # check
        if max(np.abs(X.sum(axis=1) - row_marg).max(),
               np.abs(X.sum(axis=0) - col_marg).max()) <= tol:
            break
    return X, (time.time() - t0)
