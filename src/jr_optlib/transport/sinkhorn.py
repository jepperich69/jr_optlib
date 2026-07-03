# -*- coding: utf-8 -*-
"""
Entropic optimal transport (Sinkhorn) for balanced transportation.

VETTED FUNCTION -- registry id: transport.sinkhorn_balanced (+ _uv)
Migrated numerics-preserving from
  Pub_MIPEntropy_MPC/code/mip_hybrid/apps/population_transport.py
Migration verified by an old-vs-new differential on the paper's own instances
(tests/test_sinkhorn.py::test_migration_matches_old_copy): identical output.

Sinkhorn solves  min <C,X> + tau*KL(X||1)  s.t. X1=a, X^T1=b, X>=0.
The solution is a diagonal scaling of the Gibbs kernel K=exp(-C/tau):
    X = diag(u) K diag(v).
This is matrix scaling, so the same certificate as IPF applies -- matching
marginals plus scaling form relative to K uniquely certify the plan
(see jr_optlib.oracles.transport.certify_sinkhorn).
"""

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = ["sinkhorn_balanced", "sinkhorn_balanced_uv",
           "make_transport", "TransportInstance"]


@dataclass
class TransportInstance:
    supply: np.ndarray  # (m,)
    demand: np.ndarray  # (n,)
    C: np.ndarray       # costs (m,n)
    m: int
    n: int
    N: int


def make_transport(m: int, n: int, N: int, seed: int = 1,
                   cost_scale: float = 1.0) -> TransportInstance:
    """Seeded balanced transport instance: integer supply/demand summing to N."""
    rng = np.random.default_rng(seed)
    # random integer supply/demand summing to N
    a = rng.dirichlet(np.ones(m)) * N
    b = rng.dirichlet(np.ones(n)) * N
    supply = np.round(a).astype(int); demand = np.round(b).astype(int)
    ds = N - int(supply.sum()); dd = N - int(demand.sum())
    if ds != 0: supply[np.argmax(supply)] += ds
    if dd != 0: demand[np.argmax(demand)] += dd
    C = cost_scale * rng.random((m, n))
    return TransportInstance(supply, demand, C, m, n, N)


def sinkhorn_balanced_uv(a: np.ndarray, b: np.ndarray, C: np.ndarray,
                         tau: float = 0.05, iters: int = 500,
                         tol: float = 1e-9) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Entropic OT with scalings. Returns (X, u, v, time)."""
    t0 = time.time()
    K = np.exp(-C / max(tau, 1e-12))
    u = np.ones_like(a, dtype=float)
    v = np.ones_like(b, dtype=float)
    for _ in range(iters):
        u = a / np.maximum(K @ v, 1e-18)
        v = b / np.maximum(K.T @ u, 1e-18)
        if np.max(np.abs((u * (K @ v)) - a)) < tol and np.max(np.abs((v * (K.T @ u)) - b)) < tol:
            break
    # X = diag(u) K diag(v). The outer rescaling is bit-identical to the
    # dense diag-matmul (matmul only ever adds exact zeros) but O(mn) rather
    # than O(m^2 n + m n^2); ~10-30x faster on the reconstruction step.
    X = u[:, None] * K * v[None, :]
    return X, u, v, (time.time() - t0)


def sinkhorn_balanced(a: np.ndarray, b: np.ndarray, C: np.ndarray,
                      tau: float = 0.05, iters: int = 500,
                      tol: float = 1e-9) -> Tuple[np.ndarray, float]:
    """
    Entropic OT: min <C,X> + tau * KL(X||1) s.t. X1=a, X^T1=b, X>=0.
    Standard Sinkhorn. Returns (X, time).
    """
    t0 = time.time()
    K = np.exp(-C / max(tau, 1e-12))
    u = np.ones_like(a, dtype=float)
    v = np.ones_like(b, dtype=float)
    for _ in range(iters):
        u = a / np.maximum(K @ v, 1e-18)
        v = b / np.maximum(K.T @ u, 1e-18)
        if np.max(np.abs((u * (K @ v)) - a)) < tol and np.max(np.abs((v * (K.T @ u)) - b)) < tol:
            break
    # X = diag(u) K diag(v). The outer rescaling is bit-identical to the
    # dense diag-matmul (matmul only ever adds exact zeros) but O(mn) rather
    # than O(m^2 n + m n^2); ~10-30x faster on the reconstruction step.
    X = u[:, None] * K * v[None, :]
    return X, (time.time() - t0)
