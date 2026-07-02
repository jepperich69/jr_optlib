# -*- coding: utf-8 -*-
"""Entropic-risk assignment: closed-form risk, Monte Carlo check, and solvers.

Numerics-preserving extraction of ``Pub_QP_SAA_MC/code/solvers.py`` (and the
covariance/sampling helpers from ``data.py``). The paper copy reads the mean
cost matrix, the dimension, and the sampler from a module-level ``data``
instance; here every function takes the mean cost matrix ``c_bar`` explicitly,
so the primitives are instance-agnostic.

The entropic (exponential) risk of a Gaussian cost ``c(xi) ~ N(vec(c_bar),
Sigma_c)`` applied to a decision ``x`` is exactly

    rho_eta(c^T x) = c_bar^T x + (eta/2) x^T Sigma_c x,

the mean plus a risk-loaded variance term. ``rho_eta_formula`` returns it in
closed form; ``rho_eta_mc`` estimates it by sampling for verification.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize


# ---------------------------------------------------------------------------
# Instance helpers (covariance builder + Gaussian sampler)
# ---------------------------------------------------------------------------

def build_equicorrelation_cov(sigma2: np.ndarray, rho: float = 0.0) -> np.ndarray:
    """Build the ``n^2 x n^2`` cost covariance under equicorrelation.

    ``Var(c_ij) = sigma2[i, j]`` and ``Cov(c_ij, c_kl) = rho * std_ij * std_kl``
    for ``(i, j) != (k, l)``. ``rho = 0`` gives the diagonal (independent-cost)
    model. Valid range ``rho in (-1/(n^2 - 1), 1]``.

    Numerics-preserving extraction of ``data.build_Sigma_c`` with the variance
    matrix supplied explicitly instead of read from a module global.
    """
    sigma_vec = np.asarray(sigma2, dtype=float).flatten()
    s = np.sqrt(sigma_vec)
    N = sigma_vec.size
    Sigma_c = np.diag(sigma_vec.copy())
    for p in range(N):
        for q in range(N):
            if p != q:
                Sigma_c[p, q] = rho * s[p] * s[q]
    return Sigma_c


def sample_costs(c_bar: np.ndarray, Sigma_c: np.ndarray, S: int,
                 seed: Optional[int] = None) -> np.ndarray:
    """Draw ``S`` vectorised cost samples ``c(xi) ~ N(vec(c_bar), Sigma_c)``.

    Returns an ``(S, n*n)`` array, each row a row-major vectorised cost matrix.
    Numerics-preserving extraction of ``data.sample_costs``.
    """
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(np.asarray(c_bar, dtype=float).flatten(), Sigma_c, size=S)


# ---------------------------------------------------------------------------
# Entropic risk: closed form (Theorem 1) and Monte Carlo estimate
# ---------------------------------------------------------------------------

def rho_eta_formula(x_vec: np.ndarray, c_bar: np.ndarray,
                    Sigma_c: np.ndarray, eta: float) -> float:
    """Closed-form entropic risk ``c_bar^T x + (eta/2) x^T Sigma_c x``."""
    c_bar_vec = np.asarray(c_bar, dtype=float).flatten()
    x = np.asarray(x_vec, dtype=float)
    return float(c_bar_vec @ x + 0.5 * eta * x @ Sigma_c @ x)


def rho_eta_mc(x_vec: np.ndarray, c_bar: np.ndarray, Sigma_c: np.ndarray,
               eta: float, S: int, seed: Optional[int] = None) -> float:
    """Monte Carlo estimate ``(1/eta) log E[exp(eta * c(xi)^T x)]`` over ``S`` draws.

    Log-sum-exp stabilised. Converges to :func:`rho_eta_formula` as ``S -> inf``.
    """
    C_samples = sample_costs(c_bar, Sigma_c, S, seed=seed)
    z = C_samples @ np.asarray(x_vec, dtype=float)
    shift = eta * z.max()
    log_mgf = shift + np.log(np.mean(np.exp(eta * z - shift)))
    return float(log_mgf / eta)


# ---------------------------------------------------------------------------
# QP over the doubly-stochastic (Birkhoff) polytope
# ---------------------------------------------------------------------------

def solve_qp(c_bar: np.ndarray, Sigma_c: np.ndarray, eta: float,
             x0: Optional[np.ndarray] = None) -> dict:
    """Minimise ``c_bar^T x + (eta/2) x^T Sigma_c x`` over doubly-stochastic ``x``.

    Row and column sums equal one, ``x >= 0``. Convex QP solved with SLSQP from
    three warm starts (diagonal-heavy, uniform+perturb, near-cyclic); the best
    of the three is returned. Numerics-preserving extraction of
    ``solvers.solve_qp`` with the dimension inferred from ``c_bar``.

    Returns ``{x_opt, obj, success}``.
    """
    c_bar = np.asarray(c_bar, dtype=float)
    n = c_bar.shape[0]
    N = n * n
    c_bar_vec = c_bar.flatten()

    def objective(x):
        return c_bar_vec @ x + 0.5 * eta * x @ Sigma_c @ x

    def gradient(x):
        return c_bar_vec + eta * Sigma_c @ x

    eq_constraints = []
    for i in range(n):
        idx = slice(n * i, n * i + n)
        eq_constraints.append({"type": "eq", "fun": lambda x, s=idx: x[s].sum() - 1.0})
    for j in range(n):
        col_idx = list(range(j, N, n))
        eq_constraints.append({"type": "eq", "fun": lambda x, ci=col_idx: x[ci].sum() - 1.0})

    bounds = [(0.0, None)] * N

    if x0 is None:
        x0 = np.full(N, 1.0 / n)
        for i in range(n):
            x0[n * i + i] += 0.05
        row_sums = x0.reshape(n, n).sum(axis=1, keepdims=True)
        x0 = (x0.reshape(n, n) / row_sums).flatten()

    starts = [x0]
    perm_diag = np.eye(n).flatten()
    starts.append(0.7 * perm_diag + 0.3 * np.full(N, 1.0 / n))
    perm_cyc = np.zeros((n, n))
    [perm_cyc.__setitem__((i, (i + 1) % n), 1.0) for i in range(n)]
    starts.append(0.7 * perm_cyc.flatten() + 0.3 * np.full(N, 1.0 / n))

    best_result = None
    for x_init in starts:
        result = minimize(
            objective, x_init, jac=gradient, method="SLSQP",
            bounds=bounds, constraints=eq_constraints,
            options={"ftol": 1e-12, "maxiter": 5000},
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result
    result = best_result

    return {"x_opt": result.x, "obj": float(result.fun), "success": result.success}


# ---------------------------------------------------------------------------
# Hungarian assignment (exact for independent / diagonal covariance)
# ---------------------------------------------------------------------------

def solve_hungarian_qp(c_bar: np.ndarray, sigma2: np.ndarray, eta: float) -> dict:
    """Exact binary optimum for independent Gaussian costs via one Hungarian call.

    With ``Sigma_c`` diagonal and ``x`` a permutation matrix,
    ``x^T Sigma_c x = sum_ij sigma2[i,j] x_ij``, so the objective is linear:
    ``sum_ij (c_bar[i,j] + (eta/2) sigma2[i,j]) x_ij``. One assignment on the
    modified cost matrix solves it. The reported ``obj`` is the true entropic
    risk from :func:`rho_eta_formula`, not the modified-cost sum.

    Numerics-preserving extraction of ``solvers.solve_hungarian_qp``; the paper
    copy computed ``obj`` against the module-global mean matrix, whereas this
    uses the passed ``c_bar`` consistently.
    """
    c_bar = np.asarray(c_bar, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)
    t0 = time.perf_counter()
    C_mod = c_bar + 0.5 * eta * sigma2
    row_ind, col_ind = linear_sum_assignment(C_mod)
    t1 = time.perf_counter()

    x = np.zeros(c_bar.size)
    for i, j in zip(row_ind, col_ind):
        x[c_bar.shape[1] * i + j] = 1.0

    Sigma_c_diag = np.diag(sigma2.flatten())
    obj = float(c_bar.flatten() @ x + 0.5 * eta * x @ Sigma_c_diag @ x)

    return {"x_opt": x, "obj": obj, "pi": col_ind, "time_s": t1 - t0}


def solve_hungarian_rn(c_bar: np.ndarray) -> dict:
    """Risk-neutral baseline: Hungarian on the mean cost matrix.

    SAA converges to this as ``S -> inf`` under Gaussian noise. ``obj`` is the
    mean cost only, not the entropic risk. Numerics-preserving extraction of
    ``solvers.solve_hungarian_rn``.
    """
    c_bar = np.asarray(c_bar, dtype=float)
    t0 = time.perf_counter()
    row_ind, col_ind = linear_sum_assignment(c_bar)
    t1 = time.perf_counter()

    x = np.zeros(c_bar.size)
    for i, j in zip(row_ind, col_ind):
        x[c_bar.shape[1] * i + j] = 1.0

    return {"x_opt": x, "obj": float(c_bar[row_ind, col_ind].sum()),
            "pi": col_ind, "time_s": t1 - t0}


# ---------------------------------------------------------------------------
# Gurobi MIQP (exact for correlated covariance)
# ---------------------------------------------------------------------------

def solve_miqp_gurobi(c_bar: np.ndarray, Sigma_c: np.ndarray, eta: float,
                      continuous: bool = False, time_limit: float = 60.0,
                      verbose: bool = False) -> dict:
    """Solve the entropic-risk assignment MIQP with Gurobi.

    ``min c_bar^T x + (eta/2) x^T Sigma_c x`` over doubly-stochastic
    ``x in {0,1}^{n^2}`` (or ``[0,1]`` if ``continuous``). For diagonal
    ``Sigma_c`` prefer :func:`solve_hungarian_qp`. Numerics-preserving
    extraction of ``solvers.solve_miqp_gurobi``.

    Returns ``{x_opt, obj, success, runtime_s}``.
    """
    import gurobipy as gp
    from gurobipy import GRB

    c_bar = np.asarray(c_bar, dtype=float)
    n = c_bar.shape[0]
    N = n * n
    c_bar_vec = c_bar.flatten()

    t0 = time.perf_counter()

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1 if verbose else 0)
    env.setParam("LogToConsole", 1 if verbose else 0)
    env.start()

    m = gp.Model(env=env)
    m.setParam("TimeLimit", time_limit)

    vtype = GRB.CONTINUOUS if continuous else GRB.BINARY
    xv = m.addVars(N, lb=0.0, ub=1.0, vtype=vtype, name="x")

    lin = gp.LinExpr()
    for k in range(N):
        lin.add(xv[k], c_bar_vec[k])

    qobj = gp.QuadExpr()
    for p in range(N):
        for q in range(N):
            v = Sigma_c[p, q]
            if abs(v) > 1e-14:
                qobj.add(xv[p] * xv[q], 0.5 * eta * v)

    m.setObjective(lin + qobj, GRB.MINIMIZE)

    for i in range(n):
        m.addConstr(gp.quicksum(xv[n * i + j] for j in range(n)) == 1.0)
    for j in range(n):
        m.addConstr(gp.quicksum(xv[n * i + j] for i in range(n)) == 1.0)

    m.optimize()

    x_opt = np.array([xv[k].X for k in range(N)])
    obj = float(m.ObjVal)
    ok = m.Status == GRB.OPTIMAL
    rt = time.perf_counter() - t0

    env.dispose()

    return {"x_opt": x_opt, "obj": obj, "success": ok, "runtime_s": rt}
