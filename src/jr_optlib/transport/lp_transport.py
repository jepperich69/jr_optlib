# -*- coding: utf-8 -*-
"""
Min-cost transport: exact LP solver, integer-LP rounding, and reduced-cost
greedy push.

VETTED FUNCTIONS -- registry ids: transport.solve_transport_opt,
transport.round_transport_min_cost_lp, transport.round_transport_min_cost_lp_restricted,
transport.round_transport_greedy_push.

Migrated numerics-preserving from
  Pub_MIPEntropy_MPC/code/mip_hybrid/apps/population_transport.py.

DOUBLE-DEFINITION BUG RESOLVED: in the paper file `round_transport_greedy_push`
was defined twice (population_transport.py:789 and :882); Python kept only the
second, so the first (the simple `(a,b,C,u,v)` variant) was dead code. The
migrated version here is the ACTIVE one (line 882: signature with tau/topk and
the LP repair + full-LP fallback). The dead variant is intentionally NOT
migrated.

Migration verified by an old-vs-new differential on OBJECTIVE VALUE and
feasibility, not raw plan X: transport LPs are degenerate, so two solver runs
can return different optimal plans of equal cost. Cost is unique and certifiable;
the plan is not. See tests/test_lp_transport.py.
"""

import time
from typing import Optional, Tuple

import numpy as np

from jr_optlib.transport._backend import backend_name

__all__ = [
    "solve_transport_opt",
    "round_transport_min_cost_lp",
    "round_transport_min_cost_lp_restricted",
    "round_transport_greedy_push",
]


def solve_transport_opt(a: np.ndarray, b: np.ndarray,
                        C: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Solve the exact min-cost transport (LP with TU -> integral).
    Returns (X*, obj, time).
    """
    t0 = time.time()
    m, n = len(a), len(b)
    if backend_name() == "ortools":
        from ortools.linear_solver import pywraplp
        solver = pywraplp.Solver.CreateSolver("GLOP")  # LP sufficient
        x = [[solver.NumVar(0.0, solver.infinity(), f"x_{i}_{j}") for j in range(n)] for i in range(m)]
        for i in range(m):
            ct = solver.RowConstraint(float(a[i]), float(a[i]), "")
            for j in range(n):
                ct.SetCoefficient(x[i][j], 1.0)
        for j in range(n):
            ct = solver.RowConstraint(float(b[j]), float(b[j]), "")
            for i in range(m):
                ct.SetCoefficient(x[i][j], 1.0)
        obj = solver.Objective()
        for i in range(m):
            for j in range(n):
                obj.SetCoefficient(x[i][j], float(C[i, j]))
        obj.SetMinimization()
        solver.Solve()
        X = np.array([[x[i][j].solution_value() for j in range(n)] for i in range(m)])
        T = time.time() - t0
        return X, float((C * X).sum()), T
    else:
        import pulp
        prob = pulp.LpProblem("transport", pulp.LpMinimize)
        x = [[pulp.LpVariable(f"x_{i}_{j}", lowBound=0, cat="Continuous") for j in range(n)] for i in range(m)]
        for i in range(m):
            prob += pulp.lpSum(x[i][j] for j in range(n)) == a[i]
        for j in range(n):
            prob += pulp.lpSum(x[i][j] for i in range(m)) == b[j]
        prob += pulp.lpSum(C[i, j] * x[i][j] for i in range(m) for j in range(n))
        prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=30))
        X = np.array([[pulp.value(x[i][j]) for j in range(n)] for i in range(m)])
        T = time.time() - t0
        return X, float((C * X).sum()), T


def round_transport_min_cost_lp(a: np.ndarray, b: np.ndarray,
                                C: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Integerization by solving the min-cost transport LP with costs C.
    TU => LP solution is integral; returns Xint (integer) and wall time.
    """
    t0 = time.time()
    m, n = len(a), len(b)
    if backend_name() == "ortools":
        from ortools.linear_solver import pywraplp
        solver = pywraplp.Solver.CreateSolver("GLOP")  # LP is sufficient (TU)
        x = [[solver.NumVar(0.0, solver.infinity(), f"x_{i}_{j}") for j in range(n)] for i in range(m)]
        for i in range(m):
            ct = solver.RowConstraint(float(a[i]), float(a[i]), "")
            for j in range(n):
                ct.SetCoefficient(x[i][j], 1.0)
        for j in range(n):
            ct = solver.RowConstraint(float(b[j]), float(b[j]), "")
            for i in range(m):
                ct.SetCoefficient(x[i][j], 1.0)
        obj = solver.Objective()
        for i in range(m):
            for j in range(n):
                obj.SetCoefficient(x[i][j], float(C[i, j]))
        obj.SetMinimization()
        solver.Solve()
        X = np.array([[x[i][j].solution_value() for j in range(n)] for i in range(m)])
    else:
        import pulp
        prob = pulp.LpProblem("transport_costed", pulp.LpMinimize)
        x = [[pulp.LpVariable(f"x_{i}_{j}", lowBound=0, cat="Continuous") for j in range(n)] for i in range(m)]
        for i in range(m):
            prob += pulp.lpSum(x[i][j] for j in range(n)) == a[i]
        for j in range(n):
            prob += pulp.lpSum(x[i][j] for i in range(m)) == b[j]
        prob += pulp.lpSum(C[i, j] * x[i][j] for i in range(m) for j in range(n))
        prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=30))
        X = np.array([[pulp.value(x[i][j]) for j in range(n)] for i in range(m)])
    Xint = np.rint(X).astype(int)   # Should already be integer by TU
    # Sanity: marginals exact
    assert np.all(Xint.sum(axis=1) == a)
    assert np.all(Xint.sum(axis=0) == b)
    return Xint, (time.time() - t0)


def round_transport_min_cost_lp_restricted(a: np.ndarray, b: np.ndarray,
                                           C: np.ndarray,
                                           E: np.ndarray) -> Tuple[np.ndarray, float, bool]:
    """
    Solve min-cost transport LP on restricted support E (bool mask).
    Returns (Xint, time_s, ok). If ok=False, caller should fall back to full LP.
    """
    t0 = time.time()
    m, n = len(a), len(b)
    if backend_name() == "ortools":
        from ortools.linear_solver import pywraplp
        solver = pywraplp.Solver.CreateSolver("GLOP")
        if solver is None:
            return np.zeros((m, n), dtype=int), 0.0, False
        x = [[None] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if E[i, j]:
                    x[i][j] = solver.NumVar(0.0, solver.infinity(), f"x_{i}_{j}")
        for i in range(m):
            if a[i] == 0:
                continue
            ct = solver.RowConstraint(float(a[i]), float(a[i]), "")
            has = False
            for j in range(n):
                if x[i][j] is not None:
                    ct.SetCoefficient(x[i][j], 1.0); has = True
            if not has:
                return np.zeros((m, n), dtype=int), time.time() - t0, False
        for j in range(n):
            if b[j] == 0:
                continue
            ct = solver.RowConstraint(float(b[j]), float(b[j]), "")
            has = False
            for i in range(m):
                if x[i][j] is not None:
                    ct.SetCoefficient(x[i][j], 1.0); has = True
            if not has:
                return np.zeros((m, n), dtype=int), time.time() - t0, False
        obj = solver.Objective()
        for i in range(m):
            for j in range(n):
                if x[i][j] is not None:
                    obj.SetCoefficient(x[i][j], float(C[i, j]))
        obj.SetMinimization()
        stat = solver.Solve()
        if stat != pywraplp.Solver.OPTIMAL:
            return np.zeros((m, n), dtype=int), time.time() - t0, False
        X = np.zeros((m, n), dtype=float)
        for i in range(m):
            for j in range(n):
                if x[i][j] is not None:
                    X[i, j] = x[i][j].solution_value()
    else:
        import pulp
        prob = pulp.LpProblem("transport_costed_sparse", pulp.LpMinimize)
        x = {}
        for i in range(m):
            for j in range(n):
                if E[i, j]:
                    x[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", lowBound=0, cat="Continuous")
        for i in range(m):
            vars_i = [x[(i, j)] for j in range(n) if (i, j) in x]
            if a[i] > 0 and not vars_i:
                return np.zeros((m, n), dtype=int), time.time() - t0, False
            if vars_i:
                prob += pulp.lpSum(vars_i) == float(a[i])
        for j in range(n):
            vars_j = [x[(i, j)] for i in range(m) if (i, j) in x]
            if b[j] > 0 and not vars_j:
                return np.zeros((m, n), dtype=int), time.time() - t0, False
            if vars_j:
                prob += pulp.lpSum(vars_j) == float(b[j])
        prob += pulp.lpSum(C[i, j] * x[(i, j)] for (i, j) in x)
        prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=10))
        if pulp.LpStatus[prob.status] != "Optimal":
            return np.zeros((m, n), dtype=int), time.time() - t0, False
        X = np.zeros((m, n), dtype=float)
        for (i, j), var in x.items():
            X[i, j] = pulp.value(var)

    Xint = np.rint(X).astype(int)
    if not (np.all(Xint.sum(axis=1) == a) and np.all(Xint.sum(axis=0) == b)):
        return np.zeros((m, n), dtype=int), time.time() - t0, False
    return Xint, (time.time() - t0), True


def round_transport_greedy_push(a: np.ndarray, b: np.ndarray, C: np.ndarray,
                                u: np.ndarray, v: np.ndarray, tau: float,
                                topk: Optional[int] = None) -> Tuple[np.ndarray, float, bool]:
    """
    Greedy push by reduced cost r_ij = C_ij - tau*(log u_i + log v_j).
    Returns (Xint, time_s, ok). Falls back to full LP if infeasible.

    (This is the active definition from the paper file; the earlier
    same-named `(a,b,C,u,v)` variant there was dead code -- see module docstring.)
    """
    t0 = time.time()
    m, n = len(a), len(b)
    A = a.astype(int).copy()
    B = b.astype(int).copy()
    X = np.zeros((m, n), dtype=int)
    alpha = tau * np.log(np.maximum(u, 1e-18))
    beta = tau * np.log(np.maximum(v, 1e-18))

    for i in range(m):
        if A[i] == 0:
            continue
        r = C[i] - alpha[i] - beta  # smaller is better
        if topk is not None and topk < n:
            J = np.argpartition(r, topk - 1)[:topk]
            J = J[np.argsort(r[J])]
        else:
            J = np.argsort(r)
        for j in J:
            if A[i] == 0:
                break
            if B[j] == 0:
                continue
            f = min(A[i], B[j])
            X[i, j] += f
            A[i] -= f
            B[j] -= f

    # If any residue remains, try a tiny restricted LP repair on leftover rows/cols
    if A.sum() != 0 or B.sum() != 0:
        I = np.flatnonzero(A > 0)
        J = np.flatnonzero(B > 0)
        if I.size and J.size:
            mask = np.zeros_like(C, dtype=bool)
            mask[np.ix_(I, J)] = True
            Xrep, t_rep, ok = round_transport_min_cost_lp_restricted(
                a - X.sum(axis=1), b - X.sum(axis=0), C, mask
            )
            if ok:
                X += Xrep
                A = a - X.sum(axis=1)
                B = b - X.sum(axis=0)

    # If still infeasible (rare), fall back to full LP to keep demo robust.
    if not (np.all(X.sum(axis=1) == a) and np.all(X.sum(axis=0) == b)):
        X, t_lp = round_transport_min_cost_lp(a, b, C)
        return X, (time.time() - t0) + t_lp, False

    return X, (time.time() - t0), True
