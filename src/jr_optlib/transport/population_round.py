# -*- coding: utf-8 -*-
"""
Dependent rounding of a fractional 2D table (e.g. an IPF fit) to integers,
preserving both marginals exactly with per-cell deviation < 1.

VETTED FUNCTIONS -- registry ids: transport.dependent_round_2d (+ _lp / _lp_sparsified / _mcf).
Migrated numerics-preserving from
  Pub_MIPEntropy_MPC/code/mip_hybrid/apps/population_transport.py
where they were the private `_dependent_round_2d_*` helpers. Public names here.

This completes the population-synthesis pipeline: ipf_2d gives a fractional
table matching the marginals; these routines round it to a non-negative integer
table that still matches the marginals, moving each cell by < 1. The contract
(exact marginals, integrality, max|Xint-X| < 1) is certified by
jr_optlib.oracles.certify_dependent_round.
"""

import time
from typing import Tuple

import numpy as np

from jr_optlib.transport._backend import backend_name

__all__ = [
    "dependent_round_2d",
    "dependent_round_2d_lp",
    "dependent_round_2d_lp_sparsified",
    "dependent_round_2d_mcf",
]


def dependent_round_2d_lp(X: np.ndarray, row_marg: np.ndarray,
                          col_marg: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    LP on residuals with 0-1 bounds (TU => integral). Per-cell deviation < 1.
    Objective favors larger fractional parts: min sum (1-frac)*z.
    """
    t0 = time.time()
    R, C = X.shape
    F = np.floor(X).astype(int)
    dr = (row_marg - F.sum(axis=1)).astype(int)
    dc = (col_marg - F.sum(axis=0)).astype(int)
    assert dr.sum() == dc.sum(), "Row/col deficits mismatch"

    frac = X - F
    if backend_name() == "ortools":
        from ortools.linear_solver import pywraplp
        solver = pywraplp.Solver.CreateSolver("GLOP")  # LP is enough; TU + bounds -> integer
        z = [[solver.NumVar(0.0, 1.0, f"z_{i}_{j}") for j in range(C)] for i in range(R)]
        for i in range(R):
            ct = solver.RowConstraint(float(dr[i]), float(dr[i]), "")
            for j in range(C):
                ct.SetCoefficient(z[i][j], 1.0)
        for j in range(C):
            ct = solver.RowConstraint(float(dc[j]), float(dc[j]), "")
            for i in range(R):
                ct.SetCoefficient(z[i][j], 1.0)
        obj = solver.Objective()
        for i in range(R):
            for j in range(C):
                obj.SetCoefficient(z[i][j], float(1.0 - frac[i, j]))
        obj.SetMinimization()
        solver.Solve()
        Z = np.array([[z[i][j].solution_value() for j in range(C)] for i in range(R)])
    else:
        import pulp
        prob = pulp.LpProblem("residual_01", pulp.LpMinimize)
        z = [[pulp.LpVariable(f"z_{i}_{j}", lowBound=0, upBound=1, cat="Continuous") for j in range(C)] for i in range(R)]
        for i in range(R):
            prob += pulp.lpSum(z[i][j] for j in range(C)) == dr[i]
        for j in range(C):
            prob += pulp.lpSum(z[i][j] for i in range(R)) == dc[j]
        prob += pulp.lpSum((1.0 - frac[i, j]) * z[i][j] for i in range(R) for j in range(C))
        prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=10))
        Z = np.array([[pulp.value(z[i][j]) for j in range(C)] for i in range(R)])

    Z = np.rint(Z).astype(int)
    Xint = F + Z
    # checks
    assert np.all(Xint.sum(axis=1) == row_marg)
    assert np.all(Xint.sum(axis=0) == col_marg)
    assert float(np.abs(Xint - X).max()) < 1.0 + 1e-9
    return Xint, (time.time() - t0)


def dependent_round_2d_lp_sparsified(X: np.ndarray, row_marg: np.ndarray,
                                     col_marg: np.ndarray, topk_extra: int = 8,
                                     eta: float = 0.02,
                                     eps_slack: float = 0.0) -> Tuple[np.ndarray, float]:
    """
    Sparsified residual LP: build variables only on a candidate edge set cut down
    by (i) per-row top-K fractional cells and (ii) per-column top-K, with
    eta-pruning. Falls back to the dense LP if the reduced LP is infeasible.

    Guarantees: exact marginals and per-cell deviation < 1 when eps_slack == 0.
    """
    t0 = time.time()
    R, C = X.shape
    F = np.floor(X).astype(int)
    dr = (row_marg - F.sum(axis=1)).astype(int)
    dc = (col_marg - F.sum(axis=0)).astype(int)
    assert dr.sum() == dc.sum(), "Row/col deficits mismatch"

    frac = X - F

    # --- Build candidate mask E (R x C) ---
    E = np.zeros((R, C), dtype=bool)

    # (a) Per-ROW candidates
    for i in range(R):
        if dr[i] <= 0:
            continue
        J_eta = np.flatnonzero(frac[i] >= eta)
        if J_eta.size < dr[i]:
            K = min(C, dr[i] + topk_extra)
            J_top = np.argpartition(-frac[i], K - 1)[:K]
            J = np.unique(np.concatenate([J_eta, J_top]))
        else:
            K = min(J_eta.size, dr[i] + topk_extra)
            idx = np.argpartition(frac[i, J_eta], -(K))[-K:]
            J = J_eta[idx]
            J = J[np.argsort(-frac[i, J])]
        E[i, J] = True

    # (b) Per-COLUMN candidates
    for j in range(C):
        if dc[j] <= 0:
            continue
        I_eta = np.flatnonzero(frac[:, j] >= eta)
        if I_eta.size < dc[j]:
            Kc = min(R, dc[j] + topk_extra)
            I_top = np.argpartition(-frac[:, j], Kc - 1)[:Kc]
            I = np.unique(np.concatenate([I_eta, I_top]))
        else:
            Kc = min(I_eta.size, dc[j] + topk_extra)
            idx = np.argpartition(frac[I_eta, j], -(Kc))[-Kc:]
            I = I_eta[idx]
            I = I[np.argsort(-frac[I, j])]
        E[I, j] = True

    # Quick feasibility sanity: rows/cols with positive deficit must have some edges
    if np.any((dr > 0) & (E.sum(axis=1) == 0)) or np.any((dc > 0) & (E.sum(axis=0) == 0)):
        return dependent_round_2d_lp(X, row_marg, col_marg)

    # --- Build the reduced LP on edges in E ---
    use_ort = (backend_name() == "ortools")
    try:
        if use_ort:
            from ortools.linear_solver import pywraplp
            solver = pywraplp.Solver.CreateSolver("GLOP")
            if solver is None:
                raise RuntimeError("Failed to create GLOP solver")

            z = [[None] * C for _ in range(R)]
            for i in range(R):
                for j in np.flatnonzero(E[i]):
                    z[i][j] = solver.NumVar(0.0, 1.0, f"z_{i}_{j}")

            if eps_slack > 0:
                s_row = [solver.NumVar(-eps_slack, eps_slack, f"s_row_{i}") for i in range(R)]
                s_col = [solver.NumVar(-eps_slack, eps_slack, f"s_col_{j}") for j in range(C)]
            else:
                s_row = s_col = None

            for i in range(R):
                if dr[i] == 0:
                    continue
                ct = solver.RowConstraint(float(dr[i]), float(dr[i]), "")
                has_any = False
                for j in np.flatnonzero(E[i]):
                    ct.SetCoefficient(z[i][j], 1.0); has_any = True
                if s_row is not None:
                    ct.SetCoefficient(s_row[i], 1.0)
                if not has_any:
                    raise RuntimeError("Row has deficit but no candidate vars")

            for j in range(C):
                if dc[j] == 0:
                    continue
                ct = solver.RowConstraint(float(dc[j]), float(dc[j]), "")
                has_any = False
                for i in np.flatnonzero(E[:, j]):
                    ct.SetCoefficient(z[i][j], 1.0); has_any = True
                if s_col is not None:
                    ct.SetCoefficient(s_col[j], 1.0)
                if not has_any:
                    raise RuntimeError("Column has deficit but no candidate vars")

            obj = solver.Objective()
            for i in range(R):
                for j in np.flatnonzero(E[i]):
                    obj.SetCoefficient(z[i][j], float(1.0 - frac[i, j]))
            if s_row is not None and s_col is not None:
                lam = 1e3
                for i in range(R):
                    obj.SetCoefficient(s_row[i], lam)
                for j in range(C):
                    obj.SetCoefficient(s_col[j], lam)
            obj.SetMinimization()

            status = solver.Solve()
            if status != pywraplp.Solver.OPTIMAL:
                return dependent_round_2d_lp(X, row_marg, col_marg)

            Z = np.zeros_like(F, dtype=float)
            for i in range(R):
                for j in np.flatnonzero(E[i]):
                    Z[i, j] = z[i][j].solution_value()

        else:
            import pulp
            prob = pulp.LpProblem("residual_01_sparse", pulp.LpMinimize)

            z = {}
            for i in range(R):
                for j in np.flatnonzero(E[i]):
                    z[(i, j)] = pulp.LpVariable(f"z_{i}_{j}", lowBound=0, upBound=1, cat="Continuous")

            if eps_slack > 0:
                s_row = {i: pulp.LpVariable(f"s_row_{i}", lowBound=-eps_slack, upBound=eps_slack) for i in range(R)}
                s_col = {j: pulp.LpVariable(f"s_col_{j}", lowBound=-eps_slack, upBound=eps_slack) for j in range(C)}
            else:
                s_row = s_col = {}

            for i in range(R):
                if dr[i] == 0:
                    continue
                vars_i = [z[(i, j)] for j in np.flatnonzero(E[i])]
                if not vars_i and eps_slack == 0:
                    return dependent_round_2d_lp(X, row_marg, col_marg)
                prob += pulp.lpSum(vars_i) + (s_row.get(i, 0)) == float(dr[i])

            for j in range(C):
                if dc[j] == 0:
                    continue
                vars_j = [z[(i, j)] for i in np.flatnonzero(E[:, j])]
                if not vars_j and eps_slack == 0:
                    return dependent_round_2d_lp(X, row_marg, col_marg)
                prob += pulp.lpSum(vars_j) + (s_col.get(j, 0)) == float(dc[j])

            obj = []
            for i in range(R):
                for j in np.flatnonzero(E[i]):
                    obj.append((1.0 - float(frac[i, j])) * z[(i, j)])
            if eps_slack > 0:
                lam = 1e3
                obj += [lam * s_row[i] for i in s_row]
                obj += [lam * s_col[j] for j in s_col]
            prob += pulp.lpSum(obj)

            prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=15))
            if pulp.LpStatus[prob.status] != "Optimal":
                return dependent_round_2d_lp(X, row_marg, col_marg)

            Z = np.zeros_like(F, dtype=float)
            for (i, j), var in z.items():
                Z[i, j] = pulp.value(var)

        Z = np.rint(Z).astype(int)
        Xint = F + Z
        assert np.all(Xint.sum(axis=1) == row_marg)
        assert np.all(Xint.sum(axis=0) == col_marg)
        assert float(np.abs(Xint - X).max()) < 1.0 + 1e-9
        return Xint, (time.time() - t0)

    except Exception:
        return dependent_round_2d_lp(X, row_marg, col_marg)


def dependent_round_2d_mcf(X, row_marg, col_marg, cost_scale: int = 200_000,
                           topk_extra: int = 8):
    """
    Residual 0-1 min-cost flow using OR-Tools SimpleMinCostFlow (faster when
    available). Per-cell deviation < 1 guaranteed by unit capacities. Falls back
    to the dense LP if the graph module is unavailable or infeasible.
    Returns (Xint, time_sec).
    """
    try:
        from ortools.graph import pywrapgraph
    except Exception:
        return dependent_round_2d_lp(X, row_marg, col_marg)

    R, C = X.shape
    F = np.floor(X).astype(int)
    dr = (row_marg - F.sum(axis=1)).astype(int)
    dc = (col_marg - F.sum(axis=0)).astype(int)
    assert dr.sum() == dc.sum(), "Row/col deficits mismatch"

    frac = X - F
    start = time.time()

    row_edges = []
    for i in range(R):
        if dr[i] <= 0:
            row_edges.append(np.array([], dtype=int))
            continue
        K = min(C, int(dr[i]) + topk_extra)
        idx = np.argpartition(-frac[i], K - 1)[:K]
        idx = idx[np.argsort(-frac[i, idx])]
        row_edges.append(idx)

    smcf = pywrapgraph.SimpleMinCostFlow()
    S = R + C
    T = R + C + 1

    def add_arc(u, v, cap, cost):
        smcf.AddArcWithCapacityAndUnitCost(int(u), int(v), int(cap), int(cost))

    total = int(dr.sum())

    for i in range(R):
        if dr[i] > 0:
            add_arc(S, i, int(dr[i]), 0)

    for i in range(R):
        idxs = row_edges[i]
        for j in idxs:
            w = max(0.0, 1.0 - float(frac[i, j]))
            add_arc(i, R + int(j), 1, int(round(w * cost_scale)))

    for j in range(C):
        if dc[j] > 0:
            add_arc(R + j, T, int(dc[j]), 0)

    node_count = R + C + 2
    supplies = [0] * node_count
    supplies[S] = total
    supplies[T] = -total
    for v, b in enumerate(supplies):
        smcf.SetNodeSupply(v, b)

    status = smcf.Solve()
    if status != smcf.OPTIMAL or (smcf.OptimalCost() == 0 and total > 0):
        return dependent_round_2d_lp(X, row_marg, col_marg)

    Z = np.zeros_like(F, dtype=int)
    for a in range(smcf.NumArcs()):
        u = smcf.Tail(a); v = smcf.Head(a); f = smcf.Flow(a)
        if 0 <= u < R and R <= v < R + C and f > 0:
            j = v - R
            Z[u, j] = 1

    Xint = F + Z
    assert np.all(Xint.sum(axis=1) == row_marg)
    assert np.all(Xint.sum(axis=0) == col_marg)
    assert float(np.abs(Xint - X).max()) < 1.0 + 1e-9
    return Xint, time.time() - start


def dependent_round_2d(X: np.ndarray, row_marg: np.ndarray, col_marg: np.ndarray,
                       method: str = "mcf", sparsify: bool = False,
                       topk_extra: int = 8, eta: float = 0.02,
                       eps_slack: float = 0.0) -> Tuple[np.ndarray, float]:
    """
    User-facing population rounder.
      - method="mcf": try SimpleMinCostFlow; if unavailable or infeasible, fall
                      back to LP (sparsified LP if sparsify=True, else dense LP)
      - method="lp" : use LP (sparsified if sparsify=True, else dense)
    """
    mth = method.lower()
    if mth == "mcf":
        try:
            from ortools.graph import pywrapgraph  # noqa: F401
            Xint, t = dependent_round_2d_mcf(X, row_marg, col_marg)
            return Xint, t
        except Exception:
            pass

    if sparsify:
        return dependent_round_2d_lp_sparsified(
            X, row_marg, col_marg,
            topk_extra=topk_extra, eta=eta, eps_slack=eps_slack,
        )
    else:
        return dependent_round_2d_lp(X, row_marg, col_marg)
