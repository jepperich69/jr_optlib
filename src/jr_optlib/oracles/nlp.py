# -*- coding: utf-8 -*-
"""Oracles for non-linear programming primitives."""

import time
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional

def verify_with_gurobi(
    D: np.ndarray,
    L: np.ndarray,
    k_v: np.ndarray,
    L_k: np.ndarray,
    c_k_v: np.ndarray,
    B: float,
    K: int,
    beta: float,
    alpha_tilde: float,
    delta: float,
    B_coeff: float,
    G_min: float,
    G_max: float,
    f_min: float,
    f_max: float,
    G_star: np.ndarray,
    f_star: np.ndarray,
    time_limit: int = 120
) -> Tuple[Optional[Dict], pd.DataFrame]:
    """
    Gurobi verification (warm-start + analytic bounds) for bilinear transit.
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        print("gurobipy not found -- skipping Gurobi verification.")
        return None, pd.DataFrame()

    t0 = time.perf_counter()
    Z = len(D)

    G_lo = np.full(Z, G_min)
    G_hi = np.full(Z, G_max)
    f_lo = np.full(K, f_min)
    f_hi = np.full(K, f_max)

    m = gp.Model("cph_nlp_verify")
    m.Params.NonConvex  = 2
    m.Params.OutputFlag = 0
    m.Params.TimeLimit  = time_limit

    G = m.addVars(Z, lb=G_lo.tolist(), ub=G_hi.tolist(), name="G")
    f = m.addVars(K, lb=f_lo.tolist(), ub=f_hi.tolist(), name="f")
    u = m.addVars(Z, lb=(1/G_hi).tolist(), ub=(1/G_lo).tolist(), name="u")
    w = m.addVars(K, lb=(1/f_hi).tolist(), ub=(1/f_lo).tolist(), name="w")

    obj = gp.quicksum(
        D[z] * L[z] * (beta * u[z] + alpha_tilde * G[z] + delta * w[k_v[z]])
        for z in range(Z)
    )
    m.setObjective(obj, GRB.MINIMIZE)

    p = m.addVars(Z, lb=0, name="p")
    for z in range(Z):
        k = k_v[z]
        m.addConstr(p[z] == f[k] * u[z], name=f"p_{z}")

    budget_expr = (
        gp.quicksum(c_k_v[k] * L_k[k] * f[k] for k in range(K))
        + gp.quicksum(B_coeff * L[z] * p[z] for z in range(Z))
    )
    m.addConstr(budget_expr <= B, name="budget")

    for z in range(Z):
        m.addConstr(u[z] * G[z] == 1, name=f"uG_{z}")
    for k in range(K):
        m.addConstr(w[k] * f[k] == 1, name=f"wf_{k}")

    m.update()

    for z in range(Z):
        G[z].Start = np.clip(G_star[z], G_lo[z], G_hi[z])
        u[z].Start = 1.0 / np.clip(G_star[z], G_lo[z], G_hi[z])
    for k in range(K):
        f[k].Start = np.clip(f_star[k], f_lo[k], f_hi[k])
        w[k].Start = 1.0 / np.clip(f_star[k], f_lo[k], f_hi[k])
    for z in range(Z):
        k = k_v[z]
        p[z].Start = f_star[k] * (1.0 / G_star[z])

    trace = []

    def callback(cb_model, where):
        if where != GRB.Callback.MIP:
            return
        runtime = cb_model.cbGet(GRB.Callback.RUNTIME)
        incumbent = cb_model.cbGet(GRB.Callback.MIP_OBJBST)
        bound = cb_model.cbGet(GRB.Callback.MIP_OBJBND)
        if abs(incumbent) >= GRB.INFINITY:
            incumbent = np.nan
        if abs(bound) >= GRB.INFINITY:
            bound = np.nan
        if not trace or runtime - trace[-1]["time_s"] >= 0.1:
            trace.append({
                "time_s": runtime,
                "incumbent": incumbent,
                "bound": bound,
            })

    m.optimize(callback)
    t1 = time.perf_counter()
    
    status_map = {2: "OPTIMAL", 9: "TIME_LIMIT", 3: "INFEASIBLE", 5: "UNBOUNDED"}
    status_str = status_map.get(m.Status, str(m.Status))

    summary = {
        "status": status_str,
        "runtime_s": t1 - t0,
        "solver_runtime_s": m.Runtime,
        "node_count": m.NodeCount,
        "variables": m.NumVars,
        "constraints": m.NumConstrs + m.NumQConstrs,
        "coordinate_objective": np.nan,
        "incumbent": np.nan,
        "bound": np.nan,
        "global_gap_pct": np.nan,
        "coordinate_excess_vs_incumbent_pct": np.nan,
    }
    if m.SolCount > 0:
        gap = 100.0 * (m.ObjVal - m.ObjBound) / max(abs(m.ObjVal), 1e-10)
        coord_obj = np.sum(D * L * (beta / G_star + alpha_tilde * G_star + delta / f_star[k_v]))
        summary.update({
            "coordinate_objective": coord_obj,
            "incumbent": m.ObjVal,
            "bound": m.ObjBound,
            "global_gap_pct": gap,
            "coordinate_excess_vs_incumbent_pct": 100.0 * (coord_obj / m.ObjVal - 1.0)
        })

    return summary, pd.DataFrame(trace)
