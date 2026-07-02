# -*- coding: utf-8 -*-
"""Entropy-guided set-cover heuristic.

Numerics-preserving extraction from
``Pub_MIPEntropy_MPC/code/mip_hybrid/apps/synth_setcover.py`` for the
Gurobi-free core: seeded synthetic instances, row-wise entropy relaxation, and
dual-guided rounding. Gurobi restricted-polish and MH polish are intentionally
left out of this first slice.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class SetCoverInstance:
    A: List[List[int]]
    c: np.ndarray
    n: int
    m: int
    k: int = 1
    _rows_of_col: Optional[List[np.ndarray]] = None


def gen_entropy_friendly_scp(n: int, m: int, seed: int = 42):
    """Generate the synthetic set-cover instances used in MIPEntropy."""
    rng = np.random.default_rng(seed)
    density = 0.02
    A = (rng.random((n, m)) < density).astype(np.uint8)

    row_cov = A.sum(axis=1)
    for i, c in enumerate(row_cov):
        if c == 0:
            j = rng.integers(0, m)
            A[i, j] = 1

    col_coverage = A.sum(axis=0)
    base_costs = rng.random(m) * 0.5 + 0.1
    coverage_penalty = col_coverage / col_coverage.max() * 0.5
    c = base_costs + coverage_penalty
    return A, c


def build_instance_from_matrix(A_matrix, c) -> SetCoverInstance:
    """Convert an n x m incidence matrix to the row/column views used here."""
    n, m = A_matrix.shape
    A = []
    for i in range(n):
        cols = np.where(A_matrix[i, :] == 1)[0].tolist()
        A.append(cols)

    inst = SetCoverInstance(A=A, c=c, n=n, m=m, k=1)

    rows_of_col = [[] for _ in range(m)]
    for i, cols in enumerate(A):
        for j in cols:
            rows_of_col[j].append(i)

    inst._rows_of_col = [
        np.array(r, dtype=np.int32) if r else np.empty(0, dtype=np.int32)
        for r in rows_of_col
    ]
    return inst


def ipf_rowwise_entropy(inst: SetCoverInstance, tau: float, iters: int, tol: float):
    """Row-wise IPF entropy relaxation for set cover."""
    t0 = time.time()
    q = np.exp(-inst.c / max(tau, 1e-12))
    x = q.copy()
    y = np.zeros(inst.n, dtype=float)

    rhs = float(inst.k)

    for sweep in range(iters):
        max_violation = 0.0
        for i in range(inst.n):
            cols = inst.A[i]
            if not cols:
                continue
            cols = np.array(cols)
            s = float(x[cols].sum())
            if s < rhs - tol:
                alpha = rhs / max(s, 1e-12)
                x[cols] *= alpha
                y[i] += tau * math.log(alpha)
                max_violation = max(max_violation, rhs - s)

        np.minimum(x, 1.0, out=x)

        if max_violation <= tol:
            break

    cov = np.zeros(inst.n, dtype=float)
    for i in range(inst.n):
        cov[i] = x[inst.A[i]].sum() if inst.A[i] else 0.0

    lin_cost = float(inst.c @ x)
    smooth_obj = lin_cost + tau * float(np.sum(x * (np.log(np.maximum(x, 1e-12)) - 1.0)))

    return x, y, smooth_obj, time.time() - t0, lin_cost, float(cov.min()), float(cov.mean())


def entropy_relax_with_annealing(
    inst: SetCoverInstance,
    tau: float = 0.1,
    iters: int = 50,
    tol: float = 1e-3,
    tau_schedule: Optional[List[float]] = None,
):
    """Run the row-wise entropy relaxation over one or more temperatures."""
    taus = tau_schedule if tau_schedule else [tau]
    x = None
    total_time = 0.0
    prev_t = tau

    for t in taus:
        if x is not None:
            reweight = np.exp(-(1.0 / max(t, 1e-12) - 1.0 / max(prev_t, 1e-12)) * inst.c)
            x = np.clip(x * reweight, 1e-18, 1.0)

        stage_start = time.time()
        x, y, smooth_obj, dt, lin_cost, cov_min, cov_avg = ipf_rowwise_entropy(
            inst, tau=t, iters=iters, tol=tol
        )
        stage_time = time.time() - stage_start
        total_time += stage_time

        prev_t = t

    return x, y, smooth_obj, total_time, lin_cost, cov_min, cov_avg


def compute_reduced_costs(inst: SetCoverInstance, y: np.ndarray):
    """Compute reduced costs r_j = c_j - sum_i y_i over covered rows."""
    r = inst.c.copy()
    for j in range(inst.m):
        rows = inst._rows_of_col[j]
        if rows.size > 0:
            r[j] -= y[rows].sum()
    return r


def round_cover_dual_guided(inst: SetCoverInstance, x_frac: np.ndarray, y: np.ndarray):
    """Dual-guided rounding plus drop-fix pruning."""
    n, m, k = inst.n, inst.m, inst.k
    r = compute_reduced_costs(inst, y)

    x_int = np.zeros(m, dtype=int)
    cover_cnt = np.zeros(n, dtype=int)

    nz = np.where(r <= 1e-12)[0]
    if nz.size:
        x_int[nz] = 1
        for j in nz:
            rows = inst._rows_of_col[j]
            if rows.size:
                cover_cnt[rows] += 1

    while np.any(cover_cnt < k):
        best_j = -1
        best_ratio = float("inf")

        deficit_rows = np.where(cover_cnt < k)[0]
        candidates = set()
        for i in deficit_rows:
            for j in inst.A[i]:
                if x_int[j] == 0:
                    candidates.add(j)

        for j in candidates:
            rows = inst._rows_of_col[j]
            gain = np.sum(cover_cnt[rows] < k) if rows.size else 0
            if gain > 0:
                ratio = max(1e-12, r[j]) / gain
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_j = j

        if best_j == -1:
            i0 = deficit_rows[0]
            candidates = [j for j in inst.A[i0] if x_int[j] == 0]
            if not candidates:
                break
            best_j = min(candidates, key=lambda j: r[j])

        x_int[best_j] = 1
        rows = inst._rows_of_col[best_j]
        if rows.size:
            cover_cnt[rows] += 1

    on_cols = np.where(x_int == 1)[0]
    for j in on_cols:
        rows = inst._rows_of_col[j]
        if rows.size == 0:
            x_int[j] = 0
            continue
        if not np.any(cover_cnt[rows] - 1 < k):
            x_int[j] = 0
            cover_cnt[rows] -= 1

    cov_min = float(cover_cnt.min())
    feasible = bool(cov_min >= k)
    cost = float(inst.c @ x_int)

    return x_int, cost, cov_min, feasible


def polish_solution(x_int, A, c, polish_time: float = 1.0,
                    polish_pool: float = 0.3, x_frac=None):
    """Restricted Gurobi MIP polish around the current set-cover solution."""
    t0 = time.time()
    n, m = A.shape

    if x_frac is not None:
        proximity = 1.0 - 2.0 * np.abs(x_frac - 0.5)
        top_k = int(m * polish_pool)
        candidates = np.argsort(-proximity)[:top_k]
    else:
        import random
        on_vars = np.where(x_int == 1)[0]
        top_k = max(len(on_vars), int(m * polish_pool))
        candidates = random.sample(range(m), min(top_k, m))

    on_vars = set(np.where(x_int == 1)[0])
    candidates = np.array(list(set(candidates) | on_vars))

    try:
        import gurobipy as gp
        from gurobipy import GRB

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.setParam("TimeLimit", polish_time)
            env.setParam("Threads", 1)
            env.start()

            with gp.Model(env=env) as model:
                x = model.addVars(m, vtype=GRB.BINARY, name="x")

                for j in range(m):
                    if j not in candidates:
                        x[j].lb = x[j].ub = int(x_int[j])

                for j in range(m):
                    x[j].Start = int(x_int[j])

                for i in range(n):
                    idx = np.where(A[i, :] == 1)[0]
                    if len(idx) > 0:
                        model.addConstr(gp.quicksum(x[j] for j in idx) >= 1.0)

                model.setObjective(gp.quicksum(c[j] * x[j] for j in range(m)), GRB.MINIMIZE)
                model.optimize()

                if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
                    x_polished = np.array([x[j].X for j in range(m)])
                    x_polished = np.round(x_polished).astype(int)
                    cost_polished = float(c @ x_polished)
                    return x_polished, cost_polished, time.time() - t0

                return x_int, float(c @ x_int), time.time() - t0

    except Exception:
        return x_int, float(c @ x_int), time.time() - t0


def solve_mip(A, c, timelimit_s=30, gurobi_time_limit=None,
              gurobi_gap_limit=None, track_gurobi_anytime=False):
    """Solve a set-cover MIP with Gurobi, matching the paper helper API."""
    n, m = A.shape
    actual_time_limit = gurobi_time_limit if gurobi_time_limit is not None else timelimit_s

    try:
        import gurobipy as gp

        anytime_log = [] if track_gurobi_anytime else None

        def callback(model, where):
            if where == gp.GRB.Callback.MIPSOL:
                cur_obj = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
                cur_time = model.cbGet(gp.GRB.Callback.RUNTIME)
                anytime_log.append((cur_time, cur_obj, "gurobi_incumbent"))

        t0 = time.time()
        env = gp.Env(empty=True)

        for name in ("WLSAccessID", "WLSSecret", "LicenseID", "WLSToken"):
            v = os.getenv(f"GRB_{name.upper()}")
            if v:
                env.setParam(name, int(v) if name == "LicenseID" else v)

        for cand in ("gurobi.env", os.path.join(os.getcwd(), "gurobi.env")):
            if os.path.isfile(cand):
                env.readParams(cand)
                break

        if actual_time_limit is not None:
            env.setParam("TimeLimit", float(actual_time_limit))

        if gurobi_gap_limit is not None:
            env.setParam("MIPGap", float(gurobi_gap_limit))
        else:
            gap_env = os.getenv("GRB_MIPGAP") or os.getenv("MIP_GAP")
            if gap_env:
                env.setParam("MIPGap", float(gap_env))

        thr_env = os.getenv("GRB_THREADS") or os.getenv("THREADS")
        if thr_env:
            env.setParam("Threads", int(thr_env))

        env.start()
        model = gp.Model(env=env)
        model.Params.OutputFlag = 0

        x = model.addMVar(shape=m, vtype=gp.GRB.BINARY, name="x")

        for i in range(n):
            idx = np.where(A[i, :] == 1)[0]
            if idx.size:
                model.addConstr(x[idx].sum() >= 1.0)

        model.setObjective((c @ x), gp.GRB.MINIMIZE)

        if track_gurobi_anytime:
            model.optimize(callback)
        else:
            model.optimize()

        t1 = time.time()

        if model.Status not in (gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT):
            return {
                "obj": np.nan,
                "time": t1 - t0,
                "bound": np.nan,
                "gap": np.nan,
                "anytime_log": anytime_log,
            }

        obj_val = float(model.ObjVal)
        try:
            bound = float(model.ObjBound)
            denom = max(abs(obj_val), 1e-9)
            gap_pct = 100.0 * max(0.0, (obj_val - bound) / denom)
        except Exception:
            bound, gap_pct = (np.nan, np.nan)

        return {
            "obj": obj_val,
            "time": t1 - t0,
            "bound": bound,
            "gap": gap_pct,
            "anytime_log": anytime_log,
        }

    except Exception:
        return {"obj": np.nan, "time": np.nan, "bound": np.nan, "gap": np.nan, "anytime_log": None}


def solve_entropy_setcover(
    A_matrix,
    c,
    tau: float = 0.1,
    iters: int = 50,
    tol: float = 1e-3,
    tau_schedule=None,
    polish_time: float = 0.0,
    polish_pool: float = 0.3,
    do_polish_mh: bool = False,
    mh_tau=None,
    mh_steps: int = 150,
    mh_seed=None,
):
    """RICH set-cover core: entropy relaxation then dual-guided rounding.

    This first migrated slice supports Gurobi restricted polish, but not MH
    polish.
    """
    if do_polish_mh:
        raise NotImplementedError("MH polish is not migrated in this slice")

    inst = build_instance_from_matrix(A_matrix, c)

    taus = None
    if tau_schedule:
        try:
            taus = [float(t.strip()) for t in tau_schedule.split(",")]
        except Exception:
            taus = None

    x_frac, y, smooth_obj, t_relax, lin_cost, cov_min, cov_avg = entropy_relax_with_annealing(
        inst, tau=tau, iters=iters, tol=tol, tau_schedule=taus
    )

    x_int, cost, cov_min_int, feasible = round_cover_dual_guided(inst, x_frac, y)

    total_time = t_relax
    if polish_time > 0 and feasible:
        x_int, cost, t_polish = polish_solution(
            x_int,
            A_matrix,
            c,
            polish_time=polish_time,
            polish_pool=polish_pool,
            x_frac=x_frac,
        )
        total_time += t_polish

    return x_int, cost, feasible, total_time
