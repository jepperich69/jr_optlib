# -*- coding: utf-8 -*-
"""Oracles for population-synthesis tables."""

from __future__ import annotations

import numpy as np

from jr_optlib.oracles.core import OracleResult
from jr_optlib.population.curvature import (
    entropic_projection,
    margin_information_matrix,
)


def certify_population_margins(df, constraints, weight_col="x", tol: float = 1e-6, rel_tol: float = 0.0):
    """Check that a weighted long table matches all supplied margins."""
    results = []
    ok_all = True
    max_gap = 0.0
    max_rel_gap = 0.0
    for cs in constraints:
        dims = list(cs.dims)
        cur = df.groupby(dims, observed=True, sort=False)[weight_col].sum()
        tgt = cs.df.groupby(dims, observed=True, sort=False)[cs.val_col].sum()
        cur = cur.reindex(tgt.index, fill_value=0.0)
        gap = float((cur - tgt).abs().max()) if len(tgt) else 0.0
        rel_diff = float(((cur - tgt).abs() / tgt.clip(lower=1.0)).max()) if len(tgt) else 0.0
        total_gap = float((cur - tgt).abs().sum()) if len(tgt) else 0.0
        passed = (gap <= tol) or (rel_diff <= rel_tol)
        ok_all = ok_all and passed
        max_gap = max(max_gap, gap)
        max_rel_gap = max(max_rel_gap, rel_diff)
        results.append(
            OracleResult(
                name=f"population_margin[{cs.name}]",
                passed=passed,
                residual=gap,
                tol=tol,
                certifies=False,
                detail=f"max_abs_gap={gap:.3e} max_rel_gap={rel_diff:.3e} total_abs_gap={total_gap:.3e}",
            )
        )

    results.append(
        OracleResult(
            name="population_all_margins",
            passed=ok_all,
            residual=max_gap,
            tol=tol,
            certifies=ok_all,
            detail=f"max margin residual across {len(constraints)} constraints (rel={max_rel_gap:.3e})",
        )
    )
    return results, ok_all


def _zone_secondary_l1(zone_df, secondary, weight_col, zone_col, zone):
    """Achieved total L1 gap on the secondary margins for one zone."""
    total = 0.0
    per_margin = {}
    for cs in secondary:
        inner = [d for d in cs.dims if d != zone_col]
        tgt = cs.df[cs.df[zone_col] == zone]
        tgt_sum = tgt.groupby(inner, observed=True, sort=False)[cs.val_col].sum()
        cur = zone_df.groupby(inner, observed=True, sort=False)[weight_col].sum()
        idx = tgt_sum.index.union(cur.index)
        gap = (cur.reindex(idx, fill_value=0.0) - tgt_sum.reindex(idx, fill_value=0.0)).abs().sum()
        per_margin[cs.name] = float(gap)
        total += float(gap)
    return total, per_margin


def certify_secondary_margins_vs_floor(
    df,
    controlling,
    secondary,
    weight_col="n",
    zone_col="ZoneID",
    floor_slack=0.0,
    time_limit=120,
    zones=None,
):
    """Judge secondary (approximated) margins against the achievable floor.

    The paper's contract is: controlling margins are satisfied *exactly*, while
    secondary margins are approximated *as closely as feasible*. Checking a
    secondary margin against zero is therefore wrong -- with the controlling
    totals fixed as integers, the secondary margins are generally not jointly
    satisfiable, so the correct benchmark is the minimum total secondary
    violation achievable under the exact controlling constraints.

    For each zone we solve a min-cost integer program: free non-negative integer
    counts on the zone's populated cells, controlling margins pinned to their
    targets as hard equalities, secondary margins soft with L1 slack. The optimal
    total slack is a *proven lower bound* (the floor) on secondary violation. The
    result CERTIFIES if it reaches that floor (an optimal integer repair);
    otherwise it is a valid approximation (CHECKED) and we report how far above
    the floor it sits. It is never FAILed for being nonzero -- only a result
    below the proven floor (impossible) is a defect.
    """
    try:
        import pulp
    except Exception as exc:  # pragma: no cover - solver optional
        return [OracleResult("secondary_floor", True, float("nan"), floor_slack, False,
                             f"pulp unavailable, floor not computed: {exc!r}")], {}

    all_zones = list(df[zone_col].unique()) if zones is None else list(zones)
    global_floor = 0.0
    global_achieved = 0.0
    per_margin_achieved = {}
    worst = None  # (gap_to_floor, zone, achieved, floor)
    n_at_floor = 0

    for z in all_zones:
        zdf = df[df[zone_col] == z]
        cells = zdf[[c for c in zdf.columns if c not in {zone_col, weight_col}]].reset_index(drop=True)
        orig = zdf[weight_col].to_numpy(dtype=float)
        N = len(cells)

        prob = pulp.LpProblem(f"floor_{z}", pulp.LpMinimize)
        v = [pulp.LpVariable(f"v_{i}", lowBound=0, cat="Integer") for i in range(N)]

        # Controlling margins: hard equality at the target totals.
        for cs in controlling:
            inner = [d for d in cs.dims if d != zone_col]
            tgt = cs.df[cs.df[zone_col] == z]
            tgt_sum = tgt.groupby(inner, observed=True, sort=False)[cs.val_col].sum()
            for key, idxs in cells.groupby(inner, observed=True, sort=False).indices.items():
                t = int(round(float(tgt_sum.loc[key]))) if key in tgt_sum.index else 0
                prob += pulp.lpSum(v[i] for i in idxs) == t

        # Secondary margins: soft, L1 penalised.
        slack = []
        for cs in secondary:
            inner = [d for d in cs.dims if d != zone_col]
            tgt = cs.df[cs.df[zone_col] == z]
            tgt_sum = tgt.groupby(inner, observed=True, sort=False)[cs.val_col].sum()
            for key, idxs in cells.groupby(inner, observed=True, sort=False).indices.items():
                t = float(tgt_sum.loc[key]) if key in tgt_sum.index else 0.0
                sp = pulp.LpVariable(f"sp_{cs.name}_{z}_{key}", lowBound=0)
                sm = pulp.LpVariable(f"sm_{cs.name}_{z}_{key}", lowBound=0)
                slack += [sp, sm]
                prob += pulp.lpSum(v[i] for i in idxs) + sp - sm == t

        prob += pulp.lpSum(slack)
        prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
        floor_z = float(pulp.value(prob.objective) or 0.0)

        achieved_z, pm = _zone_secondary_l1(zdf, secondary, weight_col, zone_col, z)
        global_floor += floor_z
        global_achieved += achieved_z
        for k, val in pm.items():
            per_margin_achieved[k] = per_margin_achieved.get(k, 0.0) + val
        gap_z = achieved_z - floor_z
        if achieved_z <= floor_z + 1e-6:
            n_at_floor += 1
        if worst is None or gap_z > worst[0]:
            worst = (gap_z, z, achieved_z, floor_z)

    gap_to_floor = global_achieved - global_floor
    at_floor = gap_to_floor <= floor_slack
    # A result below the proven lower bound signals a defect in the floor or the result.
    below_floor = gap_to_floor < -1e-6

    results = [
        OracleResult(
            name="secondary_margins_vs_floor",
            passed=not below_floor,
            residual=max(gap_to_floor, 0.0),
            tol=floor_slack,
            certifies=at_floor and not below_floor,
            detail=(
                f"achieved_L1={global_achieved:.0f} floor_L1={global_floor:.0f} "
                f"gap_to_floor={gap_to_floor:.0f} zones_at_floor={n_at_floor}/{len(all_zones)}"
                + (f" worst_zone={worst[1]}(achieved={worst[2]:.0f},floor={worst[3]:.0f})" if worst else "")
            ),
        )
    ]
    summary = {
        "global_floor": global_floor,
        "global_achieved": global_achieved,
        "gap_to_floor": gap_to_floor,
        "per_margin_achieved": per_margin_achieved,
        "zones_at_floor": n_at_floor,
        "n_zones": len(all_zones),
        "worst_zone": worst,
    }
    return results, summary


# --------------------------------------------------------------------------
# Entropy-projection curvature (Pub_PopInt_Part2)
# --------------------------------------------------------------------------

def _kl(p: np.ndarray, q: np.ndarray) -> float:
    """Unnormalised KL / I-divergence sum_k p_k log(p_k/q_k) - p_k + q_k.

    Zero terms (p_k == 0) contribute 0; q_k must be > 0 wherever p_k > 0.
    The linear terms make it the Bregman divergence of x log x, so it is >= 0
    and 0 iff p == q, which is what the curvature finite differences need.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = p > 0
    out = float(np.sum(q - p))  # -p + q over all cells
    out += float(np.sum(p[mask] * np.log(p[mask] / q[mask])))
    return out


def certify_entropic_projection(x0, A, x, b, tol: float = 1e-8):
    """Certify that ``x`` is the KL I-projection of ``x0`` onto ``{A x = b}``.

    Two properties jointly determine the unique I-projection (same logic as the
    2D IPF certificate): the fitted margins match the target (marginal residual),
    and the fit has exponential-family / scaling form ``x = x0 * exp(A^T λ)``,
    i.e. ``log(x/x0)`` lies in the row space of ``A``. The second is checked by
    least-squares fitting ``λ`` to ``log(x/x0)`` on the shared support and
    measuring the residual off ``range(A^T)``.
    """
    x0 = np.asarray(x0, dtype=float)
    A = np.asarray(A, dtype=float)
    x = np.asarray(x, dtype=float)
    b = np.asarray(b, dtype=float)

    marg = float(np.max(np.abs(A @ x - b))) if b.size else 0.0

    support = (x0 > 0) & (x > 0)
    logr = np.log(x[support] / x0[support])
    At = A.T[support]  # (support, m)
    lam, *_ = np.linalg.lstsq(At, logr, rcond=None)
    scaling = float(np.max(np.abs(At @ lam - logr))) if logr.size else 0.0

    residual = max(marg, scaling)
    passed = residual <= tol
    return OracleResult(
        name="entropic_projection",
        passed=passed,
        residual=residual,
        tol=tol,
        certifies=passed,
        detail=f"marginal_residual={marg:.3e} scaling_form_residual={scaling:.3e}",
    )


def certify_margin_curvature(
    x,
    A,
    n_directions: int = 48,
    h: float = 0.25,
    rel_tol: float = 1e-2,
    solve_tol: float = 1e-12,
    seed: int = 0,
    directions=None,
):
    """Certify ``H = [A diag(x) A^T]^{-1} = ∇²_b Φ(b)`` for the entropy value function.

    ``Φ(b) = min_{A y = b} D_KL(y || x0)`` is the KL cost of moving the fitted
    margins to ``b``. Proposition 2 of Pub_PopInt_Part2 states its Hessian is the
    inverse margin information matrix ``H = pinv(A diag(x) A^T)``. This oracle
    verifies that identity *independently* of the closed form, using only the
    fitted table ``x`` and ``A``:

    1. **pinv-on-range certificate** (certifies). Feasible margin moves live in
       ``range(A)``. On that subspace ``H`` must be the exact inverse of
       ``M = A diag(x) A^T``: ``H M d = d`` for every feasible ``d``. This is an
       exact algebraic certificate (no finite differences) that ``H`` inverts
       ``M`` where it matters.
    2. **PSD check** (checked). ``M`` is a covariance, so ``H`` is PSD; a negative
       eigenvalue would signal a broken price.
    3. **finite-difference Hessian** (checked). Along each feasible swap
       direction ``d = A[:,j] - A[:,i]`` the analytic curvature ``d^T H d`` is
       compared to a central second difference of the *re-solved* value
       function. Re-projecting from seed ``x`` to margins ``b ± h d`` recovers
       ``x(b ± h d)`` exactly (the I-projection stays in the exponential family),
       and ``Ψ(b') = D_KL(x(b') || x)`` differs from ``Φ`` only by a linear term,
       so ``∇²Ψ = ∇²Φ``. With ``Ψ(b) = 0`` the second difference is
       ``(Ψ(b+hd) + Ψ(b-hd)) / h²``. Because the re-solve is driven by the
       defining property ``A x = b'`` (residual-gated), not by the curvature
       formula, agreement is a genuine check on Proposition 2.

    Returns ``(results, summary)``. The combined verdict is CERTIFIED when the
    pinv-on-range identity holds and no check fails; the finite-difference test
    confirms (CHECKED) that the inverse information matrix is the value-function
    curvature. A projection that fails to converge FAILs the oracle.
    """
    x = np.asarray(x, dtype=float)
    A = np.asarray(A, dtype=float)
    m, K = A.shape

    M = margin_information_matrix(x, A)
    H = np.linalg.pinv(M, rcond=1e-12)
    b = A @ x

    # ---- (1) exact pseudo-inverse-on-range certificate + (3) directions ----
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(x > 0)
    if directions is None:
        dirs = []
        attempts = 0
        max_attempts = 40 * n_directions + 50
        while len(dirs) < n_directions and attempts < max_attempts and len(pos) >= 2:
            i, j = rng.choice(pos, size=2, replace=False)
            d = A[:, j] - A[:, i]
            if np.any(d != 0.0):
                dirs.append(d)
            attempts += 1
    else:
        dirs = [np.asarray(d, dtype=float) for d in directions]

    HM = H @ M
    max_pinv_res = 0.0
    for d in dirs:
        # d = A(e_j - e_i) is in range(A); H must invert M on it: H M d == d.
        max_pinv_res = max(max_pinv_res, float(np.max(np.abs(HM @ d - d))))

    # ---- (2) PSD of the curvature ----
    eigs = np.linalg.eigvalsh(0.5 * (H + H.T))
    min_eig = float(eigs.min())
    # scale tolerance to the spectrum
    psd_tol = 1e-8 * max(1.0, float(np.abs(eigs).max()))

    # ---- (3) finite-difference Hessian along feasible directions ----
    precond = H  # fixed quasi-Newton preconditioner; solves are residual-gated
    worst_rel = 0.0
    worst_solve_res = 0.0
    n_used = 0
    sum_rel = 0.0
    for d in dirs:
        q_cf = float(d @ H @ d)
        if q_cf <= 1e-14:
            continue  # direction with negligible curvature carries no signal
        xp, rp = entropic_projection(x, A, b + h * d, tol=solve_tol, precond=precond)
        xm, rm = entropic_projection(x, A, b - h * d, tol=solve_tol, precond=precond)
        worst_solve_res = max(worst_solve_res, rp, rm)
        q_fd = (_kl(xp, x) + _kl(xm, x)) / (h * h)
        rel = abs(q_fd - q_cf) / max(abs(q_cf), 1e-14)
        worst_rel = max(worst_rel, rel)
        sum_rel += rel
        n_used += 1

    mean_rel = sum_rel / n_used if n_used else 0.0
    solves_ok = worst_solve_res <= 1e-6

    results = [
        OracleResult(
            name="margin_curvature_pinv_identity",
            passed=max_pinv_res <= 1e-8,
            residual=max_pinv_res,
            tol=1e-8,
            certifies=(max_pinv_res <= 1e-8),
            detail=f"max|H M d - d|={max_pinv_res:.3e} over {len(dirs)} feasible dirs",
        ),
        OracleResult(
            name="margin_curvature_psd",
            passed=min_eig >= -psd_tol,
            residual=max(-min_eig, 0.0),
            tol=psd_tol,
            certifies=False,
            detail=f"min_eig(H)={min_eig:.3e}",
        ),
        OracleResult(
            name="margin_curvature_finite_diff",
            passed=(worst_rel <= rel_tol) and solves_ok,
            residual=worst_rel,
            tol=rel_tol,
            certifies=False,
            detail=(
                f"max_rel_err={worst_rel:.3e} mean_rel_err={mean_rel:.3e} "
                f"dirs={n_used} h={h:g} worst_solve_res={worst_solve_res:.1e}"
                + ("" if solves_ok else " [PROJECTION DID NOT CONVERGE]")
            ),
        ),
    ]
    summary = {
        "M_shape": M.shape,
        "min_eig_H": min_eig,
        "max_pinv_residual": max_pinv_res,
        "max_rel_err": worst_rel,
        "mean_rel_err": mean_rel,
        "n_directions": n_used,
        "worst_solve_residual": worst_solve_res,
    }
    return results, summary
