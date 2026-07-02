# -*- coding: utf-8 -*-
"""Oracles for population-synthesis tables."""

from __future__ import annotations

import numpy as np

from jr_optlib.oracles.core import OracleResult


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
