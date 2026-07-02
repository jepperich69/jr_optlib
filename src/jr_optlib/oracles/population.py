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
