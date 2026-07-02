# -*- coding: utf-8 -*-
"""
Oracles for set-cover (and, by restriction, set-partition) solutions.

Verifying a set-cover solution is trivial next to finding one: check every row
is covered (feasibility) and re-sum the selected column costs (objective). With
a known optimum or a valid lower bound in hand, gap_certificate then labels the
result CERTIFIED or CHECKED.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from jr_optlib.oracles.core import OracleResult


def setcover_feasible(selected: Iterable[int],
                      covers: Sequence[Sequence[int]],
                      n_rows: int,
                      name: str = "setcover_feasible") -> OracleResult:
    """Feasibility oracle: every row is covered by at least one selected column.

    ``covers[j]`` is the set of row indices covered by column j. Independent of
    how ``selected`` was produced.
    """
    sel = list(selected)
    covered = set()
    for j in sel:
        covered.update(covers[j])
    missing = n_rows - len(covered.intersection(range(n_rows)))
    return OracleResult(
        name=name, passed=(missing == 0), residual=float(missing), tol=0.0,
        certifies=False,
        detail=f"{missing} of {n_rows} rows uncovered; {len(sel)} columns selected",
    )


def setcover_cost(selected: Iterable[int], costs: Sequence[float]) -> float:
    """Independent objective recomputation: total cost of the selected columns."""
    return float(sum(costs[j] for j in selected))
