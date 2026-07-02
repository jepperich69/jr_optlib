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

import numpy as np

from jr_optlib.oracles.core import OracleResult, differential, metamorphic


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


def matrix_to_covers(A_matrix) -> list[list[int]]:
    """Convert an n x m row incidence matrix to covers[j] = covered rows."""
    A = np.asarray(A_matrix)
    covers = []
    for j in range(A.shape[1]):
        covers.append(np.where(A[:, j] != 0)[0].tolist())
    return covers


def certify_setcover_solution(x, A_matrix, costs, known_lower_bound=None,
                              tol: float = 1e-9):
    """Check a binary set-cover solution by feasibility and objective recompute.

    If ``known_lower_bound`` is supplied, also checks lower_bound <= objective.
    If the recomputed objective matches the lower bound, the lower-bound check
    certifies optimality.
    """
    x_arr = np.asarray(x)
    selected = np.where(x_arr > 0.5)[0].tolist()
    covers = matrix_to_covers(A_matrix)
    n_rows = np.asarray(A_matrix).shape[0]

    r_feas = setcover_feasible(selected, covers, n_rows)
    obj = setcover_cost(selected, costs)
    reported = float(np.asarray(costs) @ np.rint(x_arr).astype(int))
    r_cost = differential(obj, reported, label_a="recomputed", label_b="reported",
                          rel_tol=tol, name="setcover_cost_recompute")
    results = [r_feas, r_cost]

    if known_lower_bound is not None:
        lb = float(known_lower_bound)
        r_bound = metamorphic(before=obj, after=lb, relation="<=",
                              rel_tol=tol, name="setcover_lb_le_obj")
        r_opt = differential(obj, lb, label_a="obj", label_b="lower_bound",
                             rel_tol=tol, name="setcover_matches_lower_bound")
        if r_opt.passed:
            r_opt.certifies = True
        results.extend([r_bound, r_opt])

    return results, obj
