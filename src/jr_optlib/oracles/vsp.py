# -*- coding: utf-8 -*-
"""Oracles for Vehicle Scheduling (VSP) heuristics."""

from typing import List, Sequence

import numpy as np

from jr_optlib.oracles.core import OracleResult, metamorphic
from jr_optlib.vsp.qbuzz_vsp import QbuzzInstance, ScheduleColumn, evaluate_schedule

def certify_vsp_heuristic_chain(
    instance: QbuzzInstance,
    initial_objective: float,
    best_objective: float,
    objectives: Sequence[float],
    samples: Sequence[Sequence[ScheduleColumn]],
    epsilon: float,
) -> List[OracleResult]:
    """
    Oracle for the heuristic VSP Metropolis chain.
    
    Checks:
      1. Incumbent monotonicity: best_objective <= initial_objective
      2. Band constraint: max(objectives) <= (1 + epsilon) * initial_objective
         (Since the band is based on the incumbent which only decreases, 
          the max over the whole chain is safely bounded by the initial band).
      3. Feasibility: Recomputes a subset of retained schedules using the domain oracle
         to ensure they are truly feasible and their cost matches.
    """
    results = []

    # 1. Monotonicity
    results.append(metamorphic(
        before=initial_objective, 
        after=best_objective, 
        relation="<=", 
        name="incumbent_monotonicity"
    ))

    # 2. Band constraint
    if objectives:
        max_obj = max(objectives)
        # Adding a small tolerance for floating point issues
        band_cap = initial_objective * (1.0 + epsilon) + 1e-6
        results.append(metamorphic(
            before=band_cap, 
            after=max_obj, 
            relation="<=", 
            name="hard_band_constraint"
        ))

    # 3. Feasibility & Cost recomputation
    if samples:
        step = max(1, len(samples) // 10)
        for i in range(0, len(samples), step):
            sample = samples[i]
            feasible = True
            for col in sample:
                recalc = evaluate_schedule(instance, col.trips)
                if recalc is None or not np.isclose(recalc.cost, col.cost, rtol=1e-5):
                    feasible = False
                    break
            results.append(OracleResult(
                name="sample_feasibility_check",
                passed=feasible,
                residual=0.0 if feasible else 1.0,
                tol=0.0,
                certifies=False,  # This is a CHECKED test
                detail=f"Feasibility and cost check for sample index {i}"
            ))

    return results
