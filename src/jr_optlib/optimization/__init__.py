# -*- coding: utf-8 -*-
from jr_optlib.optimization.lagrangian import subgradient_dual_ascent
from jr_optlib.optimization.nlp import solve_coord_wise
from jr_optlib.optimization.routing import dijkstra_manhattan, compute_route_choice_shares
from jr_optlib.optimization.entropic_qp import (
    build_equicorrelation_cov,
    rho_eta_formula,
    rho_eta_mc,
    sample_costs,
    solve_hungarian_qp,
    solve_hungarian_rn,
    solve_miqp_gurobi,
    solve_qp,
)

__all__ = [
    "subgradient_dual_ascent",
    "solve_coord_wise",
    "dijkstra_manhattan",
    "compute_route_choice_shares",
    "build_equicorrelation_cov",
    "rho_eta_formula",
    "rho_eta_mc",
    "sample_costs",
    "solve_hungarian_qp",
    "solve_hungarian_rn",
    "solve_miqp_gurobi",
    "solve_qp",
]
from jr_optlib.optimization.dp import backward_induction_solver, contract_transitions
from jr_optlib.optimization.choice import compute_mnl_probabilities, compute_logsum, compute_nested_logit_probabilities

__all__.extend([
    'backward_induction_solver',
    'contract_transitions',
    'compute_mnl_probabilities',
    'compute_logsum',
    'compute_nested_logit_probabilities',
])
