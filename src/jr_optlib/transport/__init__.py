"""Transport / population-synthesis primitives."""

from jr_optlib.transport.ipf import ipf_2d, make_contingency2d, Contingency2D
from jr_optlib.transport.sinkhorn import (
    sinkhorn_balanced,
    sinkhorn_balanced_uv,
    make_transport,
    TransportInstance,
)
from jr_optlib.transport.lp_transport import (
    solve_transport_opt,
    round_transport_min_cost_lp,
    round_transport_min_cost_lp_restricted,
    round_transport_greedy_push,
)
from jr_optlib.transport.rounding import (
    round_transport_min_cost_mcf,
    round_transport_min_cost_approx,
    round_transport_floor_residue_lp,
)
from jr_optlib.transport.population_round import (
    dependent_round_2d,
    dependent_round_2d_lp,
    dependent_round_2d_lp_sparsified,
    dependent_round_2d_mcf,
)

__all__ = [
    "ipf_2d", "make_contingency2d", "Contingency2D",
    "sinkhorn_balanced", "sinkhorn_balanced_uv",
    "make_transport", "TransportInstance",
    "solve_transport_opt", "round_transport_min_cost_lp",
    "round_transport_min_cost_lp_restricted", "round_transport_greedy_push",
    "round_transport_min_cost_mcf", "round_transport_min_cost_approx",
    "round_transport_floor_residue_lp",
    "dependent_round_2d", "dependent_round_2d_lp",
    "dependent_round_2d_lp_sparsified", "dependent_round_2d_mcf",
]
