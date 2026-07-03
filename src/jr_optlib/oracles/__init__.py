"""Oracles: cheap, independent checks that *verify* a solution rather than
recompute it with the same algorithm.

Ranked by generality (see registry/SCHEMA.md and the design record):
  1. feasibility + objective recomputation      -> core.recompute_*
  2. duality / relaxation optimality certificate -> core.gap_certificate
  3. exact-vs-heuristic differential test        -> core.differential
  4. planted / seeded instances                  -> instance generators + known answer
  5. small-instance exhaustive                   -> caller-supplied brute force
  6. benchmark libraries + reference impls       -> oracle_bank/
  7. monotonicity / metamorphic                  -> core.metamorphic

Each oracle returns an ``OracleResult`` so a harness can assemble a coverage map.
"""

from jr_optlib.oracles.core import (
    OracleResult,
    Verdict,
    differential,
    gap_certificate,
    metamorphic,
    summarize,
)
from jr_optlib.oracles.transport import (
    marginal_residual,
    ipf_scaling_form,
    ipf_reference,
    certify_ipf,
    certify_sinkhorn,
    certify_transport,
    certify_dependent_round,
    transport_optimal_cost,
)
from jr_optlib.oracles.setcover import (
    certify_setcover_solution,
    matrix_to_covers,
    setcover_feasible,
    setcover_cost,
)
from jr_optlib.oracles.population import (
    certify_population_margins,
    certify_secondary_margins_vs_floor,
)
from jr_optlib.oracles.sampling import certify_detailed_balance
from jr_optlib.oracles.vsp import certify_vsp_heuristic_chain
from jr_optlib.oracles.entropic_qp import (
    certify_entropic_assignment,
    certify_entropic_risk_mc,
)
from jr_optlib.oracles.nlp import verify_with_gurobi
from jr_optlib.oracles.choice import certify_mnl, certify_nested_logit
from jr_optlib.oracles.dp import (
    certify_transition_contraction,
    certify_dp_vs_brute_force,
)
from jr_optlib.oracles.rl import certify_q_learning_vs_dp
from jr_optlib.oracles.sskp import (
    certify_sskp_chain,
    delta_invariant,
    penalty_reference,
)

__all__ = [
    "OracleResult",
    "Verdict",
    "differential",
    "gap_certificate",
    "metamorphic",
    "summarize",
    "marginal_residual",
    "ipf_scaling_form",
    "ipf_reference",
    "certify_ipf",
    "certify_sinkhorn",
    "certify_transport",
    "certify_dependent_round",
    "transport_optimal_cost",
    "setcover_feasible",
    "setcover_cost",
    "matrix_to_covers",
    "certify_setcover_solution",
    "certify_population_margins",
    "certify_secondary_margins_vs_floor",
    "certify_detailed_balance",
    "certify_vsp_heuristic_chain",
    "certify_entropic_assignment",
    "certify_entropic_risk_mc",
    "verify_with_gurobi",
    "certify_mnl",
    "certify_nested_logit",
    "certify_transition_contraction",
    "certify_dp_vs_brute_force",
    "certify_q_learning_vs_dp",
    "certify_sskp_chain",
    "delta_invariant",
    "penalty_reference",
]
