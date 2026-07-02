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
)
from jr_optlib.oracles.setcover import (
    setcover_feasible,
    setcover_cost,
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
    "setcover_feasible",
    "setcover_cost",
]
