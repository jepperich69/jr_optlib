# jr_optlib index (human-readable)

Source of truth is the YAML files in this folder; this table is a summary.
Schema: `SCHEMA.md`.

## Vetted functions (`functions.yaml`)

| id | summary | oracles | certifies | status |
|----|---------|---------|-----------|--------|
| `transport.ipf_2d` | IPF / raking (2D) -- KL projection onto marginals | marginal_residual, ipf_scaling_form, ipf_reference | yes | vetted |
| `transport.sinkhorn_balanced` | entropic OT (Sinkhorn) coupling | certify_sinkhorn (marginals + scaling form vs K) | yes | vetted |
| `transport.sinkhorn_balanced_uv` | Sinkhorn returning (u, v) scalings | certify_sinkhorn | yes | vetted |
| `transport.solve_transport_opt` | exact min-cost transport (LP) | certify_transport (vs scipy.linprog) | yes | vetted |
| `transport.round_transport_min_cost_lp` | integer transport via LP | certify_transport | yes | vetted |
| `transport.round_transport_min_cost_lp_restricted` | integer transport LP on arc mask | certify_transport | no | vetted |
| `transport.round_transport_greedy_push` | reduced-cost greedy rounding (bug-fixed) | certify_transport (CHECKED) | no | vetted |
| `transport.round_transport_min_cost_mcf` | min-cost-flow integer transport (LP fallback) | certify_transport | yes | vetted |
| `transport.round_transport_min_cost_approx` | restricted-support approx rounder | certify_transport (CHECKED) | no | vetted |
| `transport.round_transport_floor_residue_lp` | alias -> round_transport_min_cost_lp | certify_transport | yes | vetted |
| `transport.dependent_round_2d` | round fractional 2D table to integers (dispatcher) | certify_dependent_round | yes | vetted |
| `transport.dependent_round_2d_lp` | dependent rounding (dense LP) | certify_dependent_round | yes | vetted |
| `transport.dependent_round_2d_lp_sparsified` | dependent rounding (sparsified LP) | certify_dependent_round | yes | vetted |
| `transport.dependent_round_2d_mcf` | dependent rounding (min-cost flow) | certify_dependent_round | yes | vetted |
| `transport.make_transport` | seeded balanced transport instance | -- | no | vetted |
| `transport.make_contingency2d` | seeded 2D marginals (planted instance) | -- | no | vetted |
| `setcover.solve_entropy_setcover` | RICH set-cover entropy relaxation + dual-guided rounding | certify_setcover_solution (CHECKED) | no | vetted |
| `setcover.gen_entropy_friendly_scp` | seeded synthetic set-cover generator | matrix_to_covers / feasibility precondition | no | vetted |
| `setcover.polish_solution` | restricted Gurobi polish for set cover | certify_setcover_solution (CHECKED) | no | vetted |
| `setcover.solve_mip` | Gurobi set-cover MIP solve with bound/gap | gap_certificate / certify_setcover_solution | yes | vetted |

## Known-answer instances (`instances.yaml`)

| id | problem | known optimum | vendored | status |
|----|---------|---------------|----------|--------|
| `setcover.scp41` | set-cover 200x1000 | 429 (exact) | yes | vetted |
| `setcover.rail582` | rail set-cover | 211 (exact) | no (download) | experimental |
| `transport.seeded_contingency2d` | transport (generator) | -- | n/a | vetted |

## Reference implementations (`references.yaml`)

| id | backend | validates | status |
|----|---------|-----------|--------|
| `ref.ipf_logspace` | custom (logsumexp) | transport.ipf_2d | vetted |
| `ref.scipy_linear_sum_assignment` | scipy | (assignment, tbd) | vetted |
| `ref.networkx_min_cost_flow` | networkx | (MCF, tbd) | vetted |
| `ref.pot_sinkhorn` | POT | (sinkhorn, tbd) | experimental |

_Update this table whenever a YAML entry is added. Keep it in sync by hand for
now; a generator can be added later._
