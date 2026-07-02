# jr_optlib index (human-readable)

Source of truth is the YAML files in this folder; this table is a summary.
Schema: `SCHEMA.md`.

## Vetted functions (`functions.yaml`)

| id | summary | oracles | certifies | status |
|----|---------|---------|-----------|--------|
| `transport.ipf_2d` | IPF / raking (2D) -- KL projection onto marginals | marginal_residual, ipf_scaling_form, ipf_reference | yes | vetted |
| `transport.make_contingency2d` | seeded 2D marginals (planted instance) | -- | no | vetted |

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
