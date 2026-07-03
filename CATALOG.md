# jr_optlib Catalog — the gold, by problem

This is the navigable, usage-first map of everything in the library: **which
problem each primitive solves, which paper it feeds, how you call it, and which
oracle certifies it.**

- **Source of truth** for machine-readable metadata is `registry/functions.yaml`
  (signature, oracle, provenance, status). This file is the *human* view and
  adds the two things YAML can't: problem grouping and worked usage.
- **Every primitive listed here is `vetted`** (ships with at least one oracle)
  unless marked otherwise. Runnable examples live in `tests/` — each domain
  points at its test file, which is the executable version of its usage.
- **Golden rule (registry-first):** before writing any optimization or sampling
  code, look here or in the registry. If it exists, reuse it. If it doesn't,
  build it *and add it back* with an oracle + test + registry entry (see
  "Adding a primitive").

---

## 0. Quickstart

```python
# runtime is numpy-only; oracles need scipy; fast sampling needs numba
pip install -e .            # library
pip install -e .[test]      # + scipy, pulp, gurobipy, pandas, numba, pytest

from jr_optlib.transport import sinkhorn_balanced
from jr_optlib.oracles import certify_sinkhorn, summarize

X, _ = sinkhorn_balanced(a, b, C, tau=0.05)     # solve
results, certified = certify_sinkhorn(X, a, b, C, tau=0.05)   # verify, don't re-solve
print(summarize(results))   # CERTIFIED / CHECKED / FAIL / UNVALIDATED
```

**The contract:** you solve, then hand the result to an oracle. The oracle
re-derives a property the true answer *must* have (feasibility, a duality
certificate, agreement with an independent solver, a metamorphic invariant) and
labels the result on a four-level coverage map. Verifying is far cheaper than
solving; a `FAIL` is a provable defect.

---

## 1. Transportation & optimal transport
**Problem:** move mass between marginals at minimum cost, or find the
entropic-regularized (Sinkhorn) coupling; then round a fractional plan to
integers without breaking the marginals.
**Feeds / from:** `Pub_MIPEntropy_MPC`.
**Test:** `tests/test_sinkhorn.py`, `test_ipf.py`, `test_lp_transport.py`, `test_rounding.py`.

| primitive | what it does | oracle |
|---|---|---|
| `transport.ipf_2d` | IPF/raking: KL projection of a seed onto row/col marginals | `certify_ipf` (marginals + scaling form + logspace ref) |
| `transport.sinkhorn_balanced` / `_uv` | entropic OT coupling (`_uv` also returns `u,v` scalings) | `certify_sinkhorn` (marginals + scaling form vs `K`) |
| `transport.solve_transport_opt` | exact min-cost transport (LP) | `certify_transport` (vs `scipy.linprog`) |
| `transport.dependent_round_2d` | round fractional table → integers (LP/MCF dispatcher) | `certify_dependent_round` (marginals + integrality + <1 dev) |
| `transport.round_transport_*` | rounder family (LP, restricted-LP, greedy reduced-cost push, MCF, approx) | `certify_transport` |
| `transport.make_transport` / `make_contingency2d` | seeded instances | — |

**Pipeline:** `marginals + cost C` → **relax** (`sinkhorn_balanced` or `ipf_2d`)
→ **round** (`dependent_round_2d`) → **certify** (`certify_dependent_round`).

```python
from jr_optlib.transport import sinkhorn_balanced_uv, dependent_round_2d
X, u, v, _ = sinkhorn_balanced_uv(supply, demand, C, tau=0.05)
Xint, _ = dependent_round_2d(X, supply, demand)
```

---

## 2. Population synthesis & integerization
**Problem:** fit a high-dimensional weighted seed table to many marginals
(N-way IPF), then convert fractional weights to a whole-person integer
population that still hits the controlling margins.
**Feeds / from:** `Pub_PopInt_PartB`.
**Test:** `tests/test_population_hard_ipf.py`, `test_population_round.py`.

| primitive | what it does | oracle |
|---|---|---|
| `population.HardIPF` | N-dimensional raking over long record tables (sparse: bincount scatter + gather, never materializes the full tensor) | `certify_population_margins` |
| `population.step1_split` | deterministic floor split | precondition for `certify_population_margins` |
| `population.step2_anchor_pps` | anchor-conditioned PPS integer additions | `certify_population_margins` |
| `population.swap_repair_zone` | anchor-preserving integer swap repair | `certify_population_margins` + `certify_secondary_margins_vs_floor` |
| `population.optimize_repair_zone` | **bound/diagnostic**, not a drop-in repair: min secondary-margin L1 with anchor pinned | `certify_secondary_margins_vs_floor` (sits at floor) |

**Pipeline:** `seed + margins` → `HardIPF.fit()` → `step1_split` →
`step2_anchor_pps` → `swap_repair_zone` → `certify_population_margins`.

> Note: `optimize_repair_zone` is a lower-bound/diagnostic — it collapses
> fractional overlap and is **not** a substitute for `swap_repair_zone`.

---

## 3. Set cover (entropy-guided)
**Problem:** minimum-cost set cover via an entropy relaxation + dual-guided
rounding, with an exact MIP and a polish step available.
**Feeds / from:** `Pub_MIPEntropy_MPC`.
**Known instances:** `setcover.scp41` (opt 429, vendored), `rail582` (opt 211, download).
**Test:** `tests/`; end-to-end demo `oracle_bank/demo_scp41.py`.

| primitive | what it does | oracle |
|---|---|---|
| `setcover.solve_entropy_setcover` | RICH entropy relaxation + dual-guided rounding (Gurobi-free core) | `certify_setcover_solution` (CHECKED) |
| `setcover.solve_mip` | Gurobi set-cover MIP with bound/gap | `gap_certificate` / `certify_setcover_solution` (CERTIFIED at gap 0) |
| `setcover.polish_solution` | restricted Gurobi polish | `certify_setcover_solution` |
| `setcover.gen_entropy_friendly_scp` | seeded synthetic generator | feasibility precondition |

---

## 4. Assignment under risk (entropic QP)
**Problem:** assignment minimizing an entropic (exponential) risk of Gaussian
costs — closed form for the risk, QP over the Birkhoff polytope, exact
Hungarian for the independent/risk-neutral cases, MIQP for correlated costs.
**Feeds / from:** `Pub_QP_SAA_MC`.
**Test:** `tests/test_entropic_qp.py`.

| primitive | what it does | oracle |
|---|---|---|
| `optimization.rho_eta_formula` / `rho_eta_mc` | entropic risk of Gaussian cost: closed form + Monte-Carlo | `certify_entropic_risk_mc` (formula vs MC) |
| `optimization.solve_qp` | convex QP over Birkhoff polytope (SLSQP, multi-start) | `certify_entropic_assignment` (feasible + recompute + relaxation bound) |
| `optimization.solve_hungarian_qp` / `_rn` | exact assignment via `linear_sum_assignment` | `certify_entropic_assignment` (brute-force optimum, small n) |
| `optimization.solve_miqp_gurobi` | correlated-case entropic-risk MIQP (Gurobi) | `certify_entropic_assignment` |

---

## 5. Chance-constrained selection sampling (SSKP)
**Problem:** sample selections `x ∈ {0,1}^k` from `π_β ∝ exp(-β F)` where
`F = -revenue + c·E[(load-q)^+]` — a smooth chance-constraint surrogate — to
bound an SAA optimum.
**Feeds / from:** `Pub_SAA_PMIP_MC` (formulation extracted from the Java backend; JVM host not ported).
**Test:** `tests/test_sskp_mh.py`.

| primitive | what it does | oracle |
|---|---|---|
| `sampling.sskp_mh_chain` | delta-update MH: O(1) sufficient-stat update per single-flip step; optional numba JIT (~50× over pure Python, bit-identical) | `certify_sskp_chain` (delta invariant + penalty ref) |
| `sampling.sskp_objective` / `sskp_penalty` | full-recompute objective and A&S penalty (oracle reference) | — |

```python
from jr_optlib.sampling import sskp_mh_chain
from jr_optlib.oracles import certify_sskp_chain
res = sskp_mh_chain(r, mu, sigma, c, q, beta=5.0,
                    t_burn=20000, t_sample=200000, backend="auto")  # numba if present
results, ok = certify_sskp_chain(res, r, mu, sigma, c, q)
res.z_upper   # E_{π_β}[F], the upper bound
```
> `backend="auto"|"numba"|"python"`. numba is an optional `fast-sampling`
> extra; without it the pure-Python path runs, bit-identical.

---

## 6. Generic MCMC, annealing & dual ascent
**Problem:** reusable samplers/optimizers driven by user `energy_fn`/`propose_fn`
callbacks, plus a subgradient dual-ascent driver for soft constraints.
**Feeds / from:** `Pub_PMIP_VSP`, `Pub_PMIP_AOR`, `Pub_SAA_PMIP_MC`.
**Test:** `tests/test_mcmc.py`, `test_vsp.py`, `test_lagrangian.py`.

| primitive | what it does | oracle |
|---|---|---|
| `sampling.metropolis_hastings` | generic MH chain over any state space | `certify_detailed_balance` (TV vs Boltzmann on small spaces) |
| `sampling.simulated_annealing` | geometric-cooling SA optimizer | — |
| `sampling.ladder_burn_in` | temperature-ladder burn-in | — |
| `vsp.run_vsp_mh_chain` | VSP Metropolis local search | `certify_vsp_heuristic_chain` |
| `vsp.randomized_greedy_solution` | VSP randomized-greedy warm start | — |
| `optimization.subgradient_dual_ascent` | Lagrangian dual ascent for inequality constraints | — |

> These callback-based samplers trade speed for generality. When a chain is
> long and per-step work is small, follow the SSKP pattern instead: hard-code
> the incremental delta and JIT it (see §5).

---

## 7. Discrete choice
**Problem:** logit choice probabilities, expected max utility (logsum), nested
logit, and a full SUE route-choice model.
**Feeds / from:** standard (McFadden); route choice from `Pub_CongestionPMIP_TBA`.
**Test:** `tests/test_choice.py`.

| primitive | what it does | oracle |
|---|---|---|
| `optimization.compute_mnl_probabilities` | MNL probs (max-trick stable, availability mask) | `certify_mnl` |
| `optimization.compute_logsum` | expected max utility of a choice set | `certify_mnl` |
| `optimization.compute_nested_logit_probabilities` | two-level nested logit (probs, nest, conditional) | `certify_nested_logit` |
| `optimization.compute_route_choice_shares` | SUE car+transit (direct + one-transfer) route shares | — |

---

## 8. Dynamic programming & reinforcement learning
**Problem:** finite-horizon stochastic DP by exact backward induction over a
tensor grid, and its episodic tabular Q-learning analogue (risk-neutral or
exponential/risk-sensitive).
**Feeds / from:** generic (built for the library).
**Test:** `tests/test_optimization_dp.py`, `test_sampling_rl.py`.

| primitive | what it does | oracle |
|---|---|---|
| `optimization.backward_induction_solver` | finite-horizon DP, backward induction | `certify_dp_vs_brute_force` (vs exhaustive policy search) |
| `optimization.contract_transitions` | expectation step (tensordot over independent exogenous transitions) | `certify_transition_contraction` (vs brute-force expectation) |
| `sampling.run_q_learning_episode` | one episode of tabular Q-learning (risk-neutral / exponential) | `certify_q_learning_vs_dp` (converges to DP optimum) |

---

## 9. Nonlinear programming & routing
**Problem:** coordinate-wise NLP for a granularity/access model, and grid
shortest-distance helpers.
**Feeds / from:** `Pub_NapstiGranularity_TBA`.
**Test:** `tests/test_nlp.py`, `test_routing.py`.

| primitive | what it does | oracle |
|---|---|---|
| `optimization.solve_coord_wise` | coordinate-wise NLP block solve | `verify_with_gurobi` (differential vs Gurobi, when available) |
| `optimization.dijkstra_manhattan` | min Manhattan distance to nearest station per mode | — |

---

## Paper ↔ primitive feeder map

| paper | primitives it feeds / came from |
|---|---|
| `Pub_MIPEntropy_MPC` | all of `transport.*` (ipf, sinkhorn, rounders), all of `setcover.*` |
| `Pub_PopInt_PartB` | `population.*` (HardIPF, step1/2, swap/optimize repair) |
| `Pub_QP_SAA_MC` | `optimization.entropic_qp` (rho_eta, solve_qp, hungarian, miqp) |
| `Pub_SAA_PMIP_MC` | `sampling.sskp_mh_chain`, `optimization.subgradient_dual_ascent` |
| `Pub_PMIP_VSP` | `vsp.*`, `sampling.simulated_annealing` |
| `Pub_PMIP_AOR` | `sampling.ladder_burn_in` |
| `Pub_CongestionPMIP_TBA` | `optimization.compute_route_choice_shares` |
| `Pub_NapstiGranularity_TBA` | `optimization.solve_coord_wise`, `dijkstra_manhattan` |
| generic (library-native) | discrete choice (MNL/logsum/nested), DP, Q-learning |

---

## Oracle catalog (`jr_optlib.oracles`)

Generic (problem-agnostic): `gap_certificate` (UB==LB, the strongest cheap
check), `differential` (exact-vs-alternative), `metamorphic` (invariant under a
known transform), `summarize` (roll results into a verdict).

Domain: `certify_ipf` · `certify_sinkhorn` · `certify_transport` ·
`certify_dependent_round` · `certify_setcover_solution` ·
`certify_population_margins` · `certify_secondary_margins_vs_floor` ·
`certify_detailed_balance` · `certify_vsp_heuristic_chain` ·
`certify_entropic_assignment` · `certify_entropic_risk_mc` ·
`verify_with_gurobi` · `certify_mnl` · `certify_nested_logit` ·
`certify_transition_contraction` · `certify_dp_vs_brute_force` ·
`certify_q_learning_vs_dp` · `certify_sskp_chain`.

---

## Conventions

**Backends / extras.** Runtime is numpy-only. Optional extras (`pyproject.toml`):
`oracles` (scipy), `transport-lp` (pulp), `setcover-gurobi` (gurobipy),
`population` (pandas), `fast-sampling` (numba), `oracles-full` (POT, networkx).
A paper importing one function never pulls the whole stack.

**Reproducibility freeze.** Develop against live `main`. At submission, pin the
`jr_optlib` commit SHA in the paper's `requirements.txt` and vendor a snapshot.
Bit-identical performance changes (verified by an oracle) are safe on `main`;
numerics-changing edits require a new registry id. See `Pub_MIPEntropy_MPC`'s
pinned `requirements.txt` for the template.

**Adding a primitive (oracle-on-add).** No `vetted` entry without an oracle.
When you add a function: (1) write it, (2) add an oracle (certificate,
differential, metamorphic, or known-answer), (3) add an oracle-backed test,
(4) add a `registry/functions.yaml` entry, (5) add rows to `INDEX.md` and this
catalog. Until it has an oracle it is `experimental`, never `vetted`.
```
