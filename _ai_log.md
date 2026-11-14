# AI Session Log - jr_optlib

---

## Compressed sessions

- **2026-07-02** (Claude): Build the first slice of the jr_optlib robustness system (design wa... -> All 16 tests pass; end-to-end demo on real OR-Library scp41 works (greedy=CHECKED, inje...
- **2026-07-02 (migration 1: sinkhorn)** (Claude): First library migration -- move a mip_hybrid transport function int... -> Migration proven bit-for-bit: `test_migration_matches_old_copy` asserts exact `np.array...
- **2026-07-02 (migration 2: LP-family + double-def fix)** (Claude): Migrate the LP-family transport functions and resolve the round_tra... -> 64 tests pass total. Migration verified vs the live paper copy on OBJECTIVE+feasibility...
- **2026-07-02 (migration 3: remaining rounders)** (Claude): Migrate all remaining transport + population-synthesis rounders. -> 91 tests pass total. All MCF paths fall back to LP in base (no ortools graph) and match...
- **2026-07-02 (Codex: set-cover + PopInt integration)** (Codex GPT-5): Continue the common-codebase harnessing work: verify the repaired G... -> Gurobi license and command-line/Python solves are working. Set-cover core, Gurobi polis...
- **2026-07-02 (Codex: PopInt PPS + verification)** (Codex GPT-5): Finish the current PopInt integerization migration slice and run a ... -> Full `jr_optlib` suite passes: 103 tests. PPS migration committed as `e573bdb`. PopInt ...
- **2026-07-02 (Codex: PopInt residual diagnosis / close)** (Codex GPT-5): Diagnose why the PopInt model-check failed on non-anchor margins an... -> Worst repaired residual is zone `336000`, margin `AgeIncome`, L1 gap `184`. Global repa...
- **2026-07-02 (handoff prep for Codex)** (Claude): User is low on tokens and wants to continue the jr_optlib migration... -> jr_optlib now has an AGENTS.md so a Codex/Gemini session opens with full context. Verdi...
- **2026-07-02 (Gemini: VSP heuristic integration)** (Gemini CLI (Gemini 3.1 Pro (High))): Migrate the heuristic VSP (Vehicle Scheduling) chains from Pub_PMIP... -> The Metropolis feasible-state search for vehicle scheduling has been migrated, proven b...
- **2026-07-02 (Gemini: SA logic extraction)** (Gemini CLI (Gemini 3.1 Pro (High))): Extract the geometric Simulated Annealing optimization logic from P... -> The generic SA loop logic is now safely centralized in mcmc.py. VSP code is simplified ...
- **2026-07-02 (Gemini: Ladder Burn-in extraction)** (Gemini CLI (Gemini 3.1 Pro (High))): Migrate the Pub_PMIP_AOR temperature-ladder MCMC warmup logic as a ... -> The AOR temperature-ladder heuristic is now a fully generic jr_optlib primitive. mh_exa...

---

## Session 2026-07-02 (Gemini: Dual Ascent extraction)
**Agent:** Gemini CLI (Gemini 3.1 Pro (High))
**Goal:** Extract the soft-feasibility subgradient dual ascent algorithm from Pub_SAA_PMIP_MC into jr_optlib.
**Files touched:**
- src/jr_optlib/optimization/lagrangian.py (new module) -- added subgradient_dual_ascent generic driver for updating shadow prices.
- src/jr_optlib/optimization/__init__.py -- exported the primitive.
- 	ests/test_optimization_lagrangian.py -- added a differential test running the generic driver on the Pub_SAA_PMIP_MC 2x2 asymmetric expected-violation trace, proving it matches the legacy paper copy bit-for-bit.
- egistry/functions.yaml, INDEX.md -- registered optimization.subgradient_dual_ascent.
**Outcome:** We now have a clean, decoupled optimization module for Lagrangian methods. The exact subgradient loop used in the SAA paper is encapsulated and independently verifiable.
**Next steps:** Push jr_optlib to GitHub.
**Git ref:**

---

## Session 2026-07-02 (Claude: secondary-margin floor oracle + optimal repair)
**Agent:** Claude Opus 4.8
**Goal:** Fix the PopInt secondary-margin oracle to match the paper's contract (controlling margins exact, secondary margins approximate) and add an optimal repair primitive.
**Files touched:**
- src/jr_optlib/oracles/population.py -- added certify_secondary_margins_vs_floor: per-zone min-violation MIP (controlling pinned exactly, secondary soft L1) yields the proven floor; secondary margins are judged against that floor, never against zero or a hand-picked tolerance.
- src/jr_optlib/population/integerize.py -- added optimize_repair_zone: min-total-secondary-L1 MIP with the controlling margin held exactly. Optimal counterpart to swap_repair_zone; output attains the floor by construction.
- src/jr_optlib/oracles/__init__.py, src/jr_optlib/population/__init__.py -- exports.
- tests/test_population_hard_ipf.py -- new test: optimize_repair_zone reaches the floor on an infeasible-secondary zone, conserves population, holds the anchor exactly, and is certified by certify_secondary_margins_vs_floor.
- registry/functions.yaml, registry/INDEX.md -- registered both.
**Outcome:** The 4 prior PopInt FAILs were an oracle-specification error -- they demanded exact secondary margins, which is provably infeasible (floor = 489 L1 across 98 zones; 0/98 zones can reach zero). With the corrected oracle the PopInt model-check is CERTIFIED=3 CHECKED=4 FAIL=0 (controlling AgeGender certified exact before and after repair). New methodological finding: the greedy swap leaves 40855 secondary L1, ~83x above the 489 floor; optimize_repair_zone reaches the floor exactly in all 98/98 zones (~34s) with the anchor still exact. Per-margin swap -> LP: AgeChildren 9235 -> 15, AgeFamily 4996 -> 0, AgeIncome 13894 -> 128, AgeLma 12730 -> 346. Full suite 110 tests pass.
**Next steps:** Paper decision -- either soften the "as closely as feasible" wording or replace the greedy swap with optimize_repair_zone and report the 40855 -> 489 improvement as a result. Push jr_optlib (helpi 23) when ready.
**Git ref:** d714e07

---

## Session 2026-07-02 (Claude: Pub_QP_SAA_MC entropic-risk assignment migration)
**Agent:** Claude Opus 4.8
**Goal:** Migrate Pub_QP_SAA_MC/code/solvers.py (entropic-risk assignment: QP / Hungarian / MIQP) into jr_optlib with a differential test vs the live paper copy and oracle wiring.
**Files touched:**
- src/jr_optlib/optimization/entropic_qp.py (new) -- rho_eta_formula (closed-form entropic risk), rho_eta_mc (Monte Carlo check), solve_qp (SLSQP over Birkhoff polytope, 3 warm starts), solve_hungarian_qp / solve_hungarian_rn (scipy linear_sum_assignment), solve_miqp_gurobi (Gurobi), plus build_equicorrelation_cov and sample_costs helpers. Decoupled from the paper's module-global data (mean matrix / dimension / sampler now explicit params).
- src/jr_optlib/oracles/entropic_qp.py (new) -- certify_entropic_risk_mc (claimed value vs MC, convergence check) and certify_entropic_assignment (feasibility + objective recompute + brute-force-over-permutations exact optimum for small n).
- src/jr_optlib/optimization/__init__.py, src/jr_optlib/oracles/__init__.py -- exports.
- tests/test_optimization_entropic_qp.py (new) -- differential vs live paper copy (formula + seeded MC exact; solve_qp to 1e-9; Hungarian x/pi/obj exact; MIQP == Hungarian in diagonal case) + oracle tests.
- registry/functions.yaml, registry/INDEX.md -- registered the family.
**Outcome:** Full suite 117 tests pass (110 -> 117). Gurobi license is currently WORKING (MIQP differential ran, not skipped). Oracle note: the entropic-risk MC estimator (log-MGF) converges slowly for large eta*std, so certify_entropic_risk_mc is a mild-regime convergence check (certifies=False); the real optimality certificate is the brute-force-over-permutations oracle. Latent inconsistency fixed in migration: paper's solve_hungarian_qp computed obj against the module-global mean matrix regardless of the passed C_bar; library uses the passed c_bar.
**Next steps:** Continue the four-project scouting order -- next is the Napsti block-coordinate fixed-point primitive (solve_coord_wise / solve_continuous, oracle = verify_with_gurobi), then the Dijkstra + SUE route-choice bundle, then Pub_ML_Entropy MH review. Push jr_optlib (helpi 23) when ready.
**Git ref:** 391794e

---

## Session 2026-07-03 (Claude: perf audit, SSKP extraction, catalog, GitHub push)
**Agent:** Claude Opus 4.8
**Goal:** Audit jr_optlib for low-hanging numerical/vectorization wins before papers pin; extract the reusable delta-update sampler "gold"; write down everything the library contains (usage + problem map).
**Files touched:**
- `src/jr_optlib/transport/sinkhorn.py` -- replaced `X = diag(u) @ K @ diag(v)` with the broadcast outer rescaling `u[:,None]*K*v[None,:]` in both sinkhorn_balanced and _uv; ~10-30x faster on the reconstruction step, bit-for-bit identical (matmul only adds exact zeros), verified by the 18 migration cases.
- `src/jr_optlib/sampling/sskp_mh.py` (new) -- SSKP chance-constrained delta-update MH chain: O(1) sufficient-statistic update per single-flip step; optional numba JIT (~51x over pure Python, bit-identical); pure-Python fallback. Formulation extracted from Pub_SAA_PMIP_MC's Java backend (JVM host NOT ported).
- `src/jr_optlib/oracles/sskp.py` (new) -- delta_invariant (maintained stats/objective == full O(k) recompute; certificate) + penalty_reference (A&S penalty vs scipy partial expectation) + certify_sskp_chain.
- `tests/test_sskp_mh.py` (new) -- delta-invariant certify, python-vs-numba bit-identity, Boltzmann-distribution match, corrupted-stats catch.
- `src/jr_optlib/sampling/__init__.py`, `src/jr_optlib/oracles/__init__.py` -- exports.
- `registry/functions.yaml`, `registry/INDEX.md` -- registered sampling.sskp_mh_chain; also re-synced INDEX (had drifted: missing all choice/DP/RL/NLP/routing entries).
- `pyproject.toml` -- added `fast-sampling` (numba) optional extra + numba in test deps.
- `CATALOG.md` (new) -- problem-oriented, usage-first map of all 41 primitives, with pipelines, worked examples, and the paper<->primitive feeder table.
- `README.md` -- refreshed stale bits (16->139 tests, layout, status).
- (outside repo) `Publikationer/Pub_MIPEntropy_MPC/code/requirements.txt` -- pinned `jr_optlib @ git+...@3ea1d4a` so the paper's Sinkhorn/IPF results freeze to this commit.
**Outcome:** Library audit: already well-vectorized; the one real win was the Sinkhorn reconstruction (bit-identical, now pinned into Pub_MIPEntropy_MPC -- the paper's repro already imports jr_optlib live, so the pin was the durable fix). SSKP delta-update sampler is now a vetted primitive (JVM baggage left behind; numba gives JVM-class speed in-process). CATALOG.md now captures the "gold" the registry YAML couldn't. Pushed all three commits to GitHub (origin/main @ 76f36ec); pinned commit 3ea1d4a confirmed reachable so the pin resolves. 139 tests pass.
**Next steps:** (1) set-cover MH chain (scMhChain in the same Pub_SAA_PMIP_MC Java backend) -- deferred, same delta-update pattern, extract when a paper needs it; (2) optional: a generator emitting INDEX.md/CATALOG tables from functions.yaml to prevent doc drift (drift recurred this session); (3) standing workflow now recorded in memory -- check jr_optlib before any new optimization/sampling code, reuse or build-and-integrate.
**Git ref:** 76f36ec
## Session 2026-09-01 (recursive-logit baseline)
**Agent:** Codex GPT-5
**Goal:** Add the established Fosgerau--Frejinger--Karlstrom recursive-logit assignment as a vetted route-free primitive and validate it against the Figure 5 benchmark.
**Files touched:**
- `src/jr_optlib/optimization/recursive_logit.py` -- NEW. Linear desirability solve, recursive policy, and expected edge-flow propagation for multiple OD pairs.
- `src/jr_optlib/oracles/recursive_logit.py` -- NEW. Bellman, policy-normalization, conservation, and nonnegativity certificates.
- `tests/test_recursive_logit.py` -- NEW. Exact DAG path differential, cyclic-network certificate, and multiple-OD conservation tests.
- `src/jr_optlib/{optimization,oracles}/__init__.py`, `registry/functions.yaml`, `registry/INDEX.md`, `CATALOG.md` -- exports and explicit attribution/registration.
**Outcome:** The complete library suite passes: 146 tests, 0 failures. The primitive is identified as an implementation of Fosgerau, Frejinger and Karlstrom (2013), not a novel recursion. Its equal-outgoing-action reference probabilities enter as additive link utilities through `tau log q`.
**Next steps:** Add sparse/cached destination solves before metropolitan scale. Keep risk transformation and entropy-based active-set pruning as separate extensions, each with its own oracle.
**Git ref:** -

---
## Session 2026-09-01 (independent-Gaussian link risk)
**Agent:** Codex GPT-5
**Goal:** Add the independent-Gaussian entropic link-cost transformation needed for the Figure 5 recursive-logit risk experiment.
**Files touched:**
- `src/jr_optlib/optimization/entropic_qp.py` -- added `gaussian_entropic_costs(mean, variance, eta)`.
- `tests/test_optimization_entropic_qp.py` -- added exact eta-zero and route-aggregation tests against `rho_eta_formula`.
- `src/jr_optlib/optimization/__init__.py`, `registry/functions.yaml`, `registry/INDEX.md`, `CATALOG.md` -- export and documentation.
- `_ai_log.md` -- recorded the session.
**Outcome:** The transformation is exact at eta=0 and its sum over an arbitrary route incidence vector equals the full entropic-risk formula under diagonal covariance. The full library suite passes 147 tests, 0 failures.
**Next steps:** Correlated delays cannot use this elementwise transformation; treat them through an augmented disruption state or the full covariance criterion. Add sparse destination-cached recursive-logit solves before scale-up.
**Git ref:** -

---
## Session 2026-09-01 (recursive-logit support pruning)
**Agent:** Codex GPT-5
**Goal:** Permit recursive-logit active-set restriction without changing surviving action utilities.
**Files touched:**
- `src/jr_optlib/optimization/recursive_logit.py` -- explicit reference weights may now form subprobability rows after pruning; default unpruned rows remain normalized.
- `tests/test_recursive_logit.py` -- added a pruning differential proving that fixed surviving weights raise the logsum cost, whereas renormalization would change the model.
- `_ai_log.md` -- recorded the session.
**Outcome:** Full library suite passes 148 tests. The solver now supports exact restriction of an existing recursive-logit model: removed alternatives lose their mass, while surviving `tau log q` utilities remain unchanged.
**Next steps:** Sparse/cached linear systems remain the main library performance task. A global pruning-loss certificate requires path-length or occupancy information; a one-step local mass bound is insufficient.
**Git ref:** -

---

## Session 2026-11-10 (IPF performance baseline)
**Agent:** Codex
**Goal:** Benchmark existing ipf_2d against the proposed scientific-code performance standard.
**Files touched:**
- `_ai_log.md` -- record only; no library source changed.
- Evidence and runnable benchmark stored in `JR/AI_auto/drafts/ipf_benchmark/`; reusable template in `JR/AI_auto/prompts/numerical_benchmark_template.md`.
**Outcome:** Five synthetic planted-answer workloads met marginal tolerance 1e-9 and relative cell-error tolerance 1e-7; 17 existing IPF tests passed. Retained NumPy median 0.095/0.473/33.199 ms for dense 32/256/1024 squared, with 16.10 MiB traced allocations for largest case excluding live inputs. Thread settings 1, Core Ultra 5 135U, NumPy 2.4.1. Measured baseline only: consumer speed/memory budgets unspecified. Negative control confirms silent iteration exhaustion requires caller validation. Zero seeds are epsilon-clamped, not exact structural zeros. General scaling-form oracle dense design is unsuitable for largest benchmark; planted exact answers used instead. Unrelated existing dirty files preserved.
**Next steps:** See AI_auto benchmark REPORT.md. Clarify convergence and zero semantics compatibly; profile large workloads if consumer targets require improvement. No new accelerator dependency justified by this comparison with a diagnostic log-space reference.
**Git ref:** HEAD 08eee8f; exact source hash in benchmark results.json; no commit.
