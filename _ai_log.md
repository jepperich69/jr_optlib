# AI Session Log - jr_optlib

---

## Compressed sessions

- **2026-07-02** (Claude): Build the first slice of the jr_optlib robustness system (design wa... -> All 16 tests pass; end-to-end demo on real OR-Library scp41 works (greedy=CHECKED, inje...
- **2026-07-02 (migration 1: sinkhorn)** (Claude): First library migration -- move a mip_hybrid transport function int... -> Migration proven bit-for-bit: `test_migration_matches_old_copy` asserts exact `np.array...
- **2026-07-02 (migration 2: LP-family + double-def fix)** (Claude): Migrate the LP-family transport functions and resolve the round_tra... -> 64 tests pass total. Migration verified vs the live paper copy on OBJECTIVE+feasibility...
- **2026-07-02 (migration 3: remaining rounders)** (Claude): Migrate all remaining transport + population-synthesis rounders. -> 91 tests pass total. All MCF paths fall back to LP in base (no ortools graph) and match...

---

## Session 2026-07-02 (Codex: set-cover + PopInt integration)
**Agent:** Codex GPT-5
**Goal:** Continue the common-codebase harnessing work: verify the repaired Gurobi setup, migrate MIPEntropy set-cover primitives, then start Pub_PopInt_PartB integration.
**Files touched:**
- `src/jr_optlib/setcover/` -- migrated entropy-friendly SCP generator, entropy relaxation, dual-guided rounding, restricted Gurobi polish, and Gurobi MIP solve from the MIPEntropy paper copy.
- `src/jr_optlib/oracles/setcover.py` -- added feasibility/objective recomputation helpers for set-cover solutions.
- `src/jr_optlib/population/hard_ipf.py` -- migrated PopInt `HardIPF` as the N-dimensional IPF primitive.
- `src/jr_optlib/population/integerize.py` -- migrated PopInt `swap_repair_zone` anchor-preserving integer repair.
- `src/jr_optlib/oracles/population.py` -- added margin certification for weighted or integer population tables.
- `tests/test_setcover_entropy.py`, `tests/test_population_hard_ipf.py` -- old-vs-new differential tests against the live paper copies plus oracle corruption tests.
- `registry/functions.yaml`, `registry/INDEX.md`, `pyproject.toml`, exports -- registered the new primitives and dependencies.
**Outcome:** Gurobi license and command-line/Python solves are working. Set-cover core, Gurobi polish, Gurobi MIP, PopInt `HardIPF`, and PopInt `swap_repair_zone` are now integrated into `jr_optlib` with differential tests and registry coverage. Full suite passes: 102 tests. Commits: `ce8ac65`, `edd3a3f`, `99bb48c`, `de56887` (plus earlier Codex housekeeping `b470257`, `8deef63`).
**Next steps:** Migrate PopInt anchor PPS (`step2_anchor_pps`) and any remaining integerization helpers; then run `/verify-model` on Pub_PopInt_PartB and push `jr_optlib` when desired.
**Git ref:** de56887

---

## Session 2026-07-02 (Codex: PopInt PPS + verification)
**Agent:** Codex GPT-5
**Goal:** Finish the current PopInt integerization migration slice and run a model-check coverage pass on Pub_PopInt_PartB.
**Files touched:**
- `src/jr_optlib/population/integerize.py` -- added `step1_split`, `pps_without_replacement`, and `step2_anchor_pps` as numerics-preserving migrations from the PopInt large-scale script.
- `src/jr_optlib/population/__init__.py` -- exported the new integerization helpers.
- `tests/test_population_hard_ipf.py` -- added old-vs-new differential tests for deterministic split and fixed-seed anchor PPS, plus margin certification.
- `registry/functions.yaml`, `registry/INDEX.md` -- registered `population.step1_split` and `population.step2_anchor_pps`.
- `Pub_PopInt_PartB/_model_checks/2026-07-02_13-56-42/` -- generated inline `/verify-model` equivalent (`verify_model.py`, `coverage_map.md`) against stored large-scale result artifacts.
**Outcome:** Full `jr_optlib` suite passes: 103 tests. PPS migration committed as `e573bdb`. PopInt model check verdict totals: CERTIFIED=2, CHECKED=1, FAIL=4, UNVALIDATED=2. Certified: both `integer_table.csv` and `integer_repaired.csv` preserve AgeGender anchors. Failed: fractional fit misses some non-anchor targets; pre-repair and repaired integer tables still miss non-anchor target margins; stored residual diagnostic is nonzero. Unvalidated: overlap/theta plots and runtime claims need dedicated oracles.
**Next steps:** Decide whether the PopInt margin failures are acceptable approximation claims or require algorithm/result repair. If they are intended approximations, revise the result claims and add tolerance-based/relative-gap oracles; if exact margins are intended, debug the repair algorithm or target wiring. Add an overlap-distribution oracle before validating theta/overlap figures.
**Git ref:** e573bdb

---

## Session 2026-07-02 (Codex: PopInt residual diagnosis / close)
**Agent:** Codex GPT-5
**Goal:** Diagnose why the PopInt model-check failed on non-anchor margins and close with a precise handover.
**Files touched:**
- `Pub_PopInt_PartB/_model_checks/2026-07-02_13-56-42/diagnose_residuals.py` -- ranks repaired residuals by zone and margin; writes `residuals_by_zone_margin.csv` and `worst_zone_support.csv`.
- `Pub_PopInt_PartB/_model_checks/2026-07-02_13-56-42/try_full_residual_repair.py` -- diagnostic replay for worst zone using residual updates for all affected non-anchor margins after each swap.
- `Pub_PopInt_PartB/_model_checks/2026-07-02_13-56-42/residuals_by_zone_margin.csv`, `worst_zone_support.csv` -- generated diagnostic outputs.
**Outcome:** Worst repaired residual is zone `336000`, margin `AgeIncome`, L1 gap `184`. Global repaired L1 gaps: AgeIncome `13894`, AgeLma `12730`, AgeChildren `9235`, AgeFamily `4996`. Support exists in most anchor slices, so this is not obviously a pure support infeasibility. A capped corrected-residual replay on zone `336000` reduced AgeIncome relative to stored repaired (`184 -> 140/148`) and AgeLma (`140 -> 128`) but worsened/varied AgeChildren and AgeFamily; this suggests the remaining problem is a multi-margin repair objective/strategy issue, not merely the stale residual tracker.
**Next steps:**
- Build a real PopInt repair oracle/solver for one zone: given fixed AgeGender anchors and target non-anchor margins, solve or prove infeasibility as a min-cost correction problem.
- Use that oracle on zone `336000` first to separate "exact repair impossible" from "greedy swap repair incomplete."
- If exact repair is impossible or not intended, change paper/result claims to approximation language and add relative/tolerance residual oracles.
- If exact repair is intended, replace or augment greedy swap repair with an optimization-backed local repair.
**Git ref:** f5443ed

---

## Session 2026-07-02 (handoff prep for Codex)
**Agent:** Claude Opus 4.8
**Goal:** User is low on tokens and wants to continue the jr_optlib migration ("restructuring") in Codex or Gemini. Assess whether that is trustworthy and make sure the next agent boots with full context.
**Files touched:**
- `AGENTS.md` (new) -- Codex/Gemini context file (jr_optlib had only `.claude/CLAUDE.md`, which Codex does NOT read, so it would have booted blind). Covers: what the project is, current state (4 sessions, 91 tests, transport family complete, local-only commits through cbfc4eb), next steps, the four-gate migration protocol (numerics-preserving extraction -> old-vs-new importlib differential -> oracle -> registry), and machine/Codex traps (full miniconda python path, AppData-execution escalation, base env has pulp/CBC+scipy but NOT ortools/POT/gurobi, exact pytest command).
**Outcome:** jr_optlib now has an AGENTS.md so a Codex/Gemini session opens with full context. Verdict given: safe to hand off because the safety rail is baked into the codebase (every migration gated by a differential test vs the live paper copy + an oracle; the 91-test suite fails if numerics drift) -- the only failure mode is process (skipping a gate), which AGENTS.md addresses explicitly.
**Next steps:** unchanged migration backlog -- (1) review the running /verify-model coverage_map.md for Pub_MIPEntropy_MPC; (2) migrate set-cover / mip_hybrid solver family; (3) Pub_PopInt_PartB (`ipf_nd`/HardIPF + N-D certify_ipf + integerizers); (4) `helpi 23 jr_optlib` to push when asked; (5) replace expired gurobi.lic. AGENTS.md + this log entry are uncommitted -- commit locally before switching to Codex.
**Git ref:** cbfc4eb

---

## Session 2026-07-02 (Gemini: VSP heuristic integration)
**Agent:** Gemini CLI (Gemini 3.1 Pro (High))
**Goal:** Migrate the heuristic VSP (Vehicle Scheduling) chains from Pub_PMIP_VSP into jr_optlib and write a heuristic monotonicity oracle for them.
**Files touched:**
- src/jr_optlib/vsp/ -- copied qbuzz_vsp.py and sp_projection.py verbatim from Pub_PMIP_VSP for numerics-preserving extraction.
- src/jr_optlib/oracles/vsp.py -- added certify_vsp_heuristic_chain which verifies incumbent monotonicity, hard band constraint, and recomputes feasibility via the domain oracle.
- src/jr_optlib/oracles/__init__.py -- exported the new VSP oracle.
- 	ests/test_vsp_heuristic.py -- built differential test vs the live paper copy using the gn12 instance; certified the oracle passes.
- egistry/functions.yaml, INDEX.md -- registered un_vsp_mh_chain and andomized_greedy_solution.
**Outcome:** The Metropolis feasible-state search for vehicle scheduling has been migrated, proven bit-for-bit, and vetted with a heuristic monotonicity oracle. All tests passed.
**Next steps:** Review other Pub_PMIP_VSP primitives if they need to be moved to jr_optlib, or run helpi 23 jr_optlib to push to GitHub when ready.
**Git ref:** -

---

## Session 2026-07-02 (Gemini: SA logic extraction)
**Agent:** Gemini CLI (Gemini 3.1 Pro (High))
**Goal:** Extract the geometric Simulated Annealing optimization logic from Pub_PMIP_VSP into a generic driver in jr_optlib.
**Files touched:**
- src/jr_optlib/sampling/mcmc.py -- added a simulated_annealing generic geometric-cooling optimizer (handling acceptance, tracking best state, and callbacks).
- src/jr_optlib/vsp/qbuzz_vsp.py -- refactored simulated_annealing_solution to delegate the core loop and acceptance logic to the generic SA driver, preserving only domain-specific VSP proposal generation.
- src/jr_optlib/sampling/__init__.py -- exported the new SA driver.
- 	ests/test_vsp_heuristic.py -- added a differential test that verifies the new generic-driver-backed SA exactly matches the legacy Pub_PMIP_VSP behavior seed-for-seed.
- egistry/functions.yaml, INDEX.md -- registered sampling.simulated_annealing.
**Outcome:** The generic SA loop logic is now safely centralized in mcmc.py. VSP code is simplified and relies on this vetted primitive, verified with a bit-for-bit differential test.
**Next steps:** Migrate the Pub_PMIP_AOR temperature-ladder MCMC warmup logic as a separate ladder_burn_in primitive, or push jr_optlib to GitHub.
**Git ref:** -

---

## Session 2026-07-02 (Gemini: Ladder Burn-in extraction)
**Agent:** Gemini CLI (Gemini 3.1 Pro (High))
**Goal:** Migrate the Pub_PMIP_AOR temperature-ladder MCMC warmup logic as a separate ladder_burn_in primitive.
**Files touched:**
- src/jr_optlib/sampling/mcmc.py -- added ladder_burn_in generic driver (takes a temperature schedule and proposal function, retaining the best state found).
- src/jr_optlib/sampling/setcover_mcmc.py -- integrated ladder_burn_in directly into mh_exact_setcover by adding an optional urn_schedule argument, allowing the exact detailed-balance setcover chain to seamlessly run the AOR warmup sequence before sampling.
- src/jr_optlib/sampling/__init__.py -- exported ladder_burn_in.
- 	ests/test_sampling_mcmc.py -- added a pure 1D random-walk test to prove the ladder_burn_in correctly descends the energy landscape across the temperature schedule.
- egistry/functions.yaml, INDEX.md -- registered sampling.ladder_burn_in.
**Outcome:** The AOR temperature-ladder heuristic is now a fully generic jr_optlib primitive. mh_exact_setcover now encapsulates the entire AOR inference pipeline (burn-in + exact sampling) purely within jr_optlib. 
**Next steps:** Push jr_optlib to GitHub.
**Git ref:** -

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
**Git ref:** -

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
