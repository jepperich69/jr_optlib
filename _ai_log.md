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
**Git ref:**

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
