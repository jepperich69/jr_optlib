# AI Session Log - jr_optlib

---

## Compressed sessions

- **2026-07-02** (Claude): Build the first slice of the jr_optlib robustness system (design wa... -> All 16 tests pass; end-to-end demo on real OR-Library scp41 works (greedy=CHECKED, inje...

---

## Session 2026-07-02 (migration 1: sinkhorn)
**Agent:** Claude Opus 4.8
**Goal:** First library migration -- move a mip_hybrid transport function into jr_optlib, verified by an old-vs-new differential.
**Chosen target:** `sinkhorn_balanced` (+ `sinkhorn_balanced_uv`, `make_transport`). Picked because it is pure numpy (runnable in the base env; ortools/pulp/POT are NOT installed there, so the LP-based rounders and the double-defined `round_transport_greedy_push` can't be run/verified yet) and its correctness certificate already existed (Sinkhorn output is matrix scaling of K=exp(-C/tau), so IPF's marginals + scaling-form oracle certifies it).
**Files touched:**
- `src/jr_optlib/transport/sinkhorn.py` (new) -- extracted numerics-preserving.
- `src/jr_optlib/oracles/transport.py` -- added `certify_sinkhorn` (marginals + scaling form vs K).
- `src/jr_optlib/{__init__,transport/__init__,oracles/__init__}.py` -- exports.
- `tests/test_sinkhorn.py` (new) -- 28 tests incl. 18 old-vs-new differential cases (imports the real paper file via importlib and asserts array equality).
- `registry/functions.yaml`, `registry/INDEX.md` -- registered 3 functions.
**Outcome:** Migration proven bit-for-bit: `test_migration_matches_old_copy` asserts exact `np.array_equal` between jr_optlib and the live paper copy across 3 seeds x 3 shapes x 2 tau. 44 tests pass total. Committed 715914f. Still local-only (no GitHub).
**Next steps:** LP-family migration (`solve_transport_opt`, `round_transport_min_cost_lp`, and the buggy double-defined `round_transport_greedy_push`) needs an env with ortools or pulp -- check the `pyopt` env or add a scipy.optimize.linprog reference path as the exact-transport oracle. Then resolve the 789/882 double-definition during that migration.
**Git ref:** 715914f

---

## Session 2026-07-02 (migration 2: LP-family + double-def fix)
**Agent:** Claude Opus 4.8
**Goal:** Migrate the LP-family transport functions and resolve the round_transport_greedy_push double-definition.
**Env note:** base/pyopt lacked ortools+pulp; installed `pulp` 3.3.2 (bundles CBC) into base per user choice. (pyopt has gurobipy but the paper transport code uses ortools/pulp, not gurobi.)
**Files touched:**
- `src/jr_optlib/transport/_backend.py` (new) -- ortools-preferred / pulp fallback detection.
- `src/jr_optlib/transport/lp_transport.py` (new) -- solve_transport_opt, round_transport_min_cost_lp, round_transport_min_cost_lp_restricted, and the ACTIVE round_transport_greedy_push (:882). The dead :789 shadow-copy is intentionally not migrated (double-def bug resolved).
- `src/jr_optlib/oracles/transport.py` -- `transport_optimal_cost` (independent scipy.linprog HiGHS reference) + `certify_transport` (feasibility + optimal-cost; require_optimal toggle for exact vs heuristic).
- exports; `tests/test_lp_transport.py` (20 tests); `registry/functions.yaml`+`INDEX.md`+`references.yaml`; `pyproject.toml` (pulp/pyyaml test deps, transport-lp extra).
**Outcome:** 64 tests pass total. Migration verified vs the live paper copy on OBJECTIVE+feasibility (LP degeneracy -> plan not unique, cost is). Exact solvers CERTIFIED against scipy; greedy heuristic CHECKED; corruption FAILs. Committed 7532234.
**Next steps:** migrate the remaining rounders (min_cost_mcf, min_cost_approx, floor_residue) + _mask/_reopt helpers; then the population-synthesis rounding (round_transport_* on contingency tables). Consider a /verify-model run against Pub_MIPEntropy_MPC now that the transport oracles exist. GitHub push still pending user request.
**Git ref:** 7532234

---

## Session 2026-07-02 (migration 3: remaining rounders)
**Agent:** Claude Opus 4.8
**Goal:** Migrate all remaining transport + population-synthesis rounders.
**Files touched:**
- `src/jr_optlib/transport/rounding.py` (new) -- round_transport_min_cost_mcf, round_transport_min_cost_approx, round_transport_floor_residue_lp (alias), + _mask_from_topk_and_mass, _reopt_from_greedy.
- `src/jr_optlib/transport/population_round.py` (new) -- dependent_round_2d dispatcher + _lp / _lp_sparsified / _mcf (were the private `_dependent_round_2d_*`); completes the ipf_2d -> integer-table pipeline.
- `src/jr_optlib/oracles/transport.py` -- `certify_dependent_round` (exact marginals + integrality + per-cell deviation < 1).
- exports; `tests/test_rounding.py` (9) + `tests/test_population_round.py` (18); `registry/functions.yaml`+`INDEX.md` (7 new entries).
**Outcome:** 91 tests pass total. All MCF paths fall back to LP in base (no ortools graph) and match the old copy on objective+feasibility; dependent rounders certified by the rounding contract and matched on contract+residual objective. Committed cbfc4eb. Transport primitive migration is now COMPLETE -- only the experiment drivers (bench_transport/bench_population) remain in the paper repo (correctly, per the inclusion rule: not reusable primitives).
**Next steps:** run `/verify-model --project Pub_MIPEntropy_MPC` end-to-end now that transport oracles exist; migrate the set-cover / mip_hybrid solver family next; vendor rail582; git-push (helpi 23) when the user asks. Gurobi WLS license (LicenseID 2711408 in C:\Users\rich\gurobi.lic) still reports expired on re-check -- local .lic likely needs replacing with freshly-issued credentials from the portal.
**Git ref:** cbfc4eb

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

## Session 2026-07-02 (handoff prep for Codex)
**Agent:** Claude Opus 4.8
**Goal:** User is low on tokens and wants to continue the jr_optlib migration ("restructuring") in Codex or Gemini. Assess whether that is trustworthy and make sure the next agent boots with full context.
**Files touched:**
- `AGENTS.md` (new) -- Codex/Gemini context file (jr_optlib had only `.claude/CLAUDE.md`, which Codex does NOT read, so it would have booted blind). Covers: what the project is, current state (4 sessions, 91 tests, transport family complete, local-only commits through cbfc4eb), next steps, the four-gate migration protocol (numerics-preserving extraction -> old-vs-new importlib differential -> oracle -> registry), and machine/Codex traps (full miniconda python path, AppData-execution escalation, base env has pulp/CBC+scipy but NOT ortools/POT/gurobi, exact pytest command).
**Outcome:** jr_optlib now has an AGENTS.md so a Codex/Gemini session opens with full context. Verdict given: safe to hand off because the safety rail is baked into the codebase (every migration gated by a differential test vs the live paper copy + an oracle; the 91-test suite fails if numerics drift) -- the only failure mode is process (skipping a gate), which AGENTS.md addresses explicitly.
**Next steps:** unchanged migration backlog -- (1) review the running /verify-model coverage_map.md for Pub_MIPEntropy_MPC; (2) migrate set-cover / mip_hybrid solver family; (3) Pub_PopInt_PartB (`ipf_nd`/HardIPF + N-D certify_ipf + integerizers); (4) `helpi 23 jr_optlib` to push when asked; (5) replace expired gurobi.lic. AGENTS.md + this log entry are uncommitted -- commit locally before switching to Codex.
**Git ref:** cbfc4eb
