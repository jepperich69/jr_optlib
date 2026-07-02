# AI Session Log - jr_optlib

<!-- Claude updates this file at the start and end of every working session. -->
<!-- Format: one ## Session block per date. -->

## Session 2026-07-02
**Agent:** Claude Opus 4.8
**Goal:** Build the first slice of the jr_optlib robustness system (design was agreed in the AI_auto session 2026-07-02). Create the shared library + three-registry index + oracle bank + robustness harness, with ipf_2d as the pilot vetted function.
**Files touched:** whole project scaffolded --
- `pyproject.toml`, `src/jr_optlib/__init__.py` -- numpy-only runtime; scipy/pot/networkx as optional [oracles]/[oracles-full] extras.
- `src/jr_optlib/transport/ipf.py` -- `ipf_2d` + `make_contingency2d`, extracted numerics-preserving from Pub_MIPEntropy_MPC/code/mip_hybrid/apps/population_transport.py.
- `src/jr_optlib/oracles/{core,transport,setcover}.py` -- generic oracles (gap_certificate, differential, metamorphic), IPF certificate (marginals + diagonal-scaling form via least-squares log-interaction, uniqueness per Sinkhorn), set-cover feasibility/cost.
- `src/jr_optlib/harness.py` -- CoverageMap: CERTIFIED / CHECKED / FAIL / UNVALIDATED.
- `registry/{SCHEMA,INDEX}.md` + `{functions,instances,references}.yaml` -- three-registry index, one schema.
- `oracle_bank/` -- SCP loader, vendored scp41 (opt 429), known_optima.yaml, _retrieved_sources.md provenance, demo_scp41.py.
- `tests/test_ipf.py` + conftest -- 16 oracle-backed tests.
- `.claude/CLAUDE.md` filled; `.gitignore`.
**Outcome:** All 16 tests pass; end-to-end demo on real OR-Library scp41 works (greedy=CHECKED, injected fault=FAIL). `ipf_2d` vetted and certified. Delivery skill `/verify-model` built at `~/.claude/skills/verify-model/`. Installed pytest + pyyaml into miniconda base env. Confirmed the `mip_hybrid` 6x-duplication and the double-defined `round_transport_greedy_push` (population_transport.py:789 and :882).
**Next steps:** migrate the transport rounding routines and mip_hybrid one function at a time (library-vs-old-copy differential as the migration test); vendor rail582; add assignment/MCF functions wired to the scipy/networkx references already registered. GitHub push (helpi 23) not done -- local only, awaiting user request.
**Git ref:** dd09fc6 (local; initial commit, 27 files)

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

