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
**Next steps:** git init + first commit (deferred -- not yet requested); migrate the transport rounding routines and mip_hybrid one function at a time (library-vs-old-copy differential as the migration test); vendor rail582; add assignment/MCF functions wired to the scipy/networkx references already registered.
**Git ref:** (no repo yet)

