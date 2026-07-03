# Project: jr_optlib

<!-- This file is read by Claude Code at session start. Keep it current but brief.
     Session-by-session changes go in _ai_log.md -- not here. -->

## What this project is about

`jr_optlib` is one vetted, oracle-tested library of optimization primitives
shared across JR research papers (the code analogue of the paper "feeder"
links). It replaces copy-as-you-go research code, which drifted across papers
(the `mip_hybrid` package existed in 6+ locations with divergent bug fixes).

Core idea: **verify, don't re-solve.** Verifying an optimization solution is far
cheaper than finding it, so each function ships with independent *oracles*
(feasibility recompute, duality certificate, differential vs a reference,
metamorphic, known-answer benchmarks). The `harness` turns oracle results into a
**coverage map** labelling every result CERTIFIED / CHECKED / FAIL / UNVALIDATED.

## Key files

- `CATALOG.md` -- **entry point.** Problem-oriented, usage-first map of all primitives with worked examples, pipelines, and the paper<->primitive feeder table. `registry/functions.yaml` stays the machine source of truth; `CATALOG.md` + `INDEX.md` are hand-synced views (they drift -- re-check when adding entries).
- `src/jr_optlib/transport/ipf.py` -- first vetted function (`ipf_2d`), extracted numerics-preserving from Pub_MIPEntropy_MPC.
- `src/jr_optlib/sampling/` -- `mcmc` (generic MH/SA/ladder), `sskp_mh` (SSKP chance-constrained delta-update MH, optional numba), `rl` (Q-learning).
- `src/jr_optlib/oracles/` -- `core` (generic oracles), `transport` (IPF certificate), `setcover`, `population` (margin + secondary-floor), `entropic_qp` (formula-vs-MC + brute-force-optimum).
- `src/jr_optlib/optimization/` -- `lagrangian` (subgradient dual ascent), `entropic_qp` (QP/Hungarian/MIQP entropic-risk assignment; uses scipy.optimize and, for MIQP, gurobipy).
- `src/jr_optlib/harness.py` -- coverage-map builder (the deliverable).
- `registry/` -- the index: `functions.yaml`, `instances.yaml`, `references.yaml`, `SCHEMA.md`, `INDEX.md`.
- `oracle_bank/` -- benchmark instances + reference impls + provenance; `demo_scp41.py` proves the harness end-to-end.
- `tests/` -- oracle-backed (assert the oracles certify, not hand-computed values).

## Standing constraints

- **numpy-only runtime.** Oracles/tests may use scipy; POT/networkx/numba are optional extras. Keep the runtime dep set minimal so a paper importing one function doesn't pull the whole validation stack. Accelerators (e.g. numba for `sskp_mh_chain`, the `fast-sampling` extra) must be optional with a bit-identical pure-Python fallback.
- **Numerics-preserving extraction.** When migrating a function from a paper, preserve its exact behavior (even quirks like `ipf_2d` returning `(X, elapsed)`) so migration is verified by library-vs-old-copy equality.
- **No vetted entry without an oracle** (see `registry/SCHEMA.md`).
- **Reproducibility:** develop against the live library; at submission pin the commit SHA and vendor a snapshot into the paper's repro package.
- Python: `C:\Users\rich\AppData\Local\miniconda3\python.exe` (base env; NOT on PATH). Has numpy, scipy, pytest, pyyaml.

## What NOT to touch

- Live paper code under `Publikationer\` -- migration is forward-first and one function at a time, only when explicitly requested.
