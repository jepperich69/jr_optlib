# jr_optlib

One vetted, oracle-tested library of optimization primitives shared across JR
research papers. The code equivalent of the paper "feeder" links: if a second
paper could plausibly use a function, it lives here (vetted + tested +
documented + indexed); paper-specific glue stays in the paper repo.

**Start at [`CATALOG.md`](CATALOG.md)** — the navigable, usage-first map of
every primitive by problem domain, with worked examples and the paper↔primitive
feeder table. This README covers the philosophy and rules.

## Why

Research code was copied per paper and drifted (the `mip_hybrid` package existed
in 6+ locations, a bug fixed in one copy staying alive in the others). Copy-as-
you-go fails at both reuse and reproducibility. This library fixes reuse; the
submission-freeze rule below fixes reproducibility.

## Layout

```
src/jr_optlib/
  transport/     IPF, Sinkhorn, exact/rounded min-cost transport
  population/    high-dimensional IPF + integerization pipeline
  setcover/      entropy-relaxation set cover + MIP/polish
  optimization/  entropic-risk QP/assignment, dual ascent, discrete choice, DP, NLP, routing
  sampling/      MCMC, simulated annealing, SSKP delta-update MH, Q-learning
  vsp/           vehicle-scheduling Metropolis + greedy
  oracles/       independent checks: core + one module per domain
  harness.py     turns oracle results into a coverage map
registry/        the index: functions.yaml, instances.yaml, references.yaml, SCHEMA.md
oracle_bank/     benchmark instances with known optima + reference impls + provenance
tests/           oracle-backed tests (assert the oracles certify, not hand-computed values)
```

## The idea: verify, don't re-solve

Finding an optimum is expensive; *verifying* a candidate is cheap and often
already possible with what the code computes. Oracles, by generality:

1. feasibility + objective recomputation
2. duality/relaxation optimality certificate (UB == LB) -- strongest cheap check
3. exact-vs-heuristic differential test
4. planted / seeded instances
5. small-instance exhaustive
6. benchmark libraries + trusted reference impls
7. monotonicity / metamorphic

The deliverable is a **coverage map** labelling every result CERTIFIED /
CHECKED / FAIL / UNVALIDATED -- honest about what is actually guaranteed.

## Orient before writing (start of every new paper)

Search `registry/` before implementing anything:
- `functions.yaml` -- "is there a trusted implementation of X?"
- `instances.yaml` -- "is there a known-optimum instance to test against?"
- `references.yaml` -- "what independent impl can cross-check X?"

`registry/INDEX.md` is the human-readable summary. `registry/SCHEMA.md` is the
entry schema.

## Reproducibility rule (submission freeze)

Develop against the live library. **At submission, pin the commit hash**
(`jr_optlib.__version__` + git SHA) and vendor a snapshot into the paper's
reproduction package. Record the pinned SHA in the paper's repro notes. This is
how a shared library and a standalone-reproducible paper coexist.

## Install / run

```
pip install -e .            # runtime (numpy only)
pip install -e .[test]      # + pytest, scipy, pulp, gurobipy, pandas, numba
pytest                      # 139 oracle-backed tests
python oracle_bank/demo_scp41.py   # end-to-end harness on a real OR-Library instance
```

Optional extras (`pyproject.toml`): `oracles` (scipy), `transport-lp` (pulp),
`setcover-gurobi` (gurobipy), `population` (pandas), `fast-sampling` (numba),
`oracles-full` (POT + networkx). A paper importing one function never pulls the
whole stack.

## Status

41 vetted primitives across transport, population synthesis, set cover,
entropic-risk assignment, sampling (MCMC/SSKP/RL), VSP, discrete choice, DP,
NLP and routing — each certified by at least one oracle; 139 oracle-backed
tests pass. Fed by 8 papers (see the feeder map in `CATALOG.md`). Growth is
demand-driven: primitives are extracted, one at a time with an oracle, when a
paper needs them.
