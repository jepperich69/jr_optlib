# jr_optlib

One vetted, oracle-tested library of optimization primitives shared across JR
research papers. The code equivalent of the paper "feeder" links: if a second
paper could plausibly use a function, it lives here (vetted + tested +
documented + indexed); paper-specific glue stays in the paper repo.

## Why

Research code was copied per paper and drifted (the `mip_hybrid` package existed
in 6+ locations, a bug fixed in one copy staying alive in the others). Copy-as-
you-go fails at both reuse and reproducibility. This library fixes reuse; the
submission-freeze rule below fixes reproducibility.

## Layout

```
src/jr_optlib/
  transport/     vetted functions (ipf_2d, make_contingency2d, ...)
  oracles/       independent checks: core + transport + setcover
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
pip install -e .[test]      # + pytest, scipy for the oracle test suite
pytest                      # 16 oracle-backed tests
python oracle_bank/demo_scp41.py   # end-to-end harness on a real OR-Library instance
```

Optional extras: `.[oracles-full]` adds POT + networkx for transport/MCF
reference oracles.

## Status

First slice (2026-07-02): `ipf_2d` vetted and certified; set-cover oracles +
scp41 wired end-to-end. Next: migrate the transport rounding routines and the
`mip_hybrid` solver, one function at a time, using library-vs-old-copy
comparison as the migration test.
