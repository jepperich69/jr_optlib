# Oracle bank

Benchmark instances with known optima and trusted reference implementations,
used to validate `jr_optlib` functions and live paper results. Every instance's
provenance is recorded in `_retrieved_sources.md`.

## Contents

```
setcover/
  loader.py           OR-Library SCP format parser -> SetCoverInstance
  known_optima.yaml   optima table (scp*, rail*) with provenance flags
  orlib/scp41.txt     vendored instance (proven optimum 429)
demo_scp41.py         end-to-end harness proof on scp41
_retrieved_sources.md provenance for all instances + references
```

## Run the demo

```
python demo_scp41.py
```

Loads scp41, builds a feasible greedy cover, and runs the set-cover oracles:
feasibility (all rows covered), objective recomputation, and bound-consistency
(cost >= known optimum). It then injects a fault (drops a column) and confirms
the harness flags it FAIL. Expected coverage: 1 CHECKED, 1 FAIL.

## Adding an instance

1. Drop the raw file under the appropriate problem folder.
2. Add its optimum + provenance to the problem's `known_optima.yaml`.
3. Register it in `registry/instances.yaml` and `registry/INDEX.md`.
4. Record the download in `_retrieved_sources.md`.
