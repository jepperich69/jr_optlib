# jr_optlib registry schema

The registry answers one question at the start of every new paper:
**"Do we already have this, and can we trust it?"** It is three files sharing
one core schema:

| file | registry | answers |
|------|----------|---------|
| `functions.yaml`  | vetted functions           | "Is there a trusted implementation of X?" |
| `instances.yaml`  | known-answer benchmarks    | "Is there an instance with a known optimum to test against?" |
| `references.yaml` | trusted reference impls    | "What independent implementation can cross-check X?" |

Both humans and agents read these. Keep them YAML (browsable + parseable).
`INDEX.md` is a generated human-readable summary; the YAML files are the source
of truth.

## Core fields (every entry, every registry)

```yaml
- id:        unique dotted id, e.g. transport.ipf_2d
  name:      short human name
  kind:      function | instance | reference
  summary:   one line, what it is
  tags:      [domain, method, problem-type]      # for search
  status:    vetted | experimental | deprecated
  source:    where it came from (repo path, library, benchmark URL)
  added:     YYYY-MM-DD
  provenance: how correctness is established (free text; see per-kind below)
```

## Per-kind extra fields

### function (functions.yaml)
```yaml
  module:    import path, e.g. jr_optlib.transport.ipf
  signature: one-line call signature
  oracles:   [ids of oracles that validate it]   # from jr_optlib.oracles
  tests:     path to the test file
  certifies: true|false   # can any oracle certify optimality, not just check
```

### instance (instances.yaml)
```yaml
  problem_type: e.g. set-cover | transport | assignment
  file:         path under oracle_bank/ (or generator call)
  known_optimum: numeric value (or null if only bounds known)
  optimum_kind: exact | best-known | bound
```

### reference (references.yaml)
```yaml
  backend:   scipy | networkx | pot | ortools | custom
  validates: [ids of jr_optlib functions it cross-checks]
  import:    how to call it
```

## Rules

1. **Inclusion rule.** If a second paper could plausibly use a function, it
   goes in `jr_optlib` and gets a `functions.yaml` entry. Paper-specific glue
   stays in the paper repo.
2. **No entry without an oracle.** A `vetted` function must list at least one
   oracle. Un-oracled code is `experimental` at best.
3. **Reproducibility.** Papers import live during development; at submission,
   pin the commit hash (`__version__` + git SHA) and vendor a snapshot into the
   reproduction package. Record the pinned SHA in the paper's repro notes.
