# Oracle bank -- retrieved sources

Provenance for every benchmark instance and reference optimum in the oracle
bank, following the JR `Literature/_retrieved_sources.md` convention.

## OR-Library Set-Cover (SCP)

- **Citation:** Beasley, J.E. (1990). "OR-Library: distributing test problems by
  electronic mail." *Journal of the Operational Research Society* 41(11), 1069-1072.
- **Source URL:** http://people.brunel.ac.uk/~mastjjb/jeb/orlib/scpinfo.html
- **Retrieved:** scp41-scp49 already vendored in
  `Pub_MIPEntropy_MPC/code/data/orlib_scp/`; scp41 copied into
  `oracle_bank/setcover/orlib/scp41.txt` on 2026-07-02.
- **Facts supported:** proven optimal values used as known-answer oracles.
  scp41=429, scp42=512, scp43=516, scp44=494, scp45=512, scp46=560,
  scp47=430, scp48=492, scp49=641 (see `setcover/known_optima.yaml`).
- **Used in:** `oracle_bank/demo_scp41.py` end-to-end harness proof;
  registry entry `setcover.scp41`.

## OR-Library Rail (unicost set-cover)

- **Citation:** Beasley, J.E.; OR-Library rail instances.
- **Source URL:** http://people.brunel.ac.uk/~mastjjb/jeb/orlib/scpinfo.html
- **Status:** NOT yet vendored. Only run logs exist locally
  (`Pub_MIPEntropy_MPC/.../experiments/rail/`). Download the raw `rail582.txt`
  before use and record it here.
- **Facts supported:** proven optima rail507=174, rail516=182, rail582=211;
  rail2536=691 (best-known). See `setcover/known_optima.yaml`.

## Reference implementations (differential oracles)

Recorded in `registry/references.yaml`. scipy (Hungarian, linprog) and, when
installed, networkx (min-cost flow) and POT (Sinkhorn/EMD) serve as independent
cross-checks. The in-house log-space raking (`ipf_reference`) is a deliberately
different numerical route for IPF.
