# -*- coding: utf-8 -*-
"""
End-to-end oracle-bank demo: OR-Library set-cover instance scp41.

Proves the harness on real benchmark data with a known optimum (429):
  1. load the vendored instance,
  2. produce a feasible solution with a simple independent greedy,
  3. verify it with the set-cover oracles (feasibility + objective recompute),
  4. compare to the known optimum -> coverage map.

Run:
  python demo_scp41.py     (adds ../src to sys.path automatically)
"""

import sys
from pathlib import Path

# Make jr_optlib importable when run in-place, and this folder for the loader.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "setcover"))

import yaml  # noqa: E402

from loader import load_scp, rows_by_column_to_columns_by_row  # noqa: E402
from jr_optlib.oracles import setcover_feasible, setcover_cost  # noqa: E402
from jr_optlib.oracles.core import metamorphic  # noqa: E402
from jr_optlib.harness import CoverageMap  # noqa: E402


def greedy_cover(inst):
    """Independent cost-per-newly-covered greedy. Not optimal -- just feasible."""
    by_row = rows_by_column_to_columns_by_row(inst)
    uncovered = set(range(inst.n_rows))
    # rows with no covering column would make the instance infeasible
    assert all(by_row[i] for i in range(inst.n_rows)), "instance has an uncoverable row"
    selected = []
    col_rows = [set(r) for r in inst.covers]
    while uncovered:
        best_j, best_ratio = None, float("inf")
        for j in range(inst.n_cols):
            new = len(col_rows[j] & uncovered)
            if new == 0:
                continue
            ratio = inst.costs[j] / new
            if ratio < best_ratio:
                best_ratio, best_j = ratio, j
        selected.append(best_j)
        uncovered -= col_rows[best_j]
    return selected


def main():
    setcover_dir = Path(__file__).resolve().parent / "setcover"
    known = yaml.safe_load((setcover_dir / "known_optima.yaml").read_text())
    meta = known["instances"]["scp41"]
    opt = meta["optimum"]

    inst = load_scp(setcover_dir / meta["file"])
    print(f"Loaded {inst.name}: {inst.n_rows} rows x {inst.n_cols} cols; "
          f"known optimum = {opt}")

    sol = greedy_cover(inst)
    ub = setcover_cost(sol, inst.costs)

    cov = CoverageMap()
    # A feasible heuristic: it should be feasible and cost >= the true optimum.
    # It is NOT claimed optimal, so the correct verdict is CHECKED, not FAIL.
    cov.add("scp41 greedy solution (heuristic)", [
        setcover_feasible(sol, inst.covers, inst.n_rows),
        # a feasible cover can never cost less than the optimum; opt <= ub
        metamorphic(before=ub, after=opt, relation="<=",
                    name="cost_ge_known_optimum"),
    ])

    # Fault injection: drop a column to make the cover infeasible. The harness
    # must catch this -> FAIL. This is the check that the oracles have teeth.
    broken = sol[:-1]
    cov.add("scp41 corrupted solution (column dropped)", [
        setcover_feasible(broken, inst.covers, inst.n_rows),
    ])

    print()
    print(cov.render())
    print()
    print(f"greedy cost (UB) = {ub:.0f}   known optimum = {opt}   "
          f"greedy excess = {ub - opt:.0f} ({100*(ub-opt)/opt:.1f}%)")
    print("\nHarness end-to-end OK: real benchmark loaded, feasibility verified, "
          "objective recomputed independently, bound-consistency checked, and "
          "an injected fault correctly flagged.")
    # Contract: greedy is CHECKED (feasible, consistent) and the corruption FAILs.
    assert cov.claims[0].results[0].passed, "greedy produced an infeasible cover!"
    assert cov.claims[0].verdict.value == "CHECKED", cov.claims[0].verdict
    assert cov.claims[1].verdict.value == "FAIL", cov.claims[1].verdict


if __name__ == "__main__":
    main()
