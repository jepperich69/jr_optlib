# -*- coding: utf-8 -*-
"""
Generic oracle framework.

An oracle is a cheap, independent verifier: it establishes that a solution has
a property the true answer must have, without re-running the algorithm that
produced it. Verifying is far cheaper than solving, and a mismatch is a
provable defect.

Every oracle returns an ``OracleResult``. A harness collects results into a
coverage map that labels each claimed result:
    CERTIFIED  -- an optimality/uniqueness certificate holds (e.g. UB == LB)
    CHECKED    -- necessary properties hold, but optimality is not certified
    FAIL       -- an oracle was violated: the result is wrong
    UNVALIDATED-- no applicable oracle ran
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Sequence

import numpy as np


class Verdict(str, Enum):
    CERTIFIED = "CERTIFIED"
    CHECKED = "CHECKED"
    FAIL = "FAIL"
    UNVALIDATED = "UNVALIDATED"


@dataclass
class OracleResult:
    name: str
    passed: bool
    residual: float          # how far from ideal (0.0 == exact); lower is better
    tol: float               # threshold used for the pass/fail decision
    certifies: bool = False  # True if passing this oracle *certifies* optimality
    detail: str = ""

    @property
    def verdict(self) -> Verdict:
        if not self.passed:
            return Verdict.FAIL
        return Verdict.CERTIFIED if self.certifies else Verdict.CHECKED

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        tag = " [certifies]" if (self.passed and self.certifies) else ""
        return f"[{mark}] {self.name}: residual={self.residual:.3e} tol={self.tol:.1e}{tag}  {self.detail}"


def summarize(results: Sequence[OracleResult]) -> Verdict:
    """Roll a set of oracle results up into a single coverage verdict."""
    if not results:
        return Verdict.UNVALIDATED
    if any(not r.passed for r in results):
        return Verdict.FAIL
    if any(r.certifies for r in results):
        return Verdict.CERTIFIED
    return Verdict.CHECKED


# --------------------------------------------------------------------------
# Generic oracles (problem-agnostic)
# --------------------------------------------------------------------------

def gap_certificate(upper_bound: float, lower_bound: float,
                    rel_tol: float = 1e-6, name: str = "gap_certificate") -> OracleResult:
    """Optimality certificate from primal/dual (UB/LB) bounds.

    For a minimization: a feasible objective is an upper bound, any valid
    relaxation/dual value is a lower bound. When ``UB == LB`` (to tolerance)
    the incumbent is *certified optimal*. This is oracle type (2) and is the
    single strongest cheap check -- solvers already compute ``BestBound``/``gap``
    and typically throw it away.
    """
    denom = max(abs(upper_bound), abs(lower_bound), 1e-12)
    gap = abs(upper_bound - lower_bound) / denom
    passed = gap <= rel_tol
    return OracleResult(
        name=name, passed=passed, residual=gap, tol=rel_tol, certifies=passed,
        detail=f"UB={upper_bound:.6g} LB={lower_bound:.6g} rel_gap={gap:.3e}",
    )


def differential(value_a: float, value_b: float,
                 label_a: str = "exact", label_b: str = "heuristic",
                 rel_tol: float = 1e-6, name: str = "differential") -> OracleResult:
    """Exact-vs-alternative differential test (oracle type 3).

    Compares an objective from a trusted method against a second method on the
    same instance. Any relative disagreement above tolerance is a defect in one
    of them.
    """
    denom = max(abs(value_a), abs(value_b), 1e-12)
    rel = abs(value_a - value_b) / denom
    passed = rel <= rel_tol
    return OracleResult(
        name=name, passed=passed, residual=rel, tol=rel_tol, certifies=False,
        detail=f"{label_a}={value_a:.6g} {label_b}={value_b:.6g} rel_diff={rel:.3e}",
    )


def metamorphic(before: float, after: float, relation: str = "==",
                rel_tol: float = 1e-6, name: str = "metamorphic") -> OracleResult:
    """Metamorphic / monotonicity oracle (type 7).

    Assert a relation between an output before and after a known input
    transformation (e.g. scaling all costs by k>0 must scale the optimum by k;
    tightening a constraint must not improve a minimization objective).
    ``relation`` in {"==", "<=", ">=", "<", ">"}.
    """
    denom = max(abs(before), abs(after), 1e-12)
    if relation == "==":
        residual = abs(before - after) / denom
        passed = residual <= rel_tol
    elif relation in ("<=", "<"):
        residual = max(0.0, (after - before)) / denom
        passed = (after <= before + rel_tol * denom) if relation == "<=" else (after < before)
    elif relation in (">=", ">"):
        residual = max(0.0, (before - after)) / denom
        passed = (after >= before - rel_tol * denom) if relation == ">=" else (after > before)
    else:
        raise ValueError(f"unknown relation {relation!r}")
    return OracleResult(
        name=name, passed=passed, residual=residual, tol=rel_tol, certifies=False,
        detail=f"before={before:.6g} {relation} after={after:.6g}",
    )
