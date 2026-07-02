# -*- coding: utf-8 -*-
"""
Robustness harness: turn oracle results into a coverage map.

A coverage map is the deliverable of a robustness check. For each claimed
result it records the strongest verdict any oracle supports:

    CERTIFIED   -- an optimality/uniqueness certificate holds
    CHECKED     -- necessary properties verified, optimality not certified
    FAIL        -- an oracle was violated: the result is wrong
    UNVALIDATED -- no applicable oracle ran

The point is honesty about *what is actually guaranteed*: a paper can then say
which numbers are proven, which are checked, and which rest on trust alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from jr_optlib.oracles.core import OracleResult, Verdict, summarize


@dataclass
class Claim:
    """One result in a paper/model and the oracles run against it."""
    label: str
    results: List[OracleResult] = field(default_factory=list)

    @property
    def verdict(self) -> Verdict:
        return summarize(self.results)


@dataclass
class CoverageMap:
    claims: List[Claim] = field(default_factory=list)

    def add(self, label: str, results: Sequence[OracleResult]) -> Claim:
        c = Claim(label, list(results))
        self.claims.append(c)
        return c

    def counts(self) -> dict:
        out = {v: 0 for v in Verdict}
        for c in self.claims:
            out[c.verdict] += 1
        return out

    @property
    def ok(self) -> bool:
        """True if nothing FAILed."""
        return all(c.verdict is not Verdict.FAIL for c in self.claims)

    def render(self) -> str:
        lines = ["Coverage map", "=" * 60]
        for c in self.claims:
            lines.append(f"{c.verdict.value:<12} {c.label}")
            for r in c.results:
                lines.append(f"    {r}")
        counts = self.counts()
        lines.append("-" * 60)
        lines.append("  ".join(f"{v.value}={counts[v]}" for v in Verdict))
        return "\n".join(lines)
