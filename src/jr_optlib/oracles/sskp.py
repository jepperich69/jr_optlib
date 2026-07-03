# -*- coding: utf-8 -*-
"""
Oracles for the SSKP delta-update MH chain (jr_optlib.sampling.sskp_mh).

The reusable content of the chain is its *incremental sufficient statistics*:
it maintains ``(muX, sigmaX2, revX)`` and the objective ``curF`` by O(1) updates
on each single-coordinate flip, rather than an O(k) recompute. The defining
correctness property is therefore metamorphic and exact:

    the maintained statistics must equal a from-scratch recompute of the final
    selection.

If they do, the delta bookkeeping provably never drifted -- this *certifies*
the algebra without re-running the chain. A second, independent check validates
the chance-constraint penalty formula against a scipy.stats.norm partial
expectation (a different code path from the Abramowitz-Stegun approximation).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from jr_optlib.oracles.core import OracleResult, differential
from jr_optlib.sampling.sskp_mh import sskp_objective, sskp_penalty


def _scipy_penalty(muX: float, sigmaX2: float, c: float, q: float) -> float:
    """Independent partial expectation c*E[(S-q)^+], S~N(muX,sigmaX2).

    Uses scipy.stats.norm (a different implementation from the chain's
    Abramowitz-Stegun rational approximation), so agreement is a genuine
    differential rather than a re-run of the same arithmetic.
    """
    from scipy.stats import norm

    if sigmaX2 < 1e-12:
        return c * max(muX - q, 0.0)
    sigmaX = float(np.sqrt(sigmaX2))
    z = (muX - q) / sigmaX
    return c * ((muX - q) * float(norm.cdf(z)) + sigmaX * float(norm.pdf(z)))


def delta_invariant(result, r, mu, sigma, c, q, tol: float = 1e-9) -> OracleResult:
    """Certificate: maintained (muX, sigmaX2, revX, curF) == full recompute.

    Independent O(k) recomputation of the sufficient statistics and objective
    from the final selection ``result.final_x``. Equality (to floating
    reassociation tolerance) certifies that the O(1) delta updates carried the
    exact same value the slow path would have -- no accumulated drift.
    """
    F_ref, muX_ref, sig2_ref, rev_ref = sskp_objective(
        result.final_x, r, mu, sigma, c, q)

    def _rel(a, b):
        return abs(a - b) / max(abs(a), abs(b), 1e-12)

    res = max(
        _rel(result.muX, muX_ref),
        _rel(result.sigmaX2, sig2_ref),
        _rel(result.revX, rev_ref),
        _rel(result.curF, F_ref),
    )
    return OracleResult(
        name="delta_invariant", passed=res <= tol, residual=res, tol=tol,
        certifies=res <= tol,
        detail=(f"maintained vs recompute: dmuX={_rel(result.muX, muX_ref):.2e} "
                f"dsig2={_rel(result.sigmaX2, sig2_ref):.2e} "
                f"drev={_rel(result.revX, rev_ref):.2e} "
                f"dF={_rel(result.curF, F_ref):.2e}"),
    )


def penalty_reference(result, c, q, tol: float = 1e-6) -> OracleResult:
    """Differential: A&S penalty at the final state == scipy partial expectation."""
    as_pen = sskp_penalty(result.muX, result.sigmaX2, float(c), float(q))
    sp_pen = _scipy_penalty(result.muX, result.sigmaX2, float(c), float(q))
    return differential(as_pen, sp_pen, label_a="A&S", label_b="scipy",
                        rel_tol=tol, name="penalty_reference")


def certify_sskp_chain(result, r, mu, sigma, c, q,
                       tol: float = 1e-9) -> Tuple[List[OracleResult], bool]:
    """Run the SSKP chain oracle suite. Returns (results, certified).

    ``certified`` is True iff the delta invariant holds exactly: the maintained
    sufficient statistics and objective equal an independent full recompute of
    the final state. The penalty differential adds confidence in the objective
    formula but is not itself a certificate of the chain's bookkeeping.
    """
    r_delta = delta_invariant(result, r, mu, sigma, c, q, tol=tol)
    r_pen = penalty_reference(result, c, q, tol=max(tol, 1e-6))
    results = [r_delta, r_pen]
    return results, r_delta.passed and r_pen.passed
