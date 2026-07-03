# -*- coding: utf-8 -*-
"""
Delta-update Metropolis-Hastings for the SSKP chance-constrained selection.

VETTED FUNCTION -- registry id: sampling.sskp_mh_chain

Extracted (formulation-preserving) from the Java compute backend of
``Pub_SAA_PMIP_MC`` (``java_backend/.../SamplerCli.java::sskpMhChain``). What is
reusable there is not the JVM host but the *incremental sufficient-statistics*
move: the chain flips one coordinate of a binary selection ``x`` and updates
the running statistics ``(muX, sigmaX2, revX)`` in O(1), so each step costs a
constant amount of work instead of an O(k) objective recompute. This is the
only thing that makes a chain of millions of steps tractable, and it is
language-agnostic; the Java was merely a fast host for it.

Objective (minimised by the chain, target pi_beta prop exp(-beta*F)):

    F(x) = -sum_{i: x_i=1} r_i  +  c * E[(S - q)^+],   S ~ N(muX, sigmaX2)

where muX = sum mu_i x_i, sigmaX2 = sum sigma_i^2 x_i. The penalty is the
partial expectation of a normal, ``c*((muX-q)Phi(z) + sigmaX*phi(z))`` with
z=(muX-q)/sigmaX -- a smooth surrogate for a chance constraint on S.

Verified by (jr_optlib.oracles.sskp.certify_sskp_chain):
  - delta_invariant  : the incrementally maintained (muX, sigmaX2, revX, curF)
                       equal a from-scratch recompute of the final state -- the
                       exact metamorphic check that the O(1) bookkeeping never
                       drifts from the O(k) truth. This *certifies* the
                       delta-update algebra.
  - penalty_reference: the Abramowitz-Stegun penalty agrees with an independent
                       scipy.stats.norm partial expectation (differential).

Backends
--------
The hot loop is written in numba-njit-compatible style (scalar locals, typed
arrays, no Python objects). If ``numba`` is importable it is JIT-compiled;
otherwise the identical pure-Python source runs. Randomness is drawn up front
with a numpy Generator and consumed by the kernel, so the two backends are
*bit-identical* given the same seed -- the numba path is validated by a
differential against the pure-Python path (tests/test_sskp_mh.py), not trusted
blindly. numba is an optional accelerator, never a runtime dependency.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:  # optional accelerator; the library stays importable without it
    import numba  # type: ignore
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - depends on the environment
    _HAVE_NUMBA = False

__all__ = ["sskp_mh_chain", "sskp_objective", "sskp_penalty", "SskpMhResult"]

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _build(jit: bool):
    """Build (kernel, penalty) from a single source, optionally njit-compiled.

    With ``jit=False`` the returned callables are plain Python (the oracle
    reference). With ``jit=True`` they are numba-compiled from the *same* code,
    so any divergence is a compiler bug, caught by the differential test.
    """
    dec = numba.njit(cache=True) if jit else (lambda f: f)

    @dec
    def _norm_cdf(z):  # Abramowitz & Stegun 26.2.17, max abs err < 7.5e-8
        if z < -8.0:
            return 0.0
        if z > 8.0:
            return 1.0
        t = 1.0 / (1.0 + 0.2316419 * abs(z))
        poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
                    + t * (-1.821255978 + t * 1.330274429))))
        phi = math.exp(-0.5 * z * z) * _INV_SQRT_2PI
        p = 1.0 - phi * poly
        return p if z >= 0.0 else 1.0 - p

    @dec
    def _norm_pdf(z):
        return math.exp(-0.5 * z * z) * _INV_SQRT_2PI

    @dec
    def penalty(muX, sigmaX2, c, q):
        if sigmaX2 < 1e-12:  # deterministic limit; also guards tiny negatives
            d = muX - q
            return c * (d if d > 0.0 else 0.0)
        sigmaX = math.sqrt(sigmaX2)
        z = (muX - q) / sigmaX
        return c * ((muX - q) * _norm_cdf(z) + sigmaX * _norm_pdf(z))

    @dec
    def kernel(x, r, mu, sigma2, c, q, beta, js, us, t_burn, thin, samplesF):
        # Initial sufficient statistics for the given x.
        muX = 0.0
        sigmaX2 = 0.0
        revX = 0.0
        for i in range(x.shape[0]):
            if x[i] == 1:
                muX += mu[i]
                sigmaX2 += sigma2[i]
                revX += r[i]
        curF = -revX + penalty(muX, sigmaX2, c, q)

        n = js.shape[0]
        accepted = 0
        sidx = 0
        for s in range(n):
            j = js[s]
            if x[j] == 1:
                dMu = -mu[j]; dS = -sigma2[j]; dR = -r[j]
            else:
                dMu = mu[j]; dS = sigma2[j]; dR = r[j]
            newF = -(revX + dR) + penalty(muX + dMu, sigmaX2 + dS, c, q)
            delta = newF - curF
            if delta <= 0.0 or us[s] < math.exp(-beta * delta):
                x[j] = 1 - x[j]
                muX += dMu; sigmaX2 += dS; revX += dR
                curF = newF
                if s >= t_burn:
                    accepted += 1
            if s >= t_burn and ((s - t_burn) + 1) % thin == 0:
                samplesF[sidx] = curF
                sidx += 1
        return accepted, muX, sigmaX2, revX, curF

    return kernel, penalty


_kernel_py, sskp_penalty = _build(False)
_kernel_nb = _build(True)[0] if _HAVE_NUMBA else None


def sskp_objective(x, r, mu, sigma, c, q):
    """Full O(k) recompute of F(x) from scratch (the oracle reference).

    Returns ``(F, muX, sigmaX2, revX)``. Uses the same Abramowitz-Stegun
    penalty as the chain so it isolates the *delta bookkeeping* -- an
    independent penalty is cross-checked separately in the oracle.
    """
    x = np.asarray(x)
    r = np.asarray(r, float); mu = np.asarray(mu, float); sigma = np.asarray(sigma, float)
    sel = x == 1
    muX = float(mu[sel].sum())
    sigmaX2 = float((sigma[sel] ** 2).sum())
    revX = float(r[sel].sum())
    F = -revX + sskp_penalty(muX, sigmaX2, float(c), float(q))
    return F, muX, sigmaX2, revX


@dataclass
class SskpMhResult:
    samples_F: np.ndarray  # recorded objective per retained sample
    acc_rate: float        # acceptance rate over the sampling phase
    z_upper: float         # E_{pi_beta}[F], an upper bound on the SAA optimum
    final_x: np.ndarray    # final binary selection
    muX: float             # maintained sufficient statistics at termination
    sigmaX2: float
    revX: float
    curF: float            # maintained objective at termination
    elapsed: float
    backend: str


def _resolve_backend(backend: str) -> bool:
    if backend == "numba":
        if not _HAVE_NUMBA:
            raise RuntimeError("backend='numba' requested but numba is not installed")
        return True
    if backend == "python":
        return False
    if backend == "auto":
        return _HAVE_NUMBA
    raise ValueError(f"unknown backend {backend!r} (use 'auto'|'numba'|'python')")


def sskp_mh_chain(r, mu, sigma, c, q, beta,
                  t_burn: int, t_sample: int, thin: int = 1,
                  seed: int = 0, backend: str = "auto",
                  x0: Optional[np.ndarray] = None) -> SskpMhResult:
    """Run a delta-update MH chain on the SSKP selection objective.

    Parameters
    ----------
    r, mu, sigma : length-k arrays. Per-item revenue, mean load and load std.
    c, q : penalty weight and the chance-constraint threshold on total load.
    beta : inverse temperature; the chain targets pi_beta prop exp(-beta*F).
    t_burn, t_sample, thin : burn-in steps, retained samples, and thinning
        (the sampling phase runs ``t_sample*thin`` steps, recording every thin).
    seed : seeds a numpy Generator that draws all proposals/uniforms up front,
        so results are reproducible and backend-independent.
    backend : 'auto' (numba if available), 'numba', or 'python'.
    x0 : optional initial selection (length k, 0/1). Random if omitted.

    Returns
    -------
    SskpMhResult with the recorded objective samples, acceptance rate, the
    upper-bound estimate ``z_upper = mean(samples_F)``, and the maintained
    sufficient statistics (which the oracle checks against a full recompute).
    """
    r = np.ascontiguousarray(r, dtype=float)
    mu = np.ascontiguousarray(mu, dtype=float)
    sigma = np.ascontiguousarray(sigma, dtype=float)
    sigma2 = sigma * sigma
    k = r.shape[0]
    if not (mu.shape[0] == sigma.shape[0] == k):
        raise ValueError("r, mu, sigma must share length k")
    if t_sample <= 0 or thin <= 0 or t_burn < 0:
        raise ValueError("require t_sample>0, thin>0, t_burn>=0")

    rng = np.random.default_rng(seed)
    if x0 is None:
        x = rng.integers(0, 2, size=k).astype(np.int64)
    else:
        x = np.ascontiguousarray(x0, dtype=np.int64).copy()
        if x.shape[0] != k or not np.isin(x, (0, 1)).all():
            raise ValueError("x0 must be a length-k 0/1 vector")

    n_total = t_burn + t_sample * thin
    js = rng.integers(0, k, size=n_total).astype(np.int64)
    us = rng.random(n_total)
    samplesF = np.empty(t_sample, dtype=float)

    use_nb = _resolve_backend(backend)
    kern = _kernel_nb if use_nb else _kernel_py

    t0 = time.time()
    accepted, muX, sigmaX2, revX, curF = kern(
        x, r, mu, sigma2, float(c), float(q), float(beta),
        js, us, int(t_burn), int(thin), samplesF)
    elapsed = time.time() - t0

    return SskpMhResult(
        samples_F=samplesF,
        acc_rate=accepted / float(t_sample * thin),
        z_upper=float(np.mean(samplesF)),
        final_x=x,
        muX=float(muX), sigmaX2=float(sigmaX2), revX=float(revX), curF=float(curF),
        elapsed=elapsed,
        backend="numba" if use_nb else "python",
    )
