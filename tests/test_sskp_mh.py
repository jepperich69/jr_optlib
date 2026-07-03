# -*- coding: utf-8 -*-
"""Oracle-backed tests for the SSKP delta-update MH chain.

These assert that the oracles certify the run (the delta bookkeeping is exact,
the penalty matches an independent reference), that the numba and pure-Python
backends are bit-identical, and that the chain samples the correct Boltzmann
distribution -- not hand-computed magic numbers.
"""
import itertools

import numpy as np
import pytest

from jr_optlib.sampling.sskp_mh import (
    sskp_mh_chain, sskp_objective, _HAVE_NUMBA,
)
from jr_optlib.oracles.sskp import certify_sskp_chain, delta_invariant


def _instance(k, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.uniform(0.5, 2.0, size=k)
    mu = rng.uniform(0.5, 1.5, size=k)
    sigma = rng.uniform(0.2, 1.0, size=k)
    c = 3.0
    q = 0.5 * mu.sum()  # threshold near half the max total load
    return r, mu, sigma, c, q


@pytest.mark.parametrize("k", [8, 20, 50])
@pytest.mark.parametrize("seed", [1, 7])
def test_delta_invariant_certified(k, seed):
    r, mu, sigma, c, q = _instance(k, seed)
    res = sskp_mh_chain(r, mu, sigma, c, q, beta=4.0,
                        t_burn=2000, t_sample=4000, thin=2, seed=seed)
    results, certified = certify_sskp_chain(res, r, mu, sigma, c, q)
    assert certified, "\n".join(str(x) for x in results)


@pytest.mark.skipif(not _HAVE_NUMBA, reason="numba not installed")
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_numba_matches_python_bitforbit(seed):
    r, mu, sigma, c, q = _instance(30, seed)
    kw = dict(beta=5.0, t_burn=1500, t_sample=3000, thin=2, seed=seed)
    rp = sskp_mh_chain(r, mu, sigma, c, q, backend="python", **kw)
    rn = sskp_mh_chain(r, mu, sigma, c, q, backend="numba", **kw)
    assert np.array_equal(rp.samples_F, rn.samples_F), (
        f"max|diff|={np.abs(rp.samples_F - rn.samples_F).max():.3e}")
    assert np.array_equal(rp.final_x, rn.final_x)
    assert rp.curF == rn.curF and rp.acc_rate == rn.acc_rate


def test_chain_matches_boltzmann_expectation():
    """z_upper = mean(F over chain) must match the exact E_{pi_beta}[F]."""
    k = 10
    r, mu, sigma, c, q = _instance(k, seed=3)
    beta = 3.0

    # Exact Boltzmann expectation by enumerating all 2^k selections.
    Fs = np.array([sskp_objective(np.array(bits), r, mu, sigma, c, q)[0]
                   for bits in itertools.product((0, 1), repeat=k)])
    w = np.exp(-beta * (Fs - Fs.min()))
    p = w / w.sum()
    exact_EF = float((p * Fs).sum())

    res = sskp_mh_chain(r, mu, sigma, c, q, beta=beta,
                        t_burn=20000, t_sample=200000, thin=1, seed=11)
    rel = abs(res.z_upper - exact_EF) / max(abs(exact_EF), 1e-9)
    assert rel < 0.02, f"chain E[F]={res.z_upper:.5f} vs exact {exact_EF:.5f} (rel={rel:.3e})"


def test_oracle_catches_corrupted_stats():
    r, mu, sigma, c, q = _instance(20, seed=5)
    res = sskp_mh_chain(r, mu, sigma, c, q, beta=4.0,
                        t_burn=1000, t_sample=2000, thin=1, seed=5)
    # Tamper with the maintained statistic: the delta invariant must catch it.
    res.muX += 1.0
    r_delta = delta_invariant(res, r, mu, sigma, c, q)
    assert not r_delta.passed
