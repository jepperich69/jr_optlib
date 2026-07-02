# -*- coding: utf-8 -*-
"""Migration + oracle tests for the population-synthesis residual rounders.

These complete the ipf_2d -> integer table pipeline. The certificate is the
rounding contract: exact marginals, integrality, and per-cell deviation < 1.
Migration is checked against the old private `_dependent_round_2d_*` helpers on
the contract + achieved objective (the residual LP can have tied optima, so the
plan is not unique; the contract and objective are).
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from jr_optlib.transport import (
    ipf_2d, make_contingency2d,
    dependent_round_2d, dependent_round_2d_lp,
    dependent_round_2d_lp_sparsified,
)
from jr_optlib.oracles import certify_dependent_round, summarize
from jr_optlib.oracles.core import Verdict

OLD_FILE = Path(
    r"C:\Users\rich\OneDrive - Danmarks Tekniske Universitet\JR"
    r"\Publikationer\Pub_MIPEntropy_MPC\code\mip_hybrid\apps\population_transport.py"
)


def _load_old():
    if not OLD_FILE.exists():
        pytest.skip(f"old copy not present: {OLD_FILE}")
    spec = importlib.util.spec_from_file_location("old_population_transport", OLD_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _residual_objective(X, Xint):
    F = np.floor(X)
    frac = X - F
    Z = Xint - F
    return float(((1.0 - frac) * Z).sum())


def _fractional_table(R, C, N, seed):
    inst = make_contingency2d(R, C, N, seed=seed)
    X, _ = ipf_2d(inst.row_marg, inst.col_marg, tol=1e-12)
    return X, inst.row_marg, inst.col_marg


@pytest.mark.parametrize("seed", [1, 4, 9])
@pytest.mark.parametrize("shape", [(6, 6), (10, 8), (15, 12)])
def test_dependent_round_lp_certified(seed, shape):
    R, C = shape
    X, rm, cm = _fractional_table(R, C, 2000, seed)
    Xint, _ = dependent_round_2d_lp(X, rm, cm)
    results, certified = certify_dependent_round(Xint, X, rm, cm)
    assert certified, "\n".join(str(r) for r in results)
    assert summarize(results) is Verdict.CERTIFIED


@pytest.mark.parametrize("seed", [2, 5])
def test_dependent_round_sparsified_certified(seed):
    X, rm, cm = _fractional_table(12, 10, 3000, seed)
    Xint, _ = dependent_round_2d_lp_sparsified(X, rm, cm)
    results, certified = certify_dependent_round(Xint, X, rm, cm)
    assert certified, "\n".join(str(r) for r in results)


@pytest.mark.parametrize("method", ["mcf", "lp"])
def test_dispatcher_certified(method):
    X, rm, cm = _fractional_table(10, 8, 2500, seed=3)
    Xint, _ = dependent_round_2d(X, rm, cm, method=method, sparsify=(method == "lp"))
    results, certified = certify_dependent_round(Xint, X, rm, cm)
    assert certified, "\n".join(str(r) for r in results)


@pytest.mark.parametrize("seed", [1, 4, 9])
def test_migration_dependent_round_lp(seed):
    """Contract holds for both old and new, and the residual objective matches."""
    old = _load_old()
    X, rm, cm = _fractional_table(10, 8, 2000, seed)
    Xint_new, _ = dependent_round_2d_lp(X, rm, cm)
    Xint_old, _ = old._dependent_round_2d_lp(X, rm, cm)

    _, cert_new = certify_dependent_round(Xint_new, X, rm, cm)
    _, cert_old = certify_dependent_round(Xint_old, X, rm, cm)
    assert cert_new and cert_old
    assert np.isclose(_residual_objective(X, Xint_new),
                      _residual_objective(X, Xint_old), rtol=1e-9, atol=1e-6)


def test_oracle_catches_bad_rounding():
    X, rm, cm = _fractional_table(6, 6, 1000, seed=1)
    Xint, _ = dependent_round_2d_lp(X, rm, cm)
    bad = Xint.copy()
    bad[0, 0] += 2   # breaks both marginals and the <1 deviation contract
    results, certified = certify_dependent_round(bad, X, rm, cm)
    assert not certified
    assert summarize(results) is Verdict.FAIL
