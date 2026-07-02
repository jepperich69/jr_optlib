# -*- coding: utf-8 -*-
"""Migration + oracle tests for the costed transport rounders (mcf / approx / alias)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from jr_optlib.transport import (
    make_transport, sinkhorn_balanced,
    round_transport_min_cost_mcf, round_transport_min_cost_approx,
    round_transport_floor_residue_lp, round_transport_min_cost_lp,
)
from jr_optlib.oracles import certify_transport, summarize
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


def _feasible(X, a, b):
    return np.allclose(X.sum(axis=1), a, atol=1e-6) and np.allclose(X.sum(axis=0), b, atol=1e-6)


@pytest.mark.parametrize("seed", [1, 3, 5])
@pytest.mark.parametrize("shape", [(5, 5), (8, 6)])
def test_migration_mcf(seed, shape):
    """MCF rounder (LP fallback when no ortools graph): objective+feasibility vs old."""
    old = _load_old()
    m, n = shape
    inst = make_transport(m, n, N=400, seed=seed)
    X_new, _ = round_transport_min_cost_mcf(inst.supply, inst.demand, inst.C)
    X_old, _ = old.round_transport_min_cost_mcf(inst.supply, inst.demand, inst.C)
    assert _feasible(X_new, inst.supply, inst.demand)
    assert np.isclose((inst.C * X_new).sum(), (inst.C * X_old).sum(), rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize("seed", [2, 4])
def test_migration_approx(seed):
    old = _load_old()
    inst = make_transport(8, 6, N=400, seed=seed)
    Xtau, _ = sinkhorn_balanced(inst.supply.astype(float), inst.demand.astype(float), inst.C, tau=0.05)
    X_new, _, ok_new = round_transport_min_cost_approx(inst.supply, inst.demand, inst.C, Xtau=Xtau, topk=5, eta=0.0)
    X_old, _, ok_old = old.round_transport_min_cost_approx(inst.supply, inst.demand, inst.C, Xtau=Xtau, topk=5, eta=0.0)
    assert ok_new == ok_old
    assert _feasible(X_new, inst.supply, inst.demand)
    assert np.isclose((inst.C * X_new).sum(), (inst.C * X_old).sum(), rtol=1e-9, atol=1e-6)


def test_mcf_certified_optimal():
    inst = make_transport(7, 6, N=400, seed=3)
    a = inst.supply.astype(float); b = inst.demand.astype(float)
    X, _ = round_transport_min_cost_mcf(inst.supply, inst.demand, inst.C)
    results, certified = certify_transport(X, a, b, inst.C, require_optimal=True, tol=1e-6)
    assert certified, "\n".join(str(r) for r in results)
    assert summarize(results) is Verdict.CERTIFIED


def test_floor_residue_alias_equals_lp():
    inst = make_transport(6, 5, N=300, seed=7)
    X1, _ = round_transport_floor_residue_lp(inst.supply, inst.demand, inst.C)
    X2, _ = round_transport_min_cost_lp(inst.supply, inst.demand, inst.C)
    assert np.isclose((inst.C * X1).sum(), (inst.C * X2).sum(), rtol=1e-9, atol=1e-6)
