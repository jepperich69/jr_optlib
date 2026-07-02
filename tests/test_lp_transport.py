# -*- coding: utf-8 -*-
"""Migration + oracle tests for the LP-family transport functions.

Unlike sinkhorn (exact array equality), transport LPs are degenerate: two
solver runs can return different optimal plans of equal cost. So the migration
differential asserts equal OBJECTIVE and feasibility, which are unique and
certifiable -- not the raw plan X.

The exact solvers are certified against an INDEPENDENT scipy.linprog reference
(certify_transport), and the greedy heuristic is CHECKED (feasible + cost >=
optimum), never mislabelled FAIL for being non-optimal.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from jr_optlib.transport import (
    make_transport, solve_transport_opt, round_transport_min_cost_lp,
    round_transport_greedy_push, sinkhorn_balanced_uv,
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
@pytest.mark.parametrize("shape", [(5, 5), (8, 6), (12, 9)])
def test_migration_solve_transport_opt(seed, shape):
    """Exact solver migration: objective + feasibility preserved vs old copy."""
    old = _load_old()
    m, n = shape
    inst = make_transport(m, n, N=500, seed=seed)
    a = inst.supply.astype(float); b = inst.demand.astype(float)

    X_new, obj_new, _ = solve_transport_opt(a, b, inst.C)
    X_old, obj_old, _ = old.solve_transport_opt(a, b, inst.C)

    assert _feasible(X_new, a, b) and _feasible(X_old, a, b)
    assert np.isclose(obj_new, obj_old, rtol=1e-9, atol=1e-6), (obj_new, obj_old)


@pytest.mark.parametrize("seed", [2, 4, 6])
@pytest.mark.parametrize("shape", [(5, 5), (8, 6)])
def test_migration_greedy_push(seed, shape):
    """Greedy-push migration: objective + feasibility + ok flag preserved."""
    old = _load_old()
    m, n = shape
    inst = make_transport(m, n, N=400, seed=seed)
    a = inst.supply.astype(float); b = inst.demand.astype(float)
    _, u, v, _ = sinkhorn_balanced_uv(a, b, inst.C, tau=0.05)

    X_new, _, ok_new = round_transport_greedy_push(inst.supply, inst.demand, inst.C, u, v, tau=0.05)
    X_old, _, ok_old = old.round_transport_greedy_push(inst.supply, inst.demand, inst.C, u, v, tau=0.05)

    assert ok_new == ok_old
    assert _feasible(X_new, a, b) and _feasible(X_old, a, b)
    assert np.isclose((inst.C * X_new).sum(), (inst.C * X_old).sum(), rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize("seed", [1, 3, 5])
def test_exact_solvers_certified(seed):
    """solve_transport_opt and the LP rounder are CERTIFIED optimal vs scipy."""
    inst = make_transport(7, 6, N=500, seed=seed)
    a = inst.supply.astype(float); b = inst.demand.astype(float)

    X, _, _ = solve_transport_opt(a, b, inst.C)
    results, certified = certify_transport(X, a, b, inst.C, require_optimal=True, tol=1e-6)
    assert certified, "\n".join(str(r) for r in results)
    assert summarize(results) is Verdict.CERTIFIED

    Xint, _ = round_transport_min_cost_lp(inst.supply, inst.demand, inst.C)
    results2, certified2 = certify_transport(Xint, a, b, inst.C, require_optimal=True, tol=1e-6)
    assert certified2, "\n".join(str(r) for r in results2)


def test_greedy_is_checked_not_failed():
    """Greedy heuristic: feasible + cost >= optimum -> CHECKED (not FAIL)."""
    inst = make_transport(8, 6, N=400, seed=2)
    a = inst.supply.astype(float); b = inst.demand.astype(float)
    _, u, v, _ = sinkhorn_balanced_uv(a, b, inst.C, tau=0.05)
    X, _, ok = round_transport_greedy_push(inst.supply, inst.demand, inst.C, u, v, tau=0.05)
    assert ok
    results, certified = certify_transport(X, a, b, inst.C, require_optimal=False, tol=1e-6)
    assert not certified                       # heuristic is not claimed optimal
    assert summarize(results) is Verdict.CHECKED


def test_corrupted_transport_fails():
    inst = make_transport(6, 6, N=300, seed=4)
    a = inst.supply.astype(float); b = inst.demand.astype(float)
    X, _, _ = solve_transport_opt(a, b, inst.C)
    X_bad = X.copy()
    X_bad[0, 0] += 3.0   # breaks marginals
    results, certified = certify_transport(X_bad, a, b, inst.C, require_optimal=True, tol=1e-6)
    assert not certified
    assert summarize(results) is Verdict.FAIL
