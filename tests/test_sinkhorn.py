# -*- coding: utf-8 -*-
"""Migration + oracle tests for jr_optlib.transport.sinkhorn_balanced.

The migration is verified the way the design mandates: a differential against
the *actual old copy* in the paper repo. If the library and the old code
produce identical output on the paper's own instances, the migration provably
preserves every result. The oracle tests then certify the output independently.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from jr_optlib.transport import sinkhorn_balanced, make_transport
from jr_optlib.oracles import certify_sinkhorn, summarize
from jr_optlib.oracles.core import Verdict

# Path to the original paper implementation being migrated from.
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


@pytest.mark.parametrize("seed", [1, 2, 7])
@pytest.mark.parametrize("shape", [(5, 5), (8, 6), (20, 12)])
@pytest.mark.parametrize("tau", [0.05, 0.1])
def test_migration_matches_old_copy(seed, shape, tau):
    """Library output must equal the old paper code, bit-for-bit, on its own instances."""
    old = _load_old()
    m, n = shape
    inst = make_transport(m, n, N=1000, seed=seed)
    a = inst.supply.astype(float); b = inst.demand.astype(float)

    X_new, _ = sinkhorn_balanced(a, b, inst.C, tau=tau)
    X_old, _ = old.sinkhorn_balanced(a, b, inst.C, tau=tau)

    # Same arithmetic, deterministic -> expect exact equality.
    assert np.array_equal(X_new, X_old), (
        f"migration changed the result: max|diff|={np.abs(X_new-X_old).max():.3e}"
    )


@pytest.mark.parametrize("seed", [1, 2, 7])
@pytest.mark.parametrize("shape", [(5, 5), (8, 6), (20, 12)])
def test_sinkhorn_certified(seed, shape):
    m, n = shape
    inst = make_transport(m, n, N=1000, seed=seed)
    a = inst.supply.astype(float); b = inst.demand.astype(float)
    X, _ = sinkhorn_balanced(a, b, inst.C, tau=0.05, iters=5000, tol=1e-12)
    results, certified = certify_sinkhorn(X, a, b, inst.C, tau=0.05, tol=1e-6)
    assert certified, "\n".join(str(r) for r in results)
    assert summarize(results) is Verdict.CERTIFIED


def test_oracle_catches_corrupted_plan():
    inst = make_transport(6, 6, N=800, seed=3)
    a = inst.supply.astype(float); b = inst.demand.astype(float)
    X, _ = sinkhorn_balanced(a, b, inst.C, tau=0.05, iters=5000, tol=1e-12)
    X_bad = X.copy()
    X_bad[0, 0] *= 1.5   # breaks the scaling form (no longer diag(u) K diag(v))
    results, certified = certify_sinkhorn(X_bad, a, b, inst.C, tau=0.05, tol=1e-6)
    assert not certified
    assert summarize(results) is Verdict.FAIL
