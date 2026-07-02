# -*- coding: utf-8 -*-
"""Migration tests for Pub_QP_SAA_MC entropic-risk assignment solvers.

Differential vs the live paper copy (bit-for-bit where deterministic), plus
oracle checks (formula-vs-MC, feasibility/recompute, brute-force optimum).
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

from jr_optlib.optimization import (
    build_equicorrelation_cov,
    rho_eta_formula,
    rho_eta_mc,
    solve_hungarian_qp,
    solve_hungarian_rn,
    solve_qp,
)
from jr_optlib.oracles import (
    Verdict,
    certify_entropic_assignment,
    certify_entropic_risk_mc,
    summarize,
)

JR = Path(r"C:\Users\rich\OneDrive - Danmarks Tekniske Universitet\JR")
QP_CODE = JR / "Publikationer" / "Pub_QP_SAA_MC" / "code"


def load_old():
    sys.path.insert(0, str(QP_CODE))
    old_data = importlib.import_module("data")
    old_solvers = importlib.import_module("solvers")
    return old_solvers, old_data


def diag_perm_x(n, perm):
    x = np.zeros((n, n))
    for i, j in enumerate(perm):
        x[i, j] = 1.0
    return x.flatten()


# ---------------------------------------------------------------------------
# Differential vs the live paper copy
# ---------------------------------------------------------------------------

def test_build_cov_matches_old_copy():
    old_solvers, old_data = load_old()
    for rho in [0.0, 0.3, -0.1]:
        new_cov = build_equicorrelation_cov(old_data.SIGMA2, rho)
        old_cov = old_data.build_Sigma_c(rho)
        assert np.array_equal(new_cov, old_cov)


def test_rho_eta_formula_and_mc_match_old_copy():
    old_solvers, old_data = load_old()
    C_BAR = old_data.C_BAR
    Sigma_c = old_data.build_Sigma_c(0.3)
    x = diag_perm_x(old_data.n, (0, 2, 1))
    eta = 0.5

    assert rho_eta_formula(x, C_BAR, Sigma_c, eta) == old_solvers.rho_eta_formula(x, Sigma_c, eta)
    # Same seed -> identical Gaussian draws -> identical estimate.
    new_mc = rho_eta_mc(x, C_BAR, Sigma_c, eta, S=5000, seed=7)
    old_mc = old_solvers.rho_eta_mc(x, Sigma_c, eta, S=5000, seed=7)
    assert new_mc == old_mc


def test_solve_qp_matches_old_copy():
    old_solvers, old_data = load_old()
    Sigma_c = old_data.build_Sigma_c(0.3)
    eta = 0.5
    new = solve_qp(old_data.C_BAR, Sigma_c, eta)
    old = old_solvers.solve_qp(Sigma_c, eta)
    assert np.allclose(new["x_opt"], old["x_opt"], atol=1e-8)
    assert abs(new["obj"] - old["obj"]) < 1e-9
    assert new["success"] == old["success"]


def test_hungarian_qp_and_rn_match_old_copy():
    old_solvers, old_data = load_old()
    C_BAR, SIGMA2 = old_data.C_BAR, old_data.SIGMA2
    eta = 0.5

    new_qp = solve_hungarian_qp(C_BAR, SIGMA2, eta)
    old_qp = old_solvers.solve_hungarian_qp(C_BAR, SIGMA2, eta)
    assert np.array_equal(new_qp["x_opt"], old_qp["x_opt"])
    assert abs(new_qp["obj"] - old_qp["obj"]) < 1e-12
    assert np.array_equal(new_qp["pi"], old_qp["pi"])

    new_rn = solve_hungarian_rn(C_BAR)
    old_rn = old_solvers.solve_hungarian_rn(C_BAR)
    assert np.array_equal(new_rn["x_opt"], old_rn["x_opt"])
    assert new_rn["obj"] == old_rn["obj"]


# ---------------------------------------------------------------------------
# Oracles
# ---------------------------------------------------------------------------

def test_entropic_risk_mc_oracle_converges_and_catches_error():
    _, old_data = load_old()
    # Mild regime (small eta) so the log-MGF estimator converges tightly.
    Sigma_c = old_data.build_Sigma_c(0.2)
    x = diag_perm_x(old_data.n, (0, 1, 2))
    eta = 0.1
    good = certify_entropic_risk_mc(x, old_data.C_BAR, Sigma_c, eta, S=300_000, seed=1)
    assert good.passed
    # A wrong *claimed* risk (independent of the sampler) must be caught.
    exact = rho_eta_formula(x, old_data.C_BAR, Sigma_c, eta)
    bad = certify_entropic_risk_mc(x, old_data.C_BAR, Sigma_c, eta,
                                   claimed=exact + 5.0, S=300_000, seed=1)
    assert not bad.passed


def test_entropic_assignment_oracle_certifies_hungarian_and_bounds_qp():
    _, old_data = load_old()
    C_BAR, SIGMA2 = old_data.C_BAR, old_data.SIGMA2
    eta = 0.5

    # Independent (diagonal) case: Hungarian QP is the exact binary optimum.
    Sigma_diag = build_equicorrelation_cov(SIGMA2, 0.0)
    hres = solve_hungarian_qp(C_BAR, SIGMA2, eta)
    results, ok = certify_entropic_assignment(hres, C_BAR, Sigma_diag, eta, binary=True)
    assert ok
    assert summarize(results) is Verdict.CERTIFIED

    # Continuous relaxation must not beat the best permutation and must recompute.
    Sigma_c = build_equicorrelation_cov(SIGMA2, 0.3)
    qres = solve_qp(C_BAR, Sigma_c, eta)
    cresults, cok = certify_entropic_assignment(qres, C_BAR, Sigma_c, eta, binary=False)
    assert cok

    # A tampered objective must be caught by the recompute oracle.
    bad = dict(hres)
    bad["obj"] = hres["obj"] + 1.0
    bad_results, bad_ok = certify_entropic_assignment(bad, C_BAR, Sigma_diag, eta, binary=True)
    assert not bad_ok
    assert summarize(bad_results) is Verdict.FAIL


# ---------------------------------------------------------------------------
# Gurobi MIQP (skipped if no license)
# ---------------------------------------------------------------------------

def test_miqp_matches_hungarian_when_available():
    try:
        import gurobipy  # noqa: F401
        from jr_optlib.optimization import solve_miqp_gurobi
        _, old_data = load_old()
        C_BAR, SIGMA2 = old_data.C_BAR, old_data.SIGMA2
        eta = 0.5
        Sigma_diag = build_equicorrelation_cov(SIGMA2, 0.0)
        miqp = solve_miqp_gurobi(C_BAR, Sigma_diag, eta, time_limit=30)
    except Exception as exc:  # no gurobi / expired license
        pytest.skip(f"Gurobi unavailable: {exc!r}")

    hung = solve_hungarian_qp(C_BAR, SIGMA2, eta)
    # Diagonal covariance: MIQP binary optimum == Hungarian modified-cost optimum.
    assert abs(miqp["obj"] - hung["obj"]) < 1e-6
