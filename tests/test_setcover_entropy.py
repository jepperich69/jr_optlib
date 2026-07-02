# -*- coding: utf-8 -*-
"""Migration tests for the entropy-guided set-cover core."""

import contextlib
import importlib.util
import io
from pathlib import Path

import numpy as np

from jr_optlib.oracles import Verdict, certify_setcover_solution, summarize
from jr_optlib.setcover import (
    build_instance_from_matrix,
    compute_reduced_costs,
    gen_entropy_friendly_scp,
    polish_solution,
    round_cover_dual_guided,
    solve_entropy_setcover,
    solve_mip,
)


JR = Path(r"C:\Users\rich\OneDrive - Danmarks Tekniske Universitet\JR")
OLD_PATH = (
    JR
    / "Publikationer"
    / "Pub_MIPEntropy_MPC"
    / "code"
    / "mip_hybrid"
    / "apps"
    / "synth_setcover.py"
)


def load_old_module():
    spec = importlib.util.spec_from_file_location("old_synth_setcover", OLD_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def quiet_call(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def test_generator_matches_old_copy():
    old = load_old_module()
    for seed in [1, 7, 42]:
        A_old, c_old = old._gen_entropy_friendly_scp(25, 80, seed=seed)
        A_new, c_new = gen_entropy_friendly_scp(25, 80, seed=seed)
        assert np.array_equal(A_new, A_old)
        assert np.array_equal(c_new, c_old)


def test_reduced_costs_and_rounding_match_old_copy():
    old = load_old_module()
    A, c = gen_entropy_friendly_scp(30, 100, seed=3)
    old_inst = old._build_instance_from_matrix(A, c)
    new_inst = build_instance_from_matrix(A, c)

    x_old, y_old, *_ = quiet_call(old.ipf_rowwise_entropy, old_inst, 0.1, 20, 1e-3)
    x_new, y_new, *_ = quiet_call(
        __import__("jr_optlib.setcover.entropy", fromlist=["ipf_rowwise_entropy"]).ipf_rowwise_entropy,
        new_inst,
        0.1,
        20,
        1e-3,
    )
    assert np.array_equal(x_new, x_old)
    assert np.array_equal(y_new, y_old)
    assert np.array_equal(compute_reduced_costs(new_inst, y_new), old.compute_reduced_costs(old_inst, y_old))

    r_old = quiet_call(old.round_cover_dual_guided, old_inst, x_old, y_old)
    r_new = round_cover_dual_guided(new_inst, x_new, y_new)
    assert np.array_equal(r_new[0], r_old[0])
    assert r_new[1:] == r_old[1:]


def test_solve_entropy_setcover_matches_old_copy_without_polish():
    old = load_old_module()
    for seed in [2, 5, 11]:
        A, c = gen_entropy_friendly_scp(40, 120, seed=seed)
        old_x, old_obj, old_feas, _ = quiet_call(
            old.solve_entropy_setcover,
            A,
            c,
            tau=0.1,
            iters=30,
            tol=1e-3,
            tau_schedule="0.5,0.2,0.1",
            polish_time=0.0,
            do_polish_mh=False,
        )
        new_x, new_obj, new_feas, _ = solve_entropy_setcover(
            A,
            c,
            tau=0.1,
            iters=30,
            tol=1e-3,
            tau_schedule="0.5,0.2,0.1",
            polish_time=0.0,
            do_polish_mh=False,
        )
        assert np.array_equal(new_x, old_x)
        assert new_obj == old_obj
        assert new_feas == old_feas


def test_polish_solution_matches_old_copy_with_gurobi():
    old = load_old_module()
    A, c = gen_entropy_friendly_scp(30, 100, seed=12)
    inst = build_instance_from_matrix(A, c)
    x_frac, y, *_ = quiet_call(
        __import__("jr_optlib.setcover.entropy", fromlist=["ipf_rowwise_entropy"]).ipf_rowwise_entropy,
        inst,
        0.1,
        25,
        1e-3,
    )
    x0, *_ = round_cover_dual_guided(inst, x_frac, y)

    old_x, old_obj, _ = quiet_call(old.polish_solution, x0, A, c, 1.0, 0.3, x_frac)
    new_x, new_obj, _ = polish_solution(x0, A, c, 1.0, 0.3, x_frac)
    assert np.array_equal(new_x, old_x)
    assert new_obj == old_obj


def test_solve_entropy_setcover_matches_old_copy_with_gurobi_polish():
    old = load_old_module()
    A, c = gen_entropy_friendly_scp(30, 100, seed=13)
    old_x, old_obj, old_feas, _ = quiet_call(
        old.solve_entropy_setcover,
        A,
        c,
        tau=0.1,
        iters=25,
        tol=1e-3,
        tau_schedule="0.5,0.2,0.1",
        polish_time=1.0,
        polish_pool=0.3,
        do_polish_mh=False,
    )
    new_x, new_obj, new_feas, _ = solve_entropy_setcover(
        A,
        c,
        tau=0.1,
        iters=25,
        tol=1e-3,
        tau_schedule="0.5,0.2,0.1",
        polish_time=1.0,
        polish_pool=0.3,
        do_polish_mh=False,
    )
    assert np.array_equal(new_x, old_x)
    assert new_obj == old_obj
    assert new_feas == old_feas


def test_solve_mip_matches_old_copy_with_gurobi():
    old = load_old_module()
    A, c = gen_entropy_friendly_scp(20, 60, seed=16)
    old_r = quiet_call(old.solve_mip, A, c, timelimit_s=5, gurobi_time_limit=5)
    new_r = solve_mip(A, c, timelimit_s=5, gurobi_time_limit=5)
    assert np.isclose(new_r["obj"], old_r["obj"])
    assert np.isclose(new_r["bound"], old_r["bound"])
    assert np.isclose(new_r["gap"], old_r["gap"])


def test_setcover_oracle_checks_solution_and_catches_corruption():
    A, c = gen_entropy_friendly_scp(35, 100, seed=8)
    x, obj, feasible, _ = solve_entropy_setcover(A, c, polish_time=0.0)
    assert feasible

    results, recomputed = certify_setcover_solution(x, A, c)
    assert np.isclose(recomputed, obj)
    assert summarize(results) is Verdict.CHECKED

    x_bad = x.copy()
    row = int(np.argmax(A @ x))
    selected_covering = [j for j in np.where(A[row, :] != 0)[0] if x_bad[j] == 1]
    for j in selected_covering:
        x_bad[j] = 0
    bad_results, _ = certify_setcover_solution(x_bad, A, c)
    assert summarize(bad_results) is Verdict.FAIL
