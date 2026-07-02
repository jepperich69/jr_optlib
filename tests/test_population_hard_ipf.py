# -*- coding: utf-8 -*-
"""Migration tests for Pub_PopInt_PartB HardIPF."""

import contextlib
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from jr_optlib.oracles import Verdict, certify_population_margins, summarize
from jr_optlib.population import ConstraintSpec, HardIPF, swap_repair_zone


JR = Path(r"C:\Users\rich\OneDrive - Danmarks Tekniske Universitet\JR")
OLD_PATH = (
    JR
    / "Publikationer"
    / "Pub_PopInt_PartB"
    / "Large Scale"
    / "Intege_Paper_Minimal_SWAP_ARC.py"
)


def load_old_module():
    spec = importlib.util.spec_from_file_location("old_popint_large", OLD_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def quiet_call(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def toy_problem():
    rows = []
    for age in [1, 2]:
        for gender in [1, 2]:
            for income in [1, 2, 3]:
                rows.append(
                    {
                        "ZoneID": 10,
                        "AgeID": age,
                        "GenderID": gender,
                        "IncomeID": income,
                        "x": 1.0 + 0.1 * age + 0.2 * gender + 0.05 * income,
                    }
                )
    seed = pd.DataFrame(rows)

    age_gender = (
        seed.groupby(["AgeID", "GenderID"], as_index=False, observed=True)["x"].sum()
    )
    age_gender["Val"] = [7.0, 8.0, 6.0, 9.0]
    age_gender = age_gender[["AgeID", "GenderID", "Val"]]

    age_income = (
        seed.groupby(["AgeID", "IncomeID"], as_index=False, observed=True)["x"].sum()
    )
    age_tot = age_gender.groupby("AgeID", observed=True)["Val"].sum()
    for age in [1, 2]:
        mask = age_income["AgeID"] == age
        age_income.loc[mask, "Val"] = (
            age_income.loc[mask, "x"] / age_income.loc[mask, "x"].sum() * age_tot.loc[age]
        )
    age_income = age_income[["AgeID", "IncomeID", "Val"]]

    constraints = [
        ConstraintSpec("AgeGender", ("AgeID", "GenderID"), age_gender),
        ConstraintSpec("AgeIncome", ("AgeID", "IncomeID"), age_income),
    ]
    return seed, constraints


def toy_swap_problem(anchor_name="AgeGender"):
    rows = [
        {
            "ZoneID": 10,
            "AgeID": 1,
            "NumChildID": 1,
            "FamID": 1,
            "GenderID": 1,
            "IncomeID": 1,
            "LmaID": 1,
            "n": 1,
            "x": 0.2,
        },
        {
            "ZoneID": 10,
            "AgeID": 1,
            "NumChildID": 1,
            "FamID": 1,
            "GenderID": 1,
            "IncomeID": 2,
            "LmaID": 1,
            "n": 0,
            "x": 0.8,
        },
        {
            "ZoneID": 10,
            "AgeID": 1,
            "NumChildID": 1,
            "FamID": 1,
            "GenderID": 2,
            "IncomeID": 1,
            "LmaID": 1,
            "n": 0,
            "x": 0.0,
        },
        {
            "ZoneID": 10,
            "AgeID": 1,
            "NumChildID": 1,
            "FamID": 1,
            "GenderID": 2,
            "IncomeID": 2,
            "LmaID": 1,
            "n": 1,
            "x": 1.0,
        },
    ]
    both = pd.DataFrame(rows)
    int_df = both.drop(columns=["x"]).copy()
    frac_df = both.drop(columns=["n"]).copy()
    constraints = [
        ConstraintSpec(
            anchor_name,
            ("AgeID", "GenderID"),
            pd.DataFrame(
                [
                    {"AgeID": 1, "GenderID": 1, "Val": 1},
                    {"AgeID": 1, "GenderID": 2, "Val": 1},
                ]
            ),
        ),
        ConstraintSpec(
            "AgeIncome",
            ("AgeID", "IncomeID"),
            pd.DataFrame(
                [
                    {"AgeID": 1, "IncomeID": 1, "Val": 0},
                    {"AgeID": 1, "IncomeID": 2, "Val": 2},
                ]
            ),
        ),
    ]
    return int_df, frac_df, constraints


def test_hard_ipf_matches_old_copy():
    old = load_old_module()
    seed, constraints = toy_problem()
    old_constraints = [
        old.ConstraintSpec(c.name, c.dims, c.df.copy(), c.val_col)
        for c in constraints
    ]
    old_w, old_ok, old_it, old_err = quiet_call(
        old.HardIPF(seed, old_constraints, weight_col="x").fit,
        tol=1e-10,
        max_iters=100,
        verbose=False,
    )
    new_w, new_ok, new_it, new_err = HardIPF(seed, constraints, weight_col="x").fit(
        tol=1e-10,
        max_iters=100,
        verbose=False,
    )
    assert np.array_equal(new_w, old_w)
    assert new_ok == old_ok
    assert new_it == old_it
    assert new_err == old_err


def test_population_margin_oracle_certifies_hard_ipf_and_catches_error():
    seed, constraints = toy_problem()
    w, ok, *_ = HardIPF(seed, constraints, weight_col="x").fit(
        tol=1e-10,
        max_iters=100,
        verbose=False,
    )
    assert ok
    fitted = seed.copy()
    fitted["x"] = w
    results, certified = certify_population_margins(fitted, constraints, weight_col="x", tol=1e-6)
    assert certified
    assert summarize(results) is Verdict.CERTIFIED

    bad = fitted.copy()
    bad.loc[0, "x"] += 1.0
    bad_results, bad_certified = certify_population_margins(bad, constraints, weight_col="x", tol=1e-6)
    assert not bad_certified
    assert summarize(bad_results) is Verdict.FAIL


def test_swap_repair_zone_matches_old_copy_and_preserves_margins():
    old = load_old_module()
    int_df, frac_df, constraints = toy_swap_problem(anchor_name=old.ANCHOR_NAME)
    old_constraints = [
        old.ConstraintSpec(c.name, c.dims, c.df.copy(), c.val_col)
        for c in constraints
    ]

    old_out, old_stats = old.swap_repair_zone(
        int_df.copy(), frac_df.copy(), old_constraints, max_passes=3, max_moves_per_slice=10
    )
    new_out, new_stats = swap_repair_zone(
        int_df.copy(),
        frac_df.copy(),
        constraints,
        max_passes=3,
        max_moves_per_slice=10,
        anchor_name=old.ANCHOR_NAME,
    )

    pd.testing.assert_frame_equal(
        new_out.reset_index(drop=True),
        old_out.reset_index(drop=True),
        check_dtype=False,
    )
    assert new_stats == old_stats == {"moves_total": 1.0}

    results, certified = certify_population_margins(new_out, constraints, weight_col="n", tol=0.0)
    assert certified
    assert summarize(results) is Verdict.CERTIFIED
