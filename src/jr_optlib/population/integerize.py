# -*- coding: utf-8 -*-
"""Integerization repair primitives for population synthesis."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_KEY_COLS = ("AgeID", "NumChildID", "FamID", "GenderID", "IncomeID", "LmaID")
DEFAULT_ANCHOR_NAME = "Age\u00d7Gender"
SWAP_MAX_PASSES = 3
SWAP_MAX_MOVES_PER_SLICE = 2000
GHOST_X_EPS = 1e-6


def step1_split(frac_df_zone, zone_target: int):
    """Deterministically split fractional counts into floors and additions needed."""
    df = frac_df_zone.copy()
    df["n"] = np.floor(df["x"] + 1e-9).astype(int)
    df["frac"] = df["x"] - df["n"]
    floor_sum = int(df["n"].sum())
    k_add = int(zone_target - floor_sum)
    if k_add < 0:
        take = -k_add
        idx = df["frac"].nsmallest(take).index
        df.loc[idx, "n"] = (df.loc[idx, "n"] - 1).astype(int)
        k_add = 0
    return df, k_add


def pps_without_replacement(idx_array: np.ndarray, weights: np.ndarray, k: int, rng: np.random.Generator):
    """Probability-proportional-to-size sample without replacement."""
    w = weights.clip(min=1e-12)
    p = w / w.sum()
    k = int(min(k, len(idx_array)))
    if k <= 0:
        return np.array([], dtype=idx_array.dtype)
    return rng.choice(idx_array, size=k, replace=False, p=p)


def step2_anchor_pps(
    df_split,
    constraints,
    anchor_dims,
    k_add: int,
    seed: int = 42,
    anchor_name: str = DEFAULT_ANCHOR_NAME,
):
    """Apply anchor-conditioned PPS additions after deterministic floor split.

    Numerics-preserving extraction of ``step2_anchor_pps`` from
    ``Pub_PopInt_PartB/Large Scale/Intege_Paper_Minimal_SWAP_ARC.py``.
    ``anchor_dims`` and ``k_add`` are retained for signature compatibility with
    the paper copy; the original implementation uses the AgeID/GenderID anchor.
    """
    rng = np.random.default_rng(seed)
    df = df_split.copy()
    df["z"] = 0

    anchor = next(cz for cz in constraints if cz.name == anchor_name)
    dims = ["AgeID", "GenderID"]
    tgt = anchor.df.groupby(dims, observed=True, sort=False)[anchor.val_col].sum().rename("tgt")
    flr = df.groupby(dims, observed=True, sort=False)["n"].sum().rename("floor")
    need = (tgt - flr).reindex(tgt.index, fill_value=0).clip(lower=0).astype(int)

    chosen_records = []

    for key, need_units in need.items():
        if int(need_units) <= 0:
            continue

        mask = np.ones(len(df), dtype=bool)
        for d, v in zip(dims, key if isinstance(key, tuple) else (key,)):
            mask &= df[d].to_numpy() == v
        idx = df.index[mask].to_numpy()
        if len(idx) == 0:
            continue
        weights = df.loc[idx, "frac"].to_numpy()
        if not np.any(weights > 0):
            weights = np.ones_like(weights) * 1e-12
        chosen = pps_without_replacement(idx, weights, int(need_units), rng)
        df.loc[chosen, "z"] = 1
        for j in chosen:
            rec = {"row_idx": int(j)}
            for d, v in zip(dims, key if isinstance(key, tuple) else (key,)):
                rec[d] = int(v)
            rec["frac"] = float(df.at[j, "frac"])
            chosen_records.append(rec)

    df["n"] = (df["n"] + df["z"]).astype(int)
    return df, pd.DataFrame(chosen_records)


def compute_zone_residuals(int_df_zone, constraints, anchor_name: str = DEFAULT_ANCHOR_NAME):
    """Return target-current residuals for all non-anchor constraints in one zone."""
    resid = {}
    for cs in constraints:
        if cs.name == anchor_name:
            continue
        dims = list(cs.dims)
        tgt = cs.df.groupby(dims, observed=True, sort=False)[cs.val_col].sum()
        cur = int_df_zone.groupby(dims, observed=True, sort=False)["n"].sum()
        cur = cur.reindex(tgt.index, fill_value=0)
        resid[cs.name] = (tgt - cur).astype(float)
    return resid


def swap_repair_zone(
    int_df_zone,
    frac_df_zone,
    constraints,
    max_passes: int = SWAP_MAX_PASSES,
    max_moves_per_slice: int = SWAP_MAX_MOVES_PER_SLICE,
    ghost_x_eps: float = GHOST_X_EPS,
    anchor_name: str = DEFAULT_ANCHOR_NAME,
    key_cols: Sequence[str] = DEFAULT_KEY_COLS,
):
    """Repair a zone's integer table by anchor-preserving two-row swaps.

    Numerics-preserving extraction of ``swap_repair_zone`` from
    ``Pub_PopInt_PartB/Large Scale/Intege_Paper_Minimal_SWAP_ARC.py``.
    """
    key_cols = list(key_cols)

    merged = int_df_zone.merge(frac_df_zone[key_cols + ["x"]], on=key_cols, how="left")
    merged["x"] = merged["x"].fillna(0.0)
    merged["delta"] = merged["x"] - merged["n"]
    merged["ghost_blocked"] = merged["x"] <= ghost_x_eps

    resid = compute_zone_residuals(merged, constraints, anchor_name=anchor_name)

    margin_dims = {}
    for cs in constraints:
        if cs.name == anchor_name:
            continue
        other = [d for d in cs.dims if d != "AgeID"][0]
        margin_dims[cs.name] = other

    moves_total = 0

    for _pass in range(int(max_passes)):
        moves_in_pass = 0

        for (age, gen), sl_idx in merged.groupby(["AgeID", "GenderID"], sort=False).groups.items():
            idx = np.asarray(list(sl_idx), dtype=merged.index.dtype)
            sl = merged.loc[idx].copy()

            sl["_orig"] = sl.index.to_numpy()
            sl.reset_index(drop=True, inplace=True)

            donors_mask = (sl["n"] > 0) & (sl["delta"] < -1e-9)
            receivers_mask = (sl["delta"] > 1e-9) & (~sl["ghost_blocked"])
            if not donors_mask.any() or not receivers_mask.any():
                continue

            per_dim_index = {}
            for mname, dim in margin_dims.items():
                gb = sl.groupby(dim, sort=False, observed=True)
                per_dim_index[(mname, "val_to_rows")] = {
                    k: np.asarray(v, dtype=np.int64) for k, v in gb.indices.items()
                }

            slice_moves = 0
            while slice_moves < int(max_moves_per_slice):
                improved = False

                for mname, dim in margin_dims.items():
                    r_series = resid[mname]
                    try:
                        r_age = r_series.xs(age, level=r_series.index.names.index("AgeID"))
                    except Exception:
                        continue

                    deficits = [v for v, r in r_age.items() if r > 0.5]
                    surpluses = [v for v, r in r_age.items() if r < -0.5]
                    if not deficits or not surpluses:
                        continue

                    val_to_rows = per_dim_index[(mname, "val_to_rows")]

                    for v_def in deficits:
                        recv_rows = val_to_rows.get(v_def, np.array([], dtype=np.int64))
                        if recv_rows.size == 0:
                            continue

                        recv_cand = sl.iloc[recv_rows]
                        recv_cand = recv_cand[
                            (recv_cand["delta"] > 1e-9) & (~recv_cand["ghost_blocked"])
                        ]
                        if recv_cand.empty:
                            continue

                        recv_row = int(recv_cand["delta"].idxmax())

                        donor_row = None
                        best_delta = 0.0
                        for v_sur in surpluses:
                            don_rows = val_to_rows.get(v_sur, np.array([], dtype=np.int64))
                            if don_rows.size == 0:
                                continue

                            don_cand = sl.iloc[don_rows]
                            don_cand = don_cand[(don_cand["delta"] < -1e-9) & (don_cand["n"] > 0)]
                            if don_cand.empty:
                                continue

                            cand_row = int(don_cand["delta"].idxmin())
                            cand_delta = float(don_cand["delta"].min())
                            if (donor_row is None) or (cand_delta < best_delta):
                                donor_row = cand_row
                                best_delta = cand_delta

                        if donor_row is None:
                            continue

                        donor_orig = int(sl.at[donor_row, "_orig"])
                        recv_orig = int(sl.at[recv_row, "_orig"])

                        merged.loc[donor_orig, "n"] -= 1
                        merged.loc[recv_orig, "n"] += 1
                        merged.loc[donor_orig, "delta"] += 1.0
                        merged.loc[recv_orig, "delta"] -= 1.0

                        sl.loc[donor_row, "n"] = merged.at[donor_orig, "n"]
                        sl.loc[recv_row, "n"] = merged.at[recv_orig, "n"]
                        sl.loc[donor_row, "delta"] = merged.at[donor_orig, "delta"]
                        sl.loc[recv_row, "delta"] = merged.at[recv_orig, "delta"]

                        resid[mname].loc[(age, sl.at[recv_row, dim])] -= 1.0
                        resid[mname].loc[(age, sl.at[donor_row, dim])] += 1.0

                        slice_moves += 1
                        moves_total += 1
                        improved = True

                        if slice_moves >= int(max_moves_per_slice):
                            break

                if not improved:
                    break

            moves_in_pass += slice_moves

        if moves_in_pass == 0:
            break

    out_cols = ["ZoneID", *key_cols, "n"]
    return merged[out_cols].copy(), {"moves_total": float(moves_total)}
