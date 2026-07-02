# -*- coding: utf-8 -*-
"""Pandas-backed high-dimensional IPF used by Pub_PopInt_PartB.

Numerics-preserving extraction of ``HardIPF`` from
``Large Scale/Intege_Paper_Minimal_SWAP_ARC.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

EPS = 1e-18


@dataclass
class ConstraintSpec:
    name: str
    dims: Tuple[str, ...]
    df: object
    val_col: str = "Val"


class HardIPF:
    def __init__(self, seed_df, constraints, weight_col="x"):
        keep_cols = [c for c in seed_df.columns if c not in {"w", "n", "frac"}]
        self.df = seed_df[keep_cols].copy()
        self.weight = self.df[weight_col].to_numpy(dtype=float)
        self.cols = [c for c in self.df.columns if c != weight_col]
        self.codes = {}
        self.levels = {}
        for c in self.cols:
            cats, inv = np.unique(self.df[c].to_numpy(), return_inverse=True)
            self.codes[c] = inv.astype(np.int64)
            self.levels[c] = cats
        self.margins = []
        self.names = []
        for cs in constraints:
            dims = list(cs.dims)
            self.names.append(cs.name)
            shape = tuple(len(self.levels[d]) for d in dims)
            tdf = cs.df[dims + [cs.val_col]].copy()
            ok = np.ones(len(tdf), dtype=bool)
            maps = {}
            for d in dims:
                lvl = self.levels[d]
                mp = {v: i for i, v in enumerate(lvl)}
                maps[d] = mp
                ok &= tdf[d].map(mp).notna().to_numpy()
            if not ok.all():
                tdf = tdf.loc[ok].copy()
            idx = np.ravel_multi_index(
                tuple(tdf[d].map(maps[d]).to_numpy(int) for d in dims),
                dims=shape,
            )
            tgt = np.zeros(np.prod(shape), dtype=float)
            np.add.at(tgt, idx, tdf[cs.val_col].to_numpy(dtype=float))
            tgt = tgt.reshape(shape)
            row_flat = np.ravel_multi_index(tuple(self.codes[d] for d in dims), dims=shape)
            self.margins.append((dims, shape, row_flat, tgt))

    @staticmethod
    def _rel_err_safe(tgt: np.ndarray, cur: np.ndarray) -> float:
        denom = np.maximum(np.maximum(np.abs(tgt), np.abs(cur)), EPS)
        return float(np.max(np.abs(tgt - cur) / denom)) if denom.size else 0.0

    def fit(self, tol=1e-7, max_iters=200, verbose=True):
        w = self.weight.copy()
        for it in range(1, max_iters + 1):
            max_rel = 0.0
            for (dims, shape, row_flat, tgt) in self.margins:
                cur = np.bincount(row_flat, weights=w, minlength=np.prod(shape)).reshape(shape)
                scale = np.ones_like(cur, dtype=float)
                mask = cur > 0
                scale[mask] = tgt[mask] / cur[mask]
                w *= scale.reshape(-1)[row_flat]
                rel = self._rel_err_safe(tgt, cur)
                max_rel = max(max_rel, rel)
            if verbose and (it <= 10 or it % 5 == 0 or max_rel < tol):
                print(f"[IPF] iter={it:3d} max_rel_err={max_rel:.3e}")
            if max_rel < tol:
                return w, True, it, float(max_rel)
        return w, False, max_iters, float(max_rel)
