# -*- coding: utf-8 -*-
"""
Costed transport integer rounders: min-cost-flow, restricted-support approx,
and support helpers.

VETTED FUNCTIONS -- registry ids: transport.round_transport_min_cost_mcf,
transport.round_transport_min_cost_approx, transport.round_transport_floor_residue_lp.

Migrated numerics-preserving from
  Pub_MIPEntropy_MPC/code/mip_hybrid/apps/population_transport.py.

MCF uses OR-Tools' graph solver when available and otherwise falls back to the
LP rounder; either way the result is an optimal integer transport plan
(TU => integral), certifiable by jr_optlib.oracles.certify_transport.
"""

import time
from typing import Optional, Tuple

import numpy as np

from jr_optlib.transport._backend import backend_name
from jr_optlib.transport.lp_transport import (
    round_transport_min_cost_lp,
    round_transport_min_cost_lp_restricted,
)

__all__ = [
    "round_transport_min_cost_mcf",
    "round_transport_min_cost_approx",
    "round_transport_floor_residue_lp",
]


def _mask_from_topk_and_mass(C: np.ndarray, X: Optional[np.ndarray] = None,
                             topk: int = 30, eta: float = 0.0) -> np.ndarray:
    """Boolean mask E of candidate arcs: per-row top-k by C, plus X>=eta if provided."""
    m, n = C.shape
    E = np.zeros((m, n), dtype=bool)
    # per-row top-k by cost
    for i in range(m):
        k = min(n, int(topk))
        js = np.argpartition(C[i], k - 1)[:k]
        E[i, js] = True
    # add high-mass arcs from entropic plan
    if X is not None and eta > 0.0:
        E |= (X >= eta)
    # ensure each positive-demand column has at least one arc
    for j in range(n):
        if not E[:, j].any():
            i = int(np.argmin(C[:, j]))
            E[i, j] = True
    return E


def round_transport_min_cost_mcf(a: np.ndarray, b: np.ndarray, C: np.ndarray,
                                 topk: Optional[int] = None) -> Tuple[np.ndarray, float]:
    """
    Min-cost flow integerization (TU => integral). If OR-Tools graph is unavailable
    or a sparsified model proves infeasible, falls back to the dense LP version.
    Returns (Xint, round_time_seconds).
    """
    start = time.time()

    # Try OR-Tools graph first
    try:
        from ortools.graph import pywrapgraph
        m, n = len(a), len(b)

        # Optional sparsification: per-row keep top-k cheapest arcs.
        if topk is not None:
            row_keep = []
            for i in range(m):
                k = min(n, int(topk))
                js = np.argpartition(C[i], k - 1)[:k]
                js = js[np.argsort(C[i, js])]
                row_keep.append(set(int(j) for j in js))
        else:
            row_keep = [None] * m

        smcf = pywrapgraph.SimpleMinCostFlow()
        S = m + n
        T = m + n + 1

        def add_arc(u, v, cap, cost):
            smcf.AddArcWithCapacityAndUnitCost(int(u), int(v), int(cap), int(cost))

        # Source -> suppliers
        for i in range(m):
            if a[i] > 0:
                add_arc(S, i, int(a[i]), 0)

        # Supplier -> demand arcs (capacities large; integer unit costs)
        scale = 1_000_000
        for i in range(m):
            J = range(n) if row_keep[i] is None else row_keep[i]
            for j in J:
                add_arc(i, m + int(j), int(b[j]), int(round(float(C[i, j]) * scale)))

        # Demands -> sink
        for j in range(n):
            if b[j] > 0:
                add_arc(m + j, T, int(b[j]), 0)

        # Supplies
        total = int(np.sum(a))
        node_count = m + n + 2
        supplies = [0] * node_count
        supplies[S] = total
        supplies[T] = -total
        for v, s in enumerate(supplies):
            smcf.SetNodeSupply(v, s)

        status = smcf.Solve()
        if status != smcf.OPTIMAL:
            # Fall back to LP if sparse MCF can't route all flow
            Xlp, t_lp = round_transport_min_cost_lp(a, b, C)
            return Xlp, (time.time() - start)

        # Extract flow
        X = np.zeros((m, n), dtype=int)
        for e in range(smcf.NumArcs()):
            u = smcf.Tail(e); v = smcf.Head(e); f = smcf.Flow(e)
            if 0 <= u < m and m <= v < m + n and f > 0:
                j = v - m
                X[u, j] += int(f)

        # Sanity checks
        assert np.all(X.sum(axis=1) == a)
        assert np.all(X.sum(axis=0) == b)
        return X, (time.time() - start)

    except Exception:
        # No graph module (or any error) -> dense LP
        Xlp, t_lp = round_transport_min_cost_lp(a, b, C)
        return Xlp, t_lp


def round_transport_min_cost_approx(a: np.ndarray, b: np.ndarray, C: np.ndarray,
                                    Xtau: Optional[np.ndarray] = None,
                                    topk: int = 30, eta: float = 0.0) -> Tuple[np.ndarray, float, bool]:
    """
    Approximate transport rounder: restricted-support LP using top-k-by-cost per row,
    plus arcs with Xtau >= eta (if provided). Falls back to full LP on failure.
    Returns (Xint, time_s, used_restricted=True/False).
    """
    E = _mask_from_topk_and_mass(C, Xtau, topk=topk, eta=eta)
    Xint, t, ok = round_transport_min_cost_lp_restricted(a, b, C, E)
    if ok:
        return Xint, t, True
    # fallback
    Xint_full, t_full = round_transport_min_cost_lp(a, b, C)
    return Xint_full, t_full, False


def _reopt_from_greedy(a, b, C, Xgreedy, Xtau, topk, eta):
    """
    Re-optimize on a small support: union of
      - per-row top-k-by-cost,
      - Xtau >= eta arcs (if provided),
      - arcs used by the greedy solution.
    Falls back to full LP if the restricted model is infeasible.
    """
    E = _mask_from_topk_and_mass(C, Xtau, topk=topk, eta=eta)
    E |= (Xgreedy > 0)
    Xint, t_reopt, ok = round_transport_min_cost_lp_restricted(a, b, C, E)
    if ok:
        return Xint, t_reopt, True
    # fallback: full LP (still fast and integral by TU)
    Xfull, t_full = round_transport_min_cost_lp(a, b, C)
    return Xfull, t_reopt + t_full, False


def round_transport_floor_residue_lp(a: np.ndarray, b: np.ndarray,
                                     C: np.ndarray) -> Tuple[np.ndarray, float]:
    """Backward-compatible alias for round_transport_min_cost_lp (kept from paper)."""
    return round_transport_min_cost_lp(a, b, C)
