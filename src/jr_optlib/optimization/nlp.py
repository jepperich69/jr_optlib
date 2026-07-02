# -*- coding: utf-8 -*-
"""Non-linear programming primitives (e.g., coordinate descent for bilinear)."""

import numpy as np

def solve_coord_wise(
    D: np.ndarray,
    L: np.ndarray,
    k_v: np.ndarray,
    L_k: np.ndarray,
    c_k_v: np.ndarray,
    B: float,
    K: int,
    beta: float,
    alpha_tilde: float,
    delta: float,
    B_coeff: float,
    G_min: float,
    G_max: float,
    f_min: float,
    f_max: float,
    G0: np.ndarray = None,
    f0_k: np.ndarray = None,
    max_iter: int = 30,
    tol: float = 1e-6,
):
    """
    Block-coordinate descent for the bilinear transit spacing/frequency problem.
    """
    Z = len(D)
    
    if G0 is None:
        G_kz = np.full(Z, G_min)
    else:
        G_kz = np.clip(G0, G_min, G_max)
        
    if f0_k is None:
        f_k = np.full(K, f_min)
    else:
        f_k = np.clip(f0_k, f_min, f_max)

    def _solve_spacing(f_k_curr, mu_curr):
        return np.clip(
            np.sqrt((D * beta + mu_curr * f_k_curr[k_v] * B_coeff) / (D * alpha_tilde + 1e-12)),
            G_min, G_max
        )

    def _solve_frequency(G_kz_curr):
        stop_per_freq = np.zeros(K)
        for z in range(Z):
            stop_per_freq[k_v[z]] += B_coeff * L[z] / G_kz_curr[z]
        C_k = c_k_v * L_k + stop_per_freq

        D_tilde = np.zeros(K)
        for z in range(Z):
            D_tilde[k_v[z]] += D[z] * L[z]

        def f_of_mu(mu):
            return np.clip(np.sqrt(D_tilde * delta / (mu * C_k + 1e-12)), f_min, f_max)

        def total_cost(mu):
            return np.sum(C_k * f_of_mu(mu))

        if total_cost(1e-10) <= B:
            return f_of_mu(1e-10), 1e-10

        mu_lo, mu_hi = 1e-10, 1e6
        for _ in range(80):
            mu_mid = 0.5 * (mu_lo + mu_hi)
            if total_cost(mu_mid) > B:
                mu_lo = mu_mid
            else:
                mu_hi = mu_mid
            if mu_hi - mu_lo < 1e-12:
                break
        return f_of_mu(0.5 * (mu_lo + mu_hi)), 0.5 * (mu_lo + mu_hi)

    mu_new = 1e-10
    history = []
    
    for it in range(max_iter):
        f_at_G, mu = _solve_frequency(G_kz)
        G_new = _solve_spacing(f_at_G, mu)
        f_new, mu_new = _solve_frequency(G_new)

        dG = np.max(np.abs(G_new - G_kz))
        df = np.max(np.abs(f_new - f_k))
        G_kz, f_k = G_new, f_new

        obj = np.sum(D * L * (beta / G_kz + alpha_tilde * G_kz + delta / f_k[k_v]))
        history.append({
            "iteration": it + 1,
            "objective": float(obj),
            "max_dG": float(dG),
            "max_df": float(df),
            "mu": float(mu_new),
        })

        if dG < tol and df < tol:
            break

    return G_kz, f_k, mu_new, history
