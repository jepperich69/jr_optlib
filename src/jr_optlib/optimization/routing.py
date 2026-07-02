# -*- coding: utf-8 -*-
"""Routing primitives (SUE, Dijkstra, Mode choice)."""

import numpy as np
from collections import defaultdict

def dijkstra_manhattan(station_positions, person_positions, people_modes):
    """
    Simplified distance computation matching the legacy 'Dijkstra' logic for grid networks.
    Returns the minimum distance to any station per person mode.
    """
    distances = defaultdict(lambda: float('inf'))
    for person, mode in zip(person_positions, people_modes):
        x, y = person
        for station in station_positions:
            dist = abs(x - station[0]) + abs(y - station[1])
            if dist < distances[station, mode]:
                distances[station, mode] = dist
    return distances


def compute_route_choice_shares(
    n_zones: int,
    n_routes: int,
    car_times: np.ndarray,
    frequencies: np.ndarray,
    transit_time_tensor: np.ndarray,
    route_serves: np.ndarray,
    hubs: np.ndarray = None,
    beta_time: float = 1.0,
    beta_wait: float = 1.0,
    beta_cost: float = 0.2,
    asc_transit: float = 0.0,
    asc_car: float = 0.0,
    transit_fare: float = 0.0,
    car_cost_per_km: float = 0.20,
    car_congestion_charge: float = 0.0,
    transfer_time: float = 4.0,
    transfer_penalty: float = 1.5,
):
    """
    SUE route-choice over car + transit (direct and one-transfer).
    Returns:
      P_car (O,D), P_transit (O,D), S_route (R,O,D)
    """
    # Car utility
    car_km = (car_times / 60.0) * 40.0
    car_costs = car_km * car_cost_per_km + car_congestion_charge
    U_car = asc_car - beta_time * car_times - beta_cost * car_costs

    # Wait and active mask
    wait = np.where(frequencies >= 1.0, np.minimum(25.0 / np.maximum(frequencies, 1e-6), 60.0), 60.0)
    active = frequencies >= 1.0
    serves = route_serves.astype(bool)

    # Transit times
    time_r = np.where(active[:, None, None], transit_time_tensor, np.inf).astype(float)

    travel_no_wait = time_r - wait[:, None, None]
    U_dir = asc_transit - beta_time * travel_no_wait - beta_wait * wait[:, None, None] - beta_cost * transit_fare

    U_alts = []

    # Direct alternatives
    for r in range(n_routes):
        U_r = U_dir[r]
        if not np.isfinite(U_r).any():
            continue
        contrib = np.zeros((n_routes, n_zones, n_zones), dtype=float)
        mask = np.isfinite(U_r)
        contrib[r, mask] = 1.0
        U_alts.append((U_r, contrib))

    # One-transfer alternatives
    if hubs is not None:
        HUBS = np.array(hubs, dtype=int)
    else:
        HUBS = np.array([], dtype=int)

    transfer_disutil = - beta_time * (transfer_time * transfer_penalty)

    for h in HUBS:
        r_at_h = np.where(serves[:, h] & active)[0]
        if r_at_h.size == 0:
            continue

        t_leg1 = time_r[r_at_h, :, h]
        t_leg2 = time_r[r_at_h, h, :]

        U_leg1 = asc_transit - beta_time * (t_leg1 - wait[r_at_h][:, None]) - beta_wait * wait[r_at_h][:, None]
        U_leg2 = asc_transit - beta_time * (t_leg2 - wait[r_at_h][:, None]) - beta_wait * wait[r_at_h][:, None]

        U1_best_idx = np.argmax(U_leg1, axis=0)
        U2_best_idx = np.argmax(U_leg2, axis=0)
        U1_best = U_leg1[U1_best_idx, np.arange(n_zones)]
        U2_best = U_leg2[U2_best_idx, np.arange(n_zones)]

        U_pair = U1_best[:, None] + U2_best[None, :] + transfer_disutil - beta_cost * transit_fare

        contrib = np.zeros((n_routes, n_zones, n_zones), dtype=float)
        r1_by_o = r_at_h[U1_best_idx]
        r2_by_d = r_at_h[U2_best_idx]
        for o in range(n_zones):
            mask_d = np.isfinite(U_pair[o, :])
            if not mask_d.any():
                continue
            contrib[r1_by_o[o], o, mask_d] += 0.5
            ds = np.where(mask_d)[0]
            contrib[r2_by_d[ds], o, ds] += 0.5

        U_alts.append((U_pair, contrib))

    U_car_c = np.clip(U_car, -50.0, 50.0)
    exp_car = np.exp(U_car_c)
    exp_sum = exp_car.copy()

    S_num = np.zeros((n_routes, n_zones, n_zones), dtype=float)
    for U_alt, contrib in U_alts:
        U_c = np.clip(U_alt, -50.0, 50.0)
        e = np.exp(U_c)
        exp_sum += e
        S_num += e[None, :, :] * contrib

    denom = np.maximum(exp_sum, 1e-12)
    P_car = exp_car / denom
    P_transit = 1.0 - P_car

    exp_tr = np.maximum(denom - exp_car, 1e-12)
    S_route_cond = S_num / exp_tr[None, :, :]
    S_route = S_route_cond * P_transit[None, :, :]

    idx = np.arange(n_zones)
    P_car[idx, idx] = 0.0
    P_transit[idx, idx] = 0.0
    S_route[:, idx, idx] = 0.0

    return P_car, P_transit, S_route
