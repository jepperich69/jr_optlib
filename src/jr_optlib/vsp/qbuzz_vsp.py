from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:  # pragma: no cover
    gp = None
    GRB = None


@dataclass(frozen=True)
class QbuzzTrip:
    idx: int
    line: str
    trip_no: str
    from_stop: str
    start: int
    end: int
    to_stop: str
    distance_km: float


@dataclass(frozen=True)
class VehicleParams:
    fixed_bus_cost: float
    cost_per_minute: float
    cost_per_km: float
    energy_kwh_per_km: float
    max_charge_rate_kwh_per_min: float
    battery_capacity_kwh: float
    # van Kooten Niekerk (2017) battery replacement cost (BC). Set to ~100_000
    # for gn12/qlink instances to match ten Bosch et al. (2025). 0 = disabled.
    battery_cost: float = 0.0


@dataclass(frozen=True)
class Charger:
    location: str
    setup_minutes: int
    max_charge_speed_kwh_per_min: float
    max_soc: float


@dataclass
class QbuzzInstance:
    name: str
    root: Path
    garage: str
    garages: Tuple[str, ...]
    trips: List[QbuzzTrip]
    deadhead_times: Dict[Tuple[str, str], Tuple[int, int, int, int]]
    deadhead_distances: Dict[Tuple[str, str], float]
    time_versions: List[Tuple[int, int, int]]
    chargers: Dict[str, Charger]
    vehicle: VehicleParams


@dataclass(frozen=True)
class ScheduleColumn:
    trips: Tuple[int, ...]
    cost: float
    deadhead_km: float
    idle_minutes: int


@dataclass
class GreedySolution:
    seed: int
    columns: List[ScheduleColumn]
    objective: float
    runtime_sec: float


@dataclass
class SASolution:
    seed: int
    columns: List[ScheduleColumn]
    objective: float
    runtime_sec: float
    accepted_moves: int
    attempted_moves: int
    collected_columns: List[ScheduleColumn]


@dataclass
class RecombinationResult:
    objective: float
    selected_columns: List[ScheduleColumn]
    selected_indices: Tuple[int, ...]
    runtime_sec: float
    mip_gap: float
    status: int


@dataclass
class PMIPPoolResult:
    incumbent_objective: float
    epsilon: float
    n_samples: int
    runtime_sec: float
    sample_summary: pd.DataFrame
    column_summary: pd.DataFrame
    trip_summary: pd.DataFrame


def _parts(line: str) -> List[str]:
    return line.rstrip("\n").split(";")


def _read_lines(path: Path) -> List[str]:
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding).splitlines()
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="latin-1", errors="replace").splitlines()


def load_qbuzz_instance(instance_dir: str | Path) -> QbuzzInstance:
    root = Path(instance_dir)
    name = root.name
    trips: List[QbuzzTrip] = []
    deadhead_times: Dict[Tuple[str, str], Tuple[int, int, int, int]] = {}
    deadhead_distances: Dict[Tuple[str, str], float] = {}
    time_versions: List[Tuple[int, int, int]] = []
    chargers: Dict[str, Charger] = {}
    garages: List[str] = []
    vehicle: Optional[VehicleParams] = None

    for raw in _read_lines(root / "trips.txt"):
        p = _parts(raw)
        if not p or p[0] != "T":
            continue
        distance = float(p[17]) if len(p) > 17 and p[17] else 0.0
        trips.append(
            QbuzzTrip(
                idx=len(trips),
                line=p[1],
                trip_no=p[2],
                from_stop=p[5],
                start=int(p[6]),
                end=int(p[7]),
                to_stop=p[8],
                distance_km=distance,
            )
        )

    for raw in _read_lines(root / "dhd.txt"):
        p = _parts(raw)
        if not p:
            continue
        if p[0] == "G":
            time_versions.append((int(p[2]), int(p[3]), int(p[1])))
        elif p[0] == "D":
            a, b = p[1].split("-", 1)
            deadhead_times[a, b] = tuple(int(v) for v in p[2:6])
            deadhead_distances[a, b] = float(p[6]) if len(p) > 6 and p[6] else 0.0

    for raw in _read_lines(root / "parameters.txt"):
        p = _parts(raw)
        if not p:
            continue
        if p[0] == "G":
            garages.append(p[1])
        elif p[0] == "E":
            max_soc = float(p[5]) if len(p) > 5 and p[5] else 1.0
            chargers[p[1]] = Charger(
                location=p[1],
                setup_minutes=int(float(p[3])),
                max_charge_speed_kwh_per_min=float(p[4]),
                max_soc=max_soc,
            )
        elif p[0] == "U":
            vehicle = VehicleParams(
                fixed_bus_cost=float(p[2]),
                cost_per_minute=float(p[3]),
                cost_per_km=float(p[4]),
                energy_kwh_per_km=float(p[9]),
                max_charge_rate_kwh_per_min=float(p[10]),
                battery_capacity_kwh=float(p[12]),
            )

    if not garages:
        raise ValueError(f"No garage found in {root / 'parameters.txt'}")
    if vehicle is None:
        raise ValueError(f"No vehicle parameter row found in {root / 'parameters.txt'}")

    trips.sort(key=lambda t: (t.start, t.end, t.idx))
    trips = [QbuzzTrip(i, t.line, t.trip_no, t.from_stop, t.start, t.end, t.to_stop, t.distance_km) for i, t in enumerate(trips)]

    return QbuzzInstance(
        name=name,
        root=root,
        garage=garages[0],
        garages=tuple(garages),
        trips=trips,
        deadhead_times=deadhead_times,
        deadhead_distances=deadhead_distances,
        time_versions=sorted(time_versions),
        chargers=chargers,
        vehicle=vehicle,
    )


def _time_version(instance: QbuzzInstance, minute: int) -> int:
    for start, end, version in instance.time_versions:
        if start <= minute <= end:
            return version
    return 0


def deadhead(instance: QbuzzInstance, from_stop: str, to_stop: str, depart_minute: int) -> Tuple[int, float]:
    if from_stop == to_stop:
        return 0, 0.0
    key = (from_stop, to_stop)
    times = instance.deadhead_times.get(key)
    if times is None:
        return math.inf, math.inf
    version = _time_version(instance, depart_minute)
    return times[min(version, len(times) - 1)], instance.deadhead_distances.get(key, 0.0)


def _degradation_cost(soc_before: float, soc_after: float, battery_cost: float) -> float:
    """van Kooten Niekerk (2017): wear cost for charging from soc_before to soc_after (both in [0,1])."""
    if battery_cost == 0.0 or soc_after <= soc_before:
        return 0.0
    return (math.exp(2.519 * soc_after) - math.exp(2.519 * soc_before)) / 4825.3 * battery_cost


def _charge(instance: QbuzzInstance, stop: str, available_minutes: int, battery_kwh: float) -> float:
    charger = instance.chargers.get(stop)
    if charger is None or available_minutes <= charger.setup_minutes:
        return battery_kwh
    rate = min(charger.max_charge_speed_kwh_per_min, instance.vehicle.max_charge_rate_kwh_per_min)
    cap = charger.max_soc * instance.vehicle.battery_capacity_kwh
    return min(cap, battery_kwh + (available_minutes - charger.setup_minutes) * rate)


def _evaluate_schedule_from_garage(instance: QbuzzInstance, trip_ids: Sequence[int], garage: str) -> Optional[ScheduleColumn]:
    if not trip_ids:
        return None
    ordered = tuple(sorted(trip_ids, key=lambda i: instance.trips[i].start))
    cap = instance.vehicle.battery_capacity_kwh
    bc = instance.vehicle.battery_cost
    battery = cap
    prev_stop = garage
    prev_time = min(instance.trips[ordered[0]].start, 0)
    deadhead_km = 0.0
    idle_minutes = 0
    degradation = 0.0

    for trip_id in ordered:
        trip = instance.trips[trip_id]
        dh_time, dh_km = deadhead(instance, prev_stop, trip.from_stop, prev_time)
        if not math.isfinite(dh_time):
            return None
        arrival = prev_time + int(dh_time)
        if arrival > trip.start:
            return None
        battery -= dh_km * instance.vehicle.energy_kwh_per_km
        if battery < -1e-9:
            return None
        wait = trip.start - arrival
        soc_before = battery / cap
        battery = _charge(instance, trip.from_stop, wait, battery)
        degradation += _degradation_cost(soc_before, battery / cap, bc)
        idle_minutes += wait
        battery -= trip.distance_km * instance.vehicle.energy_kwh_per_km
        if battery < -1e-9:
            return None
        deadhead_km += dh_km
        prev_stop = trip.to_stop
        prev_time = trip.end

    dh_time, dh_km = deadhead(instance, prev_stop, garage, prev_time)
    if not math.isfinite(dh_time):
        return None
    battery -= dh_km * instance.vehicle.energy_kwh_per_km
    if battery < -1e-9:
        return None
    deadhead_km += dh_km
    # overnight charge: top up from end-of-day SoC to full
    degradation += _degradation_cost(battery / cap, 1.0, bc)
    total_trip_km = sum(instance.trips[i].distance_km for i in ordered)
    total_km = total_trip_km + deadhead_km
    cost = (
        instance.vehicle.fixed_bus_cost
        + instance.vehicle.cost_per_km * total_km
        + instance.vehicle.cost_per_minute * idle_minutes
        + degradation
    )
    return ScheduleColumn(trips=ordered, cost=float(cost), deadhead_km=float(deadhead_km), idle_minutes=int(idle_minutes))


def evaluate_schedule(instance: QbuzzInstance, trip_ids: Sequence[int]) -> Optional[ScheduleColumn]:
    candidates = [_evaluate_schedule_from_garage(instance, trip_ids, garage) for garage in instance.garages]
    feasible = [candidate for candidate in candidates if candidate is not None]
    if not feasible:
        return None
    return min(feasible, key=lambda c: c.cost)


def build_singleton_columns(instance: QbuzzInstance) -> List[ScheduleColumn]:
    out = []
    for trip in instance.trips:
        col = evaluate_schedule(instance, (trip.idx,))
        if col is None:
            raise ValueError(f"Trip {trip.idx} is not feasible as a singleton schedule.")
        out.append(col)
    return out


def randomized_greedy_solution(instance: QbuzzInstance, seed: int, alpha: float = 0.25) -> GreedySolution:
    rng = random.Random(seed)
    start = time.perf_counter()
    schedules: List[List[int]] = []
    columns: List[ScheduleColumn] = []

    for trip in instance.trips:
        candidates: List[Tuple[float, int, ScheduleColumn]] = []
        for s_idx, current in enumerate(schedules):
            if current and instance.trips[current[-1]].start > trip.start:
                continue
            old_cost = columns[s_idx].cost
            candidate = evaluate_schedule(instance, (*current, trip.idx))
            if candidate is not None:
                noise = rng.random() * alpha * max(1.0, candidate.cost - old_cost)
                candidates.append((candidate.cost - old_cost + noise, s_idx, candidate))
        if candidates and rng.random() > 0.08:
            _, s_idx, candidate = min(candidates, key=lambda x: x[0])
            schedules[s_idx] = list(candidate.trips)
            columns[s_idx] = candidate
        else:
            singleton = evaluate_schedule(instance, (trip.idx,))
            if singleton is None:
                raise ValueError(f"Trip {trip.idx} cannot start a feasible schedule.")
            schedules.append([trip.idx])
            columns.append(singleton)

    runtime = time.perf_counter() - start
    return GreedySolution(seed=seed, columns=columns, objective=sum(c.cost for c in columns), runtime_sec=runtime)


def _solution_objective(columns: Sequence[ScheduleColumn]) -> float:
    return float(sum(c.cost for c in columns))


def _drop_empty_schedules(schedules: Sequence[Sequence[int]]) -> List[List[int]]:
    return [list(s) for s in schedules if s]


def _columns_for_schedules(instance: QbuzzInstance, schedules: Sequence[Sequence[int]]) -> Optional[List[ScheduleColumn]]:
    cols: List[ScheduleColumn] = []
    for sched in schedules:
        col = evaluate_schedule(instance, sched)
        if col is None:
            return None
        cols.append(col)
    return cols


def _sa_trip_move(
    schedules: Sequence[Sequence[int]],
    rng: random.Random,
) -> List[List[int]]:
    out = [list(s) for s in schedules]
    sources = [i for i, sched in enumerate(out) if sched]
    if not sources:
        return out
    src = rng.choice(sources)
    dst = rng.randrange(len(out))
    if src == dst and len(out) > 1:
        dst = (dst + rng.randrange(1, len(out))) % len(out)
    length = len(out[src])
    a = rng.randrange(length)
    b = rng.randrange(a + 1, length + 1)
    block = out[src][a:b]
    del out[src][a:b]
    out[dst].extend(block)
    out[dst].sort(key=lambda i: (i if i >= 0 else -1, i))
    out[dst].sort(key=lambda i: i)
    return _drop_empty_schedules(out)


def _sa_suffix_swap(
    instance: QbuzzInstance,
    schedules: Sequence[Sequence[int]],
    rng: random.Random,
) -> List[List[int]]:
    if len(schedules) < 2:
        return [list(s) for s in schedules]
    out = [list(s) for s in schedules]
    i, j = rng.sample(range(len(out)), 2)
    lo = min(t.start for t in instance.trips)
    hi = max(t.start for t in instance.trips)
    split_time = rng.randint(lo, hi)

    def split(sched: Sequence[int]) -> Tuple[List[int], List[int]]:
        prefix = [t for t in sched if instance.trips[t].start <= split_time]
        suffix = [t for t in sched if instance.trips[t].start > split_time]
        return prefix, suffix

    i_pre, i_suf = split(out[i])
    j_pre, j_suf = split(out[j])
    out[i] = sorted(i_pre + j_suf, key=lambda t: instance.trips[t].start)
    out[j] = sorted(j_pre + i_suf, key=lambda t: instance.trips[t].start)
    return _drop_empty_schedules(out)


def simulated_annealing_solution(
    instance: QbuzzInstance,
    seed: int,
    iterations: int = 5000,
    initial_temperature: float = 10_000.0,
    cooling_rate: float = 3e-4,
    collect_every: int = 25,
) -> SASolution:
    from jr_optlib.sampling.mcmc import simulated_annealing

    start = time.perf_counter()
    current_solution = randomized_greedy_solution(instance, seed)
    init_schedules = [list(c.trips) for c in current_solution.columns]
    
    collected_columns = list(current_solution.columns)
    init_state = (init_schedules, current_solution.columns, current_solution.objective)
    
    def energy_fn(state):
        return state[2]

    def propose_fn(state, rng):
        schedules, _, _ = state
        if rng.random() < 0.5:
            cand_scheds = _sa_suffix_swap(instance, schedules, rng)
        else:
            cand_scheds = _sa_trip_move(schedules, rng)
            
        cand_cols = _columns_for_schedules(instance, cand_scheds)
        if cand_cols is None:
            return None
            
        cand_obj = _solution_objective(cand_cols)
        return (cand_scheds, cand_cols, cand_obj)

    final_current = [init_state[1]]

    def on_accept(state, energy, step, is_best):
        _, cols, _ = state
        final_current[0] = cols
        if step % collect_every == 0:
            collected_columns.extend(cols)
        if is_best:
            collected_columns.extend(cols)

    best_state, best_energy, stats = simulated_annealing(
        init_state=init_state,
        energy_fn=energy_fn,
        propose_fn=propose_fn,
        init_temp=initial_temperature,
        cooling_rate=cooling_rate,
        n_steps=iterations,
        seed=seed,
        on_accept=on_accept,
    )

    collected_columns.extend(final_current[0])
    collected_columns.extend(best_state[1])
    runtime = time.perf_counter() - start
    
    return SASolution(
        seed=seed,
        columns=best_state[1],
        objective=best_energy,
        runtime_sec=runtime,
        accepted_moves=stats.n_accepted,
        attempted_moves=iterations,
        collected_columns=collected_columns,
    )


def unique_columns(columns: Iterable[ScheduleColumn]) -> List[ScheduleColumn]:
    best: Dict[Tuple[int, ...], ScheduleColumn] = {}
    for col in columns:
        old = best.get(col.trips)
        if old is None or col.cost < old.cost:
            best[col.trips] = col
    return list(best.values())


def solve_recombination_mip(
    instance: QbuzzInstance,
    columns: Sequence[ScheduleColumn],
    time_limit: float = 120.0,
    objective_noise: Optional[np.ndarray] = None,
    incumbent_objective: Optional[float] = None,
    epsilon: Optional[float] = None,
    vehicle_penalty: float = 0.0,
    objective_weights: Optional[np.ndarray] = None,
    maximize_weights: bool = False,
    forbidden_selections: Optional[Sequence[Sequence[int]]] = None,
    threads: int = 1,
) -> RecombinationResult:
    if gp is None:
        raise ImportError("gurobipy is required for recombination MIP.")
    if objective_noise is not None and len(objective_noise) != len(columns):
        raise ValueError("objective_noise must match number of columns.")
    if objective_weights is not None and len(objective_weights) != len(columns):
        raise ValueError("objective_weights must match number of columns.")

    start = time.perf_counter()
    model = gp.Model(f"qbuzz_recombine_{instance.name}")
    model.Params.OutputFlag = 0
    if time_limit is not None and time_limit > 0:
        model.Params.TimeLimit = float(time_limit)
    model.Params.Threads = threads
    x = model.addVars(len(columns), vtype=GRB.BINARY, name="x")

    trip_to_cols: List[List[int]] = [[] for _ in instance.trips]
    for j, col in enumerate(columns):
        for trip_id in col.trips:
            trip_to_cols[trip_id].append(j)
    for trip_id, covering in enumerate(trip_to_cols):
        if not covering:
            raise ValueError(f"No column covers trip {trip_id}.")
        model.addConstr(gp.quicksum(x[j] for j in covering) == 1, name=f"cover[{trip_id}]")

    true_objective = gp.quicksum(columns[j].cost * x[j] for j in range(len(columns)))
    if incumbent_objective is not None and epsilon is not None:
        model.addConstr(true_objective <= (1.0 + float(epsilon)) * float(incumbent_objective), name="near_optimal")
    if forbidden_selections:
        for k, selection in enumerate(forbidden_selections):
            if selection:
                model.addConstr(gp.quicksum(x[j] for j in selection) <= len(selection) - 1, name=f"nogood[{k}]")

    if objective_weights is not None:
        objective = gp.quicksum(objective_weights[j] * x[j] for j in range(len(columns)))
        model.setObjective(objective, GRB.MAXIMIZE if maximize_weights else GRB.MINIMIZE)
    else:
        objective = gp.quicksum(
            (columns[j].cost + vehicle_penalty + (0.0 if objective_noise is None else objective_noise[j])) * x[j]
            for j in range(len(columns))
        )
        model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()

    selected_indices = tuple(j for j in range(len(columns)) if model.SolCount and x[j].X > 0.5)
    selected = [columns[j] for j in selected_indices]
    runtime = time.perf_counter() - start
    return RecombinationResult(
        objective=float(sum(c.cost for c in selected)),
        selected_columns=selected,
        selected_indices=selected_indices,
        runtime_sec=runtime,
        mip_gap=float(model.MIPGap) if model.SolCount else math.inf,
        status=int(model.Status),
    )


def sample_near_optimal_recombinations(
    instance: QbuzzInstance,
    columns: Sequence[ScheduleColumn],
    incumbent: RecombinationResult,
    n_samples: int = 50,
    epsilon: float = 0.02,
    temperature: float = 0.02,
    seed: int = 123,
    time_limit: float = 30.0,
    mode: str = "diversity",
) -> PMIPPoolResult:
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    mean_cost = float(np.mean([c.cost for c in columns]))
    noise_scale = max(1e-6, float(temperature) * mean_cost)
    selected_counts = np.zeros(len(columns), dtype=int)
    trip_vehicle_counts: List[List[int]] = [[] for _ in instance.trips]
    sample_rows = []
    forbidden: List[Tuple[int, ...]] = [incumbent.selected_indices]

    for sample_idx in range(n_samples):
        if mode == "nogood":
            result = solve_recombination_mip(
                instance=instance,
                columns=columns,
                time_limit=time_limit,
                incumbent_objective=incumbent.objective,
                epsilon=epsilon,
                forbidden_selections=forbidden,
            )
            if result.status in (3, 4) or not result.selected_indices:
                break
            forbidden.append(result.selected_indices)
        elif mode == "perturb_cost":
            noise = rng.gumbel(loc=0.0, scale=noise_scale, size=len(columns))
            result = solve_recombination_mip(
                instance=instance,
                columns=columns,
                time_limit=time_limit,
                objective_noise=noise,
                incumbent_objective=incumbent.objective,
                epsilon=epsilon,
            )
        elif mode == "diversity":
            weights = rng.gumbel(loc=0.0, scale=1.0, size=len(columns))
            result = solve_recombination_mip(
                instance=instance,
                columns=columns,
                time_limit=time_limit,
                incumbent_objective=incumbent.objective,
                epsilon=epsilon,
                objective_weights=weights,
                maximize_weights=True,
            )
        else:
            raise ValueError("mode must be 'nogood', 'diversity' or 'perturb_cost'.")
        for j in result.selected_indices:
            selected_counts[j] += 1
        for vehicle_idx, col in enumerate(result.selected_columns):
            for trip_id in col.trips:
                trip_vehicle_counts[trip_id].append(vehicle_idx)
        sample_rows.append(
            {
                "sample": sample_idx,
                "objective": result.objective,
                "objective_gap": (result.objective / incumbent.objective) - 1.0,
                "vehicles": len(result.selected_columns),
                "runtime_sec": result.runtime_sec,
                "mip_gap": result.mip_gap,
                "status": result.status,
            }
        )

    probs = selected_counts / max(1, n_samples)
    column_rows = []
    for j, (col, prob) in enumerate(zip(columns, probs)):
        if prob <= 0:
            continue
        column_rows.append(
            {
                "column": j,
                "inclusion_probability": prob,
                "n_trips": len(col.trips),
                "cost": col.cost,
                "deadhead_km": col.deadhead_km,
                "idle_minutes": col.idle_minutes,
                "trips": " ".join(str(v) for v in col.trips),
            }
        )

    trip_rows = []
    for trip in instance.trips:
        cover_probs = [
            probs[j]
            for j, col in enumerate(columns)
            if trip.idx in col.trips and probs[j] > 0
        ]
        entropy = -sum(p * math.log(p) for p in cover_probs if p > 0)
        trip_rows.append(
            {
                "trip": trip.idx,
                "line": trip.line,
                "start": trip.start,
                "end": trip.end,
                "n_positive_columns": len(cover_probs),
                "cover_entropy": entropy,
                "max_column_probability": max(cover_probs) if cover_probs else 0.0,
            }
        )

    runtime = time.perf_counter() - start
    return PMIPPoolResult(
        incumbent_objective=incumbent.objective,
        epsilon=epsilon,
        n_samples=n_samples,
        runtime_sec=runtime,
        sample_summary=pd.DataFrame(sample_rows),
        column_summary=pd.DataFrame(column_rows).sort_values("inclusion_probability", ascending=False) if column_rows else pd.DataFrame(),
        trip_summary=pd.DataFrame(trip_rows),
    )


def instance_summary(instance: QbuzzInstance) -> Dict[str, float | int | str]:
    return {
        "instance": instance.name,
        "trips": len(instance.trips),
        "stops": len({t.from_stop for t in instance.trips} | {t.to_stop for t in instance.trips}),
        "deadheads": len(instance.deadhead_times),
        "chargers": len(instance.chargers),
        "garage": instance.garage,
        "garages": len(instance.garages),
        "battery_kwh": instance.vehicle.battery_capacity_kwh,
    }


def run_column_pool_benchmark(
    instance: QbuzzInstance,
    runs: int,
    seed: int,
    mip_time_limit: float,
    pmip_samples: int = 0,
    method: str = "greedy",
    sa_iterations: int = 5000,
    pmip_epsilon: float = 0.02,
    pmip_temperature: float = 0.02,
    pmip_mode: str = "diversity",
) -> Tuple[pd.DataFrame, pd.DataFrame, RecombinationResult, pd.DataFrame, Optional[PMIPPoolResult]]:
    all_columns = build_singleton_columns(instance)
    run_rows = []
    for k in range(runs):
        if method == "greedy":
            solution = randomized_greedy_solution(instance, seed + k)
            all_columns.extend(solution.columns)
            accepted_moves = np.nan
            attempted_moves = np.nan
            collected = len(solution.columns)
        elif method == "sa":
            solution = simulated_annealing_solution(instance, seed + k, iterations=sa_iterations)
            all_columns.extend(solution.collected_columns)
            accepted_moves = solution.accepted_moves
            attempted_moves = solution.attempted_moves
            collected = len(solution.collected_columns)
        else:
            raise ValueError("method must be 'greedy' or 'sa'.")
        run_rows.append(
            {
                "run": k,
                "method": method,
                "seed": seed + k,
                "objective": solution.objective,
                "vehicles": len(solution.columns),
                "runtime_sec": solution.runtime_sec,
                "accepted_moves": accepted_moves,
                "attempted_moves": attempted_moves,
                "collected_columns": collected,
            }
        )

    pool = unique_columns(all_columns)
    best = solve_recombination_mip(instance, pool, time_limit=mip_time_limit)
    pool_rows = [
        {
            "instance": instance.name,
            "method": method,
            "runs": runs,
            "sa_iterations": sa_iterations if method == "sa" else 0,
            "columns": len(pool),
            "singleton_columns": len(instance.trips),
            "best_run_objective": min(r["objective"] for r in run_rows),
            "best_run_vehicles": min(r["vehicles"] for r in run_rows),
            "recombined_objective": best.objective,
            "recombined_vehicles": len(best.selected_columns),
            "recombined_runtime_sec": best.runtime_sec,
            "mip_gap": best.mip_gap,
        }
    ]

    pmip_result = None
    sample_frame = pd.DataFrame()
    if pmip_samples > 0:
        pmip_result = sample_near_optimal_recombinations(
            instance=instance,
            columns=pool,
            incumbent=best,
            n_samples=pmip_samples,
            epsilon=pmip_epsilon,
            temperature=pmip_temperature,
            seed=seed,
            time_limit=mip_time_limit,
            mode=pmip_mode,
        )
        sample_frame = pmip_result.sample_summary

    return pd.DataFrame(run_rows), pd.DataFrame(pool_rows), best, sample_frame, pmip_result
