"""
vsp_projection.py
-----------------
Metropolis feasible-state search for VSP framed as path-dependent set partitioning.

The solver is a cost-guided stochastic local search over complete feasible schedules:

  1. round_to_partition    -- greedy pool rounding to a feasible warm start (fallback;
                              the benchmark supplies its own randomized-greedy warm start).
  2. run_vsp_mh_chain      -- Metropolis acceptance at fixed temperature, restricted to a
                              hard near-optimal band, with three move families
                              (suffix swap, task relocation, greedy vehicle-reduction repair).

Design notes (honest description):
  - This is NOT a detailed-balance / Boltzmann sampler. Acceptance uses the symmetric
    Metropolis rule exp(-Delta/tau) at a FIXED tau (no cooling schedule), the chain is
    confined to a hard band C(z) <= (1+epsilon)*incumbent, and the vehicle-reduction move
    is an irreversible greedy repair. We therefore make no stationary-distribution or
    annealing-convergence claim. The retained band states form a DESCRIPTIVE empirical
    sample of near-optimal feasible schedules visited by the search, used for trip
    entropy, backbone-duty probabilities, and resilience diagnostics.
  - The domain oracle (qbuzz_vsp.evaluate_schedule) certifies feasibility and cost of each
    changed duty, so the move layer stays domain-blind. The module name "projection" is
    historical.

Connection to the set-partitioning view:
  - Requirements  = trips  (each must be covered exactly once)
  - Actions       = vehicle schedule columns  (each covers a subset of trips)
  - Memory        = battery SoC  (encoded in column feasibility, handled by the oracle)
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .qbuzz_vsp import (
    QbuzzInstance,
    ScheduleColumn,
    _columns_for_schedules,
    _drop_empty_schedules,
    _sa_suffix_swap,
    _sa_trip_move,
    _solution_objective,
    build_singleton_columns,
    evaluate_schedule,
    unique_columns,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TripCoverIndex:
    """Sparse incidence structure for a column pool."""
    n_trips: int
    n_cols: int
    col_to_trips: List[Tuple[int, ...]]   # col_to_trips[j] = trip indices in column j
    trip_to_cols: List[List[int]]          # trip_to_cols[i] = pool column indices covering trip i
    costs: np.ndarray                       # costs[j], shape (n_cols,)


@dataclass
class ProjectionRoundResult:
    columns: List[ScheduleColumn]
    objective: float
    n_vehicles: int
    n_singletons_used: int
    runtime_sec: float


@dataclass
class MHChainResult:
    samples: List[List[ScheduleColumn]]
    objectives: List[float]
    n_steps: int
    n_accepted: int
    acceptance_rate: float
    incumbent_objective: float
    runtime_sec: float
    final_schedules: List[List[int]] = None  # terminal chain state for continuation
    best_objective: float = math.inf
    best_schedules: Optional[List[List[int]]] = None
    best_columns: Optional[List[ScheduleColumn]] = None


@dataclass
class VspProjectionPmipResult:
    instance_name: str
    round: ProjectionRoundResult
    mh: MHChainResult
    trip_entropy: np.ndarray
    mean_trip_entropy: float
    n_trips_with_entropy: int
    column_inclusion_probs: Dict[Tuple[int, ...], float]
    trip_summary: pd.DataFrame
    sample_summary: pd.DataFrame
    obj_summary: pd.DataFrame


# ---------------------------------------------------------------------------
# Index over the column pool
# ---------------------------------------------------------------------------

def build_trip_cover_index(
    columns: Sequence[ScheduleColumn],
    n_trips: int,
) -> TripCoverIndex:
    """Build the sparse trip/column incidence index used by the greedy rounding."""
    n_cols = len(columns)
    col_to_trips: List[Tuple[int, ...]] = [col.trips for col in columns]
    trip_to_cols: List[List[int]] = [[] for _ in range(n_trips)]
    for j, col in enumerate(columns):
        for trip_id in col.trips:
            trip_to_cols[trip_id].append(j)
    costs = np.array([col.cost for col in columns], dtype=float)
    return TripCoverIndex(
        n_trips=n_trips,
        n_cols=n_cols,
        col_to_trips=col_to_trips,
        trip_to_cols=trip_to_cols,
        costs=costs,
    )


# ---------------------------------------------------------------------------
# Greedy pool rounding (feasible warm-start fallback)
# ---------------------------------------------------------------------------

def round_to_partition(
    index: TripCoverIndex,
    pool_columns: Sequence[ScheduleColumn],
    singleton_columns: List[ScheduleColumn],
    rng: np.random.Generator,
) -> ProjectionRoundResult:
    """
    Build a feasible integer set-partition from the column pool by greedy selection.

    Columns are ordered by cost-efficiency (cost / trips_covered, cheapest first) and
    accepted whenever they do not conflict with already-covered trips. This favours
    multi-trip columns that spread vehicle cost over many trips. Any trips left uncovered
    after the greedy pass receive singleton fallback columns.

    This is a deterministic warm-start fallback; the benchmark normally overrides it with
    a randomized-greedy warm start passed to the MH chain.
    """
    start = time.perf_counter()
    n_trips = index.n_trips
    n_cols = index.n_cols
    assigned = [-1] * n_trips   # assigned[i] = pool column index, or -(trip+1) for singleton
    selected: List[ScheduleColumn] = []
    n_singletons = 0

    # Cost-efficiency ordering: cost per trip covered (ascending — best first).
    trip_counts = np.array([len(index.col_to_trips[j]) for j in range(n_cols)], dtype=float)
    efficiency = index.costs / np.maximum(trip_counts, 1.0)
    col_order = list(np.argsort(efficiency))  # ascending = most efficient first

    # Phase 1: greedy column selection
    for j in col_order:
        trips_j = index.col_to_trips[j]
        if all(assigned[t] < 0 for t in trips_j):
            selected.append(pool_columns[j])
            for t in trips_j:
                assigned[t] = j

    # Phase 2: singleton fallback for any trip left uncovered
    for i in range(n_trips):
        if assigned[i] < 0:
            selected.append(singleton_columns[i])
            assigned[i] = -(i + 1)
            n_singletons += 1

    obj = float(sum(c.cost for c in selected))
    return ProjectionRoundResult(
        columns=selected,
        objective=obj,
        n_vehicles=len(selected),
        n_singletons_used=n_singletons,
        runtime_sec=time.perf_counter() - start,
    )


# ---------------------------------------------------------------------------
# Metropolis moves
# ---------------------------------------------------------------------------

def _schedule_insertion_options(
    instance: QbuzzInstance,
    trip_id: int,
    schedules: Sequence[Sequence[int]],
    columns: Sequence[ScheduleColumn],
    all_positions: bool = False,
) -> List[Tuple[float, int, List[int], ScheduleColumn]]:
    """
    Candidate insertion positions for one trip into the current schedule state.

    Returns tuples (delta_cost, recipient_idx, candidate_schedule, candidate_column).
    """
    t_start = instance.trips[trip_id].start
    options: List[Tuple[float, int, List[int], ScheduleColumn]] = []
    for v, sched in enumerate(schedules):
        if all_positions:
            positions = range(len(sched) + 1)
        else:
            pos = sum(1 for t in sched if instance.trips[t].start <= t_start)
            positions = [pos]
        for pos in positions:
            candidate_sched = list(sched[:pos]) + [trip_id] + list(sched[pos:])
            col = evaluate_schedule(instance, candidate_sched)
            if col is not None:
                options.append((float(col.cost - columns[v].cost), v, candidate_sched, col))
    options.sort(key=lambda x: (x[0], len(x[2])))
    return options


def _attempt_vehicle_absorption(
    instance: QbuzzInstance,
    schedules: Sequence[Sequence[int]],
    columns: Sequence[ScheduleColumn],
    donor_idx: int,
    rng: random.Random,
    max_donor_trips: int = 12,
    max_attempts: int = 8,
    all_positions: bool = False,
) -> Optional[List[List[int]]]:
    """
    Remove one vehicle and greedily repair its trips into the remaining duties.

    Trips are inserted in a regret-like order: the trip with the fewest feasible
    recipient insertions is repaired first. This is the core vehicle-reduction move.
    """
    if len(schedules) < 2:
        return None

    donor_trips = list(schedules[donor_idx])
    if not donor_trips or len(donor_trips) > max_donor_trips:
        return None

    base_schedules = [list(s) for i, s in enumerate(schedules) if i != donor_idx]
    base_columns = [columns[i] for i in range(len(columns)) if i != donor_idx]
    best_schedules: Optional[List[List[int]]] = None
    best_obj = float("inf")

    for _ in range(max_attempts):
        remaining_schedules = [list(s) for s in base_schedules]
        remaining_columns = list(base_columns)
        pending = list(donor_trips)
        rng.shuffle(pending)

        feasible = True
        while pending:
            ranked_choices = []
            for trip_id in pending:
                options = _schedule_insertion_options(
                    instance,
                    trip_id,
                    remaining_schedules,
                    remaining_columns,
                    all_positions=all_positions,
                )
                if not options:
                    feasible = False
                    break
                best_delta = options[0][0]
                second_delta = options[1][0] if len(options) > 1 else best_delta + 1e9
                regret = second_delta - best_delta
                ranked_choices.append((len(options), -regret, rng.random(), trip_id, options))
            if not feasible:
                break

            ranked_choices.sort(key=lambda x: (x[0], x[1], x[2]))
            _, _, _, trip_id, options = ranked_choices[0]
            chosen = options[0]
            if len(options) > 1 and rng.random() < 0.2:
                chosen = options[1]
            _, recipient_idx, candidate_sched, candidate_col = chosen
            remaining_schedules[recipient_idx] = candidate_sched
            remaining_columns[recipient_idx] = candidate_col
            pending.remove(trip_id)

        if feasible:
            obj = _solution_objective(remaining_columns)
            if obj < best_obj:
                best_obj = obj
                best_schedules = remaining_schedules

    return best_schedules


def _vehicle_reduction_move(
    instance: QbuzzInstance,
    schedules: List[List[int]],
    columns: Sequence[ScheduleColumn],
    rng: random.Random,
) -> Optional[List[List[int]]]:
    """
    Stronger fleet-reduction move.

    Tries a handful of donor duties, prioritising duties that are short and costly per
    trip. Returns the first successful one-vehicle reduction.
    """
    if len(schedules) < 2:
        return None

    mean_cost = float(np.mean([c.cost for c in columns])) if columns else 1.0
    donor_scores = []
    for idx, (sched, col) in enumerate(zip(schedules, columns)):
        if not sched:
            continue
        if len(sched) > 12:
            continue
        cost_per_trip = float(col.cost) / max(len(sched), 1)
        score = (
            0.65 * (cost_per_trip / max(mean_cost, 1.0))
            + 0.25 / max(len(sched), 1)
        )
        donor_scores.append((score, len(sched), rng.random(), idx))

    if not donor_scores:
        return None

    donor_scores.sort(key=lambda x: (-x[0], x[1], x[2]))
    for _, _, _, donor_idx in donor_scores[: min(5, len(donor_scores))]:
        repaired = _attempt_vehicle_absorption(
            instance=instance,
            schedules=schedules,
            columns=columns,
            donor_idx=donor_idx,
            rng=rng,
        )
        if repaired is not None:
            return repaired

    return None


# ---------------------------------------------------------------------------
# Metropolis chain
# ---------------------------------------------------------------------------

def run_vsp_mh_chain(
    instance: QbuzzInstance,
    initial_schedules: List[List[int]],
    n_steps: int,
    tau_mh: float,
    incumbent_obj: float,
    epsilon: float,
    burn_in: int,
    thin: int,
    seed: int,
    early_reduction_prob: float = 0.20,
    late_reduction_prob: float = 0.30,
) -> MHChainResult:
    """
    Cost-guided Metropolis local search over complete feasible schedules.

    Acceptance is the symmetric Metropolis rule min{1, exp(-Delta/tau_mh)} at a FIXED
    temperature, with a hard near-optimal band: only candidates within
    epsilon * incumbent_obj of the incumbent are considered. This is a heuristic solver,
    not a detailed-balance Boltzmann sampler (the band gate and the irreversible
    reduction-repair move break detailed balance).

    Proposals: suffix-swap, task relocation, and a greedy vehicle-reduction repair that
    deletes one vehicle and reinserts its trips into the remaining duties. Late in the
    chain the reduction move gets a larger share of proposals so the search spends more
    effort eliminating redundant vehicles once a good region has been found.
    """
    rng_py = random.Random(seed)
    start = time.perf_counter()

    current_schedules = [list(s) for s in initial_schedules]
    current_columns = _columns_for_schedules(instance, current_schedules)
    if current_columns is None:
        raise ValueError("Initial schedule passed to MH chain is infeasible.")
    current_obj = _solution_objective(current_columns)
    max_obj = incumbent_obj * (1.0 + epsilon)
    best_schedules = [list(s) for s in current_schedules]
    best_columns = list(current_columns)
    best_obj = current_obj

    samples: List[List[ScheduleColumn]] = []
    objectives: List[float] = []
    n_accepted = 0

    for step in range(n_steps):
        late_stage = step >= max(1, n_steps // 2)
        r = rng_py.random()
        if late_stage:
            reduction_prob = min(max(float(late_reduction_prob), 0.0), 0.95)
            remaining = 1.0 - reduction_prob
            suffix_cut = 0.57 * remaining
            trip_cut = remaining
        else:
            reduction_prob = min(max(float(early_reduction_prob), 0.0), 0.95)
            remaining = 1.0 - reduction_prob
            suffix_cut = 0.60 * remaining
            trip_cut = remaining

        if r < suffix_cut:
            candidate_schedules: Optional[List[List[int]]] = _sa_suffix_swap(
                instance, current_schedules, rng_py
            )
        elif r < trip_cut:
            candidate_schedules = _sa_trip_move(current_schedules, rng_py)
        else:
            candidate_schedules = _vehicle_reduction_move(
                instance,
                current_schedules,
                current_columns,
                rng_py,
            )

        if candidate_schedules is not None:
            candidate_columns = _columns_for_schedules(instance, candidate_schedules)
            if candidate_columns is not None:
                candidate_obj = _solution_objective(candidate_columns)
                if candidate_obj <= max_obj:
                    delta = candidate_obj - current_obj
                    if delta <= 0 or rng_py.random() < math.exp(-delta / max(tau_mh, 1e-9)):
                        current_schedules = candidate_schedules
                        current_columns = candidate_columns
                        current_obj = candidate_obj
                        n_accepted += 1
                        if current_obj < best_obj:
                            best_obj = current_obj
                            best_schedules = [list(s) for s in current_schedules]
                            best_columns = list(current_columns)

        # Always record at thinned intervals — current state regardless of this step's outcome
        if step >= burn_in and (step - burn_in) % max(thin, 1) == 0:
            samples.append(list(current_columns))
            objectives.append(current_obj)

    return MHChainResult(
        samples=samples,
        objectives=objectives,
        n_steps=n_steps,
        n_accepted=n_accepted,
        acceptance_rate=n_accepted / max(n_steps, 1),
        incumbent_objective=incumbent_obj,
        runtime_sec=time.perf_counter() - start,
        final_schedules=[list(s) for s in current_schedules],
        best_objective=best_obj,
        best_schedules=best_schedules,
        best_columns=best_columns,
    )


# ---------------------------------------------------------------------------
# Descriptive ensemble statistics
# ---------------------------------------------------------------------------

def _trip_entropy_from_samples(
    samples: List[List[ScheduleColumn]],
    n_trips: int,
) -> Tuple[np.ndarray, Dict[Tuple[int, ...], float]]:
    """
    From the retained samples compute per-trip coverage entropy and column inclusion probs.

    Trip entropy = H(covering-column distribution for trip i across samples).
    Column inclusion probability = fraction of samples in which that column appears.
    These are descriptive summaries of the near-optimal schedules the chain visited.
    """
    n_samples = len(samples)
    if n_samples == 0:
        return np.zeros(n_trips), {}

    # trip_col_counts[i][signature] = count of samples where trip i was served by signature
    trip_col_counts: List[Dict[Tuple[int, ...], int]] = [{} for _ in range(n_trips)]
    col_usage: Dict[Tuple[int, ...], int] = {}

    for sched in samples:
        for col in sched:
            sig = col.trips
            col_usage[sig] = col_usage.get(sig, 0) + 1
            for trip_i in sig:
                trip_col_counts[trip_i][sig] = trip_col_counts[trip_i].get(sig, 0) + 1

    trip_entropy = np.zeros(n_trips)
    for i in range(n_trips):
        counts = np.array(list(trip_col_counts[i].values()), dtype=float)
        total = counts.sum()
        if total > 0:
            p = counts / total
            trip_entropy[i] = float(-np.sum(p * np.log(np.maximum(p, 1e-15))))

    col_probs = {sig: cnt / n_samples for sig, cnt in col_usage.items()}
    return trip_entropy, col_probs


@dataclass
class EnsembleCompressionResult:
    schedules: List[List[int]]
    columns: List[ScheduleColumn]
    objective: float
    n_vehicles: int
    n_rounds: int
    n_successes: int
    runtime_sec: float


def compress_schedule_with_ensemble(
    instance: QbuzzInstance,
    schedules: Sequence[Sequence[int]],
    column_inclusion_probs: Dict[Tuple[int, ...], float],
    trip_entropy: Optional[np.ndarray] = None,
    objective_cap: Optional[float] = None,
    backbone_threshold: float = 0.70,
    max_donor_trips: int = 12,
    max_rounds: int = 40,
    seed: int = 1234,
) -> EnsembleCompressionResult:
    """
    Try to reduce vehicle count by freezing high-probability backbone duties and
    repeatedly removing one redundant vehicle from the remaining flexible set.

    This is a post-processing step: it uses the ensemble signal from the retained samples
    (inclusion probabilities and sample-derived trip entropy) to decide which current
    duties are treated as hard constraints and which are candidates for deletion.
    """
    rng = random.Random(seed)
    start = time.perf_counter()

    current_schedules = [list(s) for s in schedules]
    current_columns = _columns_for_schedules(instance, current_schedules)
    if current_columns is None:
        raise ValueError("Compression started from an infeasible schedule.")
    current_obj = _solution_objective(current_columns)

    n_successes = 0
    for _ in range(max_rounds):
        if len(current_schedules) < 2:
            break

        donor_candidates = []
        for idx, (sched, col) in enumerate(zip(current_schedules, current_columns)):
            if not sched or len(sched) > max_donor_trips:
                continue
            exact_prob = float(column_inclusion_probs.get(col.trips, 0.0))
            if exact_prob >= backbone_threshold and len(current_schedules) > 2:
                continue
            mean_entropy = float(np.mean([trip_entropy[t] for t in sched])) if trip_entropy is not None else 0.0
            cost_per_trip = float(col.cost) / max(len(sched), 1)
            score = (
                1.75 * (1.0 - exact_prob)
                + 0.50 * mean_entropy
                + 0.25 * cost_per_trip / max(current_obj / max(len(current_columns), 1), 1.0)
                + 0.10 / max(len(sched), 1)
            )
            donor_candidates.append((score, len(sched), rng.random(), idx))

        if not donor_candidates:
            break

        donor_candidates.sort(key=lambda x: (-x[0], x[1], x[2]))

        repaired: Optional[List[List[int]]] = None
        for _, _, _, donor_idx in donor_candidates[: min(5, len(donor_candidates))]:
            repaired = _attempt_vehicle_absorption(
                instance=instance,
                schedules=current_schedules,
                columns=current_columns,
                donor_idx=donor_idx,
                rng=rng,
                max_donor_trips=max_donor_trips,
                max_attempts=max(12, max_rounds // 2),
                all_positions=True,
            )
            if repaired is not None:
                break

        if repaired is None:
            break

        repaired_columns = _columns_for_schedules(instance, repaired)
        if repaired_columns is None:
            continue
        repaired_obj = _solution_objective(repaired_columns)

        if objective_cap is not None and repaired_obj > objective_cap + 1e-9:
            continue
        if len(repaired_columns) < len(current_columns) or repaired_obj < current_obj - 1e-9:
            current_schedules = repaired
            current_columns = repaired_columns
            current_obj = repaired_obj
            n_successes += 1

    return EnsembleCompressionResult(
        schedules=current_schedules,
        columns=current_columns,
        objective=current_obj,
        n_vehicles=len(current_columns),
        n_rounds=max_rounds,
        n_successes=n_successes,
        runtime_sec=time.perf_counter() - start,
    )


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_vsp_projection_pmip(
    instance: QbuzzInstance,
    column_pool: Sequence[ScheduleColumn],
    n_mh_steps: int = 3000,
    tau_mh: float = 500.0,
    epsilon: float = 0.02,
    burn_in: int = 500,
    thin: int = 10,
    seed: int = 42,
    warmstart_columns: Optional[Sequence[ScheduleColumn]] = None,
    early_reduction_prob: float = 0.20,
    late_reduction_prob: float = 0.30,
) -> VspProjectionPmipResult:
    """
    Full Metropolis feasible-state search pipeline for VSP-as-set-partitioning.

    Parameters
    ----------
    tau_mh             : MH temperature in cost units (fixed; no cooling schedule)
    epsilon            : near-optimal band relative to incumbent (0.02 = 2% above)
    burn_in            : MH steps before collecting samples
    thin               : collect one sample every `thin` steps past burn-in
    warmstart_columns  : if given, use as MH starting schedule; overrides the greedy round
    """
    rng = np.random.default_rng(seed)
    n_trips = len(instance.trips)

    # Deduplicate column pool; always add singleton fallback columns
    pool = list(unique_columns(column_pool))
    singletons = build_singleton_columns(instance)

    # Build index and a greedy feasible round (fallback warm start)
    index = build_trip_cover_index(pool, n_trips)
    rd = round_to_partition(index, pool, singletons, rng)

    # Starting point for MH: warmstart takes priority over the greedy round
    if warmstart_columns is not None:
        mh_start_schedules = [list(col.trips) for col in warmstart_columns]
        incumbent_obj = float(sum(c.cost for c in warmstart_columns))
    else:
        mh_start_schedules = [list(col.trips) for col in rd.columns]
        incumbent_obj = rd.objective

    # Metropolis polish and sampling
    mh = run_vsp_mh_chain(
        instance=instance,
        initial_schedules=mh_start_schedules,
        n_steps=n_mh_steps,
        tau_mh=tau_mh,
        incumbent_obj=incumbent_obj,
        epsilon=epsilon,
        burn_in=burn_in,
        thin=thin,
        seed=int(rng.integers(0, 2**31)),
        early_reduction_prob=early_reduction_prob,
        late_reduction_prob=late_reduction_prob,
    )

    # Descriptive statistics from the retained near-optimal sample
    trip_entropy, col_probs = _trip_entropy_from_samples(mh.samples, n_trips)
    mean_ent = float(trip_entropy.mean())
    n_with_ent = int((trip_entropy > 0.01).sum())

    # Summary DataFrames
    trips = instance.trips
    trip_rows = [
        {
            "trip_idx": i,
            "from_stop": trips[i].from_stop,
            "to_stop": trips[i].to_stop,
            "start_min": trips[i].start,
            "end_min": trips[i].end,
            "n_pool_columns": len(index.trip_to_cols[i]),
            "coverage_entropy": float(trip_entropy[i]),
        }
        for i in range(n_trips)
    ]
    trip_summary = pd.DataFrame(trip_rows)

    sorted_cols = sorted(col_probs.items(), key=lambda kv: -kv[1])
    sample_rows = [
        {"column_trips": str(sig), "n_trips_in_col": len(sig), "inclusion_prob": prob}
        for sig, prob in sorted_cols
    ]
    sample_summary = pd.DataFrame(sample_rows)

    obj_rows = [
        {"sample_idx": k, "objective": obj}
        for k, obj in enumerate(mh.objectives)
    ]
    obj_summary = pd.DataFrame(obj_rows)

    return VspProjectionPmipResult(
        instance_name=instance.name,
        round=rd,
        mh=mh,
        trip_entropy=trip_entropy,
        mean_trip_entropy=mean_ent,
        n_trips_with_entropy=n_with_ent,
        column_inclusion_probs=col_probs,
        trip_summary=trip_summary,
        sample_summary=sample_summary,
        obj_summary=obj_summary,
    )
