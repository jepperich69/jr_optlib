# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import numpy as np
import pytest

JR = Path(r"C:\Users\rich\OneDrive - Danmarks Tekniske Universitet\JR")
OLD_CODE_DIR = JR / "Publikationer" / "Pub_PMIP_VSP" / "code"
sys.path.insert(0, str(OLD_CODE_DIR))

from pmip_vsp import qbuzz_vsp as old_qbuzz, vsp_projection as old_vsp
from jr_optlib.vsp import load_qbuzz_instance, randomized_greedy_solution, run_vsp_mh_chain, simulated_annealing_solution
from jr_optlib.oracles import certify_vsp_heuristic_chain

INSTANCE_DIR = JR / "Publikationer" / "Pub_PMIP_VSP" / "Literature" / "evsp-instances" / "gn12"

@pytest.fixture(scope="module")
def instance():
    return load_qbuzz_instance(INSTANCE_DIR)
    
@pytest.fixture(scope="module")
def old_instance():
    return old_qbuzz.load_qbuzz_instance(INSTANCE_DIR)

def test_randomized_greedy_matches_old(instance, old_instance):
    seed = 42
    old_sol = old_qbuzz.randomized_greedy_solution(old_instance, seed)
    new_sol = randomized_greedy_solution(instance, seed)
    
    assert old_sol.objective == new_sol.objective
    assert len(old_sol.columns) == len(new_sol.columns)
    for old_c, new_c in zip(old_sol.columns, new_sol.columns):
        assert old_c.trips == new_c.trips
        assert old_c.cost == new_c.cost

def test_mh_chain_matches_old_and_certifies(instance, old_instance):
    seed = 123
    
    # Generate warm starts
    old_warm = old_qbuzz.randomized_greedy_solution(old_instance, seed)
    new_warm = randomized_greedy_solution(instance, seed)
    
    old_mh = old_vsp.run_vsp_mh_chain(
        instance=old_instance,
        initial_schedules=[list(c.trips) for c in old_warm.columns],
        n_steps=100,
        tau_mh=500.0,
        incumbent_obj=old_warm.objective,
        epsilon=0.05,
        burn_in=10,
        thin=5,
        seed=seed
    )
    
    new_mh = run_vsp_mh_chain(
        instance=instance,
        initial_schedules=[list(c.trips) for c in new_warm.columns],
        n_steps=100,
        tau_mh=500.0,
        incumbent_obj=new_warm.objective,
        epsilon=0.05,
        burn_in=10,
        thin=5,
        seed=seed
    )
    
    assert old_mh.best_objective == new_mh.best_objective
    assert old_mh.n_accepted == new_mh.n_accepted
    assert np.allclose(old_mh.objectives, new_mh.objectives)
    
    # Certify the chain
    results = certify_vsp_heuristic_chain(
        instance=instance,
        initial_objective=new_warm.objective,
        best_objective=new_mh.best_objective,
        objectives=new_mh.objectives,
        samples=new_mh.samples,
        epsilon=0.05
    )
    
    assert all(r.passed for r in results)

def test_simulated_annealing_matches_old(instance, old_instance):
    seed = 55
    iters = 100
    
    old_sa = old_qbuzz.simulated_annealing_solution(old_instance, seed, iterations=iters, collect_every=10)
    new_sa = simulated_annealing_solution(instance, seed, iterations=iters, collect_every=10)
    
    assert old_sa.objective == new_sa.objective
    assert old_sa.accepted_moves == new_sa.accepted_moves
    assert len(old_sa.columns) == len(new_sa.columns)
    assert len(old_sa.collected_columns) == len(new_sa.collected_columns)
    
    for old_c, new_c in zip(old_sa.columns, new_sa.columns):
        assert old_c.trips == new_c.trips
        assert old_c.cost == new_c.cost
