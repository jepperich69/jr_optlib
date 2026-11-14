"""Oracle and differential tests for route-free recursive logit."""

import numpy as np

from jr_optlib.optimization import recursive_logit_assignment
from jr_optlib.oracles import certify_recursive_logit


def test_dag_matches_explicit_path_distribution():
    # 0->1->3 and 0->2->3; q=1/2 at node 0 and one thereafter.
    tails = np.array([0, 1, 0, 2])
    heads = np.array([1, 3, 2, 3])
    costs = np.array([1.0, 2.0, 2.0, 2.0])
    tau = 2.5
    result = recursive_logit_assignment(
        range(4), tails, heads, costs, [0], [3], np.array([10.0]), tau
    )
    route_weights = np.array([
        0.5 * np.exp(-3.0 / tau),
        0.5 * np.exp(-4.0 / tau),
    ])
    route_prob = route_weights / route_weights.sum()
    expected = np.array([
        10 * route_prob[0], 10 * route_prob[0],
        10 * route_prob[1], 10 * route_prob[1],
    ])
    assert np.allclose(result.edge_flows[0], expected, atol=1e-12)
    expected_cost = -tau * np.log(route_weights.sum())
    assert np.isclose(result.od_cost[0], expected_cost, atol=1e-12)


def test_cycle_is_solved_and_certified():
    tails = np.array([0, 0, 1, 1, 2])
    heads = np.array([1, 2, 0, 2, 3])
    costs = np.array([1.0, 4.0, 1.0, 2.0, 1.0])
    result = recursive_logit_assignment(
        range(4), tails, heads, costs, [0], [3], np.array([7.0]), tau=1.5
    )
    checks, certified = certify_recursive_logit(
        result, tails, heads, costs, np.array([0]), np.array([3]),
        np.array([7.0]), tau=1.5,
    )
    assert certified, [(x.name, x.residual) for x in checks]
    assert result.spectral_radius[0] < 1.0
    assert result.edge_flows[0, 0] > 0


def test_multiple_od_conservation():
    tails = np.array([0, 1, 0, 2, 1, 2])
    heads = np.array([1, 3, 2, 3, 2, 1])
    costs = np.ones(6)
    origins = np.array([0, 1])
    destinations = np.array([3, 3])
    demand = np.array([11.0, 5.0])
    result = recursive_logit_assignment(
        range(4), tails, heads, costs, origins, destinations, demand, tau=2.0
    )
    _, certified = certify_recursive_logit(
        result, tails, heads, costs, origins, destinations, demand, tau=2.0
    )
    assert certified


def test_pruned_subprobability_prior_preserves_original_utilities():
    tails = np.array([0, 0, 1, 2])
    heads = np.array([1, 2, 3, 3])
    costs = np.array([1.0, 4.0, 1.0, 1.0])
    full = recursive_logit_assignment(
        range(4), tails, heads, costs, [0], [3], np.array([1.0]), tau=2.0
    )
    # Remove the costly 0->2 path but keep q(0->1)=1/2, rather than
    # renormalising it to one. The pruned logsum cost must rise.
    keep = np.array([True, False, True, False])
    pruned = recursive_logit_assignment(
        range(4), tails[keep], heads[keep], costs[keep], [0], [3],
        np.array([1.0]), tau=2.0, prior=np.array([0.5, 1.0]),
    )
    assert pruned.od_cost[0] > full.od_cost[0]
