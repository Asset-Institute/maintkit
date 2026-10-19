"""Tests for maintkit.mdp: model construction, finite-horizon solve, evaluation, induced chains."""
import itertools

import numpy as np
import pytest

from maintkit.mdp import MDP
from maintkit.time_indexed import TimeIndexed


def random_mdp_arrays(S, A, seed, zero_frac=0.3):
    rng = np.random.default_rng(seed)
    P = rng.random((A, S, S)) * (rng.random((A, S, S)) > zero_frac)
    P[:, np.arange(S), np.arange(S)] += 0.05           # no all-zero rows
    P /= P.sum(axis=2, keepdims=True)
    C = rng.random((S, A)) * 10
    return P, C


def maintenance_mdp(cost_replace=5.0, cost_fail=50.0):
    """States 0 (new) .. 3 (failed); actions 0 = continue, 1 = replace.

    Continue degrades stochastically and costs cost_fail if in the failed
    state; replace returns to new. Replace is not admissible in state 0.
    """
    cont = np.array([
        [0.7, 0.3, 0.0, 0.0],
        [0.0, 0.6, 0.3, 0.1],
        [0.0, 0.0, 0.5, 0.5],
        [0.0, 0.0, 0.0, 1.0],
    ])
    replace = np.zeros((4, 4))
    replace[1:, 0] = 1.0                                # row 0 left as zeros: inadmissible
    P = np.stack([cont, replace])
    C = np.array([[0.0, np.nan], [0.0, cost_replace], [0.0, cost_replace],
                  [cost_fail, cost_replace + 10.0]])
    admissible = np.array([[True, False], [True, True], [True, True], [True, True]])
    return MDP(P, C, admissible, states=["new", "worn", "bad", "failed"],
               actions=["continue", "replace"])


def brute_force(mdp, horizon, discount, terminal, t0=0):
    """Best value per initial state over every deterministic Markov policy."""
    S, A = mdp.n_states, mdp.n_actions
    best = np.full(S, np.inf)
    for flat in itertools.product(range(A), repeat=S * horizon):
        rules = np.array(flat).reshape(horizon, S)
        try:
            v = mdp.evaluate(TimeIndexed.finite(rules), horizon, discount, terminal, t0)
        except ValueError:                              # inadmissible somewhere
            continue
        best = np.minimum(best, v[0])
    return best


# ---- construction ----------------------------------------------------------

def test_basic_properties():
    P, C = random_mdp_arrays(3, 2, 0)
    mdp = MDP(P, C)
    assert (mdp.n_states, mdp.n_actions) == (3, 2)
    assert mdp.period == 1 and mdp.length is None
    assert mdp.states == (0, 1, 2) and mdp.actions == (0, 1)


def test_joint_period_and_length():
    P, C = random_mdp_arrays(3, 2, 0)
    P2, C2 = random_mdp_arrays(3, 2, 1)
    assert MDP(TimeIndexed.periodic([P, P2]), TimeIndexed.periodic([C, C2, C])).period == 6
    assert MDP(TimeIndexed.periodic([P, P2]), TimeIndexed.finite([C] * 5)).length == 5


def test_unwrapped_stack_refused():
    P, C = random_mdp_arrays(3, 2, 0)
    with pytest.raises(TypeError, match="periodic.*finite"):
        MDP(np.stack([P, P]), C)


def test_bad_row_names_action_state_and_step():
    P, C = random_mdp_arrays(3, 2, 0)
    bad = P.copy()
    bad[1, 2, 0] += 0.2
    with pytest.raises(ValueError, match="action 1, state 2 at step 1"):
        MDP(TimeIndexed.periodic([P, bad]), C)


def test_shape_mismatches():
    P, C = random_mdp_arrays(3, 2, 0)
    with pytest.raises(ValueError, match="costs must have shape"):
        MDP(P, C.T)
    with pytest.raises(ValueError, match=r"\(A, S, S\)"):
        MDP(np.ones((2, 3, 4)) / 4, C)
    with pytest.raises(ValueError, match="admissible must have shape"):
        MDP(P, C, np.ones((2, 3), dtype=bool))


def test_admissible_must_be_boolean_and_leave_an_action():
    P, C = random_mdp_arrays(3, 2, 0)
    with pytest.raises(TypeError, match="boolean"):
        MDP(P, C, np.ones((3, 2)))
    mask = np.ones((3, 2), dtype=bool)
    mask[1] = False
    with pytest.raises(ValueError, match="state 1 has no admissible action"):
        MDP(P, C, mask)


def test_inadmissible_rows_and_costs_are_not_validated():
    mdp = maintenance_mdp()                             # zero row and nan cost for replace in "new"
    assert mdp.n_states == 4


def test_non_finite_cost_on_admissible_pair_refused():
    P, C = random_mdp_arrays(3, 2, 0)
    C[0, 1] = np.inf
    with pytest.raises(ValueError, match="not finite"):
        MDP(P, C)


def test_per_transition_costs_reduce_to_expected():
    P, _ = random_mdp_arrays(3, 2, 4)
    c3 = np.random.default_rng(9).random((2, 3, 3))
    c3[P == 0] = np.nan                                 # impossible transitions may be undefined
    mdp = MDP(P, c3)
    expected = np.nansum(P * c3, axis=2).T
    np.testing.assert_allclose(mdp.costs[0], expected)


def test_sense_validated():
    P, C = random_mdp_arrays(3, 2, 0)
    with pytest.raises(ValueError, match="sense"):
        MDP(P, C, sense="minimise")


# ---- solve -----------------------------------------------------------------

@pytest.mark.parametrize("seed, discount", [(0, 1.0), (1, 0.9), (2, 0.5)])
def test_solve_matches_brute_force_constant(seed, discount):
    P, C = random_mdp_arrays(3, 2, seed)
    mdp = MDP(P, C)
    terminal = np.array([1.0, -2.0, 4.0])
    sol = mdp.solve(3, discount, terminal)
    np.testing.assert_allclose(sol.values[0], brute_force(mdp, 3, discount, terminal))


@pytest.mark.parametrize("t0", [0, 1, 2])
def test_solve_matches_brute_force_periodic(t0):
    P0, C0 = random_mdp_arrays(3, 2, 10)
    P1, C1 = random_mdp_arrays(3, 2, 11)
    mdp = MDP(TimeIndexed.periodic([P0, P1]), TimeIndexed.periodic([C0, C1, C0 + 1]))
    sol = mdp.solve(3, 0.95, t0=t0)
    np.testing.assert_allclose(sol.values[0], brute_force(mdp, 3, 0.95, None, t0))


def test_solve_with_admissibility_matches_brute_force():
    mdp = maintenance_mdp()
    sol = mdp.solve(3, 1.0)
    np.testing.assert_allclose(sol.values[0], brute_force(mdp, 3, 1.0, None))
    assert np.all(sol.policy.window(0, 3)[:, 0] == 0)   # never "replace" a new unit


def test_evaluating_optimal_policy_reproduces_values():
    P, C = random_mdp_arrays(4, 3, 5)
    mdp = MDP(TimeIndexed.periodic([P, P[::-1]]), C)
    sol = mdp.solve(6, 0.9, np.arange(4.0), t0=1)
    np.testing.assert_allclose(mdp.evaluate(sol.policy, 6, 0.9, np.arange(4.0), t0=1), sol.values)


def test_q_values_consistent_with_values_and_policy():
    mdp = maintenance_mdp()
    sol = mdp.solve(5)
    assert sol.q_values.shape == (5, 4, 2)
    np.testing.assert_allclose(sol.q_values.min(axis=2), sol.values[:5])
    assert np.isinf(sol.q_values[:, 0, 1]).all()        # inadmissible marked inf


def test_maximise_is_minimise_of_negated():
    P, C = random_mdp_arrays(3, 2, 6)
    lo = MDP(P, C).solve(4, 0.9, np.ones(3))
    hi = MDP(P, -C, sense="max").solve(4, 0.9, -np.ones(3))
    np.testing.assert_allclose(hi.values, -lo.values)
    np.testing.assert_array_equal(hi.policy.steps, lo.policy.steps)


def test_zero_discount_is_myopic():
    P, C = random_mdp_arrays(3, 2, 7)
    sol = MDP(P, C).solve(2, discount=0.0)
    np.testing.assert_allclose(sol.values[0], C.min(axis=1))


def test_zero_horizon_returns_terminal():
    P, C = random_mdp_arrays(3, 2, 0)
    sol = MDP(P, C).solve(0, terminal=[1.0, 2.0, 3.0])
    np.testing.assert_array_equal(sol.values, [[1.0, 2.0, 3.0]])
    assert sol.policy is None


def test_ties_go_to_lowest_action():
    P = np.stack([np.eye(2), np.eye(2)])
    sol = MDP(P, np.zeros((2, 2))).solve(3)
    np.testing.assert_array_equal(sol.policy.steps, 0)


def test_solve_horizon_checks():
    P, C = random_mdp_arrays(3, 2, 0)
    mdp = MDP(TimeIndexed.finite([P, P, P]), C)
    assert mdp.solve(2, t0=1).horizon == 2
    with pytest.raises(ValueError, match="transitions"):
        mdp.solve(3, t0=1)
    with pytest.raises(ValueError, match="not implemented"):
        MDP(P, C).solve(None)
    with pytest.raises(ValueError, match="discount"):
        MDP(P, C).solve(2, discount=1.5)


def test_solution_policy_is_finite_time_indexed():
    sol = maintenance_mdp().solve(4)
    assert sol.policy.length == 4 and sol.policy.shape == (4,)
    assert sol.policy.dtype.kind == "i"


def test_maintenance_policy_is_a_control_limit():
    # Long horizon, so the first decision rule is close to stationary: once
    # replacement is worth it, it stays worth it in every worse state.
    rule = maintenance_mdp().solve(50, 0.95).policy[0]
    assert rule[0] == 0
    switch = np.argmax(rule == 1)
    assert switch > 0 and np.all(rule[switch:] == 1)


# ---- evaluate, policies ------------------------------------------------------

def test_evaluate_stationary_policy_from_plain_array():
    P, C = random_mdp_arrays(3, 2, 8)
    mdp = MDP(P, C)
    d = np.array([1, 0, 1])
    v = mdp.evaluate(d, 2, 0.9)
    rows = np.arange(3)
    Pd, cd = P[d, rows], C[rows, d]
    np.testing.assert_allclose(v[0], cd + 0.9 * Pd @ cd)


def test_evaluate_rejects_bad_policies():
    mdp = maintenance_mdp()
    with pytest.raises(ValueError, match="not admissible"):
        mdp.evaluate(np.array([1, 0, 0, 0]), 2)
    with pytest.raises(ValueError, match="action indices"):
        mdp.evaluate(np.array([0, 0, 0, 2]), 2)
    with pytest.raises(TypeError, match="integer"):
        mdp.evaluate(np.zeros(4), 2)
    with pytest.raises(ValueError, match="only defined"):
        mdp.evaluate(TimeIndexed.finite([np.zeros(4, dtype=int)] * 2), 3)


def test_policy_costs():
    mdp = maintenance_mdp()
    c = mdp.policy_costs(np.array([0, 1, 1, 1]))
    np.testing.assert_allclose(c[0], [0.0, 5.0, 5.0, 15.0])


# ---- induced chain ----------------------------------------------------------

def test_induced_chain_rows_come_from_chosen_actions():
    mdp = maintenance_mdp()
    d = np.array([0, 0, 1, 1])
    chain = mdp.induced_chain(d)
    P = mdp.transitions[0]
    np.testing.assert_allclose(chain.transitions[0], np.stack([P[di, i] for i, di in enumerate(d)]))
    assert chain.states == mdp.states and chain.is_homogeneous


def test_induced_chain_period_combines():
    P0, C = random_mdp_arrays(3, 2, 1)
    P1, _ = random_mdp_arrays(3, 2, 2)
    mdp = MDP(TimeIndexed.periodic([P0, P1]), C)
    policy = TimeIndexed.periodic([np.array([0, 1, 0]), np.array([1, 1, 0]), np.array([0, 0, 1])])
    chain = mdp.induced_chain(policy, t0=1)
    assert chain.period == 6
    np.testing.assert_allclose(chain.transitions[0], P1[[0, 1, 0], [0, 1, 2]])   # model time 1
    finite = mdp.induced_chain(mdp.solve(4).policy)
    assert finite.length == 4


def test_induced_chain_rejects_inadmissible_policy():
    with pytest.raises(ValueError, match="not admissible"):
        maintenance_mdp().induced_chain(np.array([1, 1, 1, 1]))


def test_simulated_cost_matches_evaluation():
    mdp = maintenance_mdp()
    sol = mdp.solve(8, 1.0)
    chain = mdp.induced_chain(sol.policy)
    paths = chain.simulate("worn", 8, n_paths=20000, rng=2)
    costs = mdp.policy_costs(sol.policy)
    totals = sum(costs[k][paths[:, k]] for k in range(8))
    se = totals.std() / np.sqrt(len(totals))
    assert abs(totals.mean() - sol.values[0, 1]) < 5 * se


def test_long_run_behaviour_of_optimal_stationary_rule():
    # The chain under the near-stationary rule is a repairable degradation
    # chain: irreducible, so availability = 1 - pi(failed) is well defined.
    mdp = maintenance_mdp()
    rule = mdp.solve(50, 0.95).policy[0]
    chain = mdp.induced_chain(rule)
    assert chain.is_irreducible()
    pi = chain.stationary_distribution()
    assert abs(pi.sum() - 1) < 1e-12 and pi[3] < 0.2
