"""Tests for the constant-chain analysis in maintkit.markov_chain.

Checks are against closed-form results: the two-state chain, gambler's ruin,
a pure-birth degradation chain (sum of geometrics), and fundamental-matrix
identities.
"""
import numpy as np
import pytest

from maintkit.markov_chain import MarkovChain
from maintkit.time_indexed import TimeIndexed


def two_state(a=0.1, b=0.4):
    return np.array([[1 - a, a], [b, 1 - b]])


def gamblers_ruin(n, p):
    """States 0..n, absorbing at both ends, up with probability p."""
    P = np.zeros((n + 1, n + 1))
    P[0, 0] = P[n, n] = 1.0
    for i in range(1, n):
        P[i, i + 1] = p
        P[i, i - 1] = 1 - p
    return P


def degradation(stay):
    """Pure-birth chain: stay with stay[i], else move to i+1; last state absorbing."""
    n = len(stay) + 1
    P = np.zeros((n, n))
    for i, s in enumerate(stay):
        P[i, i] = s
        P[i, i + 1] = 1 - s
    P[-1, -1] = 1.0
    return P


def reducible():
    """0,1 transient; {2} absorbing; {3,4} a closed class with period 2."""
    return np.array([
        [0.5, 0.3, 0.2, 0.0, 0.0],
        [0.2, 0.4, 0.0, 0.4, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
    ])


def random_irreducible(n, seed):
    rng = np.random.default_rng(seed)
    P = rng.random((n, n)) + 0.01
    return P / P.sum(axis=1, keepdims=True)


# ---- classes -------------------------------------------------------------------

def test_classes_of_reducible_chain():
    chain = MarkovChain(reducible())
    assert [c.tolist() for c in chain.communicating_classes()] == [[0, 1], [2], [3, 4]]
    assert [c.tolist() for c in chain.closed_classes()] == [[2], [3, 4]]
    assert chain.transient_states().tolist() == [0, 1]
    assert chain.absorbing_states().tolist() == [2]
    assert not chain.is_irreducible()


def test_irreducible_chain_has_one_class():
    chain = MarkovChain(two_state())
    assert chain.is_irreducible()
    assert chain.transient_states().size == 0
    assert chain.absorbing_states().size == 0


def test_tiny_positive_entry_still_connects():
    P = np.array([[1 - 1e-15, 1e-15], [0.5, 0.5]])
    assert MarkovChain(P).is_irreducible()


# ---- period --------------------------------------------------------------------

def test_period_of_cycle_and_aperiodic_chain():
    cycle = np.roll(np.eye(3), 1, axis=1)               # 0 -> 1 -> 2 -> 0
    assert MarkovChain(cycle).state_period() == 3
    assert MarkovChain(two_state()).state_period() == 1


def test_period_gcd_of_cycle_lengths():
    # cycles of length 2 (0-1-0) and 4 (0-1-2-3-0) through state 0: period 2
    P = np.zeros((4, 4))
    P[0, 1] = 1.0
    P[1, 0] = P[1, 2] = 0.5
    P[2, 3] = 1.0
    P[3, 0] = 1.0
    assert MarkovChain(P).state_period() == 2


def test_period_in_reducible_chain_needs_a_state():
    chain = MarkovChain(reducible())
    with pytest.raises(ValueError, match="reducible"):
        chain.state_period()
    assert chain.state_period(3) == 2
    assert chain.state_period(2) == 1                    # absorbing: self-loop
    assert chain.state_period(0) == 1                    # 0 has a self-loop


def test_period_none_for_state_that_never_returns():
    P = np.array([[0.0, 1.0], [0.0, 1.0]])
    assert MarkovChain(P).state_period(0) is None


def test_period_accepts_labels():
    chain = MarkovChain(reducible(), states=list("abcde"))
    assert chain.state_period("d") == 2


# ---- stationary distributions --------------------------------------------------

def test_two_state_stationary_closed_form():
    a, b = 0.1, 0.4
    np.testing.assert_allclose(MarkovChain(two_state(a, b)).stationary_distribution(),
                               [b / (a + b), a / (a + b)])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_stationary_is_left_eigenvector(seed):
    P = random_irreducible(6, seed)
    pi = MarkovChain(P).stationary_distribution()
    np.testing.assert_allclose(pi @ P, pi, atol=1e-14)
    assert abs(pi.sum() - 1) < 1e-14 and np.all(pi > 0)


def test_stationary_accurate_for_nearly_decomposable_chain():
    # Two 2-state blocks coupled by eps both ways; by symmetry pi is uniform.
    eps = 1e-13
    P = np.array([
        [0.5 - eps, 0.5, eps, 0.0],
        [0.5, 0.5 - eps, 0.0, eps],
        [eps, 0.0, 0.5 - eps, 0.5],
        [0.0, eps, 0.5, 0.5 - eps],
    ])
    np.testing.assert_allclose(MarkovChain(P).stationary_distribution(), 0.25, rtol=1e-12)


def test_reducible_stationary_distributions_one_per_closed_class():
    chain = MarkovChain(reducible())
    pis = chain.stationary_distributions()
    np.testing.assert_allclose(pis, [[0, 0, 1, 0, 0], [0, 0, 0, 0.5, 0.5]])
    P = reducible()
    np.testing.assert_allclose(pis @ P, pis)
    with pytest.raises(ValueError, match="not unique"):
        chain.stationary_distribution()


def test_absorbing_chain_stationary_is_point_mass():
    np.testing.assert_allclose(MarkovChain(degradation([0.9, 0.8])).stationary_distribution(),
                               [0, 0, 1])


# ---- absorption ----------------------------------------------------------------

def test_degradation_mttf_and_variance_are_sums_of_geometrics():
    stay = np.array([0.9, 0.8, 0.95])
    res = MarkovChain(degradation(stay)).absorption()
    q = 1 - stay
    for i in range(3):
        np.testing.assert_allclose(res.expected_steps[i], np.sum(1 / q[i:]))
        np.testing.assert_allclose(res.variance_steps[i], np.sum(stay[i:] / q[i:] ** 2))
    assert res.expected_steps[3] == 0 and res.variance_steps[3] == 0


@pytest.mark.parametrize("p", [0.3, 0.5, 0.6])
def test_gamblers_ruin_probabilities_and_duration(p):
    n = 6
    res = MarkovChain(gamblers_ruin(n, p)).absorption()
    i = np.arange(n + 1)
    if p == 0.5:
        win = i / n
        duration = i * (n - i)
    else:
        r = (1 - p) / p
        win = (1 - r ** i) / (1 - r ** n)
        duration = i / (1 - 2 * p) - n / (1 - 2 * p) * win
    # closed classes are {0} then {n}
    np.testing.assert_allclose(res.class_probabilities[:, 1], win, atol=1e-12)
    np.testing.assert_allclose(res.class_probabilities.sum(axis=1), 1.0)
    np.testing.assert_allclose(res.expected_steps, duration, atol=1e-10)


def test_fundamental_matrix_counts_visits():
    res = MarkovChain(reducible()).absorption()
    Q = reducible()[np.ix_(res.transient, res.transient)]
    np.testing.assert_allclose(res.fundamental @ (np.eye(2) - Q), np.eye(2), atol=1e-14)
    assert np.all(np.diag(res.fundamental) >= 1)


def test_absorption_matches_simulation():
    P = reducible()
    chain = MarkovChain(P)
    res = chain.absorption()
    paths = chain.simulate(0, horizon=200, n_paths=20000, rng=5)
    absorbed = np.isin(paths, [2, 3, 4])
    steps = absorbed.argmax(axis=1)
    assert absorbed[:, -1].all()
    se = np.sqrt(res.variance_steps[0] / len(steps))
    assert abs(steps.mean() - res.expected_steps[0]) < 5 * se
    p2 = np.mean(paths[:, -1] == 2)
    assert abs(p2 - res.class_probabilities[0, 0]) < 5 * np.sqrt(p2 * (1 - p2) / len(steps))


def test_irreducible_chain_absorption_is_trivial():
    res = MarkovChain(two_state()).absorption()
    assert res.transient.size == 0 and res.fundamental.shape == (0, 0)
    np.testing.assert_array_equal(res.expected_steps, 0)


# ---- hitting -------------------------------------------------------------------

def test_hitting_time_of_repairable_failure_equals_absorbing_mttf():
    stay = [0.9, 0.8, 0.95]
    absorbing = degradation(stay)
    repairable = absorbing.copy()
    repairable[-1] = [0.7, 0, 0, 0.3]                # repair back to new
    k = MarkovChain(repairable).expected_hitting_times(3)
    np.testing.assert_allclose(k, MarkovChain(absorbing).absorption().expected_steps)


def test_hitting_probabilities_gamblers_ruin():
    p, n = 0.4, 5
    h = MarkovChain(gamblers_ruin(n, p)).hitting_probabilities(n)
    r = (1 - p) / p
    i = np.arange(n + 1)
    np.testing.assert_allclose(h, (1 - r ** i) / (1 - r ** n), atol=1e-12)


def test_hitting_time_infinite_where_not_certain():
    k = MarkovChain(gamblers_ruin(4, 0.5)).expected_hitting_times(4)
    assert k[4] == 0
    assert np.all(np.isinf(k[:4]))                    # ruin is possible from every other state


def test_hitting_unreachable_target():
    chain = MarkovChain(reducible())
    np.testing.assert_allclose(chain.hitting_probabilities(0), [1, 1 / 3, 0, 0, 0], atol=1e-12)
    assert np.isinf(chain.expected_hitting_times(0)[[1, 2, 3, 4]]).all()


def test_hitting_set_target_and_labels():
    chain = MarkovChain(reducible(), states=list("abcde"))
    np.testing.assert_allclose(chain.hitting_probabilities(["c", "d", "e"]), 1.0)
    k = chain.expected_hitting_times(["c", "d", "e"])
    np.testing.assert_allclose(k, MarkovChain(reducible()).absorption().expected_steps)
    with pytest.raises(KeyError):
        chain.hitting_probabilities(["c", "z"])


def test_mean_first_passage_two_state_and_consistency():
    a, b = 0.1, 0.4
    chain = MarkovChain(two_state(a, b))
    M = chain.mean_first_passage_times()
    np.testing.assert_allclose(M, [[(a + b) / b, 1 / a], [1 / b, (a + b) / a]])

    chain = MarkovChain(random_irreducible(5, 3))
    M = chain.mean_first_passage_times()
    for j in range(5):
        k = chain.expected_hitting_times(j)
        off = np.arange(5) != j
        np.testing.assert_allclose(M[off, j], k[off], rtol=1e-10)
    np.testing.assert_allclose(np.diag(M), 1 / chain.stationary_distribution())


def test_mean_first_passage_needs_irreducible():
    with pytest.raises(ValueError, match="irreducible"):
        MarkovChain(reducible()).mean_first_passage_times()


# ---- non-constant chains refused ------------------------------------------------

@pytest.mark.parametrize("method, args", [
    ("communicating_classes", ()), ("closed_classes", ()), ("state_period", (0,)),
    ("stationary_distribution", ()), ("absorption", ()), ("hitting_probabilities", (1,)),
    ("expected_hitting_times", (1,)), ("mean_first_passage_times", ()),
])
def test_analysis_refuses_time_varying_chain(method, args):
    chain = MarkovChain(TimeIndexed.periodic([two_state(), two_state(0.2, 0.3)]))
    with pytest.raises(ValueError, match="constant chain only"):
        getattr(chain, method)(*args)


def test_periodic_sequence_of_one_matrix_counts_as_constant():
    chain = MarkovChain(TimeIndexed.periodic([two_state()]))
    np.testing.assert_allclose(chain.stationary_distribution(), [0.8, 0.2])
