"""Tests for maintkit.markov_chain: construction, propagation and simulation."""
import numpy as np
import pytest

from maintkit.markov_chain import MarkovChain
from maintkit.schedule import Schedule

P2 = np.array([[0.9, 0.1],
               [0.4, 0.6]])
P_DRY = np.array([[0.9, 0.1, 0.0],
                  [0.0, 0.8, 0.2],
                  [0.0, 0.0, 1.0]])
P_WET = np.array([[0.6, 0.3, 0.1],
                  [0.0, 0.5, 0.5],
                  [0.0, 0.0, 1.0]])


# ---- construction and validation -------------------------------------------

def test_plain_matrix_is_a_homogeneous_chain():
    chain = MarkovChain(P2)
    assert chain.is_homogeneous
    assert chain.n_states == 2 and chain.period == 1 and chain.length is None
    assert chain.states == (0, 1)


def test_schedule_input_keeps_its_period_or_length():
    assert MarkovChain(Schedule.periodic([P_DRY, P_WET])).period == 2
    assert MarkovChain(Schedule.finite([P_DRY, P_WET, P_DRY])).length == 3


def test_unwrapped_stack_refused():
    with pytest.raises(TypeError, match="periodic.*finite"):
        MarkovChain(np.stack([P_DRY, P_WET]))


def test_non_square_refused():
    with pytest.raises(ValueError, match="square"):
        MarkovChain(np.full((2, 3), 1 / 3))


def test_bad_row_sum_names_step_and_row():
    bad = P_WET.copy()
    bad[1, 1] = 0.4
    with pytest.raises(ValueError, match="row 1 .* step 1"):
        MarkovChain(Schedule.periodic([P_DRY, bad]))


def test_negative_entry_names_location():
    bad = np.array([[1.2, -0.2], [0.0, 1.0]])
    with pytest.raises(ValueError, match=r"step 0 .* P\[0, 1\]"):
        MarkovChain(bad)


def test_non_finite_refused():
    bad = np.array([[np.nan, 1.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="non-finite"):
        MarkovChain(bad)


def test_rounding_negatives_set_to_zero():
    p = np.array([[1.0 - 0.9 - 0.1, 1.0], [0.0, 1.0]])   # P[0,0] is about -2.8e-17
    assert p[0, 0] < 0
    chain = MarkovChain(p)
    assert chain.transitions[0][0, 0] == 0.0


def test_state_labels():
    chain = MarkovChain(P2, states=["up", "down"])
    assert chain.state_index("down") == 1
    with pytest.raises(KeyError, match="unknown state"):
        chain.state_index("sideways")


@pytest.mark.parametrize("states, msg", [(["a"], "2 states"), (["a", "a"], "unique")])
def test_bad_state_labels(states, msg):
    with pytest.raises(ValueError, match=msg):
        MarkovChain(P2, states=states)


# ---- distribution ----------------------------------------------------------

def test_distribution_homogeneous_matches_matrix_power():
    chain = MarkovChain(P2)
    pi = chain.distribution(0, horizon=5)
    assert pi.shape == (6, 2)
    for k in range(6):
        np.testing.assert_allclose(pi[k], np.linalg.matrix_power(P2, k)[0])


def test_distribution_periodic_alternates_matrices():
    chain = MarkovChain(Schedule.periodic([P_DRY, P_WET]))
    pi = chain.distribution(0, horizon=3)
    e0 = np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(pi[3], e0 @ P_DRY @ P_WET @ P_DRY)


def test_distribution_t0_selects_phase():
    chain = MarkovChain(Schedule.periodic([P_DRY, P_WET]))
    np.testing.assert_allclose(chain.distribution(0, horizon=1, t0=1)[1], P_WET[0])
    np.testing.assert_allclose(chain.distribution(0, horizon=1, t0=4)[1], P_DRY[0])


def test_distribution_accepts_label_and_vector():
    chain = MarkovChain(P2, states=["up", "down"])
    np.testing.assert_allclose(chain.distribution("down", 1)[1], P2[1])
    np.testing.assert_allclose(chain.distribution([0.5, 0.5], 1)[1], 0.5 * (P2[0] + P2[1]))


@pytest.mark.parametrize("pi0", [[0.5, 0.6], [1.0, 0.0, 0.0], [1.5, -0.5], 7])
def test_distribution_rejects_bad_initial(pi0):
    with pytest.raises(ValueError):
        MarkovChain(P2).distribution(pi0, 1)


def test_distribution_rows_stay_normalised():
    chain = MarkovChain(Schedule.periodic([P_DRY, P_WET]))
    np.testing.assert_allclose(chain.distribution(0, 50).sum(axis=1), 1.0)


def test_distribution_finite_schedule_horizon_checked():
    chain = MarkovChain(Schedule.finite([P_DRY, P_WET, P_DRY]))
    assert chain.distribution(0, horizon=3).shape == (4, 3)
    assert chain.distribution(0, horizon=1, t0=2).shape == (2, 3)
    with pytest.raises(ValueError, match="only defined for t < 3"):
        chain.distribution(0, horizon=2, t0=2)


def test_distribution_requires_finite_horizon():
    with pytest.raises(ValueError, match="finite horizon"):
        MarkovChain(P2).distribution(0, None)


def test_zero_horizon_returns_initial():
    np.testing.assert_array_equal(MarkovChain(P2).distribution(1, 0), [[0.0, 1.0]])


# ---- transition_matrix -----------------------------------------------------

def test_transition_matrix_is_ordered_product():
    chain = MarkovChain(Schedule.periodic([P_DRY, P_WET]))
    np.testing.assert_allclose(chain.transition_matrix(1, 4), P_WET @ P_DRY @ P_WET)


def test_transition_matrix_identity_and_rows():
    chain = MarkovChain(Schedule.periodic([P_DRY, P_WET]))
    np.testing.assert_array_equal(chain.transition_matrix(3, 3), np.eye(3))
    np.testing.assert_allclose(chain.transition_matrix(0, 7).sum(axis=1), 1.0)


def test_transition_matrix_agrees_with_distribution():
    chain = MarkovChain(Schedule.periodic([P_DRY, P_WET]))
    pi0 = np.array([0.2, 0.5, 0.3])
    np.testing.assert_allclose(chain.distribution(pi0, 5, t0=1)[-1],
                               pi0 @ chain.transition_matrix(1, 6))


def test_transition_matrix_order_and_range_checked():
    with pytest.raises(ValueError, match="t1 must be >= t0"):
        MarkovChain(P2).transition_matrix(3, 2)
    with pytest.raises(ValueError, match="only defined"):
        MarkovChain(Schedule.finite([P2, P2])).transition_matrix(0, 3)


# ---- simulate --------------------------------------------------------------

def test_simulate_shape_and_start():
    paths = MarkovChain(P2).simulate(1, horizon=10, n_paths=7, rng=0)
    assert paths.shape == (7, 11)
    assert np.all(paths[:, 0] == 1)
    assert paths.dtype.kind == "i"


def test_simulate_reproducible_with_seed():
    chain = MarkovChain(Schedule.periodic([P_DRY, P_WET]))
    np.testing.assert_array_equal(chain.simulate(0, 20, 5, rng=42), chain.simulate(0, 20, 5, rng=42))


def test_simulate_deterministic_periodic_chain_is_exact():
    swap = np.array([[0.0, 1.0], [1.0, 0.0]])
    stay = np.eye(2)
    paths = MarkovChain(Schedule.periodic([swap, stay])).simulate(0, 6, n_paths=3, rng=1)
    np.testing.assert_array_equal(paths, np.tile([0, 1, 1, 0, 0, 1, 1], (3, 1)))


def test_simulate_never_takes_zero_probability_transition():
    paths = MarkovChain(Schedule.periodic([P_DRY, P_WET])).simulate(0, 30, 2000, rng=3)
    assert np.all(np.diff(paths, axis=1) >= 0)          # degradation never improves
    failed = paths[:, :-1] == 2
    assert np.all(paths[:, 1:][failed] == 2)            # failure is absorbing


def test_simulate_frequencies_match_distribution():
    chain = MarkovChain(Schedule.periodic([P_DRY, P_WET]))
    n = 20000
    paths = chain.simulate(0, 6, n_paths=n, rng=7)
    exact = chain.distribution(0, 6)
    for k in (1, 3, 6):
        freq = np.bincount(paths[:, k], minlength=3) / n
        sd = np.sqrt(exact[k] * (1 - exact[k]) / n)
        assert np.all(np.abs(freq - exact[k]) <= 5 * sd + 1e-12)


def test_simulate_draws_initial_state_from_distribution():
    paths = MarkovChain(P2).simulate([0.25, 0.75], 0, n_paths=20000, rng=11)
    assert abs(paths[:, 0].mean() - 0.75) < 5 * np.sqrt(0.75 * 0.25 / 20000)


def test_simulate_finite_schedule_and_bad_n_paths():
    chain = MarkovChain(Schedule.finite([P2, P2]))
    assert chain.simulate(0, 2, rng=0).shape == (1, 3)
    with pytest.raises(ValueError, match="only defined"):
        chain.simulate(0, 3, rng=0)
    with pytest.raises(ValueError, match="n_paths"):
        chain.simulate(0, 1, n_paths=0)
