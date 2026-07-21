"""A distribution built from an arbitrary hazard function.

This class could not be evaluated at all before: the single-point branch passed
an array to quad, the multi-point branch never integrated from 0 to the first
point, and the cache compared n points against n-1 stored ones so the second
call raised. Nothing tested it.

The checks lean on two hazards with closed-form answers -- a constant hazard is
the exponential, a linear one is a Weibull with beta = 2 -- so the results are
compared against exact values rather than against the implementation.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from maintkit.distributions import ReliabilityFromHazard


RATE = 0.02          # constant hazard -> exponential with mean 1/RATE
ETA = 50.0           # linear hazard 2t/eta^2 -> Weibull(beta=2, scale=eta)


@pytest.fixture
def exponential_like():
    return ReliabilityFromHazard(lambda t: RATE)


@pytest.fixture
def weibull_like():
    return ReliabilityFromHazard(lambda t: 2.0 * t / ETA**2)


T = np.array([10.0, 20.0, 30.0, 40.0])


# ------------------------------------------------ against closed forms ----
def test_constant_hazard_is_the_exponential_cdf(exponential_like):
    """The bug that made this wrong: F(t[0]) was 0 and every value was short
    by the integral over (0, t[0])."""
    np.testing.assert_allclose(
        exponential_like.cdf(T), stats.expon.cdf(T, scale=1 / RATE), rtol=1e-8
    )


def test_constant_hazard_is_the_exponential_reliability(exponential_like):
    np.testing.assert_allclose(
        exponential_like.reliability(T), np.exp(-RATE * T), rtol=1e-8
    )


def test_constant_hazard_is_the_exponential_density(exponential_like):
    """_pdf is now analytic. It used to be a numerical derivative of _cdf,
    because only _cdf was defined."""
    np.testing.assert_allclose(
        exponential_like.pdf(T), RATE * np.exp(-RATE * T), rtol=1e-6
    )


def test_linear_hazard_is_a_weibull(weibull_like):
    np.testing.assert_allclose(
        weibull_like.cdf(T), stats.weibull_min.cdf(T, 2.0, scale=ETA), rtol=1e-8
    )
    np.testing.assert_allclose(
        weibull_like.pdf(T), stats.weibull_min.pdf(T, 2.0, scale=ETA), rtol=1e-6
    )


def test_recovers_the_hazard_it_was_given(weibull_like):
    """f/R must return h, which ties _pdf and _sf back to the input."""
    np.testing.assert_allclose(
        weibull_like.hazard(T), 2.0 * T / ETA**2, rtol=1e-6
    )


# ---------------------------------------------------------- the bugs ----
def test_a_single_point_works(exponential_like):
    """cdf_single(0, t) passed the whole array to quad, so this raised
    TypeError: only 0-dimensional arrays can be converted to Python scalars."""
    assert exponential_like.cdf(25.0) == pytest.approx(
        stats.expon.cdf(25.0, scale=1 / RATE), rel=1e-8
    )


def test_calling_twice_gives_the_same_answer(exponential_like):
    """The cache compared n points against n-1 stored, so the second call
    raised ValueError on the broadcast."""
    first = exponential_like.cdf(T)
    second = exponential_like.cdf(T)
    np.testing.assert_array_equal(first, second)


def test_evaluating_does_not_change_the_object(exponential_like):
    """It used to store the result on self and read it back, so a read
    mutated the model and the answer depended on the call history."""
    before = dict(exponential_like.__dict__)
    exponential_like.cdf(T)
    exponential_like.pdf(T)
    exponential_like.reliability(T)
    assert set(exponential_like.__dict__) == set(before)
    assert not hasattr(exponential_like, "cumulative_hazard")


def test_a_different_grid_does_not_disturb_a_later_one(exponential_like):
    """With a cache keyed on the last grid, this is where wrong numbers came
    from: the second call reused the first grid's values."""
    exponential_like.cdf(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(
        exponential_like.cdf(T), stats.expon.cdf(T, scale=1 / RATE), rtol=1e-8
    )


def test_unsorted_input_is_handled(exponential_like):
    shuffled = T[[2, 0, 3, 1]]
    np.testing.assert_allclose(
        exponential_like.cdf(shuffled),
        stats.expon.cdf(shuffled, scale=1 / RATE),
        rtol=1e-8,
    )


def test_repeated_points_are_handled(exponential_like):
    t = np.array([10.0, 10.0, 20.0])
    np.testing.assert_allclose(
        exponential_like.cdf(t), stats.expon.cdf(t, scale=1 / RATE), rtol=1e-8
    )


# ---------------------------------------------------------- properties ----
def test_cdf_is_non_decreasing_and_bounded(weibull_like):
    t = np.linspace(0.5, 200.0, 60)
    F = weibull_like.cdf(t)
    assert np.all(np.diff(F) >= -1e-12)
    assert np.all((F >= 0) & (F <= 1))


def test_reliability_and_cdf_sum_to_one(weibull_like):
    t = np.linspace(1.0, 150.0, 25)
    np.testing.assert_allclose(
        weibull_like.cdf(t) + weibull_like.reliability(t), 1.0, rtol=1e-10
    )


def test_log_reliability_is_exact_in_the_tail(exponential_like):
    """-H(t) directly, so it stays accurate where 1 - cdf has lost every
    digit. At these times the reliability underflows the difference."""
    t = np.array([2000.0, 5000.0])
    np.testing.assert_allclose(
        exponential_like.log_reliability(t), -RATE * t, rtol=1e-8
    )
    assert np.all(np.isfinite(exponential_like.log_reliability(t)))


def test_density_integrates_to_the_cdf(weibull_like):
    grid = np.linspace(1e-6, 200.0, 4001)
    integral = np.trapezoid(weibull_like.pdf(grid), grid)
    assert integral == pytest.approx(weibull_like.cdf(200.0), abs=1e-4)


def test_support_starts_at_zero(exponential_like):
    assert exponential_like.a == 0


def test_can_be_frozen_and_evaluated(exponential_like):
    """Freezing worked, but evaluating the frozen object did not."""
    frozen = exponential_like()
    assert frozen.reliability(25.0) == pytest.approx(np.exp(-RATE * 25.0), rel=1e-8)
    assert frozen.cdf(25.0) == pytest.approx(
        stats.expon.cdf(25.0, scale=1 / RATE), rel=1e-8
    )


def test_ppf_inverts_the_cdf(exponential_like):
    """scipy finds this by root-finding on _cdf, so it exercises the
    stateless path hardest: many calls on different grids."""
    for q in [0.1, 0.5, 0.9]:
        t = exponential_like.ppf(q)
        assert exponential_like.cdf(t) == pytest.approx(q, abs=1e-6)


def test_a_time_varying_hazard_with_no_closed_form_still_behaves():
    """Bathtub-ish: decreasing then increasing."""
    model = ReliabilityFromHazard(lambda t: 0.05 / (1.0 + t) + 0.0005 * t)
    t = np.linspace(1.0, 100.0, 40)
    F = model.cdf(t)
    assert np.all(np.diff(F) >= -1e-12)
    assert np.all((F >= 0) & (F <= 1))
    np.testing.assert_allclose(F + model.reliability(t), 1.0, rtol=1e-10)
