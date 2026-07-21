"""Block replacement timing. This module had no tests at all before."""
from __future__ import annotations

from math import gamma

import numpy as np
import pytest

from maintkit.distributions import Exponential, Weibull
from maintkit.maintenance_optimization import IntervalReplacement


def _exponential(mean=5.0):
    return IntervalReplacement(Exponential()(scale=mean),
                               cost_of_failure=5.0, cost_of_pm=1.0)


def _weibull(beta=2.5, eta=10.0, cf=5.0, cpm=1.0):
    return IntervalReplacement(Weibull()(beta, scale=eta),
                               cost_of_failure=cf, cost_of_pm=cpm)


# ------------------------------------------------- the renewal recursion ----
def test_exponential_renewal_function_is_lambda_t():
    """The one case with a closed form: H(t) = t/mean exactly.

    This is what pins the recursion. Everything else here only checks shape.
    """
    mean = 5.0
    t, H = _exponential(mean).expected_number_of_failures(T=40.0, dt=0.02)
    exact = t / mean
    assert np.max(np.abs(H[1:] - exact[1:]) / exact[1:]) < 0.005


def test_error_halves_when_dt_halves():
    """First-order accurate. A recursion that was merely close but converging
    to the wrong thing would fail this while passing a loose tolerance."""
    mean = 5.0
    model = _exponential(mean)
    errors = []
    for dt in [0.2, 0.1, 0.05]:
        t, H = model.expected_number_of_failures(T=20.0, dt=dt)
        errors.append(abs(np.interp(15.0, t, H) - 15.0 / mean))
    assert errors[0] / errors[1] == pytest.approx(2.0, abs=0.15)
    assert errors[1] / errors[2] == pytest.approx(2.0, abs=0.15)


def test_matches_the_renewal_theorem_asymptote():
    """For large t, H(t) -> t/mu + (var - mu^2) / (2 mu^2).

    An independent check on a distribution with no closed form.
    """
    beta, eta = 2.5, 10.0
    mu = eta * gamma(1 + 1 / beta)
    var = eta**2 * (gamma(1 + 2 / beta) - gamma(1 + 1 / beta) ** 2)
    t, H = _weibull(beta, eta).expected_number_of_failures(T=60.0, dt=0.05)
    expected = 50.0 / mu + (var - mu**2) / (2 * mu**2)
    assert np.interp(50.0, t, H) == pytest.approx(expected, abs=0.05)


def test_is_non_decreasing_and_starts_at_zero():
    t, H = _weibull().expected_number_of_failures(T=60.0, dt=0.1)
    assert H[0] == 0.0
    assert np.all(np.diff(H) >= -1e-12)


def test_finer_grid_does_not_change_the_shape():
    model = _weibull()
    t_coarse, H_coarse = model.expected_number_of_failures(T=40.0, dt=0.5)
    t_fine, H_fine = model.expected_number_of_failures(T=40.0, dt=0.05)
    np.testing.assert_allclose(
        np.interp(t_coarse[1:], t_fine, H_fine), H_coarse[1:], rtol=0.1
    )


# ------------------------------------------------------- optimal timing ----
def test_optimal_timing_returns_a_minimum():
    t, cr, idx = _weibull().optimal_timing(T=40.0, dt=0.1)
    assert cr[idx] == pytest.approx(cr.min())
    assert 0 < t[idx] < 40.0
    assert np.all(np.isfinite(cr))


def test_optimum_beats_running_to_failure():
    """With an increasing hazard and cf > cpm, scheduled replacement should
    cost less per unit time than the run-to-failure rate cf/MTTF."""
    beta, eta = 2.5, 10.0
    mttf = eta * gamma(1 + 1 / beta)
    model = _weibull(beta, eta, cf=5.0, cpm=1.0)
    t, cr, idx = model.optimal_timing(T=40.0, dt=0.05)
    assert cr[idx] < 5.0 / mttf


def test_cost_rate_is_the_stated_formula():
    model = _weibull()
    t, cr, _ = model.optimal_timing(T=30.0, dt=0.1)
    grid, H = model.expected_number_of_failures(T=30.0, dt=0.1)
    H, grid = H[grid > 0], grid[grid > 0]
    np.testing.assert_allclose(cr, (model.cpm + model.cf * H) / grid, rtol=1e-12)


def test_a_cheaper_failure_pushes_the_optimum_later():
    early = _weibull(cf=20.0, cpm=1.0).optimal_timing(T=40.0, dt=0.05)
    late = _weibull(cf=3.0, cpm=1.0).optimal_timing(T=40.0, dt=0.05)
    assert late[0][late[2]] > early[0][early[2]]


# ----------------------------------------------------------- validation ----
def test_missing_pm_cost_is_rejected():
    model = IntervalReplacement(Weibull()(2.5, scale=10.0), cost_of_failure=5.0)
    with pytest.raises(ValueError, match="cpm"):
        model.optimal_timing(T=20.0, dt=0.5)


def test_missing_failure_cost_is_rejected():
    """This one was never caught: the check tested cpm twice, so a missing cf
    surfaced later as a TypeError inside the cost arithmetic."""
    model = IntervalReplacement(Weibull()(2.5, scale=10.0), cost_of_pm=1.0)
    with pytest.raises(ValueError, match="cf"):
        model.optimal_timing(T=20.0, dt=0.5)


def test_repair_times_are_kept():
    """They were accepted by the constructor and silently discarded."""
    model = IntervalReplacement(
        Weibull()(2.5, scale=10.0), cost_of_failure=5.0, cost_of_pm=1.0,
        failure_repair_time=3.0, pm_repair_time=0.5,
    )
    assert model.failure_repair_time == 3.0
    assert model.pm_repair_time == 0.5
