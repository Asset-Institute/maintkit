"""Nonparametric estimators and Weibull plots.

This module had no tests, which is why 15 asserts were converted to raises and
`c` became `alpha` with nothing checking either.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pytest
from scipy import stats

matplotlib.use("Agg")

from maintkit.distributions import Exponential, Weibull  # noqa: E402
from maintkit.probability_plotting import (  # noqa: E402
    ecdf,
    empirical_mean_cumulative_function,
    kaplan_meier,
    weibull_probability_plot,
    weibull_reliability_confidence_interval,
)


COMPLETE = np.array([12.0, 45.0, 67.0, 88.0, 110.0, 130.0])
ALL_OBSERVED = np.ones(6)


# ------------------------------------------------------------------ ecdf ----
@pytest.mark.parametrize("pos", ["midpoint", "mean", "median"])
def test_ecdf_is_increasing_and_starts_at_zero(pos):
    x, F = ecdf(COMPLETE, ALL_OBSERVED, pos=pos, plot=False)
    assert x[0] == 0 and F[0] == 0
    assert np.all(np.diff(x) > 0)
    assert np.all(np.diff(F) > 0)
    assert np.all((F >= 0) & (F <= 1))


def test_ecdf_plotting_positions_match_their_formulas():
    """midpoint (i-0.5)/N, mean i/(N+1), median (i-0.3)/(N+0.4)."""
    n = COMPLETE.size
    i = np.arange(1, n + 1)
    for pos, expected in [
        ("midpoint", (i - 0.5) / n),
        ("mean", i / (n + 1)),
        ("median", (i - 0.3) / (n + 0.4)),
    ]:
        _, F = ecdf(COMPLETE, ALL_OBSERVED, pos=pos, plot=False)
        np.testing.assert_allclose(F[1:], expected, rtol=1e-12)


def test_ecdf_sorts_its_input():
    shuffled = COMPLETE[[3, 0, 5, 1, 4, 2]]
    x_sorted, F_sorted = ecdf(COMPLETE, ALL_OBSERVED, plot=False)
    x_shuffled, F_shuffled = ecdf(shuffled, ALL_OBSERVED, plot=False)
    np.testing.assert_allclose(x_shuffled, x_sorted)
    np.testing.assert_allclose(F_shuffled, F_sorted)


def test_ecdf_returns_only_the_observed_times():
    observed = np.array([1, 1, 0, 1, 0, 1])
    x, F = ecdf(COMPLETE, observed, plot=False)
    assert x.size == int(observed.sum()) + 1      # +1 for the prepended zero
    assert set(x[1:]) == set(COMPLETE[observed == 1])


# ---------------------------------------------------------- Kaplan-Meier ----
def test_kaplan_meier_matches_the_ecdf_without_censoring():
    """With nothing censored the product limit reduces to the step ECDF i/N."""
    t, F, _, _ = kaplan_meier(COMPLETE, ALL_OBSERVED, plot=False)
    np.testing.assert_allclose(F[1:], np.arange(1, 7) / 6.0, rtol=1e-12)
    assert t[0] == 0 and F[0] == 0


def test_kaplan_meier_is_non_decreasing_and_bounded():
    observed = np.array([1, 0, 1, 1, 0, 1])
    t, F, LB, UB = kaplan_meier(COMPLETE, observed, plot=False)
    assert np.all(np.diff(F) >= -1e-12)
    assert np.all((F >= 0) & (F <= 1))
    assert np.all(LB <= F + 1e-12) and np.all(F <= UB + 1e-12)
    assert np.all(LB >= 0) and np.all(UB <= 1)


def test_kaplan_meier_censoring_holds_the_curve_down():
    """A censored observation contributes no failure, so F is lower afterwards.

    Compared at a shared failure time, not at the end: with the largest
    observation still a failure the risk set there is 1 and the factor is
    exactly 0, so both curves reach 1 regardless of earlier censoring.
    """
    t_all, F_all, _, _ = kaplan_meier(COMPLETE, ALL_OBSERVED, plot=False)
    censored = np.array([1, 1, 0, 1, 1, 1])
    t_cens, F_cens, _, _ = kaplan_meier(COMPLETE, censored, plot=False)
    at = 110.0
    assert np.interp(at, t_cens, F_cens) < np.interp(at, t_all, F_all)


def test_kaplan_meier_does_not_reach_one_when_the_last_time_is_censored():
    """The signature of right-censoring: the product limit stops short of 1."""
    censored = np.array([1, 1, 1, 1, 1, 0])
    _, F, _, _ = kaplan_meier(COMPLETE, censored, plot=False)
    assert F[-1] == pytest.approx(1 - 1 / 6, rel=1e-12)
    assert F[-1] < 1.0


def test_kaplan_meier_reaches_one_when_the_last_time_is_a_failure():
    _, F, _, _ = kaplan_meier(COMPLETE, ALL_OBSERVED, plot=False)
    assert F[-1] == pytest.approx(1.0)


def test_kaplan_meier_product_limit_by_hand():
    """Two failures, one censoring in between: R = (3/4)(1/2)."""
    ti = np.array([1.0, 2.0, 3.0, 4.0])
    observed = np.array([1, 0, 1, 0])
    _, F, _, _ = kaplan_meier(ti, observed, plot=False)
    assert F[-1] == pytest.approx(1 - 0.75 * (1 / 2), rel=1e-12)


def test_kaplan_meier_exponential_interval_shape():
    t, F, LB, UB = kaplan_meier(COMPLETE, ALL_OBSERVED, plot=False,
                                confidence_interval="exponential")
    assert LB.shape == F.shape and UB.shape == F.shape


def test_kaplan_meier_rejects_an_unknown_interval():
    with pytest.raises(ValueError, match="confidence_interval"):
        kaplan_meier(COMPLETE, ALL_OBSERVED, plot=False,
                     confidence_interval="bootstrap")


def test_kaplan_meier_plot_returns_six_values():
    out = kaplan_meier(COMPLETE, ALL_OBSERVED, plot=True)
    assert len(out) == 6


# ------------------------------------------ mean cumulative function ----
FLEET = [[10.0, 40.0, 95.0], [22.0, 60.0], [15.0, 30.0, 70.0, 99.0]]
HORIZONS = [100.0, 100.0, 100.0]


def test_mcf_is_non_decreasing_and_starts_at_zero():
    t, M, _, _ = empirical_mean_cumulative_function(
        FLEET, HORIZONS, plot=False, confidence_interval="normal"
    )
    assert M[0] == 0
    assert np.all(np.diff(M) >= -1e-12)


def test_mcf_final_value_is_the_mean_event_count():
    """With every asset observed to the same horizon, M(T) is just the average
    number of events per asset."""
    _, M, _, _ = empirical_mean_cumulative_function(
        FLEET, HORIZONS, plot=False, confidence_interval="normal"
    )
    expected = sum(len(e) for e in FLEET) / len(FLEET)
    assert M[-1] == pytest.approx(expected, rel=1e-12)


def test_mcf_interval_brackets_the_estimate():
    t, M, LCL, UCL = empirical_mean_cumulative_function(
        FLEET, HORIZONS, plot=False, confidence_interval="normal"
    )
    assert np.all(LCL <= M[1:] + 1e-12)
    assert np.all(M[1:] <= UCL + 1e-12)


def test_mcf_logit_interval_stays_positive():
    _, _, LCL, UCL = empirical_mean_cumulative_function(
        FLEET, HORIZONS, plot=False, confidence_interval="logit"
    )
    assert np.all(LCL > 0)
    # equal where se_hat is zero, which happens wherever the assets agree
    assert np.all(UCL >= LCL)


def test_mcf_without_an_interval_does_not_raise():
    """M_LCL and M_UCL were assigned only in the single-asset and else
    branches, but both returns read them, so this raised UnboundLocalError for
    any fleet of more than one asset."""
    t, M, LCL, UCL = empirical_mean_cumulative_function(
        FLEET, HORIZONS, plot=False, confidence_interval=None
    )
    assert np.all(np.isfinite(M))
    assert np.all(np.isnan(LCL)) and np.all(np.isnan(UCL))


def test_mcf_single_asset_returns_nan_bounds():
    t, M, LCL, UCL = empirical_mean_cumulative_function(
        [[10.0, 40.0]], [100.0], plot=False
    )
    assert np.all(np.isnan(LCL)) and np.all(np.isnan(UCL))
    assert M[-1] == pytest.approx(2.0)


def test_mcf_rejects_a_flat_list():
    with pytest.raises(ValueError, match="list of lists"):
        empirical_mean_cumulative_function([10.0, 40.0], [100.0], plot=False)


def test_mcf_shorter_horizon_shrinks_the_risk_set():
    """An asset retired early stops contributing, so later increments are
    averaged over fewer assets and M rises faster."""
    _, M_equal, _, _ = empirical_mean_cumulative_function(
        FLEET, [100.0, 100.0, 100.0], plot=False, confidence_interval=None
    )
    _, M_early, _, _ = empirical_mean_cumulative_function(
        FLEET, [100.0, 25.0, 100.0], plot=False, confidence_interval=None
    )
    assert M_early[-1] > M_equal[-1]


# --------------------------------------------------- Weibull plot + band ----
@pytest.fixture(scope="module")
def frozen_weibull():
    return Weibull()(2.0, scale=100.0)


@pytest.fixture(scope="module")
def cov():
    return np.array([[25.0, 0.5], [0.5, 0.04]])


def test_probability_plot_draws(frozen_weibull):
    ax = weibull_probability_plot(frozen_weibull)
    assert len(ax.get_lines()) >= 1


def test_probability_plot_accepts_data(frozen_weibull):
    x, F = ecdf(COMPLETE, ALL_OBSERVED, plot=False)
    ax = weibull_probability_plot(frozen_weibull,
                                  data={"times": x[1:], "ecdf": F[1:]})
    assert len(ax.get_lines()) >= 2


def test_probability_plot_rejects_an_unfrozen_distribution():
    with pytest.raises(TypeError, match="frozen"):
        weibull_probability_plot(Weibull())


def test_probability_plot_rejects_a_non_weibull():
    with pytest.raises(TypeError, match="only the Weibull"):
        weibull_probability_plot(Exponential()(scale=50.0))


def test_probability_plot_rejects_a_data_dict_missing_a_key(frozen_weibull):
    """The old check read the first two keys and asked whether each was one of
    the two valid names, so {'times', 'junk'} passed and a one-key dict raised
    IndexError."""
    with pytest.raises(ValueError, match="missing the key"):
        weibull_probability_plot(frozen_weibull,
                                 data={"times": [1.0], "junk": [0.1]})


def test_probability_plot_rejects_mismatched_data_lengths(frozen_weibull):
    with pytest.raises(ValueError, match="entries"):
        weibull_probability_plot(frozen_weibull,
                                 data={"times": [1.0, 2.0], "ecdf": [0.1]})


def test_probability_plot_rejects_non_dict_data(frozen_weibull):
    with pytest.raises(TypeError, match="dict"):
        weibull_probability_plot(frozen_weibull, data=[1.0, 2.0])


def test_probability_plot_requires_a_covariance_for_bounds(frozen_weibull):
    with pytest.raises(TypeError, match="parameter_covariance"):
        weibull_probability_plot(frozen_weibull, confidence_bounds="time")


def test_probability_plot_rejects_a_wrong_shaped_covariance(frozen_weibull):
    with pytest.raises(ValueError, match="2-by-2"):
        weibull_probability_plot(frozen_weibull, confidence_bounds="time",
                                 parameter_covariance=np.eye(3))


def test_probability_plot_rejects_an_unknown_bound_kind(frozen_weibull, cov):
    with pytest.raises(ValueError, match="confidence_bounds"):
        weibull_probability_plot(frozen_weibull, confidence_bounds="sideways",
                                 parameter_covariance=cov)


# ----------------------------------------------- reliability confidence ----
@pytest.mark.parametrize("kind", ["Reliability", "time"])
def test_band_brackets_the_curve(frozen_weibull, cov, kind):
    t = np.linspace(10.0, 200.0, 40)
    RL, RU = weibull_reliability_confidence_interval(frozen_weibull, t, cov, kind=kind)
    R = frozen_weibull.reliability(t)
    assert np.all(RL <= R + 1e-9)
    assert np.all(R <= RU + 1e-9)
    assert np.all((RL >= 0) & (RU <= 1))


def test_smaller_alpha_widens_the_band(frozen_weibull, cov):
    """alpha replaced a c argument that took the critical value directly, so
    nothing checked that it does anything."""
    t = np.linspace(10.0, 200.0, 20)
    RL95, RU95 = weibull_reliability_confidence_interval(frozen_weibull, t, cov, alpha=0.05)
    RL99, RU99 = weibull_reliability_confidence_interval(frozen_weibull, t, cov, alpha=0.01)
    assert np.all(RL99 <= RL95 + 1e-12)
    assert np.all(RU99 >= RU95 - 1e-12)


def test_alpha_uses_the_normal_critical_value(frozen_weibull, cov):
    """A band at alpha should equal the old c = norm.ppf(1 - alpha/2)."""
    t = np.linspace(10.0, 200.0, 15)
    RL, RU = weibull_reliability_confidence_interval(frozen_weibull, t, cov, alpha=0.05)
    assert stats.norm.ppf(0.975) == pytest.approx(1.959963985, abs=1e-9)
    # the band must be strictly inside the one from a larger critical value
    RL_wide, RU_wide = weibull_reliability_confidence_interval(
        frozen_weibull, t, cov, alpha=0.0455
    )
    assert np.all(RL >= RL_wide - 1e-12)


def test_band_rejects_unsorted_time(frozen_weibull, cov):
    with pytest.raises(ValueError, match="non-decreasing"):
        weibull_reliability_confidence_interval(
            frozen_weibull, np.array([10.0, 5.0, 20.0]), cov
        )


def test_band_rejects_negative_time(frozen_weibull, cov):
    with pytest.raises(ValueError, match="non-negative"):
        weibull_reliability_confidence_interval(
            frozen_weibull, np.array([-1.0, 5.0]), cov
        )


def test_band_rejects_an_unknown_kind(frozen_weibull, cov):
    with pytest.raises(ValueError, match="kind must be"):
        weibull_reliability_confidence_interval(
            frozen_weibull, np.linspace(1.0, 50.0, 5), cov, kind="hazard"
        )


def test_band_prepends_nan_when_t_starts_at_zero(frozen_weibull, cov):
    t = np.linspace(0.0, 100.0, 10)
    RL, RU = weibull_reliability_confidence_interval(frozen_weibull, t, cov)
    assert np.isnan(RL[0]) and np.isnan(RU[0])
    assert RL.size == t.size
    assert np.all(np.isfinite(RL[1:]))
