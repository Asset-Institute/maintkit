"""Exponential distribution: mean parameterisation and closed-form inference.

The class previously returned a rate from ``fit`` while ``nnlf`` consumed a
scale, so composing them was wrong by a factor of theta^2 with nothing to
signal it. Everything now speaks the mean, which is exactly what scipy calls
``scale``, so there is no conversion anywhere in the class and a fitted value
drops straight into a frozen distribution.
"""
from __future__ import annotations

import numdifftools as ndt
import numpy as np
import pytest

from maintkit.distributions import Exponential
from maintkit.inference import FitResult
from tests import _datasets as ds


@pytest.fixture(scope="module")
def complete():
    return ds.exponential_lifetimes()


@pytest.fixture(scope="module")
def censored():
    return ds.censored_lifetimes()


# ------------------------------------------------------------ estimation --
def test_fit_returns_fitresult(complete):
    res = Exponential().fit(complete)
    assert isinstance(res, FitResult)
    assert res.names == ("mean",)
    assert res.ci.shape == (1, 2)


def test_fit_recovers_the_closed_form_mean(complete):
    res = Exponential().fit(complete)
    assert res.params[0] == pytest.approx(complete.sum() / len(complete))


def test_fitted_mean_is_the_sample_mean_when_uncensored(complete):
    """With no censoring the MLE of the mean is just the average."""
    res = Exponential().fit(complete)
    assert res.params[0] == pytest.approx(complete.mean())


def test_fit_uses_exact_information(complete):
    """se = theta/sqrt(r), from I(theta) = r/theta^2 -- no differencing."""
    res = Exponential().fit(complete)
    r = len(complete)
    assert res.se[0] == pytest.approx(res.params[0] / np.sqrt(r), rel=1e-12)


def test_censored_fit_divides_by_failures_not_observations(censored):
    """Total time on test over the number of failures."""
    ti, observed = censored
    res = Exponential().fit(ti, observed=observed)
    assert res.params[0] == pytest.approx(ti.sum() / observed.sum())


def test_fit_raises_without_failures():
    with pytest.raises(ValueError, match="no observed failures"):
        Exponential().fit(np.array([1.0, 2.0]), observed=np.zeros(2))


# ------------------------------------------------ parameter consistency ----
def test_fit_output_feeds_straight_into_nnlf(complete):
    """The composition that was previously wrong by theta^2.

    nnlf must be minimised at the mean fit returns, so perturbing it in either
    direction can only increase the negative log-likelihood.
    """
    dist = Exponential()
    res = dist.fit(complete)
    mean = res.params[0]
    at_mle = dist.nnlf(mean, complete)
    for factor in (0.9, 0.99, 1.01, 1.1):
        assert dist.nnlf(mean * factor, complete) > at_mle


def test_fit_output_is_usable_as_scale_directly(complete):
    """No conversion: params[0] is what scipy calls scale."""
    dist = Exponential()
    res = dist.fit(complete)
    frozen = dist(scale=res.params[0])
    assert frozen.mean() == pytest.approx(res.params[0])
    t = 25.0
    assert frozen.reliability(t) == pytest.approx(np.exp(-t / res.params[0]))


def test_score_vanishes_at_the_estimate(complete):
    dist = Exponential()
    res = dist.fit(complete)
    g = dist.nnlf_gradient(res.params, complete)
    assert abs(g[0]) < 1e-9 * abs(dist.nnlf(res.params[0], complete))


def test_score_matches_numerical_gradient(complete):
    """Compared in log space: the mean is positive, and an adaptive step in the
    natural parameter can cross zero and return nan."""
    dist = Exponential()
    for mean in (5.0, 25.8, 100.0):
        analytic = dist.nnlf_gradient(mean, complete) * mean
        numeric = ndt.Gradient(
            lambda y: dist.nnlf(np.exp(y)[0], complete)
        )(np.log([mean]))
        np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-6)


def test_nnlf_matches_the_closed_form_expression(complete):
    """l(theta) = -r log theta - sum(t)/theta, up to the sign convention."""
    dist = Exponential()
    mean, r = 30.0, float(len(complete))
    expected = -(-r * np.log(mean) - complete.sum() / mean)
    assert dist.nnlf(mean, complete) == pytest.approx(expected)


def test_rate_is_just_the_reciprocal(complete):
    """A rate is still one division away, for anyone who wants one."""
    res = Exponential().fit(complete)
    rate = 1.0 / res.params[0]
    assert rate == pytest.approx(len(complete) / complete.sum())


# ------------------------------------------------------------ intervals ----
def test_transformed_ci_is_positive_and_asymmetric(complete):
    res = Exponential().fit(complete)
    lo, hi = res.ci[0]
    assert lo > 0
    assert (hi - res.params[0]) > (res.params[0] - lo)


def test_natural_ci_reproduces_the_symmetric_interval(complete):
    """The pre-consolidation convention, still reachable."""
    res = Exponential().fit(complete, ci_method="natural")
    mean, r = res.params[0], float(len(complete))
    z = 1.959963984540054
    expected = mean + z * np.array([-1, 1]) * mean / np.sqrt(r)
    np.testing.assert_allclose(res.ci[0], expected, rtol=1e-10)


def test_transformed_ci_is_scale_invariant(complete):
    """log-space intervals are multiplicative: theta*exp(+/- z/sqrt(r))."""
    res = Exponential().fit(complete)
    r = float(len(complete))
    z = 1.959963984540054
    expected = res.params[0] * np.exp(z * np.array([-1, 1]) / np.sqrt(r))
    np.testing.assert_allclose(res.ci[0], expected, rtol=1e-10)


def test_alpha_widens_the_interval(complete):
    narrow = Exponential().fit(complete, alpha=0.05)
    wide = Exponential().fit(complete, alpha=0.01)
    assert wide.ci[0, 0] < narrow.ci[0, 0]
    assert wide.ci[0, 1] > narrow.ci[0, 1]
