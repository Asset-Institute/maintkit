"""Verify what the Poisson-process migration did and did not change.

``poisson_process.fit``, ``power_law_nhpp.fit`` and ``fit_interval`` now
delegate to ``maintkit.inference``. Unlike the distribution fitters, this
module already built its confidence intervals on the log scale, so the CI
convention does not change here -- only the critical value, from a hardcoded
1.96 to ``norm.ppf(0.975) = 1.9599639...``.

Pre-migration values are frozen as literals below. They were generated from
tests/reference/reference_values.json at full precision rather than copied
from formatted output: a truncated digit silently corrupts anything derived
from these, which is how an earlier frozen score ended up wrong.
"""
from __future__ import annotations

import numpy as np
import pytest

from maintkit.inference import FitResult
from maintkit.poisson_process import poisson_process, power_law_nhpp
from tests import _datasets as ds

# --------------------------------------------------------------------------
# Frozen pre-migration values (generated, not transcribed).
# Recorded with maintkit 0.1.0, numpy 2.5.1, scipy 1.18.0, python 3.14.3.
# --------------------------------------------------------------------------
LEGACY_PP_FIT_PARAMS = np.array([0.023442853110932256, 1.4694587482719879])
LEGACY_PP_FIT_COV = np.array(
    [[2.4015890447827990e-05, -1.9021470480667821e-04],
     [-1.9021470480667821e-04, 1.5314247178955867e-03]]
)

LEGACY_PLP_FIT_PARAMS = np.array([0.02344280154227633, 1.4694591549675484])
LEGACY_PLP_FIT_CI = np.array(
    [[0.015562061603183403, 0.035314404875388325],
     [1.3947250436334364, 1.5481977741596031]]
)
LEGACY_PLP_FIT_COV = np.array(
    [[2.4015799592911186e-05, -1.9021440542123594e-04],
     [-1.9021440542123591e-04, 1.5314256763016264e-03]]
)

LEGACY_PLP_INTERVAL_PARAMS = np.array([0.012725004518645281, 1.5863330053465095])

# Only the critical value changed for these fitters: 1.96 -> 1.9599639845...,
# a relative change of ~1.8e-5 in the exponent argument.
PARAM_RTOL = 1e-6
COV_RTOL = 1e-5
CI_RTOL = 1e-3          # must admit the z-value change, nothing larger


@pytest.fixture(scope="module")
def data():
    return ds.nhpp_events()


@pytest.fixture(scope="module")
def counts():
    return ds.nhpp_interval_counts()


# ----------------------------------------------------- poisson_process.fit --
@pytest.fixture(scope="module")
def base_fit(data):
    events, trunc = data
    model = power_law_nhpp(0.02, 1.5)
    # exercise the generic base-class fitter, not the analytic override
    return poisson_process.fit(model, events, p0=[0.01, 1.2], truncation_times=trunc)


def test_base_fit_returns_fitresult(base_fit):
    assert isinstance(base_fit, FitResult)
    assert base_fit.ci.shape == (2, 2)
    assert base_fit.names == ("a", "b")


def test_base_fit_estimates_unchanged(base_fit):
    np.testing.assert_allclose(base_fit.params, LEGACY_PP_FIT_PARAMS, rtol=PARAM_RTOL)


def test_base_fit_covariance_unchanged(base_fit):
    np.testing.assert_allclose(base_fit.cov, LEGACY_PP_FIT_COV, rtol=COV_RTOL)


def test_generic_optimiser_agrees_with_the_closed_form(base_fit, data):
    """Two independent routes to the same estimate.

    The base class optimises numerically; power_law_nhpp.fit solves in closed
    form. Agreement to ~6 significant figures checks both against each other.

    This also settles the ConvergenceWarning the base fit emits: BFGS reports
    "precision loss" because, with an exact gradient, it reaches the point
    where the *objective* can no longer be resolved -- nnlf here is order 1e4,
    so its noise floor sits above what the line search needs. Landing on the
    analytic MLE is the evidence that the stopping point is sound.
    """
    events, trunc = data
    closed_form = power_law_nhpp(0.02, 1.5).fit(events, truncation_times=trunc)
    np.testing.assert_allclose(base_fit.params, closed_form.params, rtol=1e-5)


# ---------------------------------------------------- power_law_nhpp.fit ---
@pytest.fixture(scope="module")
def plp_fit(data):
    events, trunc = data
    return power_law_nhpp(0.02, 1.5).fit(events, truncation_times=trunc)


def test_closed_form_estimate_unchanged(plp_fit):
    """The closed form is untouched; only how the result is packaged changed."""
    np.testing.assert_allclose(plp_fit.params, LEGACY_PLP_FIT_PARAMS, rtol=PARAM_RTOL)


def test_closed_form_covariance_unchanged(plp_fit):
    np.testing.assert_allclose(plp_fit.cov, LEGACY_PLP_FIT_COV, rtol=COV_RTOL)


def test_closed_form_ci_moves_only_by_the_critical_value(plp_fit):
    """Same convention as before, so the CI shifts only via 1.96 -> norm.ppf."""
    np.testing.assert_allclose(plp_fit.ci, LEGACY_PLP_FIT_CI, rtol=CI_RTOL)
    assert not np.allclose(plp_fit.ci, LEGACY_PLP_FIT_CI, rtol=1e-9)


def test_closed_form_result_reports_no_optimiser(plp_fit):
    assert plp_fit.message == "closed-form estimate"
    assert plp_fit.success


def test_ci_brackets_the_estimate(plp_fit):
    assert np.all(plp_fit.ci[:, 0] < plp_fit.params)
    assert np.all(plp_fit.params < plp_fit.ci[:, 1])


# ------------------------------------------------------------ fit_interval --
@pytest.fixture(scope="module")
def interval_fit(counts):
    ni, ins = counts
    return power_law_nhpp(0.02, 1.5).fit_interval(ni, ins, p0=[0.01, 1.2])


def test_interval_fit_estimates_unchanged(interval_fit):
    """The optimiser path is unchanged, so the estimate must not move."""
    np.testing.assert_allclose(
        interval_fit.params, LEGACY_PLP_INTERVAL_PARAMS, rtol=PARAM_RTOL
    )


def test_interval_fit_ci_now_brackets_the_estimate(interval_fit):
    """Regression test for the double-exponentiation bug.

    The previous implementation built the interval as
    ``exp(p_hat +/- z*s)`` where ``p_hat`` was already on the natural scale,
    so it exponentiated an estimate that had itself been exponentiated. The
    recorded result was ci = [[0.566, 1.811], [4.561, 5.233]] against
    params = [0.0127, 1.586] -- the estimate did not lie inside its own
    interval, and the geometric midpoints were exactly exp(params).
    """
    assert np.all(interval_fit.ci[:, 0] < interval_fit.params)
    assert np.all(interval_fit.params < interval_fit.ci[:, 1])


def test_interval_fit_ci_is_positive(interval_fit):
    assert np.all(interval_fit.ci > 0)


def test_fit_interval_has_no_estimate_ci_flag(counts):
    """The escape hatch is gone; ci and cov are always computed."""
    ni, ins = counts
    with pytest.raises(TypeError):
        power_law_nhpp(0.02, 1.5).fit_interval(
            ni, ins, p0=[0.01, 1.2], estimate_ci=False
        )


# ------------------------------------------------------------- validation --
def test_event_times_type_error_is_raised_not_asserted():
    """Assertions vanish under python -O; these checks must not."""
    model = power_law_nhpp(0.02, 1.5)
    with pytest.raises(TypeError, match="list of lists"):
        model.fit([1.0, 2.0, 3.0], truncation_times=[10.0])


def test_invalid_truncation_time_raises(data):
    events, _ = data
    model = power_law_nhpp(0.02, 1.5)
    bad = [1.0] * len(events)          # earlier than the observed events
    with pytest.raises(ValueError, match="Invalid truncation time"):
        model.fit(events, truncation_times=bad)


def test_numpy_truncation_times_do_not_raise(data):
    """`truncation_times != None` used to blow up on an ndarray."""
    events, trunc = data
    model = power_law_nhpp(0.02, 1.5)
    res = model.fit(events, truncation_times=np.asarray(trunc, dtype=float))
    assert np.all(np.isfinite(res.params))


# ------------------------------------------- known defect: unequal horizons --
@pytest.mark.xfail(
    reason="power_law_nhpp.fit uses the Crow closed form, which is the MLE "
           "only when every asset shares a truncation time. Fixed in the "
           "following commit by solving the profile score equation.",
    strict=True,
)
def test_closed_form_is_the_mle_for_unequal_truncation_times():
    """The score must vanish at a maximum-likelihood estimate.

    With a common horizon the Crow form is exact and the score is ~1e-11. With
    unequal horizons the ratio (sum T^b log T)/(sum T^b) no longer collapses to
    log T, the returned estimate is not a stationary point, and the score is
    large. Using the score rather than comparing likelihood values makes the
    defect unambiguous: a non-zero gradient cannot be a maximum.
    """
    events, trunc = ds.nhpp_events_unequal_horizons()
    model = power_law_nhpp(0.02, 1.5)
    res = model.fit(events, truncation_times=trunc)
    score = model.nnlf_gradient(res.params, events, trunc)
    scale = abs(model.nnlf(res.params, events, trunc))
    assert np.all(np.abs(score) / scale < 1e-9), (
        f"score {score} does not vanish at the reported estimate"
    )
