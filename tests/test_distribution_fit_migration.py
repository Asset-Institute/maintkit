"""Verify what the distribution-fitter migration did and did not change.

``reliability_distribution.fit`` / ``fit_interval`` now delegate to
``maintkit.inference.fit_mle``. The intended effect is narrow:

* estimates and covariance are materially unchanged -- same objective, same
  optimiser, and the delta-method covariance reproduces the legacy convention.
  They move in the last few digits because weibull now supplies an analytic
  score; see the tolerance note below.
* the confidence-interval convention changes, from a symmetric interval on the
  natural scale to one built in the unconstrained space and mapped back.

The pre-migration numbers are frozen as literals below rather than read from
the reference file, which is regenerated whenever behaviour intentionally
changes.
"""
from __future__ import annotations

import numpy as np
import pytest

from maintkit.distributions import weibull
from maintkit.inference import FitResult
from tests import _datasets as ds

# --------------------------------------------------------------------------
# Pre-migration values, FROZEN AS LITERALS.
#
# These were originally read from tests/reference/reference_values.json, which
# was a mistake: that file is regenerated whenever behaviour intentionally
# changes, so after the first --force these tests silently began comparing the
# new implementation against itself. A "before/after" test cannot take its
# "before" from a file that gets rewritten.
#
# Captured from weibull().fit(ds.censored_lifetimes(), p0=[80., 1.5]) as
# implemented before the fitters were consolidated: symmetric CIs on the
# natural scale using a hardcoded 1.96, and finite-difference gradients.
# --------------------------------------------------------------------------
LEGACY_PARAMS = np.array([113.65427676242005, 1.89244974])
LEGACY_CI = np.array([[104.25846334586014, 123.05009017897997],
                      [1.66314011275716370, 2.12175937185984900]])
LEGACY_COV = np.array([[22.980349270826547, 0.077412030930408590],
                       [0.077412030930408650, 0.013687761923410566]])

def _score_log_space(dist, params, ti, observed):
    """|score|_inf with respect to (log eta, log beta).

    ``nnlf_gradient`` returns the score in *natural* parameters; the chain rule
    into log space is multiplication by the parameters themselves (the Jacobian
    is diagonal).

    Log space is the right yardstick for "how close to the optimum":

    * it is the space the optimiser works in, so it is what BFGS's ``gtol``
      is applied to;
    * it separates the two candidate points far more clearly. Natural space
      gives 7.46e-07 vs 6.59e-07 (12% apart, a thin margin for an assertion);
      log space gives 1.04e-05 vs 2.57e-06, a factor of four.

    No legacy score is frozen alongside the parameters: a gradient evaluated
    near an optimum is ~H*delta, so it is hypersensitive to the last digits of
    the parameters and cannot survive being copied out of formatted output.
    Both scores are therefore computed live, in consistent units.
    """
    g = np.asarray(dist.nnlf_gradient(params, ti, observed), dtype=float)
    return np.abs(g * np.asarray(params, dtype=float)).max()


@pytest.fixture(scope="module")
def weibull_fit():
    ti, observed = ds.censored_lifetimes()
    return weibull().fit(ti, p0=[80.0, 1.5], observed=observed)


# ------------------------------------------------------- what must not move --
# Tolerances: the legacy values above were produced with finite-difference
# gradients. weibull now supplies an analytic score, so BFGS
# stops at a slightly different -- and demonstrably better -- point:
#
#   finite difference : |score|_inf = 1.04e-05,  scale-equation residual +5.5e-06
#   analytic gradient : |score|_inf = 2.57e-06,  scale-equation residual -1.4e-06
#
# The estimate therefore moves by ~2e-8 relative and the covariance by ~5e-7.
# That is a genuine improvement in accuracy, not drift, so the tolerances below
# admit it while still catching any change of substance.
PARAM_RTOL = 1e-6
COV_RTOL = 1e-5


def test_estimates_unchanged(weibull_fit):
    np.testing.assert_allclose(weibull_fit.params, LEGACY_PARAMS, rtol=PARAM_RTOL)


def test_covariance_unchanged(weibull_fit):
    """The delta-method covariance must reproduce the legacy J inv(H) J^T."""
    np.testing.assert_allclose(weibull_fit.cov, LEGACY_COV, rtol=COV_RTOL)


def test_analytic_gradient_finds_a_better_optimum(weibull_fit):
    """The new stopping point must have a smaller score than the legacy one.

    This is what justifies the tolerances above: the estimate moved because the
    optimiser got closer to the root of the score, not because something drifted.

    Note what is NOT asserted. The two points differ in ``nnlf`` by ~7e-13 out
    of ~877, roughly 6 ULP of float64 -- indistinguishable in function value,
    as expected for two points at the bottom of the same likelihood. Comparing
    them at that resolution would be comparing rounding noise. The score has
    the dynamic range to tell them apart, so that is what is used.
    """
    ti, observed = ds.censored_lifetimes()
    dist = weibull()
    new_score = _score_log_space(dist, weibull_fit.params, ti, observed)
    old_score = _score_log_space(dist, LEGACY_PARAMS, ti, observed)
    assert new_score < old_score, (
        f"analytic-gradient score {new_score:.3e} should beat the legacy "
        f"finite-difference score {old_score:.3e}"
    )

    new_nnlf = dist.nnlf(weibull_fit.params, ti, observed)
    old_nnlf = dist.nnlf(LEGACY_PARAMS, ti, observed)
    tol = 10 * np.spacing(abs(old_nnlf))
    assert new_nnlf <= old_nnlf + tol



def test_natural_ci_reproduces_legacy_interval():
    """ci_method='natural' recovers the old symmetric interval.

    Not bit-identical: the old code hardcoded 1.96 while fit_mle uses
    norm.ppf(0.975) = 1.9599639..., a relative difference of ~1.8e-5. The
    tolerance below is set to accommodate exactly that and nothing larger.
    """
    ti, observed = ds.censored_lifetimes()
    res = weibull().fit(ti, p0=[80.0, 1.5], observed=observed, ci_method="natural")
    np.testing.assert_allclose(res.ci, LEGACY_CI, rtol=1e-4)


# ----------------------------------------------------------- what must move --
def test_transformed_ci_differs_from_legacy(weibull_fit):
    assert not np.allclose(weibull_fit.ci, LEGACY_CI, rtol=1e-3)


def test_transformed_ci_is_positive_and_asymmetric(weibull_fit):
    assert np.all(weibull_fit.ci > 0)
    lower_gap = weibull_fit.params - weibull_fit.ci[:, 0]
    upper_gap = weibull_fit.ci[:, 1] - weibull_fit.params
    assert np.all(upper_gap > lower_gap)


# ----------------------------------------------------------------- new API --
def test_fit_returns_fitresult(weibull_fit):
    assert isinstance(weibull_fit, FitResult)
    assert weibull_fit.ci.shape == (2, 2)
    assert weibull_fit.names == ("eta", "beta")
    assert np.isfinite(weibull_fit.nnlf)


def test_optimiser_convergence_flag_is_reported(weibull_fit):
    """The convergence flag is now visible; it was not before.

    On this objective BFGS terminates with "Desired error not necessarily
    achieved due to precision loss". That is a PRE-EXISTING condition, not
    something the migration introduced: the previous implementation called
    ``opt.minimize`` with the same arguments but never inspected
    ``result.success``, so the flag was silently discarded. The estimate and
    covariance match the frozen pre-migration values (see the tests above),
    which is what establishes the migration is faithful.

    Asserted loosely on purpose -- a future scipy may converge cleanly here,
    and that should not fail the suite.
    """
    assert isinstance(weibull_fit.success, bool)
    if not weibull_fit.success:
        assert weibull_fit.message, "a failed fit must carry a reason"
        assert "warning:" in weibull_fit.summary()


def test_fit_interval_now_returns_covariance():
    """fit_interval previously returned only the estimate and CI; cov is new."""
    ti, ins, observed = ds.interval_lifetimes()
    res = weibull().fit_interval(ti, ins, p0=[80.0, 1.5], observed=observed)
    assert isinstance(res, FitResult)
    assert res.cov.shape == (2, 2)
    assert np.all(np.diag(res.cov) > 0)



def test_alpha_is_honoured():
    ti, observed = ds.censored_lifetimes()
    narrow = weibull().fit(ti, p0=[80.0, 1.5], observed=observed, alpha=0.05)
    wide = weibull().fit(ti, p0=[80.0, 1.5], observed=observed, alpha=0.01)
    assert np.all(wide.ci[:, 0] < narrow.ci[:, 0])
    assert np.all(wide.ci[:, 1] > narrow.ci[:, 1])


def test_summary_uses_parameter_names(weibull_fit):
    text = weibull_fit.summary()
    assert "eta" in text and "beta" in text


# ------------------------------------------------------ analytic gradient --
@pytest.mark.parametrize("p", [[113.65427676, 1.89244974], [80.0, 1.5],
                               [200.0, 3.2], [50.0, 0.7]])
def test_analytic_score_matches_numerical_gradient(p):
    """The closed-form score must agree with numerical differentiation."""
    import numdifftools as ndt
    ti, observed = ds.censored_lifetimes()
    dist = weibull()
    analytic = dist.nnlf_gradient(p, ti, observed)
    numeric = ndt.Gradient(lambda q: dist.nnlf(q, ti, observed))(np.asarray(p, float))
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-6)


def test_score_vanishes_at_the_mle(weibull_fit):
    """A correct score is ~0 at the optimum -- independent evidence of convergence."""
    ti, observed = ds.censored_lifetimes()
    g = weibull().nnlf_gradient(weibull_fit.params, ti, observed)
    scale = abs(weibull_fit.nnlf)
    assert np.all(np.abs(g) / scale < 1e-6), f"score {g} too large relative to nnlf {scale}"


def test_score_reproduces_textbook_mle_equations(weibull_fit):
    """sum (t/eta)^beta == r, the censored-Weibull scale equation."""
    ti, observed = ds.censored_lifetimes()
    eta, beta = weibull_fit.params
    assert np.sum((ti / eta) ** beta) == pytest.approx(observed.sum(), rel=1e-6)


@pytest.mark.filterwarnings("ignore::maintkit.inference.ConvergenceWarning")
def test_analytic_and_numeric_gradients_give_same_estimate():
    """Supplying the score changes how the optimiser gets there, not where."""
    ti, observed = ds.censored_lifetimes()
    with_score = weibull().fit(ti, p0=[80.0, 1.5], observed=observed,
                               use_analytic_gradient=True)
    without = weibull().fit(ti, p0=[80.0, 1.5], observed=observed,
                            use_analytic_gradient=False)
    np.testing.assert_allclose(with_score.params, without.params, rtol=1e-5)
    np.testing.assert_allclose(with_score.cov, without.cov, rtol=1e-4)
