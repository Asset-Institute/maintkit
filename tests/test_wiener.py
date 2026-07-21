"""Wiener and RBM after the migration to the shared MLE code."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from maintkit.inference import FitResult
from maintkit.wiener import RBM, Wiener
from tests import _datasets as ds


MU, SIGMA = 0.5, 1.0


@pytest.fixture(scope="module")
def path():
    return ds.wiener_path()


@pytest.fixture(scope="module")
def reflected():
    """RBM cannot produce a negative value, and wiener_path contains several."""
    return ds.reflected_path()


# ------------------------------------------------------------- Wiener fit ----
@pytest.fixture(scope="module")
def wiener_fit(path):
    t, x = path
    return Wiener(mu=MU, sigma=SIGMA).fit(t, x)


def test_fit_returns_fitresult(wiener_fit):
    assert isinstance(wiener_fit, FitResult)
    assert wiener_fit.names == ("mu", "sigma")
    assert wiener_fit.ci.shape == (2, 2)


def test_sigma_stays_positive_and_so_does_its_interval(wiener_fit):
    """The Log transform is what guarantees the lower bound, not luck."""
    assert wiener_fit.params[1] > 0
    assert wiener_fit.ci[1, 0] > 0


def test_recovers_the_generating_parameters(wiener_fit):
    mu_hat, sigma_hat = wiener_fit.params
    assert abs(mu_hat - MU) < 0.3
    assert abs(sigma_hat - SIGMA) < 0.2


def test_nnlf_does_not_write_to_the_model(path):
    """It used to assign self.mu/self.sigma and restore them afterwards, so an
    exception mid-loop left the object holding trial values."""
    t, x = path
    model = Wiener(mu=MU, sigma=SIGMA)
    model.nnlf([3.7, 2.9], t, x)
    assert model.mu == MU
    assert model.sigma == SIGMA


def test_nnlf_rejects_a_non_positive_sigma(path):
    t, x = path
    model = Wiener(mu=MU, sigma=SIGMA)
    assert model.nnlf([MU, 0.0], t, x) == np.inf
    assert model.nnlf([MU, -1.0], t, x) == np.inf


def test_nnlf_matches_a_direct_normal_calculation(path):
    """Independent check: the increments are normal, so sum their log-densities."""
    t, x = path
    model = Wiener(mu=MU, sigma=SIGMA)
    expected = 0.0
    for run_t, run_x in zip(t, x):
        rt, rx = np.asarray(run_t), np.asarray(run_x)
        dt = np.diff(rt)
        expected += np.sum(
            stats.norm.logpdf(rx[1:], loc=MU * dt + rx[:-1],
                              scale=SIGMA * np.sqrt(dt))
        )
    assert model.nnlf([MU, SIGMA], t, x) == pytest.approx(-expected, rel=1e-12)


def test_inplace_writes_the_estimates_and_still_returns(path):
    t, x = path
    model = Wiener(mu=MU, sigma=SIGMA)
    result = model.fit(t, x, inplace=True)
    assert isinstance(result, FitResult)
    assert model.mu == result.params[0]
    assert model.sigma == result.params[1]
    assert model.parameter_source == "estimated"
    np.testing.assert_allclose(model.parameter_covariance, result.cov)


def test_fit_leaves_the_model_alone_by_default(path):
    t, x = path
    model = Wiener(mu=MU, sigma=SIGMA)
    model.fit(t, x)
    assert model.mu == MU
    assert model.sigma == SIGMA
    assert model.parameter_source == "specified"


def test_starting_values_are_close_to_the_optimum(path, wiener_fit):
    """A wrong moment estimate still converges, so check it directly."""
    t, x = path
    p0 = Wiener(mu=MU, sigma=SIGMA)._starting_values(t, x)
    np.testing.assert_allclose(p0, wiener_fit.params, rtol=0.2)


def test_mismatched_run_lengths_are_rejected(path):
    t, x = path
    model = Wiener(mu=MU, sigma=SIGMA)
    with pytest.raises(ValueError, match="observations"):
        model.fit([t[0]], [x[0][:-3]])


def test_a_run_with_one_point_has_no_increment(path):
    model = Wiener(mu=MU, sigma=SIGMA)
    with pytest.raises(ValueError, match="fewer than two"):
        model.fit([[0.0]], [[0.0]])


# ---------------------------------------------------------------- RBM pdf ----
def _naive_rbm_pdf(x, t, x0, mu, sigma):
    """The formula as it was written before, evaluated directly."""
    m = mu * t + x0
    s = sigma * np.sqrt(t)
    f = stats.norm.pdf((x - m) / s) / s
    f += np.exp(2 * mu * x / sigma**2) * stats.norm.pdf((x + m) / s) / s
    f -= (2 * mu / sigma**2) * np.exp(2 * mu * x / sigma**2) * stats.norm.cdf(-(x + m) / s)
    return f


@pytest.mark.parametrize(
    "x,t,mu,sigma",
    [(1.0, 1.0, 0.5, 1.0), (2.0, 3.0, 1.0, 2.0), (0.5, 0.2, -0.3, 1.0),
     (3.0, 5.0, 0.1, 1.5), (0.1, 0.1, 2.0, 1.0)],
)
def test_pdf_agrees_with_the_direct_formula(x, t, mu, sigma):
    """Where the direct formula still works, the rewrite must match it."""
    model = RBM(mu, sigma)
    got = model.transition_distribution(x, t, 0.0, type="pdf")
    assert got == pytest.approx(_naive_rbm_pdf(x, t, 0.0, mu, sigma), rel=1e-12)


@pytest.mark.parametrize("x,t,mu", [(400.0, 1.0, 1.0), (80.0, 2.0, 2.0),
                                    (1000.0, 5.0, 0.5)])
def test_logpdf_stays_finite_where_the_old_form_died(x, t, mu):
    """exp(2*mu*x/sigma**2) overflows, and the density underflowed to exactly
    zero -- at which point log(f + 1e-10) returned about -23 regardless of how
    small the density really was."""
    lp = RBM(mu, 1.0).transition_distribution(x, t, 0.0, type="logpdf")
    assert np.isfinite(lp)
    assert lp < -100          # genuinely tiny, not the -23 floor


def test_zero_drift_is_the_folded_normal():
    """Reflected BM with no drift has density 2*phi(x/s)/s. Also checks that
    the third term, whose coefficient is 2*mu/sigma**2, drops out without
    producing 0 * -inf."""
    model = RBM(0.0, 1.0)
    x = np.array([0.2, 1.0, 2.5])
    got = model.transition_distribution(x, 1.0, 0.0, type="pdf")
    np.testing.assert_allclose(got, 2 * stats.norm.pdf(x), rtol=1e-12)
    assert np.all(np.isfinite(model.transition_distribution(x, 1.0, 0.0, type="logpdf")))


@pytest.mark.parametrize("mu,sigma,t", [(0.5, 1.0, 1.0), (-0.4, 1.5, 2.0),
                                        (1.0, 1.0, 0.5)])
def test_density_integrates_to_the_cdf(mu, sigma, t):
    """Ties pdf and cdf together, so neither can drift from the other."""
    model = RBM(mu, sigma)
    hi = max(mu * t, 0.0) + 7 * sigma * np.sqrt(t)
    grid = np.linspace(1e-9, hi, 20001)
    integral = np.trapezoid(
        model.transition_distribution(grid, t, 0.0, type="pdf"), grid
    )
    band = (model.transition_distribution(grid[-1], t, 0.0, type="cdf")
            - model.transition_distribution(grid[0], t, 0.0, type="cdf"))
    assert integral == pytest.approx(band, abs=1e-4)


def test_pdf_and_logpdf_agree():
    model = RBM(0.5, 1.0)
    x = np.array([0.1, 1.0, 4.0])
    np.testing.assert_allclose(
        model.transition_distribution(x, 1.0, 0.0, type="logpdf"),
        np.log(model.transition_distribution(x, 1.0, 0.0, type="pdf")),
        rtol=1e-10,
    )


def test_unknown_type_is_rejected():
    with pytest.raises(ValueError, match="type must be one of"):
        RBM(0.5, 1.0).transition_distribution(1.0, 1.0, 0.0, type="sf")


# ---------------------------------------------------------------- RBM fit ----
def test_rbm_fit_returns_fitresult(reflected):
    t, x = reflected
    result = RBM(MU, SIGMA).fit(t, x)
    assert isinstance(result, FitResult)
    assert result.names == ("mu", "sigma")
    assert result.params[1] > 0


def test_rbm_fit_produces_numbers(reflected):
    """Assert finiteness explicitly. An earlier version of this file checked
    only that cov[0, 0] != 1.0, which a NaN satisfies -- so it passed while the
    fit was returning NaN throughout."""
    t, x = reflected
    result = RBM(MU, SIGMA).fit(t, x)
    assert np.all(np.isfinite(result.params))
    assert np.all(np.isfinite(result.cov))
    assert np.all(np.isfinite(result.se))
    assert np.all(np.isfinite(result.ci))
    assert np.isfinite(result.nnlf)


def test_rbm_covariance_is_not_the_bfgs_placeholder(reflected):
    """The old default used res.hess_inv, which came back as the identity when
    BFGS made no progress -- a 'covariance' of exactly 1.0 for mu."""
    t, x = reflected
    result = RBM(MU, SIGMA).fit(t, x)
    assert np.all(np.isfinite(result.cov))
    assert result.cov[0, 0] != 1.0
    np.testing.assert_allclose(result.cov, result.cov.T, rtol=1e-10)
    assert np.all(np.diag(result.cov) > 0)


def test_rbm_recovers_the_generating_parameters(reflected):
    t, x = reflected
    result = RBM(MU, SIGMA).fit(t, x)
    assert abs(result.params[1] - SIGMA) < 0.3


def test_rbm_rejects_data_it_cannot_produce(path):
    """wiener_path dips below zero. That is outside the support, not merely
    unlikely, so it should be named rather than met later as a NaN."""
    t, x = path
    assert min(x[0]) < 0, "fixture no longer goes negative"
    with pytest.raises(ValueError, match="negative observation"):
        RBM(MU, SIGMA).fit(t, x)


def test_wiener_still_accepts_negative_data(path):
    """The support check belongs to RBM alone."""
    t, x = path
    assert np.all(np.isfinite(Wiener(MU, SIGMA).fit(t, x).params))


# ------------------------------------------------------- RBM quantile band ----
def test_quantile_band_brackets_the_median():
    model = RBM(0.5, 1.0)
    t = np.array([0.5, 1.0, 2.0])
    L, U = model.get_upper_lower(t, 0.0, alpha=0.025)
    assert np.all(L < U)
    assert np.all(L >= 0)


def test_quantile_band_inverts_the_cdf():
    """L and U must solve F(x) = alpha/2 and F(x) = 1 - alpha/2.

    alpha is a significance level, so the band is split between the two tails.
    """
    model = RBM(0.5, 1.0)
    t = np.array([0.5, 1.0, 2.0])
    alpha = 0.05
    L, U = model.get_upper_lower(t, 0.0, alpha=alpha)
    for ti, lo, hi in zip(t, L, U):
        assert model.transition_distribution(lo, ti, 0.0, type="cdf") == pytest.approx(alpha/2, abs=1e-6)
        assert model.transition_distribution(hi, ti, 0.0, type="cdf") == pytest.approx(1 - alpha/2, abs=1e-6)


@pytest.mark.parametrize("mu,sigma", [(1.0, 2.0), (0.5, 1.0), (-0.3, 1.0),
                                      (0.0, 1.0), (3.0, 0.5)])
def test_quantile_band_over_the_notebook_grid(mu, sigma):
    """The grid and parameters from the RBM example, which fsolve could not do.

    At t=2.1 with mu=1, sigma=2 the lower guess landed at 1e-8, where the cdf
    is 3.8e-08 of the 2.5% target -- flat enough that a local method reports no
    progress. The root is at 0.249, nowhere near the boundary.
    """
    model = RBM(mu, sigma)
    t = np.arange(0.1, 10.0, 0.1)
    L, U = model.get_upper_lower(t, 0.0)
    assert np.all(np.isfinite(L)) and np.all(np.isfinite(U))
    assert np.all(L >= 0)
    assert np.all(L < U)


def test_quantile_band_is_accurate_where_fsolve_stalled():
    model = RBM(1.0, 2.0)
    L, U = model.get_upper_lower(np.array([2.1]), 0.0, alpha=0.05)
    assert L[0] == pytest.approx(0.2489571025, abs=1e-8)
    assert model.transition_distribution(L[0], 2.1, 0.0, type="cdf") == pytest.approx(0.025, abs=1e-12)


def test_quantile_band_handles_a_tiny_time():
    """s = sigma*sqrt(t) collapses, so the bracket has to shrink with it."""
    L, U = RBM(1.0, 2.0).get_upper_lower(np.array([1e-6]), 0.0)
    assert 0 <= L[0] < U[0]


def test_quantile_band_is_monotone_in_alpha():
    model = RBM(1.0, 2.0)
    t = np.array([1.0, 5.0])
    narrow_L, narrow_U = model.get_upper_lower(t, 0.0, alpha=0.20)
    wide_L, wide_U = model.get_upper_lower(t, 0.0, alpha=0.01)
    assert np.all(wide_L < narrow_L)
    assert np.all(wide_U > narrow_U)


def test_quantile_band_returns_scalars_not_arrays():
    """fsolve returns a length-1 array; assigning it into a slot is an error
    under numpy 2."""
    L, U = RBM(0.5, 1.0).get_upper_lower(np.array([1.0]), 0.0)
    assert L.shape == (1,) and U.shape == (1,)
    assert np.isscalar(L[0]) or L[0].ndim == 0
