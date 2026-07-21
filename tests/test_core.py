"""Core functionality tests: fitting paths and parameter transforms."""
import numpy as np
import scipy.stats as sps

from maintkit.distributions import expdist, weibull
from maintkit.poisson_process import power_law_nhpp
from maintkit.utilities import _parameter_transform_log, _parameter_transform_identity


def test_parameter_transform_log_roundtrip():
    x = np.array([0.01, 2.5])
    fwd = _parameter_transform_log(x, direction="forward")
    back = _parameter_transform_log(fwd, direction="inverse")
    assert np.allclose(back, x)


def test_parameter_transform_identity_roundtrip():
    x = np.array([1.0, 2.0, 3.0])
    assert np.allclose(_parameter_transform_identity(x), x)


def test_expdist_closed_form_matches_analytic():
    rng = np.random.default_rng(1)
    ti = rng.exponential(scale=25.0, size=500)
    res = expdist().fit(ti, observed="all")
    assert np.isclose(res.params[0], ti.sum() / len(ti))


def test_weibull_mle_recovers_parameters():
    beta_true, eta_true = 2.0, 100.0
    rng = np.random.default_rng(2)
    ti = sps.weibull_min.rvs(beta_true, scale=eta_true, size=2000, random_state=rng)
    res = weibull().fit(ti, p0=[80.0, 1.5])
    eta_hat, beta_hat = res.params
    assert abs(eta_hat - eta_true) / eta_true < 0.1
    assert abs(beta_hat - beta_true) / beta_true < 0.1
    # 95% CIs should bracket the truth for a sample this large
    assert res.ci[0, 0] < eta_true < res.ci[0, 1]
    assert res.ci[1, 0] < beta_true < res.ci[1, 1]


def test_power_law_nhpp_fit_recovers_shape():
    a_true, b_true = 0.02, 1.5
    m = power_law_nhpp(a_true, b_true)
    np.random.seed(3)
    T = 200
    n_assets = 40
    event_times = m.random_arrival_times(T, size=n_assets)
    truncation_times = [float(T)] * n_assets
    res = m.fit(event_times, truncation_times=truncation_times)
    a_hat, b_hat = res.params
    assert abs(b_hat - b_true) / b_true < 0.25
    assert a_hat > 0
