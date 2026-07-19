"""Unit tests for maintkit.inference.

These use analytically-known likelihoods so the machinery is checked against
closed-form answers rather than against itself.
"""
import numpy as np
import pytest

from maintkit.inference import FitResult, fit_mle, result_at, hessian_at
from maintkit.transforms import Identity, Log, Composite, Logit


# ---------------------------------------------------------------- fixtures --
@pytest.fixture
def normal_sample():
    rng = np.random.default_rng(0)
    return rng.normal(loc=5.0, scale=2.0, size=2000)


def normal_nnlf_factory(x):
    """NNLF of a normal with parameters (mu, sigma). MLEs are known exactly."""
    def nnlf(p):
        mu, sigma = p
        return 0.5 * len(x) * np.log(2 * np.pi * sigma**2) + np.sum(
            (x - mu) ** 2
        ) / (2 * sigma**2)
    return nnlf


def exponential_nnlf_factory(t):
    """NNLF of an exponential with mean theta. MLE is exactly mean(t)."""
    def nnlf(p):
        theta = np.atleast_1d(p)[0]
        return len(t) * np.log(theta) + np.sum(t) / theta
    return nnlf


# -------------------------------------------------------------------- fits --
def test_recovers_normal_mle(normal_sample):
    x = normal_sample
    res = fit_mle(
        normal_nnlf_factory(x),
        p0=[0.0, 1.0],
        transform=Composite([Identity(), Log()]),
        names=["mu", "sigma"],
    )
    assert res.success
    # closed-form MLEs
    assert res.params[0] == pytest.approx(x.mean(), rel=1e-4)
    assert res.params[1] == pytest.approx(x.std(ddof=0), rel=1e-3)


def test_standard_error_matches_analytic(normal_sample):
    """se(mu) = sigma / sqrt(n) exactly for a normal."""
    x = normal_sample
    res = fit_mle(
        normal_nnlf_factory(x), p0=[0.0, 1.0],
        transform=Composite([Identity(), Log()]),
    )
    expected = x.std(ddof=0) / np.sqrt(len(x))
    assert res.se[0] == pytest.approx(expected, rel=1e-3)


def test_exponential_closed_form_via_result_at():
    rng = np.random.default_rng(1)
    t = rng.exponential(scale=25.0, size=1000)
    theta_hat = t.mean()
    res = result_at(
        exponential_nnlf_factory(t), [theta_hat],
        transform=Log(), names=["theta"],
    )
    assert res.params[0] == pytest.approx(theta_hat)
    # Fisher information for exponential mean: n/theta^2 -> se = theta/sqrt(n)
    assert res.se[0] == pytest.approx(theta_hat / np.sqrt(len(t)), rel=1e-3)
    assert res.message == "closed-form estimate"


# ------------------------------------------------------------------- CIs ----
def test_transformed_ci_is_positive_and_asymmetric():
    rng = np.random.default_rng(2)
    t = rng.exponential(scale=2.0, size=12)      # small sample on purpose
    res = fit_mle(exponential_nnlf_factory(t), p0=[1.0], transform=Log())
    lo, hi = res.ci[0]
    assert lo > 0                                  # the point of the log scale
    assert (hi - res.params[0]) > (res.params[0] - lo)


def test_natural_ci_can_go_negative_transformed_does_not():
    """Documents exactly why the default convention changed in P2."""
    rng = np.random.default_rng(3)
    t = rng.exponential(scale=2.0, size=6)
    kw = dict(p0=[1.0], transform=Log())
    nat = fit_mle(exponential_nnlf_factory(t), ci_method="natural", **kw)
    tra = fit_mle(exponential_nnlf_factory(t), ci_method="transformed", **kw)
    assert np.allclose(nat.params, tra.params)     # same estimate
    assert np.allclose(nat.cov, tra.cov)           # same covariance
    assert tra.ci[0, 0] > 0                        # only the interval differs
    assert nat.ci[0, 0] < tra.ci[0, 0]


def test_alpha_widens_interval():
    rng = np.random.default_rng(4)
    t = rng.exponential(scale=10.0, size=200)
    narrow = fit_mle(exponential_nnlf_factory(t), p0=[1.0], transform=Log(), alpha=0.05)
    wide = fit_mle(exponential_nnlf_factory(t), p0=[1.0], transform=Log(), alpha=0.01)
    assert wide.ci[0, 0] < narrow.ci[0, 0]
    assert wide.ci[0, 1] > narrow.ci[0, 1]


def test_ci_shape_is_always_n_by_2(normal_sample):
    res = fit_mle(
        normal_nnlf_factory(normal_sample), p0=[0.0, 1.0],
        transform=Composite([Identity(), Log()]),
    )
    assert res.ci.shape == (2, 2)
    assert np.all(res.ci[:, 0] < res.ci[:, 1])


def test_invalid_ci_method_raises(normal_sample):
    with pytest.raises(ValueError, match="ci_method"):
        fit_mle(
            normal_nnlf_factory(normal_sample), p0=[0.0, 1.0],
            transform=Composite([Identity(), Log()]), ci_method="nonsense",
        )


# --------------------------------------------------------------- FitResult --
def test_fitresult_unpacks_as_legacy_triple():
    rng = np.random.default_rng(5)
    t = rng.exponential(scale=3.0, size=100)
    res = fit_mle(exponential_nnlf_factory(t), p0=[1.0], transform=Log())
    p_hat, p_ci, p_cov = res                      # back-compat path
    assert np.allclose(p_hat, res.params)
    assert np.allclose(p_ci, res.ci)
    assert np.allclose(p_cov, res.cov)


def test_summary_renders(normal_sample):
    res = fit_mle(
        normal_nnlf_factory(normal_sample), p0=[0.0, 1.0],
        transform=Composite([Identity(), Log()]), names=["mu", "sigma"],
    )
    text = res.summary()
    assert "mu" in text and "sigma" in text and "95%" in text
    assert str(res) == text


def test_transform_type_is_validated(normal_sample):
    with pytest.raises(TypeError):
        fit_mle(normal_nnlf_factory(normal_sample), p0=[0.0, 1.0], transform="log")


def test_singular_hessian_raises_informative_error():
    """A likelihood flat in one direction should say so, not emit a bare LinAlgError."""
    def flat(p):
        return (p[0] - 1.0) ** 2          # p[1] does not appear
    with pytest.raises(np.linalg.LinAlgError, match="singular"):
        fit_mle(flat, p0=[0.5, 0.5], transform=Identity())


def test_hessian_at_returns_y_and_matrix():
    rng = np.random.default_rng(6)
    t = rng.exponential(scale=4.0, size=50)
    y_hat, H = hessian_at(exponential_nnlf_factory(t), [t.mean()], transform=Log())
    assert y_hat.shape == (1,)
    assert H.shape == (1, 1)
    assert H[0, 0] > 0                     # minimum -> positive curvature
