"""Regression tests for the six bugs fixed during the maintkit refactor."""
import numpy as np
import pytest

from maintkit.poisson_process import poisson_process, power_law_nhpp
from maintkit.distributions import expdist
from maintkit.wiener import Wiener, weiner
from maintkit.utilities import _parameter_transform_log


class ConstPP(poisson_process):
    """Homogeneous Poisson process used to exercise the base-class methods."""
    def intensity(self, t):
        return self.parameters[0]


def test_base_intensity_raises_not_implemented():
    pp = poisson_process([1.0])
    with pytest.raises(NotImplementedError):
        pp.intensity(1.0)


def test_power_law_intensity_is_numeric():
    # Previously misspelled 'intesity', so this silently returned a string.
    m = power_law_nhpp(a=0.01, b=1.5)
    val = m.intensity(4.0)
    assert np.isclose(val, 0.01 * 1.5 * 4.0 ** 0.5)


def test_cumulative_intensity_scalar_t0():
    # Previously crashed on len(list(t0)) when t0 was a scalar.
    pp = ConstPP([2.0])
    out = pp.cumulative_intensity([1.0, 2.0, 3.0], t0=0)
    assert np.allclose(out, [2.0, 4.0, 6.0])


def test_cumulative_intensity_vector_t0():
    pp = ConstPP([2.0])
    out = pp.cumulative_intensity([2.0, 4.0], t0=[1.0, 3.0])
    assert np.allclose(out, [2.0, 2.0])


def test_nnlf_truncation_none_does_not_crash():
    # Previously crashed indexing truncation_times[m] when it was None.
    m = power_law_nhpp(a=0.01, b=1.5)
    val = m.nnlf([0.01, 1.5], [[1.0, 2.0, 3.0]], truncation_times=None)
    assert np.isfinite(val)


def test_nnlf_with_truncation():
    m = power_law_nhpp(a=0.01, b=1.5)
    val = m.nnlf([0.01, 1.5], [[1.0, 2.0, 3.0]], truncation_times=[10.0])
    assert np.isfinite(val)


def test_expdist_fit_with_array_censoring():
    # Previously raised on `observed == "all"` when observed was an array.
    rng = np.random.default_rng(0)
    ti = rng.exponential(scale=50.0, size=100)
    observed = np.ones(100)
    observed[::5] = 0  # some right-censored
    res = expdist().fit(ti, observed=observed)
    r = observed.sum()
    assert np.isclose(res.params[0], ti.sum() / r)


def test_wiener_alias_and_symmetric_covariance():
    assert weiner is Wiener
    w = Wiener(mu=0.5, sigma=1.0)
    np.random.seed(0)
    times = list(np.linspace(0, 50, 200))
    x = w.simulate(times, num_samples=1)
    mu, sigma, p_cov = w.estimate_parameters([times], [x[0].tolist()])
    assert np.allclose(p_cov, p_cov.T)          # symmetric sandwich J.T Hi J
    assert np.all(np.diag(p_cov) > 0)
    assert abs(mu - 0.5) < 0.3


# ------------------------------------------------- frozen distribution plot ----
@pytest.mark.parametrize(
    "kind", ["pdf", "cdf", "reliability", "hazard", "conditional_reliability"]
)
def test_every_plot_type_works(kind):
    """Two bugs met here.

    The frozen class inherited from ``rv_frozen``, which has no ``pdf`` -- the
    density lives on ``rv_continuous_frozen``. And ``plot`` built its dispatch
    table out of bound methods, so asking for any curve looked up all five and
    hit the missing one regardless of what was requested.
    """
    import matplotlib
    matplotlib.use("Agg")
    from maintkit.distributions import weibull

    dist = weibull()(2.0, scale=100.0)
    kwds = {"t0": 10.0} if kind == "conditional_reliability" else {}
    ax = dist.plot(type=kind, **kwds)
    line, = ax.get_lines()
    assert np.all(np.isfinite(line.get_ydata()))


def test_frozen_distribution_has_a_density():
    from maintkit.distributions import weibull
    dist = weibull()(2.0, scale=100.0)
    assert hasattr(dist, "pdf")
    assert dist.pdf(50.0) > 0


def test_plot_rejects_an_unknown_type():
    from maintkit.distributions import weibull
    with pytest.raises(ValueError, match="type must be one of"):
        weibull()(2.0, scale=100.0).plot(type="survival")


def test_conditional_reliability_needs_t0():
    from maintkit.distributions import weibull
    with pytest.raises(ValueError, match="t0 must be specified"):
        weibull()(2.0, scale=100.0).plot(type="conditional_reliability")
