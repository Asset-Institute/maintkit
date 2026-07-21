"""Regression tests for the six bugs fixed during the maintkit refactor."""
import numpy as np
import pytest

from maintkit.poisson_process import PoissonProcess, PowerLawNHPP
from maintkit.distributions import Exponential
from maintkit.wiener import Wiener


class ConstPP(PoissonProcess):
    """Homogeneous Poisson process used to exercise the base-class methods."""
    def intensity(self, t):
        return self.parameters[0]


def test_base_intensity_raises_not_implemented():
    pp = PoissonProcess([1.0])
    with pytest.raises(NotImplementedError):
        pp.intensity(1.0)


def test_power_law_intensity_is_numeric():
    # Previously misspelled 'intesity', so this silently returned a string.
    m = PowerLawNHPP(a=0.01, b=1.5)
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
    m = PowerLawNHPP(a=0.01, b=1.5)
    val = m.nnlf([0.01, 1.5], [[1.0, 2.0, 3.0]], truncation_times=None)
    assert np.isfinite(val)


def test_nnlf_with_truncation():
    m = PowerLawNHPP(a=0.01, b=1.5)
    val = m.nnlf([0.01, 1.5], [[1.0, 2.0, 3.0]], truncation_times=[10.0])
    assert np.isfinite(val)


def test_expdist_fit_with_array_censoring():
    # Previously raised on `observed == "all"` when observed was an array.
    rng = np.random.default_rng(0)
    ti = rng.exponential(scale=50.0, size=100)
    observed = np.ones(100)
    observed[::5] = 0  # some right-censored
    res = Exponential().fit(ti, observed=observed)
    r = observed.sum()
    assert np.isclose(res.params[0], ti.sum() / r)


def test_wiener_covariance_is_symmetric():
    w = Wiener(mu=0.5, sigma=1.0)
    np.random.seed(0)
    times = list(np.linspace(0, 50, 200))
    x = w.simulate(times, num_samples=1)
    res = w.fit([times], [x[0].tolist()])
    assert np.allclose(res.cov, res.cov.T)
    assert np.all(np.diag(res.cov) > 0)
    assert abs(res.params[0] - 0.5) < 0.3


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
    from maintkit.distributions import Weibull

    dist = Weibull()(2.0, scale=100.0)
    kwds = {"t0": 10.0} if kind == "conditional_reliability" else {}
    ax = dist.plot(type=kind, **kwds)
    line, = ax.get_lines()
    assert np.all(np.isfinite(line.get_ydata()))


def test_frozen_distribution_has_a_density():
    from maintkit.distributions import Weibull
    dist = Weibull()(2.0, scale=100.0)
    assert hasattr(dist, "pdf")
    assert dist.pdf(50.0) > 0


def test_plot_rejects_an_unknown_type():
    from maintkit.distributions import Weibull
    with pytest.raises(ValueError, match="type must be one of"):
        Weibull()(2.0, scale=100.0).plot(type="survival")


def test_conditional_reliability_needs_t0():
    from maintkit.distributions import Weibull
    with pytest.raises(ValueError, match="t0 must be specified"):
        Weibull()(2.0, scale=100.0).plot(type="conditional_reliability")


# -------------------------------------------- frozen distribution support ----
def test_frozen_distribution_has_a_support():
    """__init__ never set self.a / self.b, so anything reading them raised
    AttributeError. It cannot call super().__init__() to get them: scipy
    rebuilds the distribution from its constructor parameters, which drops
    whatever a subclass needs beyond them."""
    from maintkit.distributions import Weibull
    dist = Weibull()(2.0, scale=100.0)
    lo, hi = dist.support()
    assert lo == 0
    assert np.isinf(hi)


def test_frozen_distribution_interval_works():
    from maintkit.distributions import Weibull
    dist = Weibull()(2.0, scale=100.0)
    lo, hi = dist.interval(0.95)
    assert 0 < lo < hi
    assert dist.cdf(hi) - dist.cdf(lo) == pytest.approx(0.95, abs=1e-9)


def test_freezing_keeps_the_instance_it_was_given():
    """Not a rebuilt copy -- the rebuild is what would lose subclass state."""
    from maintkit.distributions import Weibull
    d = Weibull()
    assert d(2.0, scale=100.0).dist is d


def test_reliability_from_hazard_can_be_frozen():
    """The case that makes scipy's rebuild impossible: its constructor takes a
    hazard function, which is not an rv_continuous constructor parameter, so
    dist.__class__(**dist._updated_ctor_param()) would raise TypeError.

    Only freezing is checked here. Evaluating one of these is broken for a
    separate, older reason: integrate_hazard's single-point branch passes the
    whole array to quad instead of its element, and its multi-point branch
    never integrates from 0 to t[0].
    """
    from maintkit.distributions import ReliabilityFromHazard
    dist = ReliabilityFromHazard(lambda t: 0.02 * np.ones_like(np.asarray(t, float)))
    frozen = dist()
    assert frozen.dist is dist
    lo, hi = frozen.support()
    assert lo == 0
    assert np.isinf(hi)
