"""Tests for ``PoissonProcess.reliability`` and ``PoissonProcess.pdf``.

``reliability`` previously had the signature ``(t, w)`` where ``t`` was the
start of the window and ``w`` its *end* -- names that inverted the meaning of
the arguments -- and it accepted no ``t0`` keyword, so ``pdf`` raised
``TypeError`` on every call. The signature is now ``(w, t0=0)``: a window
length measured forward from an origin.

The origin has to be an argument. For a non-homogeneous process the survival
probability over a window of length ``w`` depends on where that window sits,
so a ``reliability(w)`` that silently assumed ``t0 = 0`` would be wrong
everywhere except at the start of life.
"""
import numpy as np
import pytest
from scipy import stats
from scipy.integrate import quad

from maintkit.poisson_process import PoissonProcess, PowerLawNHPP


class ConstPP(PoissonProcess):
    """Homogeneous process, so the base-class quadrature path is exercised."""
    def intensity(self, t):
        return self.parameters[0]


# ------------------------------------------------------------- basic form ----
def test_reliability_of_a_zero_length_window_is_one():
    assert ConstPP([2.0]).reliability(0.0) == pytest.approx(1.0)
    assert PowerLawNHPP(a=0.01, b=1.5).reliability(0.0, t0=7.0) == pytest.approx(1.0)


def test_reliability_matches_the_closed_form_for_the_power_law():
    a, b = 0.01, 1.5
    m = PowerLawNHPP(a=a, b=b)
    t0, w = 4.0, 3.0
    want = np.exp(-a * ((t0 + w) ** b - t0 ** b))
    assert m.reliability(w, t0=t0) == pytest.approx(want)


def test_reliability_is_consistent_with_cumulative_intensity():
    """The definition, spelled out: R = exp(-[Lambda(t0+w) - Lambda(t0)])."""
    m = PowerLawNHPP(a=0.02, b=0.8)
    t0, w = 5.0, 11.0
    LAMBDA = m.cumulative_intensity(t0 + w, t0=t0)
    assert m.reliability(w, t0=t0) == pytest.approx(np.exp(-LAMBDA))


def test_homogeneous_reliability_is_exponential_and_ignores_the_origin():
    """The one case where an origin-free signature would have been harmless:
    an HPP has stationary increments, so R depends on w alone."""
    lam = 0.3
    pp = ConstPP([lam])
    w = 4.0
    assert pp.reliability(w) == pytest.approx(np.exp(-lam * w))
    for t0 in (0.0, 2.5, 100.0):
        assert pp.reliability(w, t0=t0) == pytest.approx(np.exp(-lam * w))


def test_reliability_depends_on_the_origin_for_a_non_homogeneous_process():
    """The reason ``t0`` exists. With b > 1 the intensity increases, so the
    same window is less survivable later in life."""
    m = PowerLawNHPP(a=0.01, b=2.0)
    w = 3.0
    early = m.reliability(w, t0=1.0)
    late = m.reliability(w, t0=50.0)
    assert late < early
    assert not np.isclose(early, m.reliability(w))


def test_reliability_decreases_with_the_window_length():
    m = PowerLawNHPP(a=0.01, b=1.5)
    R = m.reliability(np.linspace(0.0, 50.0, 25), t0=10.0)
    assert np.all(np.diff(R) < 0)
    assert np.all((R > 0) & (R <= 1))


# ------------------------------------------------------- shapes and errors ----
def test_scalar_arguments_give_back_a_scalar():
    R = PowerLawNHPP(a=0.01, b=1.5).reliability(2.0, t0=1.0)
    assert np.ndim(R) == 0


def test_vector_window_with_a_scalar_origin():
    a, b = 0.01, 1.5
    m = PowerLawNHPP(a=a, b=b)
    w = np.array([1.0, 2.0, 3.0])
    t0 = 4.0
    got = m.reliability(w, t0=t0)
    assert got.shape == w.shape
    assert np.allclose(got, np.exp(-a * ((t0 + w) ** b - t0 ** b)))


def test_vector_window_with_a_vector_origin():
    a, b = 0.01, 1.5
    m = PowerLawNHPP(a=a, b=b)
    w = np.array([1.0, 2.0])
    t0 = np.array([3.0, 30.0])
    got = m.reliability(w, t0=t0)
    assert got.shape == w.shape
    assert np.allclose(got, np.exp(-a * ((t0 + w) ** b - t0 ** b)))


def test_a_negative_window_is_rejected():
    """Guards the old calling convention: someone passing an absolute end time
    smaller than the origin used to get a silently reversed interval, i.e. a
    'reliability' above 1."""
    with pytest.raises(ValueError, match="non-negative"):
        PowerLawNHPP(a=0.01, b=1.5).reliability(-1.0, t0=5.0)


# -------------------------------------------------------------------- pdf ----
def test_pdf_no_longer_raises():
    """It used to call ``reliability(w, t0=...)`` against a signature with no
    ``t0``, so every call was a TypeError."""
    val = PowerLawNHPP(a=0.01, b=1.5).pdf(6.0, t_previous=2.0)
    assert np.isfinite(val)
    assert val > 0


def test_pdf_matches_the_intensity_times_survival_definition():
    a, b = 0.01, 1.5
    m = PowerLawNHPP(a=a, b=b)
    t, t_prev = 9.0, 4.0
    want = a * b * t ** (b - 1) * np.exp(-a * (t ** b - t_prev ** b))
    assert m.pdf(t, t_previous=t_prev) == pytest.approx(want)


def test_homogeneous_pdf_is_the_exponential_density():
    """An HPP's waiting time is exactly Exp(lambda), measured from t_previous."""
    lam = 0.3
    pp = ConstPP([lam])
    t_prev = 5.0
    t = np.array([5.5, 7.0, 12.0])
    got = np.array([pp.pdf(ti, t_previous=t_prev) for ti in t])
    assert np.allclose(got, stats.expon.pdf(t - t_prev, scale=1 / lam))


@pytest.mark.parametrize("t_previous", [0.5, 4.0, 20.0])
def test_pdf_integrates_to_one_over_the_waiting_time(t_previous):
    """A density for the next event time, so it must integrate to 1 from
    ``t_previous`` onwards -- b > 1 guarantees an event occurs eventually.

    Integrated to a finite upper limit rather than ``np.inf``: the density is
    concentrated within a few units of ``t_previous`` and quad's infinite-range
    transformation steps straight over it. The limit is where the cumulative
    intensity has grown by 40, i.e. survival below 1e-17.
    """
    a, b = 0.05, 2.0
    m = PowerLawNHPP(a=a, b=b)
    upper = (t_previous ** b + 40.0 / a) ** (1 / b)
    total, _ = quad(lambda t: m.pdf(t, t_previous=t_previous),
                    t_previous, upper, limit=200)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_pdf_is_the_derivative_of_one_minus_reliability():
    """f(t) = -d/dw R(w | t0) at w = t - t0. Checked numerically rather than
    assumed, since the two methods are written independently."""
    m = PowerLawNHPP(a=0.01, b=1.5)
    t0, w, h = 4.0, 6.0, 1e-5
    slope = -(m.reliability(w + h, t0=t0) - m.reliability(w - h, t0=t0)) / (2 * h)
    assert m.pdf(t0 + w, t_previous=t0) == pytest.approx(slope, rel=1e-6)


# ------------------------------------------------------------- simulation ----
def test_reliability_agrees_with_simulated_first_event_times():
    """End-to-end check against the process itself: the fraction of assets with
    no event in (t0, t0+w] should be R(w | t0)."""
    rng = np.random.default_rng(20260726)
    lam = 0.05
    pp = ConstPP([lam])
    t0, w, n = 10.0, 8.0, 20000

    # An HPP's count over the window is Poisson with mean lambda*w.
    counts = rng.poisson(lam * w, size=n)
    empirical = np.mean(counts == 0)
    assert empirical == pytest.approx(pp.reliability(w, t0=t0), abs=0.01)
