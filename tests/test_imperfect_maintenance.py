"""Cumulative intensity for the imperfect-maintenance model.

The previous implementation raised IndexError for every input, so none of this
was covered. The j=0 window is (s_0, tau_0) = (0, 0), empty by construction, and
the carry-forward line indexed the empty result on the first loop iteration.
"""
from __future__ import annotations

import numpy as np
import pytest

from maintkit.imperfect_maintenance import ProportionalAgeReduction
from maintkit.inference import FitResult
from tests import _datasets as ds

A, B, RHO = 0.02, 1.5, 0.4
PM = np.array([50.0, 100.0, 150.0, 200.0, 250.0])   # actual PMs only
T = 300.0                                          # observation ends here


@pytest.fixture(scope="module")
def model():
    return ProportionalAgeReduction(A, B, RHO)


def _numerical_integral(t, pm, a=A, b=B, rho=RHO, n=200_000):
    """Integrate the intensity directly, as an independent reference."""
    pm = np.asarray(pm, float)
    out = []
    for tt in np.atleast_1d(np.asarray(t, float)):
        if tt <= 0:
            out.append(0.0)
            continue
        grid = np.linspace(0.0, tt, n)
        idx = np.searchsorted(pm, grid, side="left") - 1
        last = np.where(idx < 0, 0.0, pm[np.clip(idx, 0, pm.size - 1)])
        lam = a * b * np.maximum(grid - rho * last, 1e-300) ** (b - 1)
        out.append(np.trapezoid(lam, grid))
    return np.array(out)


# --------------------------------------------------------- does it run at all --
def test_does_not_raise(model):
    """The old version raised IndexError on every call."""
    assert np.all(np.isfinite(model.cumulative_intensity(np.array([75.0]), PM, T)))


@pytest.mark.parametrize(
    "t",
    [
        np.array([120.0, 260.0]),          # sparse: most PM windows contain no points
        PM.copy(),                         # exactly on the PM times
        np.array([175.0]),                 # a single point
        np.linspace(1.0, 90.0, 20),        # grid stopping well before the last PM
    ],
)
def test_awkward_grids_are_handled(model, t):
    """Each of these left gaps or raised under the loop-and-carry approach."""
    M = model.cumulative_intensity(t, PM, T)
    assert M.shape == t.shape
    assert np.all(np.isfinite(M))
    assert np.all(M >= 0)


# ------------------------------------------------------------- correctness ----
def test_matches_numerical_integration(model):
    t = np.array([10.0, 49.9, 50.0, 50.1, 75.0, 120.0, 175.0, 240.0, 299.0])
    np.testing.assert_allclose(
        model.cumulative_intensity(t, PM, T), _numerical_integral(t, PM),
        rtol=1e-5,
    )


def test_is_non_decreasing(model):
    M = model.cumulative_intensity(np.linspace(0.0, 300.0, 4000), PM, T)
    assert np.all(np.diff(M) >= -1e-12)


def test_starts_at_zero(model):
    assert model.cumulative_intensity(np.array([0.0]), PM, T)[0] == pytest.approx(0.0)


def test_agrees_with_the_likelihood(model):
    """At the PM times M must equal the running total of a * total_exposure.

    total_exposure is the sum nnlf uses, so this ties the two together and
    stops them drifting apart.
    """
    edges = model._interval_edges(PM, T)
    s = model._last_pm(edges, edges)
    w, v = edges - RHO * s, (1.0 - RHO) * s
    running = np.cumsum(A * w**B - A * v**B)
    np.testing.assert_allclose(
        model.cumulative_intensity(edges, PM, T), running, rtol=1e-12
    )


def test_reduces_to_the_baseline_when_rho_is_zero():
    """With no age reduction the result is just the power-law cumulative."""
    m = ProportionalAgeReduction(A, B, 0.0)
    t = np.linspace(0.0, 300.0, 500)
    np.testing.assert_allclose(m.cumulative_intensity(t, PM, T), A * t**B, rtol=1e-12)


def test_points_on_pm_times_are_not_skipped(model):
    """The old windows were open at both ends, so these were never written."""
    M = model.cumulative_intensity(PM, PM, T)
    assert np.all(M > 0)


# ------------------------------------------------------------------- fit ----
# Frozen from the pre-migration implementation (generated, not transcribed).
LEGACY_FIT_PARAMS = np.array([0.020605633793343875,
                              1.52831926573264,
                              0.637672515910584])


@pytest.fixture(scope="module")
def data():
    return ds.imperfect_maintenance_data()


@pytest.fixture(scope="module")
def fit_result(data):
    failures, pm, trunc = data
    return ProportionalAgeReduction(A, B, RHO).fit(failures, pm, trunc)


def test_fit_returns_fitresult(fit_result):
    assert isinstance(fit_result, FitResult)
    assert fit_result.names == ("a", "b", "rho")


def test_fit_ci_has_the_standard_shape(fit_result):
    """Was (2, 3); every other fitter returns (n_params, 2)."""
    assert fit_result.ci.shape == (3, 2)


def test_fit_estimates_unchanged(fit_result):
    np.testing.assert_allclose(fit_result.params, LEGACY_FIT_PARAMS, rtol=1e-6)


def test_fit_now_returns_a_covariance(fit_result):
    """The 2-tuple return had no covariance at all."""
    assert fit_result.cov.shape == (3, 3)
    assert np.all(np.diag(fit_result.cov) > 0)
    np.testing.assert_allclose(fit_result.cov, fit_result.cov.T, rtol=1e-10)


def test_fit_ci_brackets_the_estimate(fit_result):
    assert np.all(fit_result.ci[:, 0] < fit_result.params)
    assert np.all(fit_result.params < fit_result.ci[:, 1])


def test_fit_rho_ci_stays_inside_the_unit_interval(fit_result):
    """The logit transform is what guarantees this."""
    lo, hi = fit_result.ci[2]
    assert 0.0 < lo < hi < 1.0


def test_fit_a_is_at_its_conditional_maximum(fit_result, data):
    """a is concentrated out, so a = N / total_exposure must hold exactly."""
    failures, pm, trunc = data
    m = ProportionalAgeReduction(A, B, RHO)
    _, edges = m._prepare(failures, pm, trunc)
    a_hat, b_hat, rho_hat = fit_result.params
    expected = len(failures) / m._total_exposure(b_hat, rho_hat, edges)
    assert a_hat == pytest.approx(expected, rel=1e-12)


def test_p0_takes_two_entries_not_three(data):
    failures, pm, trunc = data
    m = ProportionalAgeReduction(A, B, RHO)
    with pytest.raises(ValueError, match="two"):
        m.fit(failures, pm, trunc, p0=[0.02, 1.5, 0.4])


def test_p0_changes_the_start_not_the_answer(data, fit_result):
    """A different starting point should reach the same optimum."""
    failures, pm, trunc = data
    other = ProportionalAgeReduction(A, B, RHO).fit(
        failures, pm, trunc, p0=[2.0, 0.3]
    )
    np.testing.assert_allclose(other.params, fit_result.params, rtol=1e-4)


def test_alpha_widens_the_interval(data):
    failures, pm, trunc = data
    m = ProportionalAgeReduction(A, B, RHO)
    narrow = m.fit(failures, pm, trunc, alpha=0.05)
    wide = m.fit(failures, pm, trunc, alpha=0.01)
    assert np.all(wide.ci[:, 0] < narrow.ci[:, 0])
    assert np.all(wide.ci[:, 1] > narrow.ci[:, 1])


# ---------------------------------------------------------- many assets ----
@pytest.fixture(scope="module")
def fleet():
    return ds.imperfect_maintenance_fleet()


def test_promoting_one_asset_leaves_the_likelihood_untouched(data):
    """A flat sequence and a one-entry list must agree to the last bit.

    This is what makes the recorded reference values still meaningful: the
    single-asset case is not merely close after the change, it is identical.
    """
    failures, pm, trunc = data
    m = ProportionalAgeReduction(A, B, RHO)
    for p in [(0.02, 1.5, 0.4), (0.0206, 1.5283, 0.6377), (0.5, 0.8, 0.05)]:
        flat = m.nnlf(p, failures, pm, trunc)
        listed = m.nnlf(p, [failures], [pm], [trunc])
        assert flat == listed


def test_two_copies_of_one_asset_double_the_likelihood(data):
    """Assets are independent given the parameters, so -l is additive."""
    failures, pm, trunc = data
    m = ProportionalAgeReduction(A, B, RHO)
    p = (0.02, 1.5, 0.4)
    one = m.nnlf(p, [failures], [pm], [trunc])
    two = m.nnlf(p, [failures, failures], [pm, pm], [trunc, trunc])
    assert two == pytest.approx(2.0 * one, rel=1e-12)


def test_duplicated_asset_gives_the_same_estimates(data, fit_result):
    """Twice the data at the same parameters: estimates hold, errors shrink."""
    failures, pm, trunc = data
    doubled = ProportionalAgeReduction(A, B, RHO).fit(
        [failures, failures], [pm, pm], [trunc, trunc]
    )
    np.testing.assert_allclose(doubled.params, fit_result.params, rtol=1e-5)
    # Twice the information, so standard errors fall by about sqrt(2).
    np.testing.assert_allclose(doubled.se, fit_result.se / np.sqrt(2.0), rtol=1e-3)


def test_fleet_with_ragged_schedules_fits(fleet):
    failures, pm, trunc = fleet
    result = ProportionalAgeReduction(A, B, RHO).fit(failures, pm, trunc)
    assert np.all(np.isfinite(result.params))
    assert np.all(result.params[:2] > 0)
    assert 0.0 < result.params[2] < 1.0
    assert result.cov.shape == (3, 3)


def test_fleet_recovers_the_generating_parameters(fleet):
    """Loose bounds -- this checks the fleet likelihood is assembled right,
    not that 226 failures pin three parameters precisely."""
    failures, pm, trunc = fleet
    result = ProportionalAgeReduction(A, B, RHO).fit(failures, pm, trunc)
    a_hat, b_hat, rho_hat = result.params
    assert 0.005 < a_hat < 0.08
    assert 1.0 < b_hat < 2.2
    assert 0.0 < rho_hat < 1.0


def test_asset_with_no_failures_still_counts(fleet):
    """It contributes exposure but no events, so dropping it moves the fit."""
    failures, pm, trunc = fleet
    assert not failures[4], "fixture no longer has a failure-free asset"
    m = ProportionalAgeReduction(A, B, RHO)
    p = (0.02, 1.5, 0.4)
    with_it = m.nnlf(p, failures, pm, trunc)
    without = m.nnlf(p, failures[:4], pm[:4], trunc[:4])
    assert with_it != without


def test_asset_with_no_maintenance_is_allowed(fleet):
    failures, pm, trunc = fleet
    assert not pm[2], "fixture no longer has an unmaintained asset"
    result = ProportionalAgeReduction(A, B, RHO).fit(failures, pm, trunc)
    assert np.all(np.isfinite(result.params))


def test_scalar_truncation_time_applies_to_every_asset(data):
    failures, pm, trunc = data
    m = ProportionalAgeReduction(A, B, RHO)
    p = (0.02, 1.5, 0.4)
    scalar = m.nnlf(p, [failures, failures], [pm, pm], trunc)
    listed = m.nnlf(p, [failures, failures], [pm, pm], [trunc, trunc])
    assert scalar == listed


def test_asset_counts_must_agree(data):
    failures, pm, trunc = data
    m = ProportionalAgeReduction(A, B, RHO)
    with pytest.raises(ValueError, match="own schedule"):
        m.fit([failures, failures], [pm], trunc)


def test_wrong_number_of_truncation_times_is_rejected(data):
    failures, pm, trunc = data
    m = ProportionalAgeReduction(A, B, RHO)
    with pytest.raises(ValueError, match="one per asset"):
        m.fit([failures, failures], [pm, pm], [trunc, trunc, trunc])


def test_mixing_numbers_and_sequences_is_rejected(data):
    """[array, 50.0] is a typo, not a fleet. Say so rather than guessing."""
    failures, pm, trunc = data
    m = ProportionalAgeReduction(A, B, RHO)
    with pytest.raises(TypeError, match="mixes numbers and sequences"):
        m.fit(failures, [pm, 50.0], trunc)


def test_fleet_with_no_failures_anywhere_is_rejected(data):
    failures, pm, trunc = data
    m = ProportionalAgeReduction(A, B, RHO)
    with pytest.raises(ValueError, match="no failures observed"):
        m.fit([[], []], [pm, pm], trunc)


def test_validation_names_the_offending_asset(data):
    """With a fleet, "a failure is out of range" is not enough to act on."""
    failures, pm, trunc = data
    m = ProportionalAgeReduction(A, B, RHO)
    late = list(failures) + [trunc + 10.0]
    with pytest.raises(ValueError, match="asset 1"):
        m.fit([failures, late], [pm, pm], trunc)


# -------------------------------------------------- repair factor bounds ----
@pytest.mark.parametrize("rho", [0.0, 0.5, 1.0])
def test_repair_factor_endpoints_are_valid_models(rho):
    """0 is no age reduction, 1 is as-good-as-new. Both are meaningful."""
    m = ProportionalAgeReduction(A, B, rho)
    assert m.repair_factor == rho


@pytest.mark.parametrize("rho", [-0.1, 1.1, 5.0])
def test_repair_factor_outside_the_unit_interval_raises(rho):
    """Raises rather than asserts: python -O strips assert statements."""
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        ProportionalAgeReduction(A, B, rho)


def test_set_parameters_checks_the_repair_factor_too():
    m = ProportionalAgeReduction(A, B, RHO)
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        m.set_parameters(A, B, 2.0)


# -------------------------------------------------------------- validation ----
def test_rejects_unsorted_pm_times(model):
    with pytest.raises(ValueError, match="strictly increasing"):
        model.cumulative_intensity(np.array([10.0]), np.array([100.0, 50.0]), T)


def test_rejects_pm_at_the_origin(model):
    """0 is the start of observation, not a maintenance action."""
    with pytest.raises(ValueError, match="strictly positive"):
        model.cumulative_intensity(np.array([10.0]), np.array([0.0, 50.0]), T)


def test_rejects_pm_after_the_truncation_time(model):
    with pytest.raises(ValueError, match="before truncation_time"):
        model.cumulative_intensity(np.array([10.0]), np.array([50.0, 400.0]), T)


def test_rejects_non_positive_truncation_time(model):
    with pytest.raises(ValueError, match="truncation_time must be positive"):
        model.cumulative_intensity(np.array([10.0]), PM, 0.0)


def test_no_pm_times_is_allowed(model):
    """A window with no maintenance at all is just the baseline."""
    t = np.linspace(0.0, 300.0, 200)
    np.testing.assert_allclose(
        model.cumulative_intensity(t, np.array([]), T), A * t**B, rtol=1e-12
    )
