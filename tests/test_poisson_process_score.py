"""Analytic score for the power-law NHPP.

Checked against numerical differentiation of the actual ``nnlf`` rather than
against a reimplementation. That matters here because ``nnlf`` has a
non-obvious convention -- it omits the compensator term ``-Lambda(T)`` for any
asset whose truncation time is ``None`` -- and a score that disagrees with its
own objective sends the optimiser somewhere neither of them minimises.
"""
from __future__ import annotations

import numdifftools as ndt
import numpy as np
import pytest

from maintkit.poisson_process import PowerLawNHPP
from tests import _datasets as ds


@pytest.fixture(scope="module")
def data():
    return ds.nhpp_events()


@pytest.fixture(scope="module")
def model():
    return PowerLawNHPP(0.02, 1.5)


def _numeric_log(model, events, trunc, p):
    """Numerical gradient with respect to (log a, log b).

    Differentiating in the unconstrained space is deliberate, not incidental.
    ``a`` and ``b`` are strictly positive, and numdifftools chooses its own
    step then Richardson-extrapolates over several multiples of it. For a
    parameter as small as ``a = 1e-3`` some of those steps cross zero, ``nnlf``
    evaluates ``log`` of a negative number, and the entire extrapolation
    returns nan -- a defect in the numerical reference, not in the score.

    Log space has no boundary to cross, so the comparison stays valid however
    small the parameter. It is also the space ``fit_mle`` optimises in, so this
    is the gradient that actually gets used.
    """
    f = lambda y: model.nnlf(np.exp(y), events, trunc)
    return ndt.Gradient(f)(np.log(np.asarray(p, float)))


def _analytic_log(model, events, trunc, p):
    """Score in log space: chain rule is multiplication by the parameters."""
    p = np.asarray(p, float)
    return model.nnlf_gradient(p, events, trunc) * p


# ------------------------------------------------------------ correctness --
@pytest.mark.parametrize(
    "p", [[0.01, 1.2], [0.05, 2.0], [0.001, 0.8], [0.03, 1.6], [1e-06, 0.5]]
)
def test_score_matches_numerical_gradient(model, data, p):
    events, trunc = data
    analytic = _analytic_log(model, events, trunc, p)
    numeric = _numeric_log(model, events, trunc, p)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-6)


def test_score_matches_in_natural_space_too(model, data):
    """Pins the natural-space convention directly, at a well-conditioned point.

    Only one point, and deliberately not near the boundary -- see
    :func:`_numeric_log` for why natural-space differentiation is unreliable
    for small ``a``.
    """
    events, trunc = data
    p = [0.05, 2.0]
    analytic = model.nnlf_gradient(p, events, trunc)
    numeric = ndt.Gradient(lambda q: model.nnlf(q, events, trunc))(np.asarray(p, float))
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-6)


# ------------------------------------------------- conventions must match --
@pytest.mark.parametrize("p", [[0.01, 1.2], [0.05, 2.0]])
def test_score_matches_when_truncation_is_none(model, data, p):
    """nnlf drops the compensator entirely when truncation_times is None."""
    events, _ = data
    analytic = _analytic_log(model, events, None, p)
    numeric = _numeric_log(model, events, None, p)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("p", [[0.01, 1.2], [0.03, 1.6]])
def test_score_matches_with_mixed_none_truncation(model, data, p):
    """Per-asset truncation times may individually be None."""
    events, trunc = data
    mixed = list(trunc)
    mixed[0] = mixed[3] = mixed[7] = None
    analytic = _analytic_log(model, events, mixed, p)
    numeric = _numeric_log(model, events, mixed, p)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-6)


def test_score_accepts_ndarray_like_nnlf(model):
    """nnlf converts a 2D ndarray to lists; the score must accept the same."""
    arr = np.array([[10.0, 40.0, 95.0], [22.0, 60.0, 150.0]])
    trunc = [200.0, 200.0]
    analytic = _analytic_log(model, arr, trunc, [0.02, 1.5])
    numeric = _numeric_log(model, arr, trunc, [0.02, 1.5])
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-6)


# ------------------------------------------------- agreement with the MLE --
def test_score_vanishes_at_the_closed_form_mle(model, data):
    """Independent evidence that both the score and the closed form are right.

    Uses the equal-horizon dataset, where the closed form is exact. (It is not
    exact for unequal horizons -- see the profile-likelihood fix.)
    """
    events, trunc = data
    res = model.fit(events, truncation_times=trunc)
    p_hat = np.asarray(res[0] if isinstance(res, tuple) else res.params, float)
    g = model.nnlf_gradient(p_hat, events, trunc)
    scale = abs(model.nnlf(p_hat, events, trunc))
    assert np.all(np.abs(g) / scale < 1e-9), f"score {g} too large relative to {scale}"


def test_score_encodes_the_alpha_mle(model, data):
    """dl/da = 0 is exactly  a = N / sum(T^b)."""
    events, trunc = data
    b = 1.4694591549675484
    n_events = sum(len(e) for e in events)
    a_star = n_events / np.sum(np.asarray(trunc, float) ** b)
    g = model.nnlf_gradient([a_star, b], events, trunc)
    assert abs(g[0]) < 1e-9 * n_events


def test_score_sign_is_for_the_negative_loglikelihood(model, data):
    """nnlf_gradient returns minus the score, matching nnlf = -loglikelihood."""
    events, trunc = data
    p = [0.01, 1.2]
    n_events = sum(len(e) for e in events)
    # dl/da = N/a - sum(T^b); nnlf_gradient must return the negation
    dl_da = n_events / p[0] - np.sum(np.asarray(trunc, float) ** p[1])
    assert model.nnlf_gradient(p, events, trunc)[0] == pytest.approx(-dl_da)
