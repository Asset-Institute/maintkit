"""Unit tests for maintkit.transforms (numpy-only, no scipy required)."""
import numpy as np
import pytest

from maintkit.transforms import Transform, Identity, Log, Logit, Composite


ALL = [
    (Identity(), [-2.0, 0.0, 3.0]),
    (Log(), [0.01, 1.0, 250.0]),
    (Logit(), [1e-4, 0.5, 0.999]),
]


@pytest.mark.parametrize("transform,p", ALL)
def test_round_trip(transform, p):
    assert np.allclose(transform.inverse(transform.forward(p)), p)


@pytest.mark.parametrize(
    "transform,y",
    [
        (Identity(), [0.3, -1.0]),
        (Log(), [np.log(2.0), np.log(50.0)]),
        (Logit(), [0.0, 1.5]),
    ],
)
def test_jacobian_matches_finite_differences(transform, y):
    y = np.asarray(y, dtype=float)
    h = 1e-6
    eye = np.eye(y.size)
    fd = np.array([
        (transform.inverse(y + h * eye[i])[i] - transform.inverse(y - h * eye[i])[i])
        / (2 * h)
        for i in range(y.size)
    ])
    assert np.allclose(fd, transform.jacobian_diag(y), rtol=1e-6)


@pytest.mark.parametrize("transform", [Identity(), Log(), Logit()])
def test_strictly_increasing(transform):
    """Monotonicity is what lets transformed-space CI bounds keep their order."""
    ys = np.linspace(-6, 6, 200)
    ps = np.array([transform.inverse([v])[0] for v in ys])
    assert np.all(np.diff(ps) > 0)


def test_log_covariance_closed_form():
    # p = exp(y), J = diag(p); with H = I, cov(p) = J J^T = diag(p^2)
    y = np.array([np.log(3.0), np.log(7.0)])
    assert np.allclose(Log().covariance(y, np.eye(2)), np.diag([9.0, 49.0]))


def test_covariance_matches_the_original_convention():
    """The sandwich the package used before Transform existed, written out here
    rather than imported, so it stays a fixed reference even though the helper
    that implemented it has been deleted."""
    y = np.array([np.log(3.0), np.log(7.0)])
    H = np.array([[4.0, 1.0], [1.0, 3.0]])
    J = np.diag(np.exp(y))
    Ji = np.linalg.inv(J)
    legacy = np.linalg.inv(Ji.T @ H @ Ji)
    assert np.allclose(Log().covariance(y, H), legacy)


def test_covariance_symmetric_positive_definite():
    y = np.array([np.log(3.0), np.log(7.0)])
    cov = Log().covariance(y, np.array([[4.0, 1.0], [1.0, 3.0]]))
    assert np.allclose(cov, cov.T)
    assert np.all(np.linalg.eigvals(cov) > 0)


def test_composite_round_trip_and_jacobian():
    c = Composite([Log(), Log(), Logit()])
    p = [0.02, 1.5, 0.3]
    assert np.allclose(c.inverse(c.forward(p)), p)
    assert np.allclose(c.jacobian_diag(c.forward(p)), [0.02, 1.5, 0.3 * 0.7])
    assert c.n_params == 3


@pytest.mark.parametrize("bad", [[-1.0], [0.0]])
def test_log_rejects_non_positive(bad):
    with pytest.raises(ValueError):
        Log().forward(bad)


@pytest.mark.parametrize("bad", [[0.0], [1.0], [1.5], [-0.1]])
def test_logit_rejects_out_of_unit_interval(bad):
    with pytest.raises(ValueError):
        Logit().forward(bad)


def test_composite_rejects_wrong_size():
    with pytest.raises(ValueError):
        Composite([Log()]).forward([1.0, 2.0])


def test_composite_rejects_non_transform():
    with pytest.raises(TypeError):
        Composite([object()])


def test_covariance_rejects_bad_hessian_shape():
    with pytest.raises(ValueError):
        Log().covariance([0.0, 0.0], np.eye(3))


def test_logit_inverse_is_numerically_stable():
    """Large-magnitude y must not overflow."""
    out = Logit().inverse([-800.0, 800.0])
    assert np.all(np.isfinite(out))
    assert out[0] >= 0.0 and out[1] <= 1.0
