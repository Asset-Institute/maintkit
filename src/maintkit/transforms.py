"""Parameter transforms for constrained maximum-likelihood estimation.

Optimisation and Hessian evaluation are carried out in an *unconstrained* space
(``y``), while parameters are reported in their *natural* space (``p``). A
:class:`Transform` defines that mapping and supplies the Jacobian needed to move
a covariance matrix between the two spaces via the delta method.

Convention used throughout the package::

    y = transform.forward(p)      # unconstrained <- natural
    p = transform.inverse(y)      # natural       <- unconstrained

Every transform defined here is monotonically increasing and acts elementwise,
so confidence bounds computed in ``y`` map to bounds in ``p`` directly. If you 
make your own transforms, note that transforms must be monotonically increasing 
to be used with ``ci_method="transformed"``.

"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["Transform", "Identity", "Log", "Logit", "Composite"]


def _asarray(x):
    return np.atleast_1d(np.asarray(x, dtype=float))

# Abstract base class for parameter transforms, causes errors at instantiation if any 
# of the abstract methods are not implemented.
class Transform(ABC):
    """Base class for elementwise, monotonically increasing reparameterisations."""

    @abstractmethod
    def forward(self, p):
        """Map natural parameters to the unconstrained space."""

    @abstractmethod
    def inverse(self, y):
        """Map unconstrained parameters back to the natural space."""

    @abstractmethod
    def jacobian_diag(self, y):
        """Diagonal of dp/dy evaluated at ``y``."""

    def jacobian(self, y):
        """Full (diagonal) Jacobian matrix dp/dy at ``y``."""
        return np.diag(_asarray(self.jacobian_diag(y)))

    def covariance(self, y, hessian):
        """Natural-space covariance from the unconstrained-space Hessian.

        Given ``H``, the Hessian of the negative log-likelihood with respect to
        ``y``, the covariance of ``p`` follows from the delta method:

            cov(p) = J inv(H) J^T,     J = dp/dy

        See "Reparameterization" at
        https://en.wikipedia.org/wiki/Fisher_information
        """
        hessian = np.atleast_2d(np.asarray(hessian, dtype=float))
        y = _asarray(y)
        if hessian.shape != (y.size, y.size):
            raise ValueError(
                f"hessian must be {(y.size, y.size)} for {y.size} parameters, "
                f"got {hessian.shape}"
            )
        cov_y = np.linalg.inv(hessian)
        J = self.jacobian(y)
        return J @ cov_y @ J.T

    def __repr__(self):
        return f"{type(self).__name__}()"


class Identity(Transform):
    """No reparameterisation. Suitable for unbounded parameters."""

    def forward(self, p):
        return _asarray(p)

    def inverse(self, y):
        return _asarray(y)

    def jacobian_diag(self, y):
        return np.ones_like(_asarray(y))


class Log(Transform):
    """``y = log(p)``. For strictly positive parameters (scales, rates, shapes)."""

    def forward(self, p):
        p = _asarray(p)
        if np.any(p <= 0):
            raise ValueError("Log transform requires strictly positive parameters")
        return np.log(p)

    def inverse(self, y):
        return np.exp(_asarray(y))

    def jacobian_diag(self, y):
        return np.exp(_asarray(y))


class Logit(Transform):
    """``y = log(p/(1-p))``. For parameters confined to the open interval (0, 1).

    Used for quantities such as the repair-effectiveness factor ``rho``.
    """

    def forward(self, p):
        p = _asarray(p)
        if np.any(p <= 0) or np.any(p >= 1):
            raise ValueError("Logit transform requires parameters strictly in (0, 1)")
        return np.log(p / (1.0 - p))

    def inverse(self, y):
        y = _asarray(y)
        # numerically stable logistic
        out = np.empty_like(y)
        pos = y >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-y[pos]))
        ey = np.exp(y[~pos])
        out[~pos] = ey / (1.0 + ey)
        return out

    def jacobian_diag(self, y):
        p = self.inverse(y)
        return p * (1.0 - p)


class Composite(Transform):
    """Apply a different transform to each parameter.

    Example
    -------
    The imperfect-maintenance model has parameters ``(a, b, rho)`` where ``a``
    and ``b`` are positive and ``rho`` lies in (0, 1)::

        Composite([Log(), Log(), Logit()])
    """

    def __init__(self, transforms):
        transforms = list(transforms)
        if not transforms:
            raise ValueError("Composite requires at least one transform")
        for t in transforms:
            if not isinstance(t, Transform):
                raise TypeError(f"expected Transform instances, got {type(t).__name__}")
        self.transforms = transforms

    @property
    def n_params(self):
        return len(self.transforms)

    def _check_size(self, v):
        v = _asarray(v)
        if v.size != self.n_params:
            raise ValueError(
                f"expected {self.n_params} parameters, got {v.size}"
            )
        return v

    def forward(self, p):
        p = self._check_size(p)
        return np.array(
            [float(t.forward(p[i])[0]) for i, t in enumerate(self.transforms)]
        )

    def inverse(self, y):
        y = self._check_size(y)
        return np.array(
            [float(t.inverse(y[i])[0]) for i, t in enumerate(self.transforms)]
        )

    def jacobian_diag(self, y):
        y = self._check_size(y)
        return np.array(
            [float(np.atleast_1d(t.jacobian_diag(y[i]))[0])
             for i, t in enumerate(self.transforms)]
        )

    def __repr__(self):
        inner = ", ".join(repr(t) for t in self.transforms)
        return f"Composite([{inner}])"
