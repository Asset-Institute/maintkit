"""Shared maximum-likelihood machinery.

Every parametric model in :mod:`maintkit` fits by the same recipe: transform the
parameters to an unconstrained space, minimise a negative log-likelihood,
evaluate the Hessian numerically, and turn that into standard errors and
confidence intervals. This module holds that recipe once.

Models supply only the part that is genuinely model-specific -- their ``nnlf``
-- and call :func:`fit_mle`. Models with a closed-form MLE skip the optimiser
and call :func:`result_at` instead.

"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numdifftools as ndt
from scipy import optimize as opt
from scipy import stats as sps

from maintkit.transforms import Identity, Transform

# list of public symbols for ``from maintkit.inference import *``. The
__all__ = ["FitResult", "fit_mle", "result_at", "hessian_at"]


@dataclass
class FitResult:
    """Outcome of a maximum-likelihood fit.

    Supports tuple unpacking as ``params, ci, cov`` for backwards compatibility
    with the older fitters that returned a 3-tuple.
    """

    params: np.ndarray
    se: np.ndarray
    ci: np.ndarray            # always (n_params, 2): column 0 lower, column 1 upper
    cov: np.ndarray
    nnlf: float
    success: bool = True
    message: str = ""
    alpha: float = 0.05
    names: tuple | None = None

    def __iter__(self):
        """Backwards compatibility: ``p_hat, p_ci, p_cov = model.fit(...)``."""
        yield from (self.params, self.ci, self.cov)

    @property
    def n_params(self):
        return int(np.size(self.params))

    def _labels(self):
        if self.names is not None:
            return list(self.names)
        return [f"p{i}" for i in range(self.n_params)]

    def summary(self):
        """Readable parameter table."""
        pct = int(round((1.0 - self.alpha) * 100))
        lines = [
            f"Maximum-likelihood fit  (nnlf={self.nnlf:.6g}, success={self.success})",
            f"{'param':<12}{'estimate':>14}{'std. err':>14}"
            f"{f'[{pct}% CI lower':>16}{'upper]':>14}",
        ]
        for name, p, s, (lo, hi) in zip(
            self._labels(), np.atleast_1d(self.params),
            np.atleast_1d(self.se), np.atleast_2d(self.ci)
        ):
            lines.append(f"{name:<12}{p:>14.6g}{s:>14.6g}{lo:>16.6g}{hi:>14.6g}")
        if not self.success and self.message:
            lines.append(f"warning: {self.message}")
        return "\n".join(lines)

    def __str__(self):
        return self.summary()


def _invert(matrix, what):
    try:
        return np.linalg.inv(np.atleast_2d(matrix))
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            f"{what} is singular and cannot be inverted; the likelihood is flat "
            "in at least one direction. Check for insufficient data, a redundant "
            "parameter, or a starting point far from the optimum."
        ) from exc


def _build_result(y_hat, hessian, transform, *, alpha, nnlf_value,
                  success, message, names, ci_method):
    y_hat = np.atleast_1d(np.asarray(y_hat, dtype=float))
    hessian = np.atleast_2d(np.asarray(hessian, dtype=float))

    p_hat = transform.inverse(y_hat)
    cov = transform.covariance(y_hat, hessian)
    se = np.sqrt(np.diag(cov))
    z = sps.norm.ppf(1.0 - alpha / 2.0)

    if ci_method == "transformed":
        # Delta method in the unconstrained space, then map bounds back. Keeps
        # positive parameters positive and yields asymmetric intervals.
        cov_y = _invert(hessian, "Hessian")
        s_y = np.sqrt(np.diag(cov_y))
        lower = transform.inverse(y_hat - z * s_y)
        upper = transform.inverse(y_hat + z * s_y)
    elif ci_method == "natural":
        # Symmetric Wald interval on the natural scale. Can produce a negative
        # lower bound for a positive parameter; retained for reproducing
        # results published with earlier versions.
        lower = p_hat - z * se
        upper = p_hat + z * se
    else:
        raise ValueError(
            f"ci_method must be 'transformed' or 'natural', got {ci_method!r}"
        )

    return FitResult(
        params=p_hat,
        se=se,
        ci=np.column_stack([lower, upper]),
        cov=cov,
        nnlf=float(nnlf_value),
        success=bool(success),
        message=str(message),
        alpha=float(alpha),
        names=tuple(names) if names is not None else None,
    )


def fit_mle(objective, p0, *, transform=None, alpha=0.05, names=None,
            optimizer_kwds=None, ndt_kwds=None, ci_method="transformed"):
    """Maximum-likelihood fit of ``objective`` starting from ``p0``.

    Parameters
    ----------
    objective : callable
        ``objective(p) -> float``, the negative log-likelihood evaluated at
        *natural* parameters ``p``. The transform is applied internally, so the
        objective never sees the unconstrained space.
    p0 : array_like
        Starting values, in natural parameters.
    transform : Transform, optional
        Reparameterisation used for optimisation. Defaults to
        :class:`~maintkit.transforms.Identity`.
    alpha : float
        Significance level; ``alpha=0.05`` gives 95% intervals.
    names : sequence of str, optional
        Parameter names, used only for :meth:`FitResult.summary`.
    optimizer_kwds, ndt_kwds : dict, optional
        Forwarded to ``scipy.optimize.minimize`` and ``numdifftools.Hessian``.
    ci_method : {'transformed', 'natural'}
        See :func:`_build_result`.

    Returns
    -------
    FitResult
    """
    transform = Identity() if transform is None else transform
    if not isinstance(transform, Transform):
        raise TypeError(f"transform must be a Transform, got {type(transform).__name__}")
    optimizer_kwds = {} if optimizer_kwds is None else dict(optimizer_kwds)
    ndt_kwds = {} if ndt_kwds is None else dict(ndt_kwds)

    y0 = transform.forward(p0)

    def objective_y(y):
        return objective(transform.inverse(y))

    result = opt.minimize(objective_y, y0, **optimizer_kwds)
    y_hat = np.atleast_1d(result.x)
    hessian = ndt.Hessian(objective_y, **ndt_kwds)(y_hat)

    return _build_result(
        y_hat, hessian, transform,
        alpha=alpha, nnlf_value=result.fun,
        success=result.success, message=result.message,
        names=names, ci_method=ci_method,
    )


def hessian_at(objective, p_hat, *, transform=None, ndt_kwds=None):
    """Hessian of ``objective`` in the unconstrained space at ``p_hat``.

    Returns ``(y_hat, hessian)``. Useful for models whose MLE is available in
    closed form but whose standard errors still need a numerical Hessian.
    """
    transform = Identity() if transform is None else transform
    ndt_kwds = {} if ndt_kwds is None else dict(ndt_kwds)
    y_hat = transform.forward(p_hat)

    def objective_y(y):
        return objective(transform.inverse(y))

    return y_hat, ndt.Hessian(objective_y, **ndt_kwds)(y_hat)


def result_at(objective, p_hat, *, transform=None, alpha=0.05, names=None,
              ndt_kwds=None, ci_method="transformed"):
    """Build a :class:`FitResult` at a known (e.g. closed-form) estimate.

    Skips the optimiser entirely; only the Hessian is computed numerically.
    """
    transform = Identity() if transform is None else transform
    y_hat, hessian = hessian_at(
        objective, p_hat, transform=transform, ndt_kwds=ndt_kwds
    )
    return _build_result(
        y_hat, hessian, transform,
        alpha=alpha, nnlf_value=objective(np.atleast_1d(np.asarray(p_hat, float))),
        success=True, message="closed-form estimate",
        names=names, ci_method=ci_method,
    )
