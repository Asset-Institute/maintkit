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

import warnings
from dataclasses import dataclass, field

import numpy as np
import numdifftools as ndt
from scipy import optimize as opt
from scipy import stats as sps

from maintkit.transforms import Identity, Transform

# list of public symbols for ``from maintkit.inference import *``. The
__all__ = [
    "FitResult",
    "ConvergenceWarning",
    "fit_mle",
    "result_at",
    "result_from_covariance",
    "hessian_at",
]


class ConvergenceWarning(RuntimeWarning):
    """The optimiser did not report successful convergence.

    Its own class so it can be filtered independently of other warnings::

        warnings.filterwarnings("ignore", category=maintkit.ConvergenceWarning)
        warnings.filterwarnings("error",  category=maintkit.ConvergenceWarning)
    """


@dataclass
class FitResult:
    """Outcome of a maximum-likelihood fit.

    Every fitter in the package returns one of these, so ``ci`` is always
    ``(n_params, 2)`` and the fields always mean the same thing.
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

    if hessian.shape != (y_hat.size, y_hat.size):
        raise ValueError(
            f"hessian must be {(y_hat.size, y_hat.size)} for {y_hat.size} "
            f"parameters, got {hessian.shape}"
        )

    # Invert once, here, rather than calling Transform.covariance: that routes
    # the failure through _invert so a flat likelihood reports why, instead of
    # numpy's bare "Singular matrix". It also avoids inverting the Hessian
    # twice, which the transformed-CI branch below would otherwise do.
    cov_y = _invert(hessian, "Hessian")
    J = transform.jacobian(y_hat)
    cov = J @ cov_y @ J.T
    se = np.sqrt(np.diag(cov))
    z = sps.norm.ppf(1.0 - alpha / 2.0)

    if ci_method == "transformed":
        # Delta method in the unconstrained space, then map bounds back. Keeps
        # positive parameters positive and yields asymmetric intervals.
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


def fit_mle(objective, p0, *, gradient=None, transform=None, alpha=0.05,
            names=None, optimizer_kwds=None, ndt_kwds=None,
            ci_method="transformed"):
    """Maximum-likelihood fit of ``objective`` starting from ``p0``.

    Parameters
    ----------
    objective : callable
        ``objective(p) -> float``, the negative log-likelihood evaluated at
        *natural* parameters ``p``. The transform is applied internally, so the
        objective never sees the unconstrained space.
    p0 : array_like
        Starting values, in natural parameters.
    gradient : callable, optional
        ``gradient(p) -> array``, the gradient of ``objective`` with respect to
        the *natural* parameters. The chain rule into the unconstrained space
        is applied internally, so models need not know about the transform.
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

    Warns
    -----
    ConvergenceWarning
        When the optimiser does not report success. Filter it with
        ``warnings.filterwarnings("ignore", category=ConvergenceWarning)``.

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

    if gradient is not None:
        def gradient_y(y):
            # Chain rule into the unconstrained space. The Jacobian is diagonal
            # for every Transform here, so this is an elementwise product
            # rather than a matrix product.
            g = np.asarray(gradient(transform.inverse(y)), dtype=float)
            return g * transform.jacobian_diag(y)

        optimizer_kwds.setdefault("jac", gradient_y)

    result = opt.minimize(objective_y, y0, **optimizer_kwds)

    if not result.success:
        warnings.warn(
            f"Optimiser did not converge cleanly: {result.message} "
            "Estimates and standard errors may be unreliable -- the Hessian is "
            "evaluated at the point the optimiser stopped at. Inspect "
            "FitResult.success and .message, or pass optimizer_kwds to change "
            "the method or tolerance. Silence with "
            "warnings.filterwarnings('ignore', category=ConvergenceWarning).",
            ConvergenceWarning,
            stacklevel=2,
        )

    y_hat = np.atleast_1d(result.x)
    hessian = ndt.Hessian(objective_y, **ndt_kwds)(y_hat)

    return _build_result(
        y_hat, hessian, transform,
        alpha=alpha, nnlf_value=result.fun,
        success=result.success, message=result.message,
        names=names, ci_method=ci_method,
    )


def result_from_covariance(params, cov, *, transform=None, alpha=0.05, names=None,
                           nnlf_value=np.nan, ci_method="transformed",
                           message="closed-form estimate and information"):
    """Build a :class:`FitResult` from an estimate whose covariance is known exactly.

    For models where the Fisher information is available in closed form, so
    neither an optimiser nor a numerical Hessian is needed. The exponential is
    the case in hand: :math:`I(\\lambda) = r/\\lambda^{2}`, giving
    :math:`\\operatorname{se}(\\hat\\lambda) = \\hat\\lambda/\\sqrt{r}` exactly.
    Differentiating numerically to recover a quantity already known in closed
    form would only add error.

    Parameters
    ----------
    params : array_like
        The estimate, in natural parameters.
    cov : array_like
        Covariance of ``params``, in the *natural* parameterisation.
    transform : Transform, optional
        Reparameterisation used for confidence intervals. The natural-space
        covariance is mapped to the unconstrained space internally, so callers
        supply ``cov`` in the units they think in.
    nnlf_value : float, optional
        Negative log-likelihood at ``params``, if known. Recorded on the result
        but not otherwise used.

    Returns
    -------
    FitResult
    """
    transform = Identity() if transform is None else transform
    if not isinstance(transform, Transform):
        raise TypeError(f"transform must be a Transform, got {type(transform).__name__}")

    params = np.atleast_1d(np.asarray(params, dtype=float))
    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    if cov.shape != (params.size, params.size):
        raise ValueError(
            f"cov must be {(params.size, params.size)} for {params.size} "
            f"parameters, got {cov.shape}"
        )

    # _build_result works from the unconstrained-space Hessian, so invert the
    # delta-method relation cov = J cov_y J^T to recover cov_y, then H.
    y_hat = transform.forward(params)
    J = transform.jacobian(y_hat)
    J_inv = _invert(J, "transform Jacobian")
    cov_y = J_inv @ cov @ J_inv.T

    return _build_result(
        y_hat, _invert(cov_y, "covariance"), transform,
        alpha=alpha, nnlf_value=nnlf_value,
        success=True, message=message,
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
              ndt_kwds=None, ci_method="transformed",
              success=True, message="closed-form estimate"):
    """Build a :class:`FitResult` at a known (e.g. closed-form) estimate.

    Skips the optimiser entirely; only the Hessian is computed numerically.

    ``success`` and ``message`` are settable because the estimate does not
    always come from a closed form. A profile likelihood, for instance,
    optimises a reduced problem and then evaluates the Hessian of the full one
    here, and the convergence status of that earlier step needs carrying
    through rather than being replaced by a default.
    """
    transform = Identity() if transform is None else transform
    y_hat, hessian = hessian_at(
        objective, p_hat, transform=transform, ndt_kwds=ndt_kwds
    )
    return _build_result(
        y_hat, hessian, transform,
        alpha=alpha, nnlf_value=objective(np.atleast_1d(np.asarray(p_hat, float))),
        success=success, message=message,
        names=names, ci_method=ci_method,
    )
