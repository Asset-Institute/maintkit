"""Gamma process"""
import inspect
import numpy as np
import scipy.stats as sps

from maintkit.inference import fit_mle
from maintkit.transforms import Composite, Log
from maintkit.utilities import _ensure_list

class Gamma_Process:
    """Gamma process with shape f(t | alpha) and scale beta.

    Increments are independent, and the increment over ``(t0, t1]`` is Gamma
    distributed with shape ``f(t1|alpha) - f(t0|alpha)`` and scale ``beta``.
    ``alpha`` is a vector, one element per parameter of the shape function; it
    defaults to the linear law ``f(t) = alpha[0]*t``.
    """

    def __init__(self,alpha=None,beta=None):
        self.alpha = alpha
        self.beta = beta
        self.parameter_source = "specified"
        self.parameter_covariance = None

    #: Shape function f(t | alpha), set by :meth:`shape_function`.
    @staticmethod
    def _linear_shape(t,*alpha):
        """Default shape f(t) = alpha[0]*t, the stationary gamma process."""
        return alpha[0]*t

    #: Shape function f(t | alpha). Defaults to the linear law, so a model is
    #: usable without calling :meth:`shape_function`.
    _shape = _linear_shape

    def shape_function(self,function=None,validate=True):
        """Set the shape function f(t, *alpha) and return it.

        Alpha is passed unpacked, so a two-parameter law is ``f(t, a, b)``.
        No argument restores the linear default. ``validate`` checks the
        gamma-process requirements: f(0)=0, strictly increasing, finite,
        vectorised over t.
        """
        if function is None:
            function = self._linear_shape
        if not callable(function):
            raise TypeError(f"function must be callable f(t, *alpha), got {type(function).__name__}")

        if validate:
            t = np.concatenate(([0.0],np.logspace(-4,1,40)))
            if self.alpha is None:
                parameters = list(inspect.signature(function).parameters.values())
                starred = any(p.kind is p.VAR_POSITIONAL for p in parameters)
                alpha = np.ones(1 if starred else len(parameters) - 1)
            else:
                alpha = np.atleast_1d(self.alpha)
            v = np.asarray(function(t,*alpha),dtype=float)
            if v.shape != t.shape:
                raise ValueError(f"function must be vectorised over t: got shape {v.shape} for {t.shape} times")
            if not np.all(np.isfinite(v)):
                raise ValueError(f"function is not finite at alpha={alpha}, first at t={t[~np.isfinite(v)][0]:g}")
            if abs(v[0]) > 1e-12:
                raise ValueError(f"a gamma process needs f(0)=0, got {v[0]:g} at alpha={alpha}")
            if np.any(np.diff(v) <= 0):
                raise ValueError(f"function must be strictly increasing in t at alpha={alpha}, or increments have a non-positive shape")

        self._shape = function
        return function

    def shape(self,t,alpha=None):
        """Evaluate f(t | alpha), defaulting to the model's alpha.

        Alpha is an argument so ``nnlf`` can try a parameter without writing
        it to the object.
        """
        alpha = self.alpha if alpha is None else alpha
        if alpha is None:
            raise ValueError("alpha is not set; pass it here, set it on the model, or estimate it with fit")

        return np.asarray(self._shape(np.asarray(t,dtype=float),*np.atleast_1d(alpha)),dtype=float)

    def transition_distribution(self,x,t,x0,type="cdf",*,alpha=None,beta=None):
        """pdf, logpdf or cdf of the state at ``t``, starting from ``x0`` at t=0.

        Gamma with shape f(t) - f(0) and scale beta, shifted by ``x0``. Alpha
        and beta default to the model's.
        """
        alpha = self.alpha if alpha is None else alpha
        beta = self.beta if beta is None else beta
        if type.lower() not in ["logpdf","pdf","cdf"]:
            raise ValueError(f"type must be one of pdf, logpdf, cdf; got {type!r}")

        shape = self.shape(t,alpha) - self.shape(0,alpha)
        scale = beta
        if type.lower() == "pdf":
            return sps.gamma.pdf(x,shape,scale=scale, loc=x0)
        elif type.lower() == "logpdf":
            return sps.gamma.logpdf(x,shape,scale=scale, loc=x0)
        else:
            return sps.gamma.cdf(x,shape,scale=scale, loc=x0)

    def simulate(self,times,initial=0.0,num_samples=1):
        """Simulate paths on ``times``, returning a (num_samples, len(times)) array.

        Each step draws Gamma(f(t_i) - f(t_i-1), beta); paths are the
        cumulative sum, so they are monotone increasing from ``initial``.
        """
        T = len(times)
        x = np.zeros((num_samples,T))
        x[:,0] = initial
        for ii in range(1,len(times)): 
            shape = self.shape(times[ii],self.alpha) - self.shape(times[ii-1],self.alpha)
            dxs = sps.gamma(shape,scale=self.beta).rvs(num_samples)
            x[:,ii] = dxs

        return np.cumsum(x,axis=1)

    @classmethod
    def _check_runs(cls,t,x):
        """Normalise to lists of runs and check the two line up."""
        t = _ensure_list(t)
        x = _ensure_list(x)
        if len(t) != len(x):
            raise ValueError(
                f"t has {len(t)} run(s) but x has {len(x)}; each run of "
                "observations needs its own times"
            )
        for m,(run_t,run_x) in enumerate(zip(t,x)):
            if len(run_t) != len(run_x):
                raise ValueError(
                    f"run {m} has {len(run_t)} times but {len(run_x)} "
                    "observations"
                )
            if len(run_t) < 2:
                raise ValueError(
                    f"run {m} has fewer than two observations, so it contains "
                    "no increment to fit"
                )
        return t,x

    def _starting_values(self):
        """Starting values ``[[alpha...], beta]``, all ones.

        One alpha per parameter of the shape function, counted from its
        signature. The default ``_linear_shape(t, *alpha)`` cannot be counted,
        so it falls back to a single alpha.
        """
        parameters = list(inspect.signature(self._shape).parameters.values())
        starred = any(p.kind is p.VAR_POSITIONAL for p in parameters)
        n_alpha = 1 if starred else len(parameters) - 1
        return [[1.0]*n_alpha, 1.0]

    #: Floor applied to increments before evaluating the density. The gamma
    #: density is 0 or infinite at exactly zero, which would send the whole
    #: log-likelihood to +-inf and strand the optimiser at its starting point.
    #: Zero increments are ordinary observations -- no measurable degradation
    #: between visits, or an underflow when the shape is small -- so they are
    #: floored rather than dropped.
    increment_floor = 1e-12
    def nnlf(self,alpha: list[float()],beta:float,t,x):
        """Negative log-likelihood for (alpha, beta).

        ``t`` and ``x`` are lists of runs. Increments are independent, so the
        log-likelihood is the sum of transition log-densities over every step
        of every run.

        Increments are floored at :attr:`increment_floor`; see the note there.

        Returns ``inf`` for a parameter the model cannot produce.
        """

        shape = np.diff(self.shape(t,alpha))
        dxs = np.maximum(np.diff(x),self.increment_floor)
        total = np.sum(sps.gamma.logpdf(dxs, shape, scale=beta))
        nnlf = -total
        return nnlf if np.isfinite(nnlf) else np.inf

    def fit(self,t,x,p0=None,*,inplace=False,alpha=0.05,ndt_kwds=None,
            optimizer_kwds=None,ci_method="transformed"):
        """Fit ``(alpha, beta)`` by maximum likelihood.

        ``p0`` is ``[[alpha...], beta]``, defaulting to :meth:`_starting_values`.
        Both alpha and beta are positive, so the fit runs on the log scale.
        """
        t,x = self._check_runs(t,x)
        a0,b0 = self._starting_values() if p0 is None else p0
        a0 = np.atleast_1d(np.asarray(a0,dtype=float))
        n = a0.size

        result = fit_mle(
            lambda p: self.nnlf(p[:n],p[n],t,x),
            np.append(a0,b0),
            transform=Composite([Log()]*(n+1)),
            alpha=alpha,
            names=tuple(f"alpha{i}" for i in range(n))+("beta",),
            ndt_kwds=ndt_kwds,
            optimizer_kwds=optimizer_kwds,
            ci_method=ci_method,
        )

        if inplace:
            self.alpha = list(result.params[:n])
            self.beta = float(result.params[n])
            self.parameter_source = "estimated"
            self.parameter_covariance = result.cov

        return result


