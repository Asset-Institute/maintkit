"""Wiener process and regulated Brownian motion degradation models."""
import numpy as np
import scipy.stats as sps
from scipy.optimize import fsolve
from scipy.special import logsumexp

from maintkit.inference import fit_mle
from maintkit.transforms import Composite, Identity, Log
from maintkit.utilities import _ensure_list


class Wiener:
    """Brownian motion with drift.

    The state at time ``t`` given ``x0`` is normal with mean ``mu*t + x0`` and
    standard deviation ``sigma*sqrt(t)``.
    """

    #: mu is unconstrained, sigma must stay positive.
    parameter_transform = Composite([Identity(), Log()])

    parameter_names = ("mu", "sigma")

    def __init__(self,mu=None,sigma=None):
        self.mu = mu
        self.sigma = sigma
        self.parameter_source = "specified"
        self.parameter_covariance = None

    def transition_distribution(self,x,t,x0,type="cdf",*,mu=None,sigma=None):
        """Distribution of the state at ``t`` given ``x0``.

        ``mu`` and ``sigma`` default to the model's own parameters. They are
        arguments so that ``nnlf`` can evaluate at a trial point without
        writing to the object.
        """
        mu = self.mu if mu is None else mu
        sigma = self.sigma if sigma is None else sigma
        if type.lower() not in ["logpdf","pdf","cdf"]:
            raise ValueError(f"type must be one of pdf, logpdf, cdf; got {type!r}")

        loc = mu*t + x0
        scale = sigma*np.sqrt(t)
        if type.lower() == "pdf":
            return sps.norm.pdf(x,loc=loc,scale=scale)
        elif type.lower() == "logpdf":
            return sps.norm.logpdf(x,loc=loc,scale=scale)
        else:
            return sps.norm.cdf(x,loc=loc,scale=scale)

    def simulate(self,times,initial=0.0,num_samples=1):
        T = len(times)
        x = np.zeros((num_samples,T))
        x[:,0] = initial
        for ii in range(1,len(times)):
            dt = times[ii] - times[ii-1]
            dxs = sps.norm(loc=self.mu*dt,scale=self.sigma*np.sqrt(dt)).rvs(num_samples)
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

    @staticmethod
    def _increments(t,x):
        """Flatten every run to (dt, dx) pairs."""
        dt, dx = [], []
        for run_t,run_x in zip(t,x):
            for ii in range(1,len(run_t)):
                dt.append(run_t[ii] - run_t[ii-1])
                dx.append(run_x[ii] - run_x[ii-1])
        return np.asarray(dt,dtype=float), np.asarray(dx,dtype=float)

    def _starting_values(self,t,x):
        r"""Moment estimates used as the starting point.

        Over a step of length :math:`\Delta t` the increment has mean
        :math:`\mu\,\Delta t` and variance :math:`\sigma^{2}\Delta t`, so

        .. math::
            \hat\mu = \frac{\sum \Delta x}{\sum \Delta t}, \qquad
            \hat\sigma = \sqrt{
                \frac{1}{n}\sum \frac{(\Delta x - \hat\mu\,\Delta t)^{2}}
                                     {\Delta t}}.
        """
        dt, dx = self._increments(t,x)
        mu = float(np.sum(dx)/np.sum(dt))
        variance = float(np.mean((dx - mu*dt)**2/dt))
        return [mu, np.sqrt(variance)]

    def nnlf(self,p,t,x):
        """Negative log-likelihood for ``p = (mu, sigma)``.

        ``t`` and ``x`` are lists of runs. Increments are independent, so the
        log-likelihood is the sum of transition log-densities over every step
        of every run.

        Returns ``inf`` for a parameter the model cannot produce.
        """
        mu, sigma = (float(v) for v in p)
        if not sigma > 0:
            return np.inf

        total = 0.0
        for run_t,run_x in zip(t,x):
            for ii in range(1,len(run_t)):
                dt = run_t[ii] - run_t[ii-1]
                total += self.transition_distribution(
                    run_x[ii], dt, run_x[ii-1], type="logpdf",
                    mu=mu, sigma=sigma,
                )

        nnlf = -total
        return nnlf if np.isfinite(nnlf) else np.inf

    def fit(self,t,x,p0=None,*,inplace=False,alpha=0.05,ndt_kwds=None,
            optimizer_kwds=None,ci_method="transformed"):
        """Fit ``(mu, sigma)`` by maximum likelihood.

        Parameters
        ----------
        t, x : list of lists
            Observation times and states, one list per run. A single flat list
            is taken to be one run.
        p0 : array_like, optional
            Starting values ``(mu, sigma)``. Defaults to the moment estimates
            in :meth:`_starting_values`.
        inplace : bool
            Also write the estimates onto the model, so that ``simulate`` and
            ``transition_distribution`` use them. The result is returned
            either way.

        Returns
        -------
        FitResult
        """
        t,x = self._check_runs(t,x)
        p0 = self._starting_values(t,x) if p0 is None else p0

        result = fit_mle(
            lambda p: self.nnlf(p,t,x),
            p0,
            transform=self.parameter_transform,
            alpha=alpha,
            names=self.parameter_names,
            ndt_kwds=ndt_kwds,
            optimizer_kwds=optimizer_kwds,
            ci_method=ci_method,
        )

        if inplace:
            self.mu, self.sigma = result.params
            self.parameter_source = "estimated"
            self.parameter_covariance = result.cov

        return result


class RBM(Wiener):
    """Brownian motion with drift, reflected at zero."""

    def __init__(self,mu,sigma):
        super().__init__(mu,sigma)

    @classmethod
    def _check_runs(cls,t,x):
        """Also require the observations to lie in the support.
        """
        t,x = super()._check_runs(t,x)
        for m,run in enumerate(x):
            arr = np.asarray(run,dtype=float)
            if np.any(arr < 0):
                below = arr[arr < 0]
                raise ValueError(
                    f"run {m} has {below.size} negative observation(s), the "
                    f"smallest being {below.min():.6g}. Reflected Brownian "
                    "motion is confined to [0, inf), so these cannot have come "
                    "from this model. Fit a Wiener process instead, or shift "
                    "the data if zero is not the true reflecting boundary."
                )
        return t,x

    def simulate(self,h,T,initial=0.0,num_samples=1):
        # This one is a bit tricky to simulate. See [1] for the algorithm.
        #
        # References:
        #   [1] D. P. Kroese, T. Taimre, and Z. I. Botev, Handbook of Monte Carlo Methods, 1st ed. Wiley, 2011. doi: 10.1002/9781118014967.

        # scale to sigma == 1
        x0 = initial/self.sigma
        mu = self.mu/self.sigma

        K = int(np.floor(T/h)+1)
        Y = np.sqrt(h)*np.random.randn(num_samples,K)
        U = np.random.rand(num_samples,K)
        M = Y+np.sqrt(Y**2-2*h*np.log(U))
        M *= 0.5
        X = np.zeros((num_samples,K))
        X[:,0] = x0
        for m in range(num_samples):
            for k in range(1,K):
                X[m,k] = max(M[m,k-1]-Y[m,k-1],X[m,k-1]+mu*h-Y[m,k-1])

        # re-scale to sigma != 1
        X *= self.sigma

        return X

    def transition_distribution(self,x,t,x0,type="cdf",*,mu=None,sigma=None):
        r"""Distribution of the reflected state at ``t`` given ``x0``.

        Densities and distribution functions from [1]. The density is a signed
        sum of three terms,

        .. math::
            f(x) = \frac{1}{s}\varphi\!\Bigl(\frac{x-m}{s}\Bigr)
                 + \frac{1}{s}e^{2\mu x/\sigma^{2}}
                       \varphi\!\Bigl(\frac{x+m}{s}\Bigr)
                 - \frac{2\mu}{\sigma^{2}}e^{2\mu x/\sigma^{2}}
                       \Phi\!\Bigl(-\frac{x+m}{s}\Bigr),

        with :math:`m = \mu t + x_0` and :math:`s = \sigma\sqrt{t}`.

        Each term therefore formed in log space and combined with ``logsumexp``,
        which never builds the exponential and does the cancellation at full
        precision.

        A non-positive result returns ``-inf`` and ``pdf`` returns 0.

        References
        ----------
        [1] J. Abate and W. Whitt, "Transient Behavior of Regulated Brownian
        Motion, I: Starting at the Origin", Advances in Applied Probability,
        vol. 19, no. 3, pp. 560-598, 1987, doi: 10.2307/1427408.
        """
        mu = self.mu if mu is None else mu
        sigma = self.sigma if sigma is None else sigma
        if type.lower() not in ["logpdf","pdf","cdf"]:
            raise ValueError(f"type must be one of pdf, logpdf, cdf; got {type!r}")

        x = np.asarray(x,dtype=float)
        m = mu*t + x0
        s = sigma*np.sqrt(t)
        z_minus = (x - m)/s
        z_plus = (x + m)/s
        # The exponent that overflows if exponentiated on its own.
        drift = 2.0*mu*x/sigma**2

        if type.lower() == "cdf":
            # Written as a single exponential of a sum, so the overflowing
            # factor is never formed by itself.
            return (sps.norm.cdf(z_minus)
                    - np.exp(drift + sps.norm.logcdf(-z_plus)))

        # log of each term's magnitude, and the sign it carries
        log_s = np.log(s)
        # mu == 0 kills the third term; keep its log finite so that the zero
        # weight below multiplies a number rather than -inf.
        log_coefficient = (np.log(2.0*abs(mu)/sigma**2) if mu != 0 else 0.0)

        log_terms = np.broadcast_arrays(
            -log_s + sps.norm.logpdf(z_minus),
            -log_s + drift + sps.norm.logpdf(z_plus),
            log_coefficient + drift + sps.norm.logcdf(-z_plus),
        )
        log_terms = np.stack(log_terms)
        signs = np.stack([
            np.ones_like(log_terms[0]),
            np.ones_like(log_terms[0]),
            np.full_like(log_terms[0], -np.sign(mu)),
        ])

        value, sign = logsumexp(log_terms, b=signs, axis=0, return_sign=True)
        logpdf = np.where(sign > 0, value, -np.inf)

        if type.lower() == "logpdf":
            return logpdf if logpdf.ndim else float(logpdf)
        density = np.exp(logpdf)
        return density if density.ndim else float(density)

    def _solve_quantile(self,q,ti,x0,guess):
        """Invert the cdf at one time, checking that the solver converged."""
        root, _, converged, message = fsolve(
            lambda z: q - self.transition_distribution(z,ti,x0,type="cdf"),
            guess, full_output=True,
        )
        if converged != 1:
            raise RuntimeError(
                f"could not solve for the {q:g} quantile at t={ti:g}: "
                f"{message.strip()}"
            )
        return float(root[0])

    def get_upper_lower(self,t,x0,alpha=0.05):
        """Pointwise quantile band, from the alpha/2 and 1-alpha/2 quantiles.

        ``alpha`` is a significance level, matching every other ``alpha`` in
        the package, so the default 0.05 gives a 95% band. 
        """
        t = np.atleast_1d(np.asarray(t,dtype=float))
        centre = self.mu*t + x0
        spread = 1.96*self.sigma*np.sqrt(t)

        # The process is reflected at zero, so the lower guess is started just
        # inside the support rather than on the boundary, where the cdf is
        # flattest and the solver has least to work with.
        guesses_upper = centre + spread
        guesses_lower = np.maximum(centre - spread, 1e-8)

        U = np.zeros(t.size)
        L = np.zeros(t.size)
        for i,ti in enumerate(t):
            U[i] = self._solve_quantile(1.0-alpha/2.0, ti, x0, guesses_upper[i])
            L[i] = self._solve_quantile(alpha/2.0, ti, x0, guesses_lower[i])

        return L,U
