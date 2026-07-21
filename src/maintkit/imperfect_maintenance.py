from scipy import stats as stats
from scipy import optimize as opt
from scipy.integrate import quad
import numpy as np
import numdifftools as ndt
from maintkit.utilities import _parameter_transform_log
# Import the class directly rather than `import maintkit.poisson_process as rpp`.
# The module and the class share the name `poisson_process`, so the package
# __init__ re-binds the attribute `maintkit.poisson_process` to the *class*,
# and `import ... as rpp` would then resolve to the class, not the module.
from maintkit.poisson_process import power_law_nhpp

class imperfect_pm_minimal_cm:
    # Uses a proportinal age reduction modification of a power-law NHPP for now. 
    # Only works for a single asset at the moment. 

    def __init__(self,a,b,rho):
        self.baseline_model = power_law_nhpp(a,b)
        assert rho<=1.0 and rho>=0, "Repair factor must be between 0 and 1 (inclusive)"
        self.repair_factor = rho
    
    def set_parameters(self,a,b,r):
        self.baseline_model = power_law_nhpp(a,b)
        assert r<=1.0 and r>=0, "Repair factor must be between 0 and 1 (inclusive)"
        self.repair_factor = r

    def _last_pm(self,t,pm_times):
        # returns last PM before all time points

        pm_times = np.array(pm_times)
        last_pm = []
        for tt in t:
            idx, = np.where(pm_times<tt)
            if len(idx)==0:
                last_pm.append(0)
            else:
                idx_last = np.max(idx)
                last_pm.append(pm_times[idx_last])

        return np.array(last_pm)

    @staticmethod
    def _interval_edges(pm_times, truncation_time):
        """Validate the observation window and return the interval end points.

        ``pm_times`` holds only the maintenance actions. Observation runs from
        0 to ``truncation_time``, so the periods between events end at each PM
        and then at the truncation time, which is what this returns.

        Keeping the schedule and the end of observation apart matters because
        they are different things. A PM resets the age clock; the end of
        observation does not. Passing them in one array cannot express a study
        that ran on past the last PM, and it silently accepts failures recorded
        after observation stopped.
        """
        pm_times = np.atleast_1d(np.asarray(pm_times, dtype=float))
        truncation_time = float(truncation_time)

        if truncation_time <= 0:
            raise ValueError(
                f"truncation_time must be positive, got {truncation_time}"
            )
        if pm_times.ndim != 1:
            raise ValueError("pm_times must be one-dimensional")
        if pm_times.size and np.any(np.diff(pm_times) <= 0):
            raise ValueError("pm_times must be strictly increasing")
        if pm_times.size and pm_times[0] <= 0:
            raise ValueError(
                "pm_times must be strictly positive; observation starts at 0, "
                "and a PM at 0 has no earlier PM to set the clock back from, so "
                "it does nothing. Do not include it."
            )
        if pm_times.size and pm_times[-1] >= truncation_time:
            raise ValueError(
                f"pm_times must fall before truncation_time; last PM is "
                f"{pm_times[-1]} and observation ends at {truncation_time}"
            )

        return np.concatenate([pm_times, [truncation_time]])

    @staticmethod
    def _check_failure_times(failure_times, truncation_time):
        """Failures must lie inside the observation window."""
        failure_times = np.atleast_1d(np.asarray(failure_times, dtype=float))
        if failure_times.size == 0:
            raise ValueError("cannot fit: no failures observed")
        if np.any(failure_times <= 0):
            raise ValueError("failure times must be strictly positive")
        if np.any(failure_times > truncation_time):
            late = failure_times[failure_times > truncation_time]
            raise ValueError(
                f"{late.size} failure time(s) fall after truncation_time="
                f"{truncation_time}, the latest being {late.max()}"
            )
        return failure_times

    def intensity(self,t,pm_times):
        """Failure intensity at each time in ``t``.

        Needs only the maintenance schedule, since the age clock does not
        depend on when observation stops.
        """
        t = np.atleast_1d(np.asarray(t, dtype=float))
        last_pm = self._last_pm(t,pm_times)
        return self.baseline_model.intensity(t-self.repair_factor*last_pm)

    def cumulative_intensity(self,t,pm_times,truncation_time):
        r"""Cumulative number of failures expected by each time in ``t``.

        Between two PM times the process is just the baseline power law, but
        running on a clock that the last PM set back. Each PM subtracts
        :math:`\rho` times the time of the PM before it, so on
        :math:`(\tau_{j-1}, \tau_j]` the intensity is

        .. math::
            \lambda(u) = a\,b\,\bigl(u - \rho\,\tau_{j-1}\bigr)^{b-1},

        and integrating over that interval gives

        .. math::
            \Lambda_0\bigl(\tau_j - \rho\,\tau_{j-1}\bigr)
            - \Lambda_0\bigl((1-\rho)\,\tau_{j-1}\bigr),

        where :math:`\Lambda_0` is the baseline cumulative intensity. Both
        endpoints are shifted back by the same :math:`\rho\,\tau_{j-1}`.

        Summing the finished intervals and adding the part of the one holding
        :math:`t`,

        .. math::
            M(t) = \sum_{k \le j-1}
                     \Bigl[\Lambda_0\bigl(\tau_k - \rho\,\tau_{k-1}\bigr)
                     - \Lambda_0\bigl((1-\rho)\,\tau_{k-1}\bigr)\Bigr]
                   + \Lambda_0\bigl(t - \rho\,\tau_{j-1}\bigr)
                   - \Lambda_0\bigl((1-\rho)\,\tau_{j-1}\bigr),

        for :math:`t \in (\tau_{j-1}, \tau_j]`. The sum is accumulated once
        with ``cumsum``, and each time in ``t`` looks up which :math:`j` it
        falls in.

        With PMs at 50 and 100 and :math:`\rho = 0.4`: observation starts at 0,
        so the first interval runs from 0 to 50 with nothing subtracted. The
        next, 50 to 100, follows the PM at 50, so
        :math:`0.4 \times 50 = 20` comes off both ends and it runs from 30 to
        80. The last interval runs from the final PM to ``truncation_time``.

        Parameters
        ----------
        t : array_like
            Times at which to evaluate. Need not be sorted or evenly spaced.
        pm_times : array_like
            Maintenance times, strictly increasing, all in
            ``(0, truncation_time)``. Do not include 0: observation starts
            there, and a PM at 0 has no earlier PM to set the clock back from,
            so it would do nothing.
        truncation_time : float
            When observation stopped. Unlike a PM, this does not reset the age
            clock; it just ends the last interval.

        Returns
        -------
        ndarray
            Same shape as ``t``, non-decreasing, zero at ``t = 0``.

        """
        t = np.atleast_1d(np.asarray(t, dtype=float))
        edges = self._interval_edges(pm_times, truncation_time)

        rho = self.repair_factor
        s = self._last_pm(edges, edges)           # last PM strictly before each edge
        w = edges - rho*s                         # interval upper limit, shifted clock
        v = (1.0 - rho)*s                         # interval lower limit, shifted clock

        # completed[j] = total over intervals strictly below j
        per_interval = (self.baseline_model.cumulative_intensity(w, t0=0.0)
                        - self.baseline_model.cumulative_intensity(v, t0=0.0))
        completed = np.concatenate([[0.0], np.cumsum(per_interval)])

        # side="left" is deliberate: a point in (tau_{j-1}, tau_j] belongs to
        # the interval whose UPPER endpoint is tau_j. side="right"-1 selects the
        # interval below and silently gives the wrong shift.
        j = np.clip(np.searchsorted(edges, t, side="left"), 0, edges.size - 1)
        s_j = s[j]

        return (completed[j]
                + self.baseline_model.cumulative_intensity(t - rho*s_j, t0=0.0)
                - self.baseline_model.cumulative_intensity((1.0 - rho)*s_j, t0=0.0))
    
    def nnlf(self,p,failure_times,pm_times,truncation_time):
        """Negative log-likelihood for parameters ``p = (a, b, rho)``.

        ``pm_times`` holds the maintenance actions; ``truncation_time`` is when
        observation stopped. The compensator runs over the periods ending at
        each PM and then at the truncation time.
        """
        a,b,r = p
        failure_times = np.atleast_1d(np.asarray(failure_times, dtype=float))
        edges = self._interval_edges(pm_times, truncation_time)
        N = failure_times.size

        last_pm_before_failure = self._last_pm(failure_times,edges)
        last_pm_before_edge = self._last_pm(edges,edges)

        term1 = np.sum( np.log(failure_times-r*last_pm_before_failure) )
        term2 = np.sum( (edges-r*last_pm_before_edge)**b - ((1-r)*last_pm_before_edge)**b)
        like = N*np.log(a)+N*np.log(b) + (b-1)*term1 - a*term2
        return -like

    def reduced_nnlf(self,p,failure_times,pm_times,truncation_time):
        """Profile negative log-likelihood over ``p = (b, rho)``, with ``a``
        concentrated out at its conditional maximum ``a = N / term1``."""
        b,r = p
        failure_times = np.atleast_1d(np.asarray(failure_times, dtype=float))
        edges = self._interval_edges(pm_times, truncation_time)
        N = failure_times.size

        last_pm_before_failure = self._last_pm(failure_times,edges)
        last_pm_before_edge = self._last_pm(edges,edges)

        term1 = np.sum( (edges-r*last_pm_before_edge)**b - ((1-r)*last_pm_before_edge)**b )
        term2 = np.sum( np.log(failure_times-r*last_pm_before_failure) )
        like = N*np.log(b) + N*np.log(N) - N*np.log(term1) + (b-1)*term2 - N

        return -like
    
    def fit(self,failure_times,pm_times,truncation_time,ndt_kwds={}):
        """Fit ``(a, b, rho)`` by maximum likelihood.

        Parameters
        ----------
        failure_times : array_like
            Failure times, all in ``(0, truncation_time]``.
        pm_times : array_like
            Maintenance times, strictly increasing, all in
            ``(0, truncation_time)``. Do not include 0.
        truncation_time : float
            When observation stopped.
        """
        failure_times = self._check_failure_times(failure_times, truncation_time)
        self._interval_edges(pm_times, truncation_time)   # validate before fitting

        transform = lambda u: np.r_[np.log(u[0:-1]), np.log(u[-1]/(1-u[-1]))]
        inverse_transform = lambda u: np.r_[ np.exp(u[0:-1]), 1/(1+np.exp(-u[-1])) ]
        y0 = transform([1,0.5])
        robj = lambda x: self.reduced_nnlf(inverse_transform(x),failure_times,pm_times,truncation_time)
        result = opt.minimize(robj,y0)
        y_hat = result.x
        b_hat,r_hat = inverse_transform(y_hat)

        edges = self._interval_edges(pm_times, truncation_time)
        s_edges = self._last_pm(edges, edges)
        a_hat = len(failure_times)/np.sum( (edges-r_hat*s_edges)**b_hat
                                            -((1-r_hat)*s_edges)**b_hat )
        
        p_hat = [a_hat,b_hat,r_hat]
        log_p_hat = transform(p_hat)
        full_obj = lambda x: self.nnlf(inverse_transform(x),failure_times,pm_times,truncation_time)
        H = ndt.Hessian(full_obj,**ndt_kwds)(log_p_hat)
        log_p_cov = np.linalg.inv(H)
        s = np.sqrt(np.diag(log_p_cov))
        p_ci =  log_p_hat +1.96*np.array([-s,s]) # log p_ci initially
        p_ci[0,:] = inverse_transform(p_ci[0,:])
        p_ci[1,:] = inverse_transform(p_ci[1,:])

        return p_hat,p_ci


    



