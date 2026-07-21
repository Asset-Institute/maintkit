import numpy as np
from maintkit.inference import fit_mle, result_at
from maintkit.transforms import Composite, Log, Logit
from maintkit.poisson_process import PowerLawNHPP

class ProportionalAgeReduction:
    """Power-law NHPP with proportional age reduction at each PM.

    Failures receive minimal repair; each PM sets the age clock back by
    ``rho`` times the time of the preceding PM.

    ``fit`` handles a fleet: the assets share one ``(a, b, rho)`` but each
    keeps its own maintenance schedule and its own end of observation. Passing
    a flat sequence of times fits a single asset. ``intensity`` and
    ``cumulative_intensity`` describe one asset at a time and take that asset's
    schedule directly.
    """

    #: Transform for the full parameter vector (a, b, rho).
    parameter_transform = Composite([Log(), Log(), Logit()])

    #: Transform for the reduced problem (b, rho), with a concentrated out.
    _reduced_transform = Composite([Log(), Logit()])

    parameter_names = ("a", "b", "rho")

    def __init__(self,a,b,rho):
        self.baseline_model = PowerLawNHPP(a,b)
        self.repair_factor = self._check_repair_factor(rho)

    def set_parameters(self,a,b,r):
        self.baseline_model = PowerLawNHPP(a,b)
        self.repair_factor = self._check_repair_factor(r)

    @staticmethod
    def _check_repair_factor(rho):
        """Repair factor must lie in [0, 1].

        Both endpoints are meaningful models: 0 is no age reduction, 1 is
        as-good-as-new. Raises rather than asserts because ``python -O`` strips
        assert statements, which would let an out-of-range value through
        silently.
        """
        rho = float(rho)
        if not 0.0 <= rho <= 1.0:
            raise ValueError(f"repair factor must be in [0, 1], got {rho}")
        return rho

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
    def _check_failure_times(failure_times, truncation_time, asset=None):
        """Failures must lie inside the observation window.

        An asset with no failures is allowed: it still ran, so it contributes
        exposure to the likelihood even though it contributes no events. Only
        a fleet with no failures anywhere cannot be fitted, and ``fit`` checks
        that once over the whole fleet rather than here.
        """
        failure_times = np.atleast_1d(np.asarray(failure_times, dtype=float))
        where = "" if asset is None else f" for asset {asset}"
        if failure_times.size and np.any(failure_times <= 0):
            raise ValueError(f"failure times must be strictly positive{where}")
        if np.any(failure_times > truncation_time):
            late = failure_times[failure_times > truncation_time]
            raise ValueError(
                f"{late.size} failure time(s){where} fall after "
                f"truncation_time={truncation_time}, the latest being "
                f"{late.max()}"
            )
        return failure_times

    @staticmethod
    def _as_asset_list(values, name):
        """Normalise per-asset times to a list of 1D arrays, one per asset.

        A flat sequence is taken to be a single asset, so single-asset callers
        need not wrap their data in a list. That is unambiguous here: an entry
        is either a number or a sequence of times, never both, so a flat
        sequence can only describe one asset.
        """
        if isinstance(values, np.ndarray):
            if values.ndim == 1:
                return [values.astype(float)]
            if values.ndim == 2:
                return [values[m, :].astype(float) for m in range(values.shape[0])]
            raise ValueError(
                f"{name} must be 1D (one asset) or 2D (one row per asset), "
                f"got {values.ndim} dimensions"
            )

        if isinstance(values, (list, tuple)):
            if len(values) == 0:
                # One asset with no times, not a fleet of no assets. An empty
                # pm_times is a real case: an asset that was never maintained.
                return [np.empty(0, dtype=float)]
            nested = [isinstance(v, (list, tuple, np.ndarray)) for v in values]
            if all(nested):
                return [np.atleast_1d(np.asarray(v, dtype=float)) for v in values]
            if any(nested):
                raise TypeError(
                    f"{name} mixes numbers and sequences; pass either a flat "
                    "sequence of times (one asset) or a sequence of sequences "
                    "(one per asset)"
                )
            return [np.asarray(values, dtype=float)]

        raise TypeError(
            f"{name} must be a sequence of times or a sequence of those, got "
            f"{type(values).__name__}"
        )

    @staticmethod
    def _resolve_truncation_times(truncation_times, n_assets):
        """Per-asset end of observation, broadcasting a single value.

        A scalar means every asset was observed to the same time.
        """
        values = np.atleast_1d(np.asarray(truncation_times, dtype=float))
        if values.ndim != 1:
            raise ValueError("truncation_times must be a scalar or 1D")
        if values.size == 1:
            return [float(values[0])] * n_assets
        if values.size != n_assets:
            raise ValueError(
                f"got {values.size} truncation times for {n_assets} assets; "
                "pass one per asset, or a single value for all of them"
            )
        return [float(v) for v in values]

    def _prepare(self, failure_times, pm_times, truncation_times):
        """Validate and line up the per-asset inputs.

        Returns ``(failures, edges)``, both lists of length ``n_assets``.
        """
        failures = self._as_asset_list(failure_times, "failure_times")
        schedules = self._as_asset_list(pm_times, "pm_times")

        if len(schedules) != len(failures):
            raise ValueError(
                f"got failure times for {len(failures)} asset(s) but "
                f"maintenance times for {len(schedules)}; each asset needs its "
                "own schedule, since assets are not maintained on the same days"
            )

        taus = self._resolve_truncation_times(truncation_times, len(failures))

        edges = [self._interval_edges(s, tau) for s, tau in zip(schedules, taus)]
        failures = [
            self._check_failure_times(f, tau, asset=m)
            for m, (f, tau) in enumerate(zip(failures, taus))
        ]
        return failures, edges

    def _total_exposure(self, b, rho, edges):
        """Cumulative intensity over the whole observation window, with a = 1.

        Summed over every interval of every asset, so for a single asset

            cumulative_intensity(truncation_time) = a * total_exposure

        Leaving ``a`` out is what makes it reusable: it is the denominator of
        the closed-form estimate ``a_hat = N / total_exposure``. Shared by nnlf,
        reduced_nnlf and fit so the three cannot drift apart.

        ``edges`` is the list of interval end points per asset, as returned by
        :meth:`_prepare`.
        """
        total = 0.0
        for e in edges:
            s = self._last_pm(e, e)
            total += np.sum((e - rho*s)**b - ((1.0 - rho)*s)**b)
        return total

    def _sum_log_shifted_ages(self, rho, failures, edges):
        """``sum log(t - rho * s)`` over every failure of every asset.

        ``s`` is the last PM strictly before the failure. Assets with no
        failures contribute nothing.
        """
        total = 0.0
        for f, e in zip(failures, edges):
            if f.size:
                total += np.sum(np.log(f - rho*self._last_pm(f, e)))
        return total

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
    
    def nnlf(self,p,failure_times,pm_times,truncation_times):
        """Negative log-likelihood for parameters ``p = (a, b, rho)``.

        ``pm_times`` holds the maintenance actions; ``truncation_times`` is
        when observation stopped. The exposure runs over the periods ending at
        each PM and then at the truncation time.

        The assets share ``(a, b, rho)`` but not their schedules, so the
        likelihood is a sum over assets of the single-asset likelihood.
        """
        a,b,r = p
        failures, edges = self._prepare(failure_times, pm_times, truncation_times)
        N = sum(f.size for f in failures)

        term1 = self._sum_log_shifted_ages(r, failures, edges)
        term2 = self._total_exposure(b, r, edges)
        like = N*np.log(a)+N*np.log(b) + (b-1)*term1 - a*term2
        return -like

    def reduced_nnlf(self,p,failure_times,pm_times,truncation_times):
        """Profile negative log-likelihood over ``p = (b, rho)``, with ``a``
        concentrated out at its conditional maximum ``a = N / total_exposure``."""
        b,r = p
        failures, edges = self._prepare(failure_times, pm_times, truncation_times)
        N = sum(f.size for f in failures)

        term1 = self._total_exposure(b, r, edges)
        term2 = self._sum_log_shifted_ages(r, failures, edges)
        like = N*np.log(b) + N*np.log(N) - N*np.log(term1) + (b-1)*term2 - N

        return -like
    
    def fit(self,failure_times,pm_times,truncation_times,p0=None,*,alpha=0.05,
            ndt_kwds=None,optimizer_kwds=None,ci_method="transformed"):
        """Fit ``(a, b, rho)`` by maximum likelihood across one or more assets.

        Runs in two steps. ``a`` has a closed-form conditional maximum, so it
        is concentrated out and only ``(b, rho)`` are optimised. ``a`` is then
        recovered and the covariance comes from the Hessian of the full
        three-parameter likelihood at that point -- the reduced problem cannot
        supply it, since it has no ``a`` to be uncertain about.

        All assets share one ``(a, b, rho)``; each keeps its own maintenance
        schedule and its own end of observation.

        Parameters
        ----------
        failure_times : array_like
            Failure times. A flat sequence is one asset; a sequence of
            sequences is one entry per asset. An asset may have none, in which
            case it contributes exposure but no events. All times must fall in
            ``(0, truncation_time]`` for that asset.
        pm_times : array_like
            Maintenance times, in the same layout as ``failure_times`` and with
            the same number of assets. Strictly increasing within an asset, all
            in ``(0, truncation_time)``. Do not include 0. Schedules need not
            agree across assets, and usually will not.
        truncation_times : float or array_like
            When observation stopped. A single value applies to every asset;
            otherwise pass one per asset.
        p0 : array_like, optional
            Starting values for ``(b, rho)`` only -- two entries, not three.
            ``a`` needs no starting value because it is never optimised.
            Defaults to ``[1.0, 0.5]``.
        alpha : float
            Significance level; 0.05 gives 95% intervals.

        Returns
        -------
        FitResult
            ``params`` is ``(a, b, rho)``.
        """
        failures, edges = self._prepare(failure_times, pm_times, truncation_times)
        n_failures = sum(f.size for f in failures)
        if n_failures == 0:
            raise ValueError("cannot fit: no failures observed")

        p0 = [1.0, 0.5] if p0 is None else np.atleast_1d(
            np.asarray(p0, dtype=float)
        )
        if len(p0) != 2:
            raise ValueError(
                f"p0 must hold starting values for (b, rho) only, so two "
                f"entries; got {len(p0)}. `a` is concentrated out and is not "
                "optimised."
            )

        # Step 1: optimise the reduced likelihood over (b, rho).
        # fit_mle also returns a covariance for these two, which is discarded:
        # the reported uncertainty must come from the full likelihood below.
        profile = fit_mle(
            lambda q: self.reduced_nnlf(
                q, failure_times, pm_times, truncation_times
            ),
            p0,
            transform=self._reduced_transform,
            optimizer_kwds=optimizer_kwds,
            ndt_kwds=ndt_kwds,
        )
        b_hat, rho_hat = profile.params

        # Step 2: recover a at its conditional maximum.
        a_hat = n_failures / self._total_exposure(b_hat, rho_hat, edges)

        # Step 3: covariance from the full three-parameter likelihood.
        return result_at(
            lambda p: self.nnlf(p, failure_times, pm_times, truncation_times),
            [a_hat, b_hat, rho_hat],
            transform=self.parameter_transform,
            alpha=alpha,
            names=self.parameter_names,
            ndt_kwds=ndt_kwds,
            ci_method=ci_method,
            success=profile.success,
            message=profile.message,
        )


    



