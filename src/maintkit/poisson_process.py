from scipy import stats as stats
from scipy import optimize as opt
from scipy.integrate import quad
import numpy as np
import numdifftools as ndt
from maintkit.utilities import _parameter_transform_log
from maintkit.inference import fit_mle, result_at
from maintkit.transforms import Log

class poisson_process:

    #: Transform used when fitting. Both power-law parameters are positive.
    parameter_transform = Log()

    #: Parameter names, used only for FitResult.summary().
    parameter_names = None

    def __init__(self,parameters):
        self.parameters = parameters
    
    def intensity(self,t):
        raise NotImplementedError("Intensity must be defined via subclassing.")

    def random_counts(self,t,s=0,size=1):
        if s != min(t):
            t = np.insert(t,0,s)

        dN = np.zeros((size,len(t)))
        for ii,tii in enumerate(t):
            if ii > 0:
                M = self.cumulative_intensity(t[ii],t0=t[ii-1])
                dN[:,ii] = stats.poisson.rvs(M,size=size)
        
        return np.cumsum(dN,axis=1)

    def cumulative_intensity(self,t1,t0=0):
        t1 = np.atleast_1d(t1)
        t0 = np.atleast_1d(t0)
        if t0.size == 1:
            t0 = np.full(t1.shape, t0.item())
        elif t0.size != t1.size:
            raise ValueError("t0 must be a scalar or the same length as t1")

        LAMBDA = [quad(self.intensity, t0[ii], t1[ii])[0] for ii in range(t1.size)]
        return LAMBDA
    
    def log_intensity(self,t):
        return np.log(self.intensity(t))

    def reliability(self,t,w):
        return np.exp(-self.cumulative_intensity(w,t0=t))
    
    def pdf(self,t,t_previous=0):
        w = t-t_previous
        return self.intensity(t)*self.reliability(w,t0=t_previous)
    
    def nnlf(self,p,event_times,truncation_times=None):
        # event_times[asset][failure time index], truncation_time=None means that last index is a failure.

        original_parameters = self.parameters
        self.parameters = p

        # turn into a list if tim is a numpy array. Lists are preferred
        # since they can be ragged and have different numbers of event times. 
        if isinstance(event_times,np.ndarray):
            event_times = event_times.tolist()

        # check for valid truncation time
        if truncation_times is not None:
            for m,_ in enumerate(event_times):
                if len(event_times[m]) > 0 and truncation_times[m] is not None:
                    if not truncation_times[m] > max(event_times[m]):
                        raise ValueError("Invalid truncation time for asset "+str(m))

        like = 0
        for m,_ in enumerate(event_times):
            for f in event_times[m]:
                like += self.log_intensity(f)

            if truncation_times is not None and truncation_times[m] is not None:
                T = truncation_times[m]
                like += -np.sum(self.cumulative_intensity(T,t0=0))
        
        self.parameters = original_parameters
        return -like

    def fit(self,event_times,p0,truncation_times=None,*,alpha=0.05,
            ndt_kwds=None,optimizer_kwds=None,ci_method="transformed",
            use_analytic_gradient=True):
        """Fit the process by maximum likelihood.

        Parameters
        ----------
        event_times : list of lists, or 2D ndarray
            ``event_times[asset][k]`` is the k-th event time for that asset.
            A ragged list is preferred, since assets rarely have equal counts.
        p0 : array_like
            Starting values, in natural parameters.
        truncation_times : list, optional
            End of observation per asset. ``None`` -- for the whole argument or
            for an individual asset -- means that asset contributes no
            compensator term, i.e. its last event is a failure rather than the
            end of observation.
        alpha : float
            Significance level; 0.05 gives 95% intervals.
        ci_method : {'transformed', 'natural'}
            'transformed' (default) matches what this module already did:
            intervals built on the log scale and mapped back.
        use_analytic_gradient : bool
            Use the subclass's ``nnlf_gradient`` when it defines one.

        Returns
        -------
        FitResult
        """
        event_times = self._validate_event_times(event_times)

        score = getattr(self, "nnlf_gradient", None) if use_analytic_gradient else None
        gradient = (
            (lambda p: score(p,event_times,truncation_times))
            if score is not None else None
        )

        return fit_mle(
            lambda p: self.nnlf(p,event_times,truncation_times=truncation_times),
            p0,
            gradient=gradient,
            transform=self.parameter_transform,
            alpha=alpha,
            names=self.parameter_names,
            optimizer_kwds=optimizer_kwds,
            ndt_kwds=ndt_kwds,
            ci_method=ci_method,
        )

    @staticmethod
    def _validate_event_times(event_times):
        """Normalise event times to a list of lists.

        Raises rather than asserts: assertions are stripped under ``python -O``,
        which would turn a clear message into an obscure failure later.
        """
        if isinstance(event_times,np.ndarray):
            return [event_times[m,:].tolist() for m in range(event_times.shape[0])]
        if isinstance(event_times,list):
            if not all(isinstance(e,(list,np.ndarray)) for e in event_times):
                raise TypeError(
                    "event_times must be a list of lists (one list of event "
                    "times per asset) or a 2D numpy array"
                )
            return [list(e) for e in event_times]
        raise TypeError(
            "event_times must be a list of lists or a 2D numpy array, got "
            f"{type(event_times).__name__}"
        )

    def transform_scale(self,x,likelihood_hessian=None,direction="inverse"):
         return _parameter_transform_log(x,likelihood_hessian=likelihood_hessian,\
            direction=direction)

class power_law_nhpp(poisson_process):

    parameter_names = ("a", "b")

    def __init__(self,a,b):
        self.parameters = [a,b]
    
    def intensity(self,t):
        a,b = self.parameters
        return a*b*t**(b-1)
    
    def random_arrival_times(self,T,t0=0,size=1):

        msg = "Suspension times must be either an int>0 or a list of len == size"
        if isinstance(T,int):
            T = [T]*size
        else:
            assert isinstance(T,list), msg
            assert len(T) == size, msg

        t = [ [] for m in range(size)]
        a,b = [*self.parameters]
        for m in range(size):
            t_next = t0*1.0
            while t_next < T[m]:
                t_last = t_next*1.0
                t[m].append(t_last)
                U = np.random.rand()
                t_next = (-np.log(1-U)/a + t_last**b)**(1/b)
        
        t = [t[m][1::] for m in range(size)]
        return t

    def cumulative_intensity(self, t1, t0=0):
        a,b = [*self.parameters]
        return a*(t1**b-t0**b)
    
    def log_intensity(self, t):
        a,b = [*self.parameters]
        return np.log(a)+np.log(b)+(b-1)*np.log(t)

    def nnlf_gradient(self,p,event_times,truncation_times=None):
        r"""Analytic score of the power-law NHPP negative log-likelihood.

        With intensity :math:`\lambda(t)=a b t^{b-1}` and cumulative intensity
        :math:`\Lambda(T)=a T^{b}`, the log-likelihood across assets is

        .. math::
            \ell = N\log a + N\log b + (b-1)S - a\sum_m T_m^{b}

        where :math:`N` is the total number of events and
        :math:`S=\sum_m\sum_k \log t_{mk}`. The compensator sum runs only over
        assets that actually have a truncation time, matching :meth:`nnlf`,
        which omits the :math:`-\Lambda(T)` term when the truncation time is
        ``None``. Keeping the two consistent matters: an objective and a
        gradient that disagree will send the optimiser somewhere neither of
        them minimises.

        .. math::
            \partial\ell/\partial a &= N/a - \sum_m T_m^{b} \\
            \partial\ell/\partial b &= N/b + S - a\sum_m T_m^{b}\log T_m

        Setting these to zero recovers :math:`\hat a = N/\sum_m T_m^{b}` and
        the profile equation for :math:`b`.

        Returns the gradient of the *negative* log-likelihood -- that is,
        minus the score -- ordered ``(a, b)`` to match :meth:`nnlf`.
        """
        a, b = p[0], p[1]

        if isinstance(event_times,np.ndarray):
            event_times = event_times.tolist()

        n_events = 0
        sum_log_t = 0.0
        for events in event_times:
            n_events += len(events)
            for t in events:
                sum_log_t += np.log(t)

        sum_Tb = 0.0
        sum_Tb_logT = 0.0
        if truncation_times is not None:
            for m,_ in enumerate(event_times):
                T = truncation_times[m]
                if T is None:
                    continue
                Tb = T**b
                sum_Tb += Tb
                sum_Tb_logT += Tb*np.log(T)

        dl_da = n_events/a - sum_Tb
        dl_db = n_events/b + sum_log_t - a*sum_Tb_logT

        return -np.array([dl_da, dl_db])
    
    def fit(self,event_times,truncation_times=None,*,alpha=0.05,
            ndt_kwds=None,ci_method="transformed"):
        """Fit by maximum likelihood using the closed-form estimate.

        The MLE reduces to a single equation in the shape parameter, so no
        general-purpose optimiser runs; see :meth:`_profile_mle`. Only the
        Hessian is computed numerically, to obtain standard errors.

        Parameters
        ----------
        event_times : list of lists, or 2D ndarray
            ``event_times[asset][k]`` is the k-th event time for that asset.
        truncation_times : list, optional
            End of observation per asset. If omitted, each asset's last event
            is treated as its truncation time.
        alpha : float
            Significance level; 0.05 gives 95% intervals.

        Returns
        -------
        FitResult
        """
        event_times = self._validate_event_times(event_times)
        tau = self._resolve_truncation_times(event_times, truncation_times)
        p_hat = self._profile_mle(event_times, tau)

        return result_at(
            lambda p: self.nnlf(p,event_times,truncation_times=truncation_times),
            p_hat,
            transform=self.parameter_transform,
            alpha=alpha,
            names=self.parameter_names,
            ndt_kwds=ndt_kwds,
            ci_method=ci_method,
        )

    @staticmethod
    def _resolve_truncation_times(event_times, truncation_times):
        """Per-asset end of observation, defaulting to the last observed event."""
        tau = []
        for m, events in enumerate(event_times):
            if truncation_times is None:
                if not events:
                    raise ValueError(
                        f"asset {m} has no events and no truncation time, so "
                        "its observation window is undefined"
                    )
                tau.append(max(events))
            else:
                T = truncation_times[m]
                if events and not T > max(events):
                    raise ValueError(
                        f"Invalid truncation time for asset {m}: {T} is not "
                        f"after its last event {max(events)}"
                    )
                tau.append(T)
        return tau

    #: Upper limit of the shape-parameter bracket search. A root beyond this
    #: means the data cannot identify a shape at all, not that b is large.
    _MAX_SHAPE = 1e4

    @staticmethod
    def _profile_score(b, n_events, sum_log_t, tau):
        r"""Score for :math:`b` after eliminating :math:`a`.

        Substituting :math:`\hat a(b) = N/\sum_m T_m^b` into the likelihood
        leaves a single equation in :math:`b`:

        .. math::
            N/b + S - N\frac{\sum_m T_m^b \log T_m}{\sum_m T_m^b} = 0

        The ratio is evaluated with weights :math:`(T_m/T_{max})^b \le 1`
        rather than :math:`T_m^b` directly. Written the obvious way it
        overflows for even moderate ``b`` (400**200 is inf), which turns the
        bracket search into a hunt through nan.
        """
        log_tau = np.log(tau)
        w = np.exp(b * (log_tau - log_tau.max()))
        return n_events/b + sum_log_t - n_events*np.sum(w*log_tau)/np.sum(w)

    @classmethod
    def _profile_mle(cls, event_times, tau):
        r"""Maximum-likelihood estimate of ``(a, b)``.

        Solves the profile score above for :math:`b` by bracketed root
        finding, then recovers :math:`\hat a = N/\sum_m T_m^{\hat b}`.

        When every :math:`T_m` is equal the ratio collapses to :math:`\log T`
        and the root is exactly the Crow closed form
        :math:`N/\sum_m\sum_k\log(T_m/t_{mk})`; this generalises that estimator
        rather than replacing it.

        A bracket always exists for identifiable data: the score tends to
        :math:`+\infty` as :math:`b\to0^+`, and to
        :math:`-\sum_m\sum_k\log(T_{max}/t_{mk}) < 0` as :math:`b\to\infty`.
        """
        tau = np.asarray(tau, dtype=float)
        n_events = sum(len(e) for e in event_times)
        if n_events == 0:
            raise ValueError(
                "cannot fit: no events observed across any asset"
            )
        sum_log_t = sum(np.log(t) for events in event_times for t in events)

        def score(b):
            return cls._profile_score(b, n_events, sum_log_t, tau)

        lo, hi = 1e-8, 1.0
        while score(hi) > 0:
            hi *= 2.0
            if hi > cls._MAX_SHAPE:
                raise ValueError(
                    "the likelihood has no interior maximum in the shape "
                    "parameter: every event coincides with its asset's "
                    "truncation time, so the data cannot identify a shape"
                )

        b_hat = opt.brentq(score, lo, hi)
        a_hat = n_events / np.sum(tau ** b_hat)
        return np.array([a_hat, b_hat])
    
    def nnlf_interval(self,p,ni,ins,cumulative=False):
        
        assert isinstance(ni,list), "number of events must be a list of lists"
        assert len(ni)>0, "number of events must be a list of lists"
        assert all([isinstance(ni[ii],list) for ii in range(len(ni))]), "number of events must be a list of lists"
            
        assert isinstance(ins,list), "inspections must be a list of lists"
        assert len(ins)>0, "inspections must be a list of lists"
        assert all([isinstance(ins[ii],list) for ii in range(len(ins))]), "inspections must be a list of lists"
        
        original_parameters = self.parameters
        self.parameters = p
        
        loglike = 0
        for m in range(len(ni)):
            # difference to obtain number of arrivals in the time interval since last inspection
            if cumulative:
                nim = np.diff(ni[m])
            else:
                nim = ni[m]
        
            if len(nim)>0: # otherwise there is no data :(
                inspec_m = np.array(ins[m])
                LAMBDA = self.cumulative_intensity(inspec_m[1::],t0=inspec_m[0:-1])
                for ii,nimii in enumerate(nim):
                    loglike += stats.poisson(mu=LAMBDA[ii]).logpmf(nimii)
        
        self.parameters = original_parameters

        return -loglike
    
    def fit_interval(self,ni,ins,p0,cumulative=False,*,alpha=0.05,
                     ndt_kwds=None,optimizer_kwds=None,ci_method="transformed"):
        """Fit from interval counts rather than exact event times.

        Parameters
        ----------
        ni : list of lists
            Number of events observed in each inter-inspection interval.
        ins : list of lists
            Inspection times per asset; ``len(ins[m]) == len(ni[m]) + 1``.
        p0 : array_like
            Starting values, in natural parameters.
        cumulative : bool
            True if ``ni`` holds cumulative counts rather than per-interval
            counts.
        alpha : float
            Significance level; 0.05 gives 95% intervals.

        Returns
        -------
        FitResult
            Confidence intervals and covariance are always computed; the
            previous ``estimate_ci`` flag has been removed, since it returned
            a result whose ``ci`` and ``cov`` were arrays of nan with an
            inconsistent shape.
        """
        return fit_mle(
            lambda p: self.nnlf_interval(p,ni,ins,cumulative=cumulative),
            p0,
            transform=self.parameter_transform,
            alpha=alpha,
            names=self.parameter_names,
            optimizer_kwds=optimizer_kwds,
            ndt_kwds=ndt_kwds,
            ci_method=ci_method,
        )

    def transform_scale(self,x,likelihood_hessian=None,direction="inverse"):
        return _parameter_transform_log(x,likelihood_hessian=likelihood_hessian,\
            direction=direction)

    def mcf_confidence_interval(self,t,p_cov,kind="mcf",*,alpha=0.05,
                                ndt_kwds=None):
    
        if kind.lower() not in ["time", "mcf"]:
            raise ValueError(f"kind must be 'time' or 'mcf', got {kind!r}")
        if not all((t[ii+1]-t[ii]) >= 0 for ii in range(len(t)-1)):
            raise ValueError("time vector must be sorted")
        if t[0] < 0:
            raise ValueError("times must be non-negative")
        ndt_kwds = {} if ndt_kwds is None else dict(ndt_kwds)
        c = stats.norm.ppf(1.0 - alpha/2.0)
        if t[0] == 0:
            print('Warning: inserting nan for t==0 since logM(t) is undefined.')
            prependNaN = True
            t = t[1::]
        else:
            prependNaN = False

        #Confidence intervals using the Delta Method on the log(M(t)) and then 
        # transforming back. 
        #  The below is a bit lazy and uses numerical gradients. Might use analytical gradients later.
        a,b = self.parameters
        p = np.array([a,b])
        M = self.cumulative_intensity(t)
        if kind.lower() == "time":
            u = np.log(t)
            fun = lambda x: 1/x[1] * (np.log(M)-np.log(x[0]))
            g = ndt.Gradient(fun,**ndt_kwds)(p)
            w = c*np.sqrt( np.sum(g@p_cov*g,axis=1) )
            ML,MU = self.cumulative_intensity(np.exp(u+w)),self.cumulative_intensity(np.exp(u-w))

        elif kind.lower() == "mcf":
            u = np.log(M)
            fun = lambda x: x[1]*np.log(t)+np.log(x[0])
            g = ndt.Gradient(fun,**ndt_kwds)(p)
            w = c*np.sqrt( np.sum(g@p_cov*g,axis=1) )
            ML,MU = np.exp(u-w),np.exp(u+w)
        
        if prependNaN:
            ML = np.insert(ML,0,np.nan)
            MU = np.insert(MU,0,np.nan)
        
        return ML,MU
