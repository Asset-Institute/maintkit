# note that all packages must have licenses that permit commercial use!
from scipy import stats as stats
from scipy import optimize as opt
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numdifftools as ndt
from maintkit.utilities import _parameter_transform_log,_parameter_transform_identity
from maintkit.inference import fit_mle, result_from_covariance
from maintkit.transforms import Identity, Log

class reliability_distribution(stats.rv_continuous):

    #: Transform used when fitting. Subclasses override for constrained
    #: parameters (see weibull, which uses Log for positive eta/beta).
    parameter_transform = Identity()

    #: Parameter names, used only for FitResult.summary().
    parameter_names = None

    def __init__(self,*args,**kwargs): #need *args and **kwargs so that I pass these into the methods inherited from the parent class!
        stats.rv_continuous.__init__(self,*args,**kwargs)
        self.a = 0

    def reliability(self,x,*args,**kwargs):
        return self.sf(x,*args,**kwargs)
    
    def log_reliability(self,x,*args,**kwargs):
        return self.logsf(x,*args,**kwargs)
    
    def hazard(self,x,*args,**kwargs):
        return np.exp(self.logpdf(x,*args,**kwargs)-self.log_reliability(x,*args,**kwargs))
   
    def conditional_reliability(self,tau,t0):
        return np.exp(self.log_reliability(t0+tau) - self.log_reliability(t0))
    
    def fit(self,ti,p0,observed="all",*,alpha=0.05,ndt_kwds=None,
            optimizer_kwds=None,ci_method="transformed",
            use_analytic_gradient=True):
        """Fit by maximum likelihood, handling right-censored observations.

        Overrides ``scipy.stats`` fitting, which does not handle censoring.

        Parameters
        ----------
        ti : array_like
            Observed times.
        p0 : array_like
            Starting values, in natural parameters.
        observed : array_like or "all"
            1 for an observed failure, 0 for a right-censored observation.
        alpha : float
            Significance level; 0.05 gives 95% intervals.
        ci_method : {'transformed', 'natural'}
            'transformed' (default) builds intervals in the unconstrained
            space and maps them back, so bounds respect parameter constraints.
            'natural' reproduces the symmetric intervals returned by versions
            before the fitters were consolidated.
        use_analytic_gradient : bool
            Use the subclass's ``nnlf_gradient`` when it defines one (weibull
            does). Set False to fall back to finite differences, e.g. to
            compare against results produced before the score was available.

        Returns
        -------
        FitResult
        """
        # Use the closed-form score when the subclass provides one. Removes the
        # finite-difference noise that otherwise stops BFGS short of its gtol.
        score = getattr(self, "nnlf_gradient", None) if use_analytic_gradient else None
        gradient = (lambda p: score(p,ti,observed)) if score is not None else None

        return fit_mle(
            lambda p: self.nnlf(p,ti,observed),
            p0,
            gradient=gradient,
            transform=self.parameter_transform,
            alpha=alpha,
            names=self.parameter_names,
            optimizer_kwds=optimizer_kwds,
            ndt_kwds=ndt_kwds,
            ci_method=ci_method,
        )

    def fit_interval(self,ti,ins,p0,observed="all",bnds=None,*,alpha=0.05,
                     ndt_kwds=None,optimizer_kwds=None,ci_method="transformed"):
        """Fit by maximum likelihood from interval-censored observations.

        ``ins`` holds the lower bound of each interval and ``ti`` the upper
        bound. See :meth:`fit` for the remaining parameters.

        Returns
        -------
        FitResult
        """
        return fit_mle(
            lambda p: self.nnlf_interval(p,ti,ins,observed),
            p0,
            transform=self.parameter_transform,
            alpha=alpha,
            names=self.parameter_names,
            optimizer_kwds=optimizer_kwds,
            ndt_kwds=ndt_kwds,
            ci_method=ci_method,
        )


    def freeze(self, *args, **kwds):
        return reliability_distribution_frozen(self, *args, **kwds) # freeze using new reliabilty class, otherwise new functions won't be defined (e.g. reliability)
    
    def transform_scale(self,x,likelihood_hessian=None,direction="inverse"):
        # define as unity transform unless overwritten
         return _parameter_transform_identity(x,likelihood_hessian=likelihood_hessian,\
            direction=direction)

# rv_continuous_frozen, not rv_frozen: scipy splits the two, and rv_frozen
# carries cdf, sf and ppf but not pdf or logpdf. Inheriting from it gives a
# "distribution" with no density.
class reliability_distribution_frozen(stats._distn_infrastructure.rv_continuous_frozen):
    def __init__(self, dist, *args, **kwds):
        self.args = args
        self.kwds = kwds
        self.dist = dist
    def reliability(self,x):
        return self.dist.sf(x,*self.args, **self.kwds)
    def log_reliability(self,x):
        return self.dist.logsf(x,*self.args, **self.kwds)
    def hazard(self,x):
        return self.dist.hazard(x,*self.args, **self.kwds)
    def conditional_reliability(self,tau,t0):
        return np.exp(self.dist.log_reliability(t0+tau,*self.args, **self.kwds) - self.dist.log_reliability(t0,*self.args, **self.kwds))
    
    def plot(self,type:str='pdf',figkwds=None,pltkwds=None,ax=None,t0=None):
        """
        Plot one characteristic of the frozen distribution.

        Parameters
        ----------
        type : {'pdf','cdf','reliability','hazard','conditional_reliability'}
            Which curve to plot.
        figkwds : dict, optional
            Passed to ``matplotlib.pyplot.subplots``; used only when ``ax is None``.
        pltkwds : dict, optional
            Passed to ``matplotlib.axes.Axes.plot``.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. Created if not supplied.
        t0 : float, optional
            Required when ``type='conditional_reliability'``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        valid = ['pdf','cdf','reliability','hazard','conditional_reliability']
        if type not in valid:
            raise ValueError(f"type must be one of {valid}, got {type!r}")
        if type == 'conditional_reliability' and t0 is None:
            raise ValueError("t0 must be specified for conditional_reliability")

        # avoid mutable default arguments
        figkwds = {} if figkwds is None else figkwds
        pltkwds = {} if pltkwds is None else pltkwds

        # Bound methods are looked up lazily, by name, so asking for one curve
        # does not require every other one to exist. Storing self.pdf directly
        # here would evaluate all five on every call.
        curves = {
            'pdf':         ('pdf',         "f(t)"),
            'cdf':         ('cdf',         "F(t)"),
            'reliability': ('reliability', "R(t)"),
            'hazard':      ('hazard',      r"$\lambda(t)$"),
            'conditional_reliability': (
                'conditional_reliability', f"R(t+{t0}|{t0})",
            ),
        }
        name,ylabel = curves[type]
        method = getattr(self, name)
        fun = ((lambda x: method(x, t0)) if type == 'conditional_reliability'
               else method)

        t = np.linspace(self.ppf(0.001),self.ppf(0.999),1000)
        if ax is None:
            _,ax = plt.subplots(**figkwds)

        ax.plot(t,fun(t),**pltkwds)
        ax.set_xlabel("t")
        ax.set_ylabel(ylabel)

        return ax

class reliability_from_hazard(reliability_distribution):
    def __init__(self,h,*args,**kwargs):
        self.hazard = h
        self.cumulative_hazard = None
        self = reliability_distribution.__init__(self,*args,**kwargs)
    
    def integrate_hazard(self,t,verb=False):
        if verb:
            print("Integrating hazard ... ")

        cdf_single = lambda t1,t2: quad(self.hazard,t1,t2)[0]
        if len(t)>1:
            L = np.vectorize(cdf_single)
            ints = np.array(L(t[0:-1],t[1::]))
            c = np.cumsum(np.append([0],ints))
        else:
            c = [cdf_single(0,t)]

        self.cumulative_hazard = np.stack((np.array(t),np.array(c)),axis=1)

        if verb:
            print("Done!")
        
    def _cdf(self,times):
        if len(times)<2 or np.all(self.cumulative_hazard == None) or (not np.all(times==self.cumulative_hazard[1::,0])): # not sure why, but times doesn't contain zero. There must be somthing about the cdf call that causes this.
            self.integrate_hazard(times)
        else:
            print("Cumulative hazard already computed")

        return 1-np.exp(-self.cumulative_hazard[:,1])

class expdist(reliability_distribution):
    r"""Exponential distribution, parameterised by the mean.

    The single parameter is the **mean** :math:`\theta = \mathbb{E}[T]`, which
    is also the reciprocal of the failure rate, :math:`\theta = 1/\lambda`.

    This is deliberately the same quantity that ``scipy.stats`` calls
    ``scale``, so there is no conversion anywhere in the class: ``fit``,
    ``nnlf``, ``nnlf_gradient`` and the inherited distribution methods all
    speak the same parameter. A fitted value drops straight into a frozen
    distribution::

        res  = expdist().fit(ti, observed=observed)
        dist = expdist()(scale=res.params[0])
        dist.reliability(t)

    It is also consistent with :class:`weibull`, whose ``eta`` is likewise a
    scale.
    """

    parameter_transform = Log()
    parameter_names = ("mean",)

    def _pdf(self,t):
        return np.exp(-t)

    def _cdf(self,t):
        return 1-np.exp(-t)

    @staticmethod
    def _scalar(p):
        """Accept a scalar or a length-1 array, as fit_mle passes an array."""
        return float(np.atleast_1d(np.asarray(p, dtype=float))[0])

    @staticmethod
    def _failure_count(ti, observed):
        if isinstance(observed,str) and observed == "all":
            return float(len(ti))
        return float(np.sum(observed))

    def nnlf(self,p,ti,observed="all"):
        r"""Negative log-likelihood as a function of the mean.

        .. math::
            \ell(\theta) = -r\log\theta - \frac{1}{\theta}\sum_i t_i

        The sum runs over all observations, censored included, since
        :math:`\log f` and :math:`\log S` share the :math:`-t/\theta` term.
        """
        mean = self._scalar(p)
        ti = np.asarray(ti, dtype=float)

        if isinstance(observed,str) and observed == "all":
            observed = np.ones(ti.shape)
        observed = np.asarray(observed, dtype=float)

        loglike = sum(self.logpdf(ti[observed==1],0,mean)) + \
            sum(self.logsf(ti[observed==0],0,mean)) # deals with right censoring

        return -loglike

    def nnlf_gradient(self,p,ti,observed="all"):
        r"""Analytic score, with respect to the mean.

        .. math::
            \partial\ell/\partial\theta
              = \frac{1}{\theta^{2}}\left(\sum_i t_i - r\theta\right)

        Returned negated, to match :meth:`nnlf`.
        """
        mean = self._scalar(p)
        ti = np.asarray(ti, dtype=float)
        r = self._failure_count(ti, observed)
        return -np.array([(np.sum(ti) - r*mean) / mean**2])

    def fit(self,ti,observed="all",*,alpha=0.05,ci_method="transformed"):
        r"""Closed-form maximum-likelihood fit of the mean.

        Both the estimate and its information are analytic,

        .. math::
            \hat\theta = \frac{\sum_i t_i}{r},
            \qquad
            I(\hat\theta) = \frac{r}{\hat\theta^{2}},
            \qquad
            \operatorname{se}(\hat\theta) = \frac{\hat\theta}{\sqrt r},

        so no optimiser runs and no Hessian is differenced.

        Returns
        -------
        FitResult
            ``params[0]`` is the mean, usable directly as ``scale=``.
        """
        ti = np.asarray(ti, dtype=float)
        r = self._failure_count(ti, observed)
        if r <= 0:
            raise ValueError("cannot fit: no observed failures")

        mean_hat = np.sum(ti)/r
        cov = np.array([[mean_hat**2 / r]])

        return result_from_covariance(
            [mean_hat], cov,
            transform=self.parameter_transform,
            alpha=alpha,
            names=self.parameter_names,
            nnlf_value=self.nnlf(mean_hat, ti, observed),
            ci_method=ci_method,
        )

class weibull(reliability_distribution):

    # eta (scale) and beta (shape) are both strictly positive, so fit on the
    # log scale. Matches the legacy transform_scale override below.
    parameter_transform = Log()
    parameter_names = ("eta", "beta")

    def _pdf(self,t,beta):
        return stats.distributions.weibull_min.pdf(t,beta)
   
    def _cdf(self,t,beta):
        return stats.distributions.weibull_min.cdf(t,beta)
    
    def _sf(self,t,beta):
        return stats.distributions.weibull_min.sf(t,beta)
   
    def _logsf(self,t,beta):
        return stats.distributions.weibull_min.logsf(t,beta)
   
    def _logpdf(self,t,beta):
        return stats.distributions.weibull_min.logpdf(t,beta)
   
    def nnlf(self,p,ti,observed="all"):
        loc = 0
        eta = p[0]
        beta = p[1]

        # handle case where all are observed
        if isinstance(observed,str) and observed == "all":
            observed = np.ones(ti.shape)
        
        loglike = sum(self.logpdf(ti[observed==1],beta,loc,eta)) + \
            sum(self.log_reliability(ti[observed==0],beta,loc,eta)) # deals with right censoring
        
        return -loglike
    
    def nnlf_interval(self,p,ti,ins,observed="all"):
        loc = 0
        eta = p[0]
        beta = p[1]

        # handle case where all are observed
        if isinstance(observed,str) and observed == "all":
            observed = np.ones(ti.shape)
        
        idxo, = np.where(observed==1)
        loglike = sum( np.log(self.cdf(ti[idxo],beta,loc,eta)-self.cdf(ins[idxo],beta,loc,eta)) ) +\
            sum(self.log_reliability(ti[observed==0],beta,loc,eta)) # deals with right censoring
        
        return -loglike
    
    def nnlf_gradient(self,p,ti,observed="all"):
        r"""Analytic score of the right-censored Weibull negative log-likelihood.

        With :math:`u_i = \log(t_i/\eta)`, :math:`z_i = (t_i/\eta)^\beta`,
        :math:`\delta_i` the failure indicator and :math:`r = \sum \delta_i`,
        the log-likelihood is

        .. math::
            \ell = r\log\beta - r\log\eta
                   + (\beta-1)\sum \delta_i u_i - \sum z_i

        (the :math:`-\sum z_i` runs over all observations, since
        :math:`\log f` contains :math:`-z` and :math:`\log S = -z`), giving

        .. math::
            \partial\ell/\partial\eta  &= (\beta/\eta)\left(\sum z_i - r\right) \\
            \partial\ell/\partial\beta &= r/\beta + \sum \delta_i u_i
                                          - \sum z_i u_i

        Setting these to zero recovers the textbook censored-Weibull equations
        :math:`\eta^\beta = \sum t_i^\beta / r` and
        :math:`1/\beta = \sum t_i^\beta \log t_i / \sum t_i^\beta
        - r^{-1}\sum \delta_i \log t_i`.

        Returns the gradient of the *negative* log-likelihood, ordered
        ``(eta, beta)`` to match :meth:`nnlf`.
        """
        eta, beta = p[0], p[1]
        ti = np.asarray(ti, dtype=float)

        if isinstance(observed,str) and observed == "all":
            observed = np.ones(ti.shape)
        observed = np.asarray(observed, dtype=float)

        u = np.log(ti) - np.log(eta)
        z = np.exp(beta*u)
        r = observed.sum()

        dl_deta  = (beta/eta)*(z.sum() - r)
        dl_dbeta = r/beta + np.sum(observed*u) - np.sum(z*u)

        return -np.array([dl_deta, dl_dbeta])   # nnlf = -loglikelihood

    def transform_scale(self,x,likelihood_hessian=None,direction="inverse"):
        return _parameter_transform_log(x,likelihood_hessian=likelihood_hessian,\
            direction=direction)

    def anderson_darling_test(self,ti,observed="all"):
        """
            Uses transformation noted here: https://en.wikipedia.org/wiki/Anderson%E2%80%93Darling_test (see Tests for other distributions). 
            This currently only works for samples without censored data.
        """
        if observed != "all":
            ValueError("This function does not yet work for censored samples, so observed must be ""all"" ")
        else:
            x = np.log(1.0/ti)
            return stats.anderson(x,dist='gumbel_r')

