# note that all packages must have licenses that permit commercial use!
from scipy import stats as stats
from scipy import optimize as opt
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numdifftools as ndt
from ReliabilityAnalysis.utilities import _parameter_transform_log,_parameter_transform_identity

class reliability_distribution(stats.rv_continuous):
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
    
    def fit(self,ti,p0,observed="all",ndt_kwds={}): # Overwriting scipy.stats fitting because it seems that it doesn't handle censoring
        y0 = self.transform_scale(p0,direction="forward")
        obj = lambda x: self.nnlf(self.transform_scale(x),ti,observed)
        result = opt.minimize(obj,y0)
        y_hat = result.x
        H = ndt.Hessian(obj,**ndt_kwds)(y_hat)
        p_hat,Ht = self.transform_scale(y_hat,likelihood_hessian=H,direction="inverse")
        p_cov = np.linalg.inv(Ht)
        s = np.sqrt(np.diag(p_cov))
        p_ci = p_hat + 1.96*np.array([-s,s])
        
        return p_hat, p_ci.transpose(), p_cov
    
    def fit_interval(self,ti,ins,p0,observed="all",bnds=None,ndt_kwds={}): # Overwriting scipy.stats fitting because it seems that it doesn't handle censoring
        y0 = self.transform_scale(p0,direction="forward")
        obj = lambda x: self.nnlf_interval(self.transform_scale(x),ti,ins,observed)
        result = opt.minimize(obj,y0)
        y_hat = result.x
        H = ndt.Hessian(obj,**ndt_kwds)(y_hat)
        p_hat,Ht = self.transform_scale(y_hat,likelihood_hessian=H,direction="inverse")
        p_cov = np.linalg.inv(Ht)
        s = np.sqrt(np.diag(p_cov))
        p_ci = p_hat + 1.96*np.array([-s,s]) 

        return p_hat, p_ci.transpose()    
    
    def freeze(self, *args, **kwds):
        return reliability_distribution_frozen(self, *args, **kwds) # freeze using new reliabilty class, otherwise new functions won't be defined (e.g. reliability)
    
    def transform_scale(self,x,likelihood_hessian=None,direction="inverse"):
        # define as unity transform unless overwritten
         return _parameter_transform_identity(x,likelihood_hessian=likelihood_hessian,\
            direction=direction)

class reliability_distribution_frozen(stats._distn_infrastructure.rv_frozen):
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
    
    def _pdf(self,t):
        return np.exp(-t)
    
    def _cdf(self,t):
        return 1-np.exp(-t)
    
    def nnlf(self,mu0,ti,observed="all"):
        loc = 0
        scale = mu0

        # handle case where all are observed
        if isinstance(observed,str) and observed == "all":
            observed = np.ones(ti.shape)
        
        loglike = sum(self.logpdf(ti[observed==1],loc,scale)) + \
            sum(self.logsf(ti[observed==0],loc,scale)) # deals with right censoring
        
        return -loglike
    
    def fit(self,ti,observed="all",bnds=None):
        if observed == "all":
            r = len(ti)
        else:
            r = np.sum(observed)

        p_hat = r/np.sum(ti)
        s = p_hat/np.sqrt(r)
        p_ci = p_hat + 1.96*np.array([-s,s])
        
        return p_hat, p_ci
        
class weibull(reliability_distribution):
   
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

