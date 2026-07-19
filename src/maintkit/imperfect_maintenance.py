from scipy import stats as stats
from scipy import optimize as opt
from scipy.integrate import quad
import numpy as np
import numdifftools as ndt
from ReliabilityAnalysis.utilities import _parameter_transform_log
import ReliabilityAnalysis.poisson_process as rpp

class imperfect_pm_minimal_cm:
    # Uses a proportinal age reduction modification of a power-law NHPP for now. 
    # Only works for a single asset at the moment. 

    def __init__(self,a,b,rho):
        self.baseline_model = rpp.power_law_nhpp(a,b)
        assert rho<=1.0 and rho>=0, "Repair factor must be between 0 and 1 (inclusive)"
        self.repair_factor = rho
    
    def set_parameters(self,a,b,r):
        self.baseline_model = rpp.power_law_nhpp(a,b)
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

    def intensity(self,t,pm_times):
        last_pm = self._last_pm(t,pm_times)
        return self.baseline_model.intensity(t-self.repair_factor*last_pm)

    def cumulative_intensity(self,t,pm_times):
        last_pm = self._last_pm(pm_times,pm_times)
        M = np.zeros_like(t)

        for ii,tau in enumerate(pm_times):
            idx, = np.where( (t>last_pm[ii]) & (t<tau) )
            t0 = (1-self.repair_factor)*last_pm[ii]
            M[idx] += self.baseline_model.cumulative_intensity(t[idx]-self.repair_factor*last_pm[ii],t0=t0)
            M[idx[-1]+1::] = 1.0*M[idx[-1]]
        return M
    
    def nnlf(self,p,failure_times,pm_times):
        N = len(failure_times)
        a,b,r = p
        last_pm_before_failure = self._last_pm(failure_times,pm_times)
        last_pm_before_pm = self._last_pm(pm_times,pm_times)

        term1 = np.sum( np.log(failure_times-r*last_pm_before_failure) )
        term2 = np.sum( (pm_times-r*last_pm_before_pm)**b - ((1-r)*last_pm_before_pm)**b)
        like = N*np.log(a)+N*np.log(b) + (b-1)*term1 - a*term2
        return -like

    def reduced_nnlf(self,p,failure_times,pm_times):
        
        N = len(failure_times)
        b,r = p
        last_pm_before_failure = self._last_pm(failure_times,pm_times)
        last_pm_before_pm = self._last_pm(pm_times,pm_times)

        term1 = np.sum( (pm_times-r*last_pm_before_pm)**b - ((1-r)*last_pm_before_pm)**b )
        term2 = np.sum( np.log(failure_times-r*last_pm_before_failure) )
        like = N*np.log(b) + N*np.log(N) - N*np.log(term1) + (b-1)*term2 - N
        
        return -like
    
    def fit(self,failure_times,pm_times,ndt_kwds={}):

        transform = lambda u: np.r_[np.log(u[0:-1]), np.log(u[-1]/(1-u[-1]))]
        inverse_transform = lambda u: np.r_[ np.exp(u[0:-1]), 1/(1+np.exp(-u[-1])) ]
        y0 = transform([1,0.5])
        robj = lambda x: self.reduced_nnlf(inverse_transform(x),failure_times,pm_times)
        result = opt.minimize(robj,y0)
        y_hat = result.x
        b_hat,r_hat = inverse_transform(y_hat)

        pm_times = np.array(pm_times)
        a_hat = len(failure_times)/np.sum( (pm_times[1::]-r_hat*pm_times[0:-1])**b_hat 
                                            -((1-r_hat)*pm_times[0:-1])**b_hat )
        
        p_hat = [a_hat,b_hat,r_hat]
        log_p_hat = transform(p_hat)
        full_obj = lambda x: self.nnlf(inverse_transform(x),failure_times,pm_times)
        H = ndt.Hessian(full_obj,**ndt_kwds)(log_p_hat)
        log_p_cov = np.linalg.inv(H)
        s = np.sqrt(np.diag(log_p_cov))
        p_ci =  log_p_hat +1.96*np.array([-s,s]) # log p_ci initially
        p_ci[0,:] = inverse_transform(p_ci[0,:])
        p_ci[1,:] = inverse_transform(p_ci[1,:])

        return p_hat,p_ci


    



