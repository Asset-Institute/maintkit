from scipy import stats as stats
from scipy import optimize as opt
from scipy.integrate import quad
import numpy as np
import numdifftools as ndt
from maintkit.utilities import _parameter_transform_log

class poisson_process:
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

    def fit(self,event_times,p0,truncation_times=None,ndt_kwds={}):

        msg = "event_times must be a list of lists or a 2D numpy arrays"
        if isinstance(event_times,list):
            assert all([isinstance(event_times[m],list) for m in range(len(event_times))]),msg
        elif isinstance(event_times,np.ndarray):
            event_times = [event_times[m,:] for m in range(event_times.shape[0])]

        y0 = self.transform_scale(p0,direction="forward")
        obj = lambda x: self.nnlf(self.transform_scale(x),event_times,truncation_times=truncation_times)
        result = opt.minimize(obj,y0)
        y_hat = result.x
        H = ndt.Hessian(obj,**ndt_kwds)(y_hat)
        p_hat,Ht = self.transform_scale(y_hat,likelihood_hessian=H,direction="inverse")
        log_p_cov = np.linalg.inv(H)
        s = np.sqrt(np.diag(log_p_cov))
        p_ci =  np.exp(y_hat +1.96*np.array([-s,s]))

        p_cov = np.linalg.inv(Ht)
        return p_hat, p_ci.transpose(),p_cov

    def transform_scale(self,x,likelihood_hessian=None,direction="inverse"):
         return _parameter_transform_log(x,likelihood_hessian=likelihood_hessian,\
            direction=direction)

class power_law_nhpp(poisson_process):
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
    
    def fit(self,event_times,truncation_times=None,ndt_kwds={}):

        msg = "event_times must be a list of lists or a 2D numpy array"
        if isinstance(event_times,list):
            assert all([isinstance(event_times[m],list) for m in range(len(event_times))]),msg
        elif isinstance(event_times,np.ndarray):
            event_times = [event_times[m,:] for m in range(event_times.shape[0])]


        # check for valid truncation time
        tau = []
        if truncation_times != None:
            for m,_ in enumerate(event_times):
                if len(event_times[m]) > 0:
                    assert truncation_times[m] > max(event_times[m]), "Invalid truncation time for asset "+str(m)
                tau.append(truncation_times[m])
        else:
            for m,_ in enumerate(event_times):
                tau.append(max(event_times[m]))
        
        # analytical computation of MLE for observed failure times
        num_failures = 0
        den_beta_hat = 0
        for n,_ in enumerate(event_times):
            failures = event_times[n]
            num_failures += len(failures)
            for k,_ in enumerate(failures):
                den_beta_hat += (np.log(tau[n])-np.log(failures[k]))
        beta_hat = num_failures/den_beta_hat

        sum_truncation_times = 0
        for n,_ in enumerate(event_times):
            sum_truncation_times += tau[n]**beta_hat
        alpha_hat = num_failures/sum_truncation_times
        p_hat = [alpha_hat,beta_hat]

        # lazy numerical computation of the Hessian and parameter CIs
        y_hat = self.transform_scale(p_hat,direction='forward')
        obj = lambda x: self.nnlf(self.transform_scale(x),event_times,truncation_times=truncation_times)
        H_log = ndt.Hessian(obj,**ndt_kwds)(y_hat)
        log_p_cov = np.linalg.inv(H_log)
        s = np.sqrt(np.diag(log_p_cov))
        p_ci = np.exp(y_hat + 1.96*np.array([-s,s]))
        _,H = self.transform_scale(y_hat,likelihood_hessian=H_log)

        return p_hat, p_ci.transpose(),np.linalg.inv(H)
    
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
    
    def fit_interval(self,ni,ins,p0,cumulative=False,estimate_ci=False,ndt_kwds={}):
        y0 = self.transform_scale(p0,direction="forward")
        obj = lambda x: self.nnlf_interval(self.transform_scale(x),ni,ins,cumulative=cumulative)
        result = opt.minimize(obj,y0)
        y_hat = result.x
        
        if estimate_ci:
            H = ndt.Hessian(obj,**ndt_kwds)(y_hat)
            p_hat,Ht = self.transform_scale(y_hat,likelihood_hessian=H,direction="inverse")
            log_p_cov = np.linalg.inv(H)
            s = np.sqrt(np.diag(log_p_cov))
            p_ci = np.exp(p_hat + 1.96*np.array([-s,s]))
            p_cov = np.linalg.inv(Ht)
        else:
            p_hat = self.transform_scale(y_hat,likelihood_hessian=None,direction="inverse")
            n = p_hat.shape[0]
            p_ci = np.nan*np.ones((n,2))
            p_cov = np.nan*np.ones((2,2))

        return p_hat, p_ci.transpose(),p_cov

    def transform_scale(self,x,likelihood_hessian=None,direction="inverse"):
        return _parameter_transform_log(x,likelihood_hessian=likelihood_hessian,\
            direction=direction)

    def mcf_confidence_interval(self,t,p_cov,kind="mcf",c=1.96,ndt_kwds={}):
    
        assert kind.lower() in ["time",'mcf'],"kind must be ""time"" or ""mcf""."
        assert all([(t[ii+1]-t[ii])>=0 for ii in range(len(t)-1)]), "time vector must be sorted"
        assert t[0]>=0, "Negative time doesn't make sense!"
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
