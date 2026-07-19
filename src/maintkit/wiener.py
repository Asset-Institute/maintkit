import numpy as np
import scipy.stats as sps
from ReliabilityAnalysis.utilities import _ensure_list
from scipy.optimize import minimize, fsolve

class weiner:
    def __init__(self,mu=None,sigma=None):
        self.mu = mu
        self.sigma = sigma
        self.parameter_source = "specified"
        self.parameter_covariance = None
    
    def transition_distribution(self,x,t,x0,type="cdf"):
        mu = self.mu*t + x0
        s = self.sigma*np.sqrt(t)
        assert type.lower() in ["logpdf","pdf","cdf"], "need to specify type as cdf, pdf, or logpdf"
        if type.lower() == "pdf":
            return sps.norm.pdf(x,loc=mu,scale=s)
        elif type.lower() == "logpdf":
            return sps.norm.logpdf(x,loc=mu,scale=s)
        else:
            return sps.norm.cdf(x,loc=mu,scale=s)        
            
    
    def simulate(self,times,initial=0.0,num_samples=1):
        T = len(times)
        x = np.zeros((num_samples,T))
        x[:,0] = initial
        for ii in range(1,len(times)):
            dt = times[ii] - times[ii-1]
            dxs = sps.norm(loc=self.mu*dt,scale=self.sigma*np.sqrt(dt)).rvs(num_samples)
            x[:,ii] = dxs
        
        return np.cumsum(x,axis=1)

    
    def nnlf(self,t,x,m,log_s):
        old_mu = self.mu
        old_sigma = self.sigma

        self.mu = m
        self.sigma = np.exp(log_s)
        nloglike = 0
        for mm in range(len(x)):
            for ii in range(1,len(x[mm])):
                dt = t[mm][ii] - t[mm][ii-1]
                nloglike += self.transition_distribution(x[mm][ii],dt,x[mm][ii-1],type='logpdf')


        self.mu = old_mu
        self.sigma = old_sigma
            
        return -nloglike

    def estimate_parameters(self,t,x,x0=None,inplace=False,use_numdifftools=False,ndt_kwds={}):
        
        # error checking
        t = _ensure_list(t)
        x = _ensure_list(x)        
        assert len(t)==len(x), "t and x must have same number of runs"
        assert all([len(t[mm])==len(x[mm]) for mm,_ in enumerate(t)]), "All runs in t and x must have same number of elements"
        
        M = len(t)
        N = [len(t[mm]) for mm in range(M)]

        if x0 is None: # obtain initial guesses for parameters
            flat_deltas = lambda x: [x[mm][ii]-x[mm][ii-1] for mm in range(M) for ii in range(1,N[mm])]
            dt = np.array(flat_deltas(t))
            dx = np.array(flat_deltas(x))
            mu_guess = np.mean(dx/dt)
            sig_guess = np.mean( (dx-mu_guess)**2/dt )
            y0 = [np.mean(dx/dt),np.log(sig_guess)]
        else:
            y0 = [x0[0],np.log(x0[1])]

        # MLE
        obj = lambda z: self.nnlf(t,x,z[0],z[1])
        res = minimize(obj,y0,method='BFGS')
        mu = res.x[0]
        sigma = np.exp(res.x[1])

        if use_numdifftools:
            print("Using numdifftools to approximate hessian")
            import numdifftools as ndt
            H = ndt.Hessian(obj,**ndt_kwds)(res.x)
            Hi = np.linalg.inv(H)
        else:
            print("Using hessian inverse approximation from scipy.minimize")
            Hi = res.hess_inv

        # estimate parameter covariance. See Reparameterization at https://en.wikipedia.org/wiki/Fisher_information 
        J = np.array([     [1,0], 
                            [0,np.exp(res.x[1])] 
                        ])
        p_cov = J.T @ Hi @ J.T
    
        if inplace is True:
            self.mu = mu 
            self.sigma = sigma
            self.parameter_source = "estimated"
            self.parameter_covariance = p_cov

            p_sigma = np.sqrt(np.diag(p_cov))
            muL = mu - 1.96*p_sigma[0]
            muU = mu + 1.96*p_sigma[0]
            sL = sigma - 1.96*p_sigma[1]
            sU = sigma + 1.96*p_sigma[1]

            print("Parameters updated: ")
            print(f"\t mu={self.mu:.2e} [{muL:.2e},{muU:.2e}]")
            print(f"\t sigma={self.sigma:.2e} [{sL:.2e},{sU:.2e}]")
        else:
            return mu,sigma,p_cov          
        

class RBM(weiner):
    def __init__(self,mu,sigma):
        super().__init__(mu,sigma)
    
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

    def transition_distribution(self,x,t,x0,type="cdf"):
        # Distributions from [1]
        #
        # References:
        #   [1] J. Abate and W. Whitt, “Transient Behavior of Regulated Brownian Motion, I: Starting at the Origin,” Advances in Applied Probability, vol. 19, no. 3, pp. 560–598, 1987, doi: 10.2307/1427408.

        mu = self.mu
        sigma = self.sigma
        m = mu*t + x0
        s = sigma*np.sqrt(t)
        assert type.lower() in ["logpdf","pdf","cdf"], "need to specify type as cdf, pdf, or logpdf"
        if type.lower() in ["pdf","logpdf"]:
            f = 1/s * sps.norm.pdf((-x+m)/s)
            f += np.exp( -np.log(s) +2*mu*x/sigma**2 + sps.norm.logpdf( (-x-m)/s ) )
            f -= (2*mu/sigma**2)*np.exp(2*mu*x/sigma**2 + sps.norm.logcdf((-x-m)/s) )
            if type.lower() == "logpdf":
                return np.log(f+1e-10)
            else:
                return f
        else: # type.lower() == "cdf"
            F = 1 - sps.norm.cdf((-x+m)/s) 
            F -= np.exp(2*mu*x/sigma**2)*sps.norm.cdf( (-x-m)/s )
            return F

    def estimate_parameters(self,t,x,x0=None,inplace=False,use_numdifftools=False):
        mu,sigma,p_cov = super().estimate_parameters(t,x,x0=x0,inplace=False,use_numdifftools=use_numdifftools)

        p_sigma = np.sqrt(np.diag(p_cov))
        muL = mu - 1.96*p_sigma[0]
        muU = mu + 1.96*p_sigma[0]
        sL = sigma - 1.96*p_sigma[1]
        sU = sigma + 1.96*p_sigma[1]

        if inplace is True:
            self.mu = mu 
            self.sigma = sigma
            self.parameter_source = "estimated"
            self.parameter_covariance = p_cov

            p_sigma = np.sqrt(np.diag(p_cov))
            muL = mu - 1.96*p_sigma[0]
            muU = mu + 1.96*p_sigma[0]
            sL = sigma - 1.96*p_sigma[1]
            sU = sigma + 1.96*p_sigma[1]

            print("Parameters updated: ")
            print(f"\t mu={self.mu:.2e} [{muL:.2e},{muU:.2e}]")
            print(f"\t sigma={self.sigma:.2e} [{sL:.2e},{sU:.2e}]")
        else:
            return mu,sigma,p_cov 

    def get_upper_lower(self,t,x0,alpha=0.025):

        guessesU = self.mu*t + x0 + 1.96*self.sigma*np.sqrt(t)
        U = np.zeros(len(t))
        L = np.zeros(len(t))
        guessesL = np.zeros(len(t))
        for i,ti in enumerate(t):
            U[i] = fsolve( lambda x: (1-alpha)-self.transition_distribution(x,ti,x0,type="cdf"),guessesU[i])
            L[i] = fsolve( lambda x: alpha-self.transition_distribution(x,ti,x0,type="cdf"),guessesL[i])
        
        return L,U

