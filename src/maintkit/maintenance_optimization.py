import numpy as np
import ReliabilityAnalysis.distributions as radist


class interval_replacement:
    def __init__(   self,
                    dist,
                    cost_of_failure=None,
                    cost_of_pm=None,
                    failure_repair_time=None,
                    pm_repair_time=None):
        
        self.failure_time_distribution = dist
        self.cpm = cost_of_pm
        self.cf = cost_of_failure
    
    def expected_number_of_failures(self,
                                    T=None,
                                    tol=1e-10,
                                    dt=None):
        dist = self.failure_time_distribution

        if T is None: # simulate for 100*MTTF
            T = 10*dist.ppf(1-tol) # likely to see at least 10 failures

        if dt is None:
            dt = dist.ppf(1-tol)/1000 

        time_grid = np.arange(0,T,dt)
        
        DF = np.diff(dist.cdf(time_grid))
        H = np.zeros(len(time_grid))
        for kk,_ in enumerate(time_grid):
            H[kk] = np.sum( (1+H[0:kk])*np.flip(DF[0:kk]) )

        return time_grid,H
    
    def optimal_timing(self,T=None,tol=1e-10,dt=None):
        assert self.cpm is not None, "Define the cost of pm (cpm) before using this function."
        assert self.cpm is not None, "Define the cost of failure (cf) before using this function."

        t,H = self.expected_number_of_failures(T=T,tol=tol,dt=dt)
        
        H = H[t>0]
        t = t[t>0]
        CR = (self.cpm + self.cf*H)/t
        idx_min = np.argmin(CR)

        return t,CR,idx_min




        

