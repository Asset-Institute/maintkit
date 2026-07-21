import numpy as np


class IntervalReplacement:
    def __init__(   self,
                    dist,
                    cost_of_failure=None,
                    cost_of_pm=None,
                    failure_repair_time=None,
                    pm_repair_time=None):
        
        self.failure_time_distribution = dist
        self.cpm = cost_of_pm
        self.cf = cost_of_failure
        # Accepted and stored but not used by the cost rate yet, which counts
        # only the direct costs. They were previously accepted and silently
        # discarded, so a caller supplying them got no error and no effect.
        self.failure_repair_time = failure_repair_time
        self.pm_repair_time = pm_repair_time
    
    def expected_number_of_failures(self,
                                    T=None,
                                    tol=1e-10,
                                    dt=None):
        r"""Expected cumulative failures on a grid, by the renewal equation.

        Solves :math:`H(t) = F(t) + \int_0^t H(t-u)\,dF(u)` on a uniform grid
        by the usual discrete recursion. First-order accurate in ``dt``: for an
        exponential, where :math:`H(t) = \lambda t` exactly, halving ``dt``
        halves the error.

        Returns ``(time_grid, H)``.
        """
        dist = self.failure_time_distribution

        if T is None:
            # Far enough out to see many failures. Not a multiple of the MTTF:
            # ppf(1-tol) is an extreme quantile of the time to ONE failure.
            T = 10*dist.ppf(1-tol)

        if dt is None:
            dt = dist.ppf(1-tol)/1000 

        time_grid = np.arange(0,T,dt)
        
        DF = np.diff(dist.cdf(time_grid))
        H = np.zeros(len(time_grid))
        for kk,_ in enumerate(time_grid):
            H[kk] = np.sum( (1+H[0:kk])*np.flip(DF[0:kk]) )

        return time_grid,H
    
    def optimal_timing(self,T=None,tol=1e-10,dt=None):
        # The second check tested cpm twice, so a missing cf was never caught
        # here and surfaced later inside the cost arithmetic instead.
        if self.cpm is None:
            raise ValueError("set the cost of preventive maintenance (cpm) before optimising")
        if self.cf is None:
            raise ValueError("set the cost of failure (cf) before optimising")

        t,H = self.expected_number_of_failures(T=T,tol=tol,dt=dt)
        
        H = H[t>0]
        t = t[t>0]
        CR = (self.cpm + self.cf*H)/t
        idx_min = np.argmin(CR)

        return t,CR,idx_min




        

