# note that all packages must have licenses that permit commercial use!
from scipy import stats as stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numdifftools as ndt
from ReliabilityAnalysis.distributions import weibull,reliability_distribution_frozen

def ecdf(ti,observed,pos="midpoint",plot=True):
    ti = np.array(ti)
    observed = np.array(observed)
    idx = np.argsort(ti)
    ti = ti[idx]
    observed = observed[idx]
    i = np.arange(0,observed.sum())
    N = len(ti)
    if pos == "midpoint":
        x = ti[np.where(observed==1)]
        Fhat = (i+1-0.5)/N
    elif pos == "mean":
        x = ti[np.where(observed==1)]
        Fhat = (i+1)/(N+1)
    elif pos == "median":
        x = ti[np.where(observed==1)]
        Fhat = (i+1-0.3)/(N+0.4)

    x = np.insert(x,0,0)
    Fhat = np.insert(Fhat,0,0)

    if plot:
        fig, ax = plt.subplots()
        ax.step(x,Fhat,where="post")
        ax.set_xlabel("Time")
        ax.set_ylabel("$\hat{F}$")

    return x,Fhat

def kaplan_meier(ti,observed,plot=True,confidence_interval="greenwood"):
    
    # ensure that inputs are numpy arrays
    ti = np.array(ti)
    observed = np.array(observed)
    
    # sort times in ascending order
    idx = np.argsort(ti)
    ti = ti[idx]
    observed = observed[idx]
    
    # obtain unique time points
    uti,idxu = np.unique(ti,return_index=True)
    obsu = observed[idxu] 
    uti = uti[obsu==1] # remove censored samples from the unique times

    N = len(ti)
    Nu = len(uti) 
    ni = N #assets at risk
    Rhat = np.ones(Nu+1)
    S = np.zeros(Nu+1)
    S[0] = 0
    for i in range(0,Nu):
        droppedOut = (ti==uti[i]) # number dropped out at time uti 
        failed = np.logical_and(droppedOut,observed)  
        di = np.sum(failed) # number of failures at time uti

        Rhat[i+1] = Rhat[i]*(ni-di)/ni # Product limit estimation
        if Rhat[i+1]!=0:
            S[i+1] = S[i]+di/(ni*(ni-di)) # Greenwood formula
        else:
            print("Warning: Reliability is zero so no variance can be calculated.")

        ni -= np.sum(droppedOut)
    
    uti = np.insert(uti,0,0)
    Fhat = 1-Rhat   

    if confidence_interval.lower() == "greenwood":
        v = (Rhat**2)*S
        UB = np.clip(Fhat+1.96*np.sqrt(v),None,1)
        LB = np.clip(Fhat-1.96*np.sqrt(v),0,None)
    elif confidence_interval.lower() == "exponential":
        v = 1/np.log(Rhat)**2 * S
        cp = np.log(-np.log(Rhat))+1.96*np.sqrt(v)
        cm = np.log(-np.log(Rhat))-1.96*np.sqrt(v)
        LB = 1-np.exp(-np.exp(cm))
        UB = 1-np.exp(-np.exp(cp))
    else:
        raise ValueError("confidence_interval not recognized. Use ""greenwood"" or ""exponential"" """)

    if plot:
        fig, ax = plt.subplots()
        ax.step(uti,Fhat,where="post",label=r"$\hat{F}(t)$",color="blue")
        ax.fill_between(uti,LB,y2=UB,linestyle='--',color="blue",step="post",label="95% CI",alpha=0.1)
        ax.set_xlabel("Time")
        ax.set_ylabel("$\hat{F}(t)$")
        ax.set_ylim((0,ax.get_ylim()[1]))
        plt.legend()
        return uti,Fhat,LB,UB,fig,ax
    else:
        return uti,Fhat,LB,UB

def empirical_mean_cumulative_function(event_times,suspension_times,plot=True,confidence_interval="normal"):
    
    # [1] Chapter 12.1A of Tobias, P.A., Trindade, D., 2011. Applied Reliability, Third. ed. CRC Press LLC, London, United Kingdom.

    # ensure that we have a list of lists
    if (not isinstance(event_times,list)) or (not any(isinstance(el, list) for el in event_times)):
        raise ValueError("event_times must be a list of lists")
        
        
    n_systems = len(event_times)
    tau = suspension_times
    
    # create a single time grid from the flattened event times
    t = np.array([item for sublist in event_times for item in sublist])
    t.sort()
    t = np.unique(t)
    t = t.astype(np.float64)
    
    n = np.zeros((n_systems,len(t)))
    d = np.zeros((n_systems,len(t)))
    for ii in range(n_systems):
        d[ii,t<=tau[ii]] = 1
        for tij in event_times[ii]:
            n[ii,tij==t] = 1
    
    m_hat = n.sum(axis=0)/d.sum(axis=0)
    M_hat = m_hat.cumsum()
    
    if n_systems == 1:
        print('Warning: Confidence intervals cannot be estimated for a single asset')
        M_UCL = np.nan*np.ones(M_hat.shape)
        M_LCL = M_UCL
    elif confidence_interval is None:
        print('Skipping confidence interval ... ')
    else:
        V_hat = np.sum( np.cumsum( d/d.sum(axis=0)*(n-m_hat),axis=1)**2, axis=0)
        se_hat = np.sqrt(V_hat)
        if confidence_interval.lower() == "normal":
            M_UCL = M_hat + 1.96*se_hat
            M_LCL = M_hat - 1.96*se_hat
        elif confidence_interval == "logit":
            w = np.exp(1.96*se_hat/M_hat)
            M_UCL = w*M_hat
            M_LCL = M_hat/w
        else:
            raise ValueError("confidence_interval not recognized")

    # append zero
    t = np.insert(t,0,1e-8)
    M_hat = np.insert(M_hat,0,0)

    if plot:
        fig, ax = plt.subplots()
        ax.step(t,M_hat,where="post",label=r"$\hat{M}(t)$",linewidth=2,color="blue")
        if confidence_interval is not None:
            ax.fill_between(t[1::],M_LCL,y2=M_UCL,linestyle='--',linewidth=2,color="blue",step="post",label="95% CI",alpha=0.1)
        ax.set_xlabel("Time")
        ax.set_ylabel(r"$\hat{M}(t)$") 
        ax.set_ylim((0,ax.get_ylim()[1]))
        plt.legend()  
        return t,M_hat, M_LCL, M_UCL, fig, ax
    else:
        return t,M_hat, M_LCL, M_UCL

def weibull_probability_plot(dist,data=None,ax=None,confidence_bounds=None,parameter_covariance=None,figsize=(7,7)):  

    assert isinstance(dist,reliability_distribution_frozen),"Distribution must be frozen before using this function."
    assert type(dist.dist) in [weibull], "Distribution not supported. Must be Weibull for now."
        
    t = np.linspace( dist.ppf(1e-3),dist.ppf(1-1e-3),100 )
    Y = np.log10(-np.log(dist.reliability(t)))

    ############################ Nominal plot ##################################
    if ax == None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.semilogx(t,Y,linestyle="--",color="red",label="Distribution")
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"$F(t)$")

    if (data is not None):
        assert isinstance(data,dict), "Data needs to be a dict with keys [""times"",""ecdf""]."
        k = list(data.keys())
        assert k[0] in ['times','ecdf'], "Data must be a dict with keys [""times"",""ecdf""]."
        assert k[1] in ['times','ecdf'], "Data must be a dict with keys [""times"",""ecdf""]."
        assert len(data['times'])==len(data['ecdf']), "time and Fhat lists must be the same length"

        Yd = np.log10(-np.log(1-data['ecdf']))
        ax.semilogx(data['times'],Yd,'.',color="blue",label="Data")
        plt.legend()
    
    ######################### confidence bounds ##########################
    if confidence_bounds!=None:
        assert confidence_bounds.lower() in ["time","reliability"], "confidence_bounds must be either ""Time"" or ""Reliability""."
        assert isinstance(parameter_covariance,np.ndarray), "You must supply parameter_covariance to get confidence bounds"
        assert parameter_covariance.shape[0]==2 and parameter_covariance.shape[1] == 2,"Parameter covariance must be 2-by-2"

        RL,RU = weibull_reliability_confidence_interval(dist,t,parameter_covariance,kind=confidence_bounds,c=1.96) 
        FL,FU = np.log10(-np.log(RL)),np.log10(-np.log(RU))       
        ax.fill_between(t,FL,FU,label=f"CI ({confidence_bounds})",color='red',alpha=0.1)

    ######################### format plot ################################
    ytc = np.log10(-np.log([0.995, 0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,0.01, 0.001, 0.00001]))
    ax.set_ylim((ytc.min(),ytc.max()))
    ax.set_yticks(ytc)
    ax.yaxis.set_major_locator(mticker.FixedLocator(ytc))
    ax.set_yticklabels([f"{(1-np.exp(-10**x))*100:.1f}%" for x in ytc])  
    ax.grid(visible=True,which="major")
    ax.legend(loc='upper left')

    return ax

def weibull_reliability_confidence_interval(dist,t,p_cov,kind="Reliability",c=1.96):
        
        assert type(dist.dist) is weibull and isinstance(dist,reliability_distribution_frozen),\
            "The distribution must be a frozen Weibull distribution."
        assert kind.lower() in ["time",'reliability'],"kind must be ""time"" or ""reliability""."
        assert all([(t[ii+1]-t[ii])>=0 for ii in range(len(t)-1)]), "time vector must be sorted"
        assert t[0]>=0, "Negative time doesn't make sense!"
        if t[0] == 0:
            print('Warning: inserting nan for t==0 since logM(t) is undefined.')
            prependNaN = True
            t = t[1::]
        else:
            prependNaN = False

        # The below is a bit lazy and uses numerical gradients. Might use analytical gradients later.
        a,b = dist.kwds['scale'],dist.args[0]
        p = np.array([a,b])
        R = dist.reliability(t)
        if kind.lower() == "time":
            u = np.log(t)
            fun = lambda x: 1/x[1] * np.log(-np.log(R))+np.log(x[0])
            g = ndt.Gradient(fun)(p)
            w = c*np.sqrt( np.sum(g@p_cov*g,axis=1) )
            RL,RU = dist.reliability(np.exp(u+w)),dist.reliability(np.exp(u-w))

        elif kind.lower() == "reliability":
            u = np.log(-np.log(R))
            fun = lambda x: x[1]*(np.log(t)-np.log(x[0]))
            g = ndt.Gradient(fun)(p)
            w = c*np.sqrt( np.sum(g@p_cov*g,axis=1) )
            RL,RU = np.exp(-np.exp(u-w)),np.exp(-np.exp(u+w))
        
        if prependNaN:
            RL = np.insert(RL,0,np.nan)
            RU = np.insert(RU,0,np.nan)

        return RL,RU
