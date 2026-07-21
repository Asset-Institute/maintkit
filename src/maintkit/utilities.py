from scipy import stats as stats
import numpy as np

def _ensure_list(x):
    
    # ensure that t and x are list of lists
    assert isinstance(x,list), "values must be a list or a list of lists"
    x_valid = [isinstance(x[mm],(list,int,float)) for mm in range(len(x))]
    assert all(x_valid), "values must be a list of (int,float) or a list of lists"
    if isinstance(x[0],(float,int)):
        x = [x]
    
    return x
