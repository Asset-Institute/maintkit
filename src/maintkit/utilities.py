from scipy import stats as stats
import numpy as np

def _ensure_list(x):
    
    # ensure that t and x are list of lists
    if not isinstance(x,list):
        raise TypeError(
            f"values must be a list or a list of lists, got {type(x).__name__}"
        )
    if not all(isinstance(x[mm],(list,int,float)) for mm in range(len(x))):
        raise TypeError(
            "values must be a list of numbers (one run) or a list of lists "
            "(one per run)"
        )
    if isinstance(x[0],(float,int)):
        x = [x]
    
    return x
