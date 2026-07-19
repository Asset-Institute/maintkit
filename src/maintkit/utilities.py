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

def _parameter_transform_log(x,likelihood_hessian=None,direction="inverse"):
        # direction is either "forward" (to log-scaled space) or "inverse" (back to original scale)
        x = np.array(x)
        if direction == "inverse":
            z = np.exp(x)
        elif direction == "forward":
            z = np.log(x)
        else:
            raise ValueError("Transformation direction not recognized.")

        if not isinstance(likelihood_hessian,np.ndarray): # can't use likelihood_hessian == None because it is an array if supplied
            # print("No valid Hessian supplied. Returning only parameter estimates")
            return z
        else:
            #Jacobian for transformation. See Reparameterization at https://en.wikipedia.org/wiki/Fisher_information 
            J = np.diag(np.exp(x))

            if direction == "inverse":
                Ji = np.linalg.inv(J)                            
                H = Ji.transpose() @ likelihood_hessian @ Ji 
            elif direction == "forward":
                H = J.transpose() @ likelihood_hessian @ J

            return z,H  

def _parameter_transform_identity(x,likelihood_hessian=None,direction="inverse"):
        z = np.array(x)

        if not isinstance(likelihood_hessian,np.ndarray): # can't use likelihood_hessian == None because it is an array if supplied
            # print("No valid Hessian supplied. Returning only parameter estimates")
            return z
        else:
            H = likelihood_hessian
            return z,H 