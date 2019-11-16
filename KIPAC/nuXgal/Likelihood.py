
import numpy as np



def log_prior(f):
    f_diff = f
    if 0. < f_diff < 2.:
        return 0.
    return -np.inf
    
