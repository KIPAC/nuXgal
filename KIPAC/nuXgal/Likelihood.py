"""Defintion of Likelihood function"""

import numpy as np



def log_prior(f):
    """Logrithm of a flat prior, defined from 0 to 2"""
    f_diff = f
    if 0. < f_diff < 2.:
        return 0.
    return -np.inf
