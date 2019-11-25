"""Defintion of Likelihood function"""

import os
import numpy as np
import healpy as np
import emcee
import corner
import matplotlib.pyplot as plt

from .Utilityfunc import *
from .EventGenerator import *
from .Analyze import *
from . import Defaults


 
def log_prior(f):
    """Logrithm of a flat prior, defined from 0 to 2"""
    f_diff = f
    if 0. < f_diff < 2.:
        return 0.
    return -np.inf
