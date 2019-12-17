"""Utility function for analysis"""

import os
import numpy as np

from scipy.stats import norm, distributions

def makedir_safe(filepath):
    """Make a directory for a file and catch exceptions if it already exists"""
    try:
        os.makedirs(os.path.dirname(filepath))
    except OSError:
        pass



def overdensityMap(countsMap):
    """Construct an overdensity map from event map"""
    mean = np.mean(countsMap)
    return (countsMap-mean)/mean


def overdensityMap_mask(countsMap, idx):
    """Construct an overdensity map from event map. Pixels under mask are not taken into account. Masked region has value = 1"""
    localmap = np.ma.array(countsMap, mask=False)
    localmap.mask[idx] = True
    mean = localmap.mean()
    return countsMap / mean - 1.



def significance(chi_square, dof):
    """Construct an significance for a chi**2 distribution

    Parameters
    ----------
    chi_square : `float`
    dof : `int`

    Returns
    -------
    significance : `float`
    """
    p_value = distributions.chi2.sf(chi_square, dof-1)
    significance_twoTailNorm = norm.isf(p_value/2)
    return significance_twoTailNorm


def significance_from_chi(chi):
    """Construct an significance set of chi values

    Parameters
    ----------
    chi : `array`
    dof : `int`

    Returns
    -------
    significance : `float`
    """
    chi2 = chi*chi
    dof = len(chi2)
    return significance(np.sum(chi2), dof)
