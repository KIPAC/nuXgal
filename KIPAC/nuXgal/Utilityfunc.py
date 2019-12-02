"""Utility function for analysis"""

import os
import numpy as np
import healpy as hp

from scipy.stats import norm, distributions

def makedir_safe(filepath):
    """Make a directory for a file and catch exceptions if it already exists"""
    try:
        os.makedirs(os.path.dirname(filepath))
    except OSError:
        pass


def density_cl(cl, nside, randomSeed):
    """Generate a density map using cl

    Parameters
    ----------
    cl : `np.ndarray`
        Array of cl
    nside : `int`
        `healpy` nside parameter
    randomSeed : `int`
        Random number generator seed

    Returns
    -------
    density : `np.ndarray`
        The synthetic map
    """
    np.random.seed(randomSeed)
    alm = hp.sphtfunc.synalm(cl, lmax=3*nside-1)
    density = hp.sphtfunc.alm2map(alm, nside, verbose=False)
    return density


def poisson_sampling(hpmap, n_sample):
    """Generate a synthetic map with a given number of expected events

    Parameters
    ----------
    hpmap : `np.ndarray`
        Input map
    n_sample : `int`
        Desired number of events

    Returns
    -------
    count_map : `np.ndarray`
        The synthetic map
    """

    norm_map = np.exp(hpmap)
    norm_map *= (n_sample/norm_map.sum())
    count_map = np.random.poisson(norm_map.clip(0, np.inf))
    return count_map


def overdensityMap(countsMap):
    """Construct an overdensity map from event map"""
    mean = np.mean(countsMap)
    return (countsMap-mean)/mean



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
