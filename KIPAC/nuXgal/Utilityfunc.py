import os
import numpy as np
import healpy as hp

from scipy.stats import norm, distributions

def makedir_safe(filepath):
    try:
        os.makedirs(os.path.dirname(filepath))
    except OSError:
        pass
                   


# generate a density map using cl
def density_cl(cl, nside, randomSeed):
    np.random.seed(randomSeed)
    alm = hp.sphtfunc.synalm(cl, lmax=3*nside-1)
    density = hp.sphtfunc.alm2map(alm, nside, verbose=False)
    return density


def poisson_sampling(hpmap, n_sample):
    norm_map = np.exp(hpmap)
    norm_map *= (n_sample/norm_map.sum())
    count_map = np.random.poisson(norm_map.clip(0, np.inf))
    return count_map




# overdensity map from event map
def overdensityMap(countsMap):
    mean = np.mean(countsMap)
    return (countsMap-mean)/mean





def significance(chi_square, dof):
    p_value = distributions.chi2.sf(chi_square, dof-1)
    significance_twoTailNorm = norm.ppf(1.-p_value/2)
    return significance_twoTailNorm
