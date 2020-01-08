"""Contains GalaxySample class to organize galaxy samples for cross correlation"""


import os

import numpy as np

import healpy as hp

import matplotlib.pyplot as plt

from astropy import units as u
from astropy.coordinates import SkyCoord

from . import Defaults



class GalaxySample():
    """Class to organize galaxy samples for cross correlation
    """
    def __init__(self, galaxyName, idx_galaxymask, **kwargs):
        """C'tor

        Currently this implements:
        WISE : WISE-2MASS galaxy sample map based on ~5M galaixes
        analy : simulated galaxy sample based on analytical power spectrum
        flat : Poisson generated flat galaxy sample

        Parameters
        ----------
        galaxyName : `str`
            Name for the sample, used to define the sample and specify output file paths
        idx_galaxymask : 
            Used to ma
        
        """
        self.galaxyName = galaxyName
        galaxymap_path = Defaults.GALAXYMAP_FORMAT.format(galaxyName=galaxyName)
        overdensityalm_path = Defaults.GALAXYALM_FORMAT.format(galaxyName=galaxyName)
        self.galaxymap = hp.fitsfunc.read_map(galaxymap_path, verbose=False)
        self.overdensityalm = hp.fitsfunc.read_alm(overdensityalm_path)
        self.density = self.galaxymap / np.sum(self.galaxymap)
        self.idx_galaxymask = idx_galaxymask
        self.f_sky = 1. - len(self.idx_galaxymask[0]) / float(Defaults.NPIXEL)
        self.analyCL = kwargs.get('analyCL', None)

        #_galaxymap = self.galaxymap.copy()
        #_galaxymap[self.idx_galaxymask] = hp.UNSEEN
        #_galaxymap = hp.ma(_galaxymap)
        #self.overdensity = _galaxymap / np.mean(_galaxymap) - 1.



    def plotGalaxymap(self, plotmax=100):
        """Plot galaxy counts map for a particular sample

        Parameters
        ----------
        plotmax : `float`
            Maximum value in the plot
        """
        hp.mollview(self.galaxymap, title=self.galaxyName, max=plotmax)
        testfigpath = Defaults.GALAXYMAP_FIG_FORMAT.format(galaxyName=self.galaxyName)
        plt.savefig(testfigpath)




def GalaxyMask_WISE():
    c_icrs = SkyCoord(ra=Defaults.exposuremap_phi * u.radian,
                      dec=(np.pi/2 - Defaults.exposuremap_theta)*u.radian, frame='icrs')
    return np.where(np.abs(c_icrs.galactic.b.degree) < 10)

def GalaxyMask_AllSky():
    return np.where(False)


GALAXY_MASK_FUNC_DICT = dict(WISE=GalaxyMask_WISE,
                             analy=GalaxyMask_AllSky)

class GalaxySampleLibrary:
    """Library of galaxy samples"""

    def __init__(self, randomseed_galaxy=Defaults.randomseed_galaxy):
        """C'tor"""
        self._gs_dict = {}
        self.randomseed_galaxy = randomseed_galaxy
        
    def keys(self):
        """Return the names of exposure maps"""
        return self._gs_dict.keys()

    def values(self):
        """Returns the exposure maps"""
        return self._gs_dict.values()

    def items(self):
        """Return the name : map pairs"""
        return self._gs_dict.items()

    def __getitem__(self, key):
        """Return a particular exposure map by name"""
        return self._gs_dict[key]
    
    def get_sample(self, sampleName):
        """Get a particular sample by name"""       
        if sampleName in self._gs_dict:
            return self._gs_dict[sampleName]

        mask_func = GALAXY_MASK_FUNC_DICT[sampleName]
        mask = mask_func()
        kw_ctor = {}
        if sampleName == 'analy':
            #simulated galaxy sample based on analytical power spectrum
            kw_ctor['analyCL'] = np.loadtxt(Defaults.ANALYTIC_CL_PATH)
        gs = GalaxySample(sampleName, mask, **kw_ctor)
        self._gs_dict[sampleName] = gs
        return gs

    def generateGalaxy(self, N_g=2000000, write_map=True):
        """Generate a synthetic galaxy sample

        Parameters
        ----------
        N_g : `int`
            Number of galaxies to generate
        write_map: `bool`
            if True write the generate map to the ancilary data area
        """
        analyCL = np.loadtxt(Defaults.ANALYTIC_CL_PATH)
        np.random.seed(self.randomseed_galaxy)
        alm = hp.sphtfunc.synalm(analyCL, lmax=Defaults.MAX_L)
        density_g = hp.sphtfunc.alm2map(alm, Defaults.NSIDE, verbose=False)
        #density_g = hp.sphtfunc.synfast(analyCL, Defaults.NSIDE)
        density_g = np.exp(density_g)
        density_g /= density_g.sum()
        expected_counts_map = density_g * N_g
        np.random.seed(Defaults.randomseed_galaxy)

        analy_galaxymap = np.random.poisson(expected_counts_map)
        if write_map:
            analy_galaxymap_path = Defaults.GALAXYMAP_FORMAT.format(galaxyName='analy')
            hp.fitsfunc.write_map(analy_galaxymap_path, analy_galaxymap, overwrite=True)
            overdensityalm_path = Defaults.GALAXYALM_FORMAT.format(galaxyName='analy')
            hp.fitsfunc.write_alm(overdensityalm_path, alm, overwrite=True)


GALAXY_LIBRARY = GalaxySampleLibrary(Defaults.randomseed_galaxy)

