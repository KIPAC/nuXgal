"""Contains GalaxySample class to organize galaxy samples for cross correlation"""


import os

import numpy as np

import healpy as hp

import matplotlib.pyplot as plt


from . import Defaults



class GalaxySample():
    """Class to organize galaxy samples for cross correlation
    """
    def __init__(self, galaxyName, computeGalaxy=False):
        """C'tor

        Currently this implements:
        WISE : WISE-2MASS galaxy sample map based on ~5M galaixes
        analy : simulated galaxy sample based on analytical power spectrum
        flat : Poisson generated flat galaxy sample

        Parameters
        ----------
        galaxyName : `str`
            Name for the sample, used to define the sample and specify output file paths
        computeGalaxy: `bool`
            If True, generate synthetic maps from CL
        """
        if computeGalaxy:
            self.generateGalaxy()

        self.galaxyName = galaxyName
        galaxymap_path = os.path.join(Defaults.NUXGAL_ANCIL_DIR, galaxyName + '_galaxymap.fits')
        overdensityalm_path = os.path.join(Defaults.NUXGAL_ANCIL_DIR, galaxyName + '_overdensityalm.fits')
        self.galaxymap = hp.fitsfunc.read_map(galaxymap_path, verbose=False)
        self.overdensityalm = hp.fitsfunc.read_alm(overdensityalm_path)
        self.density = self.galaxymap / np.sum(self.galaxymap)
        self.initiateGalaxySample()
        self.f_sky = 1. - len(self.idx_galaxymask[0]) / float(Defaults.NPIXEL)

        #_galaxymap = self.galaxymap.copy()
        #_galaxymap[self.idx_galaxymask] = hp.UNSEEN
        #_galaxymap = hp.ma(_galaxymap)
        #self.overdensity = _galaxymap / np.mean(_galaxymap) - 1.



    def initiateGalaxySample(self):
        """Internal method to initialize a particular sample"""
        if self.galaxyName == 'WISE':
            from astropy import units as u
            from astropy.coordinates import SkyCoord
            #WISE-2MASS galaxy sample map based on ~5M galaixes
            c_icrs = SkyCoord(ra=Defaults.exposuremap_phi * u.radian, dec=(np.pi/2 - Defaults.exposuremap_theta)*u.radian, frame='icrs')
            self.idx_galaxymask = np.where(np.abs(c_icrs.galactic.b.degree) < 10)

        if self.galaxyName == 'analy':
            #simulated galaxy sample based on analytical power spectrum
            analyCLpath = os.path.join(Defaults.NUXGAL_ANCIL_DIR, 'Cl_ggRM.dat')
            self.analyCL = np.loadtxt(analyCLpath)
            self.idx_galaxymask = np.where(False)



    def generateGalaxy(self, N_g=2000000, write_map=True):
        """Generate a synthetic galaxy sample

        Parameters
        ----------
        N_g : `int`
            Number of galaxies to generate
        write_map: `bool`
            if True write the generate map to the ancilary data area
        """

        analyCLpath = os.path.join(Defaults.NUXGAL_ANCIL_DIR, 'Cl_ggRM.dat')
        analyCL = np.loadtxt(analyCLpath)
        np.random.seed(Defaults.randomseed_galaxy)
        alm = hp.sphtfunc.synalm(analyCL,lmax = Defaults.MAX_L)
        density_g = hp.sphtfunc.alm2map(alm, Defaults.NSIDE,  verbose=False)
        #density_g = hp.sphtfunc.synfast(analyCL, Defaults.NSIDE)
        density_g = np.exp(density_g)
        density_g /= density_g.sum()
        expected_counts_map = density_g * N_g
        np.random.seed(Defaults.randomseed_galaxy)

        analy_galaxymap = np.random.poisson(expected_counts_map)
        if write_map:
            analy_galaxymap_path = os.path.join(Defaults.NUXGAL_ANCIL_DIR, 'analy_galaxymap.fits')
            hp.fitsfunc.write_map(analy_galaxymap_path, analy_galaxymap, overwrite=True)
            overdensityalm_path = os.path.join(Defaults.NUXGAL_ANCIL_DIR, 'analy_overdensityalm.fits')
            hp.fitsfunc.write_alm(overdensityalm_path, alm, overwrite=True)


    def plotGalaxymap(self, plotmax=100):
        """Plot galaxy counts map for a particular sample

        Parameters
        ----------
        plotmax : `float`
            Maximum value in the plot
        """
        hp.mollview(self.galaxymap, title=self.galaxyName, max=plotmax)
        testfigpath = os.path.join(Defaults.NUXGAL_PLOT_DIR, 'test_')
        plt.savefig(testfigpath+self.galaxyName+'_galaxy.pdf')
