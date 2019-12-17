"""Contains GalaxySample class to organize galaxy samples for cross correlation"""


import os

import numpy as np

import healpy as hp

import matplotlib.pyplot as plt

from astropy import units as u
from astropy.coordinates import SkyCoord

from . import Defaults

from . import Utilityfunc


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
        self.initiateGalaxySample()
        self.overdensity = Utilityfunc.overdensityMap_mask(self.galaxymap, self.idx_galaxymask)
        self.density = self.galaxymap / np.sum(self.galaxymap)


    def initiateGalaxySample(self):
        """Internal method to initialize a particular sample"""
        if self.galaxyName == 'WISE':
            #WISE-2MASS galaxy sample map based on ~5M galaixes
            galaxymap_path = os.path.join(Defaults.NUXGAL_ANCIL_DIR, 'WISE_galaxymap.fits')
            self.galaxymap = hp.fitsfunc.read_map(galaxymap_path, verbose=False)
            c_icrs = SkyCoord(ra=(2 * np.pi - Defaults.exposuremap_phi) * u.radian, dec=(np.pi/2 - Defaults.exposuremap_theta)*u.radian, frame='icrs')
            self.idx_galaxymask = np.where(np.abs(c_icrs.galactic.b.degree) < 10)


        if self.galaxyName == 'analy':
            #simulated galaxy sample based on analytical power spectrum
            analyCLpath = os.path.join(Defaults.NUXGAL_ANCIL_DIR, 'Cl_ggRM.dat')
            self.analyCL = np.loadtxt(analyCLpath)
            analy_galaxymap_path = os.path.join(Defaults.NUXGAL_ANCIL_DIR, 'analy_galaxymap.fits')
            self.galaxymap = hp.fitsfunc.read_map(analy_galaxymap_path, verbose=False)
            self.idx_galaxymask = np.where(False)

        if self.galaxyName == 'nonGal':
            density_nonGal = hp.sphtfunc.synfast(self.analyCL * 0.6, Defaults.NSIDE, verbose=False)
            density_nonGal = np.exp(density_nonGal)
            #self.density = density_nonGal / density_nonGal.sum()

        if self.galaxyName == 'flat':
            self.galaxymap = np.random.poisson(10., size=Defaults.NPIXEL)



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
        density_g = hp.sphtfunc.synfast(analyCL, Defaults.NSIDE)
        density_g = np.exp(density_g)
        density_g /= density_g.sum()
        expected_counts_map = density_g * N_g

        self.analy_galaxymap = np.random.poisson(expected_counts_map)
        if write_map:
            analy_galaxymap_path = os.path.join(Defaults.NUXGAL_ANCIL_DIR, 'analy_galaxymap.fits')
            hp.fitsfunc.write_map(analy_galaxymap_path, self.analy_galaxymap, overwrite=True)
            #analy_density_path = os.path.join(Defaults.NUXGAL_ANCIL_DIR, 'analy_density.fits')
            #hp.fitsfunc.write_map(analy_density_path, self.analy_density, overwrite=True)

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
