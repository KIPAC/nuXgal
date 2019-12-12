import os

import numpy as np

import healpy as hp

from . import Defaults

from . import Utilityfunc

from . import FigureDict

import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord

class GalaxySample():
    """Class to organize galaxy samples for cross correlation
    """
    def __init__(self, computeGalaxy = False):


        # WISE-2MASS galaxy sample map based on ~5M galaixes
        galaxymap_path = os.path.join(Defaults.NUXGAL_ANCIL_DIR,'WISE_galaxymap.fits')
        self.WISE_galaxymap = hp.fitsfunc.read_map(galaxymap_path, verbose=False)
        self.WISE_galaxymap_overdensity = Utilityfunc.overdensityMap(self.WISE_galaxymap)
        self.WISE_density = self.WISE_galaxymap / np.sum(self.WISE_galaxymap)

        c_icrs = SkyCoord(ra=(2 * np.pi - Defaults.exposuremap_phi) * u.radian, dec=(np.pi/2 - Defaults.exposuremap_theta)*u.radian, frame='icrs')
        idx = np.where(np.abs(c_icrs.galactic.b.degree) < 10)
        self.WISE_mask = np.zeros(Defaults.NPIXEL, dtype=np.bool)
        self.WISE_mask[idx] = 1.

        # simulated galaxy sample based on analytical power spectrum
        analyCLpath = os.path.join(Defaults.NUXGAL_ANCIL_DIR, 'Cl_ggRM.dat')
        self.analyCL = np.loadtxt(analyCLpath)
        if computeGalaxy:
            self.generateGalaxy()
        else:
            analy_galaxymap_path = os.path.join(Defaults.NUXGAL_ANCIL_DIR, 'analy_galaxymap.fits')
            self.analy_galaxymap = hp.fitsfunc.read_map(analy_galaxymap_path, verbose=False)


        self.analy_galaxymap_overdensity = Utilityfunc.overdensityMap(self.analy_galaxymap)
        self.analy_density = self.analy_galaxymap / np.sum(self.analy_galaxymap)
        #analy_density_path = os.path.join(Defaults.NUXGAL_ANCIL_DIR, 'analy_density.fits')
        #self.analy_density = hp.fitsfunc.read_map(analy_density_path, verbose=False)
        self.analy_mask = None




        # simulated neutrino source density that do NOT share random seed with galaxies
        #density_nonGal = hp.sphtfunc.synfast(self.analyCL * 0.6, Defaults.NSIDE, verbose=False)
        #density_nonGal = np.exp(density_nonGal)
        #self.nonGal_density = density_nonGal / density_nonGal.sum()

        # galaxy sample that follows a flat CL, with total count ~ 2M, for testing
        #self.flat_CL_galaxymap = np.random.poisson(10., size=Defaults.NPIXEL)
        #self.flat_CL_galaxymap_overdensity =  Utilityfunc.overdensityMap(self.flat_CL_galaxymap)
        #self.flat_CL_galaxymap_overdensity_CL = hp.anafast(self.flat_CL_galaxymap_overdensity)
        #self.flat_CL_galaxymap_density = self.flat_CL_galaxymap / np.sum(self.flat_CL_galaxymap)


    def generateGalaxy(self, N_g = 2000000, write_map = True):
        np.random.seed(Defaults.randomseed_galaxy)
        density_g = hp.sphtfunc.synfast(self.analyCL, Defaults.NSIDE)
        density_g = np.exp(density_g)
        density_g /= density_g.sum()
        expected_counts_map = density_g * N_g
        self.analy_galaxymap = np.random.poisson(expected_counts_map)
        if write_map:
            analy_galaxymap_path = os.path.join(Defaults.NUXGAL_ANCIL_DIR, 'analy_galaxymap.fits')
            hp.fitsfunc.write_map(analy_galaxymap_path, self.analy_galaxymap, overwrite=True)


        np.random.seed(Defaults.randomseed_galaxy)
        density_g = hp.sphtfunc.synfast(self.analyCL * 0.6, Defaults.NSIDE)
        density_g = np.exp(density_g)
        density_g /= density_g.sum()
        self.analy_density = density_g
        if write_map:
            analy_density_path = os.path.join(Defaults.NUXGAL_ANCIL_DIR, 'analy_density.fits')
            hp.fitsfunc.write_map(analy_density_path, self.analy_density, overwrite=True)

        return self.analy_galaxymap



    def getOverdensity(self, key):

        if key == 'analy':
            return self.analy_galaxymap_overdensity
        if key == 'WISE':
            return self.WISE_galaxymap_overdensity
        #if key == 'flatCL':
        #    return self.flat_CL_galaxymap_overdensity
        print ('please use one of the following keywords: WISE, analy, flatCL')




    def getDensity(self, key):

        if key == 'analy':
            return self.analy_density
        if key == 'WISE':
            return self.WISE_density
        #if key == 'nonGal':
        #    return self.nonGal_density
        #if key == 'flatCL':
        #    return self.flat_CL_galaxymap_density

        print ('please use one of the following keywords: WISE, analy, nonGal, flatCL')

    def getMask(self, key):
        if key == 'analy':
            return self.analy_mask
        if key == 'WISE':
            return self.WISE_mask


    def plotGalaxymap(self, keyword, plotmax=100):
        if keyword == 'WISE':
            galaxymap = self.WISE_galaxymap
        elif keyword == 'analy':
            galaxymap = self.analy_galaxymap_overdensity
        else:
            print ('please use one of the following keywords: WISE, analy')

        figs = FigureDict()
        hp.mollview(galaxymap, title=keyword, max=plotmax)
        testfigpath = os.path.join(Defaults.NUXGAL_PLOT_DIR, 'test_')
        plt.savefig(testfigpath+keyword+'_galaxy.pdf')



    def plotCL(self):
        figs = FigureDict()
        o_dict = figs.setup_figure('galaxyCL', xlabel='$\ell$', ylabel='$C_{\ell}$', figsize=(8, 6))
        fig, axes = o_dict['fig'], o_dict['axes']
        axes.set_xscale('log')
        axes.set_yscale('log')

        axes.plot(Defaults.ell, self.analyCL[0 : Defaults.NCL], label='analytical')
        axes.plot(Defaults.ell, self.WISE_galaxymap_overdensity_cl * 0.3, label='WISE-2MASS galaxies x 0.3')
        fig.legend()
        axes.set_ylim(1e-7, 1e-1)
        testfigpath = os.path.join(Defaults.NUXGAL_PLOT_DIR, 'test')
        figs.save_all(testfigpath, 'pdf')
