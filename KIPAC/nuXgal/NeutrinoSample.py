"""Analysis class"""

import os

import numpy as np

import healpy as hp

from . import Defaults

from . import file_utils

from . import hp_utils

from . import Utilityfunc

from .WeightedAeff import WeightedAeff

from KIPAC.nuXgal.plot_utils import FigureDict


class NeutrinoSample():
    """Neutrino class"""

    def __init__(self):
        """C'tor"""
        self.wAeff = WeightedAeff()
        self.countsmap = None
        self.fluxmap = None


    def inputFluxmap(self, fluxmap):
        self.fluxmap = fluxmap

    def inputCountsmap(self, countsmap, spectralIndex):
        self.countsmap = countsmap
        if spectralIndex == 3.7:
            self.fluxmap = hp_utils.vector_intensity_from_counts_and_exposure(countsmap, self.wAeff.exposuremap_atm)
        elif spectralIndex == 2.28:
            self.fluxmap = hp_utils.vector_intensity_from_counts_and_exposure(countsmap, self.wAeff.exposuremap_astro)
        else:
            exposuremap = self.wAeff.computeWeightedAeff(spectralIndex)
            self.fluxmap = hp_utils.vector_intensity_from_counts_and_exposure(countsmap, exposuremap)

    def inputCountsmapMix(self, countsmap_atm, countsmap_astro):
        fluxmap_atm = hp_utils.vector_intensity_from_counts_and_exposure(countsmap_atm, self.wAeff.exposuremap_atm)
        fluxmap_astro = hp_utils.vector_intensity_from_counts_and_exposure(countsmap_astro, self.wAeff.exposuremap_astro)
        self.fluxmap = fluxmap_atm + fluxmap_astro



    def getIntensity(self, dt_years, idx_mask=None):
        """Compute the intensity / energy flux of the neutirno sample"""
        assert self.fluxmap is not None, 'NeutrinoSample: fluxmap uninitialized'
        if idx_mask is None:
            fluxmap_unmasked = self.fluxmap
            f_sky = 1.
        else:
            fluxmap_unmasked = hp_utils.vector_apply_mask(self.fluxmap, idx_mask, copy=True)
            f_sky = 1. - len( idx_mask[0] ) / float(Defaults.NPIXEL)
            print ('f_sky =', f_sky)

        intensity = np.zeros(Defaults.NEbin)
        for i in np.arange(Defaults.NEbin):
            intensity[i] = np.sum(fluxmap_unmasked[i]) / (10.**Defaults.map_logE_center[i] * np.log(10.) * Defaults.map_dlogE) / (dt_years * Defaults.DT_SECONDS) / (4 * np.pi * f_sky)  / 1e4 ## exposure map in m^2
        return intensity


    def getPowerSpectrum(self, idx_mask=None):
        """Compute the power spectrum of the neutirno sample"""

        assert self.fluxmap is not None, 'NeutrinoSample: fluxmap uninitialized'
        w_auto = np.zeros((Defaults.NEbin, Defaults.NCL))

        if idx_mask is not None:
            for i in range(Defaults.NEbin):
                overdensitymap_nu = Utilityfunc.overdensityMap_mask(self.fluxmap[i], idx_mask)
                overdensitymap_nu[idx_mask] = hp.UNSEEN
                w_auto[i] = hp.sphtfunc.anafast(overdensitymap_nu)
            return w_auto

        else:
            overdensitymap = hp_utils.vector_overdensity_from_intensity(self.fluxmap)
            return hp_utils.vector_cl_from_overdensity(overdensitymap, Defaults.NCL)



    def getCrossCorrelation(self, overdensityMap_g, idx_mask=None):
        """Compute the cross correlation between the overdensity map and a counts map"""

        assert self.fluxmap is not None, 'NeutrinoSample: fluxmap uninitialized'
        w_cross = np.zeros((Defaults.NEbin, Defaults.NCL))

        if idx_mask is not None:
            for i in range(Defaults.NEbin):
                overdensitymap_nu = Utilityfunc.overdensityMap_mask(self.fluxmap[i], idx_mask)
                overdensitymap_nu[idx_mask] = hp.UNSEEN
                w_cross[i] = hp.sphtfunc.anafast(overdensitymap_nu, overdensityMap_g)
            return w_cross
        else:
            overdensitymap = hp_utils.vector_overdensity_from_intensity(self.fluxmap)
            odmap_2d = hp_utils.reshape_array_to_2d(overdensityMap_g)
            return hp_utils.vector_cross_correlate_maps(overdensitymap, odmap_2d, Defaults.NCL)


    def mollview_maps_mask(self, map, testfigpath, idx_mask=None):
        if idx_mask is not None:
            for i in range(Defaults.NEbin):
                map[i][idx_mask] = hp.UNSEEN

        figs = FigureDict()
        figs.mollview_maps('fluxmap', map)
        figs.save_all(testfigpath, 'pdf')


    def plotFluxmap(self, testfigpath, idx_mask=None):
        assert self.fluxmap is not None, 'NeutrinoSample: fluxmap uninitialized'
        self.mollview_maps_mask(self.fluxmap, testfigpath, idx_mask)


    def plotCountsmap(self, testfigpath, idx_mask=None):
        assert self.countsmap is not None, 'NeutrinoSample: countsmap uninitialized'
        self.mollview_maps_mask(self.countsmap, testfigpath, idx_mask)




    def getCrossCorrelation_countsmap_atm(self, countsmap_atm, overdensityMap_g, idx_mask):
        """Compute the cross correlation between the overdensity map and an atm counts map"""
        fluxmap = hp_utils.vector_intensity_from_counts_and_exposure(countsmap_atm, self.wAeff.exposuremap_atm)
        w_cross = np.zeros((Defaults.NEbin, Defaults.NCL))
        for i in range(Defaults.NEbin):
            overdensitymap_nu = Utilityfunc.overdensityMap_mask(fluxmap[i], idx_mask)
            overdensitymap_nu[idx_mask] = hp.UNSEEN
            w_cross[i] = hp.sphtfunc.anafast(overdensitymap_nu, overdensityMap_g)
        return w_cross

    def getCrossCorrelation_countsmap_mix(self, countsmap_atm, countsmap_astro, overdensityMap_g, idx_mask):
        fluxmap_atm = hp_utils.vector_intensity_from_counts_and_exposure(countsmap_atm, self.wAeff.exposuremap_atm)
        fluxmap_astro = hp_utils.vector_intensity_from_counts_and_exposure(countsmap_astro, self.wAeff.exposuremap_astro)
        fluxmap = fluxmap_atm + fluxmap_astro
        w_cross = np.zeros((Defaults.NEbin, Defaults.NCL))
        for i in range(Defaults.NEbin):
            overdensitymap_nu = Utilityfunc.overdensityMap_mask(fluxmap[i], idx_mask)
            overdensitymap_nu[idx_mask] = hp.UNSEEN
            w_cross[i] = hp.sphtfunc.anafast(overdensitymap_nu, overdensityMap_g)
        return w_cross
