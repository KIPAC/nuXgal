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

    def __init__(self, year='IC86-2012'):
        """C'tor"""
        self.wAeff = WeightedAeff(year)
        self.countsmap = None
        self.fluxmap = None
        self.idx_mask = None
        self.f_sky = 1.


    def inputFluxmap(self, fluxmap):
        self.fluxmap = fluxmap
        self.fluxmap_fullsky = fluxmap

    def inputCountsmap(self, countsmap, spectralIndex=None):
        self.countsmap = countsmap
        self.countsmap_fullsky = countsmap

        if spectralIndex is not None:
            if spectralIndex == 3.7:
                self.fluxmap = hp_utils.vector_intensity_from_counts_and_exposure(countsmap, self.wAeff.exposuremap_atm)
            elif spectralIndex == 2.28:
                self.fluxmap = hp_utils.vector_intensity_from_counts_and_exposure(countsmap, self.wAeff.exposuremap_astro)
            else:
                exposuremap = self.wAeff.computeWeightedAeff(spectralIndex)
                self.fluxmap = hp_utils.vector_intensity_from_counts_and_exposure(countsmap, exposuremap)
            self.fluxmap_fullsky = self.fluxmap


    def inputCountsmapMix(self, countsmap_atm, countsmap_astro):
        fluxmap_atm = hp_utils.vector_intensity_from_counts_and_exposure(countsmap_atm, self.wAeff.exposuremap_atm)
        fluxmap_astro = hp_utils.vector_intensity_from_counts_and_exposure(countsmap_astro, self.wAeff.exposuremap_astro)
        self.fluxmap = fluxmap_atm + fluxmap_astro
        self.countsmap = countsmap_atm + countsmap_astro
        self.fluxmap_fullsky = self.fluxmap
        self.countsmap_fullsky = self.countsmap

    def inputData(self, countsmappath, fluxmappath):
        self.fluxmap = file_utils.read_maps_from_fits(fluxmappath, Defaults.NEbin)
        self.countsmap = file_utils.read_maps_from_fits(countsmappath, Defaults.NEbin)
        self.fluxmap_fullsky = self.fluxmap
        self.countsmap_fullsky = self.countsmap

    def updateMask(self, idx_mask):
        self.idx_mask = idx_mask
        self.f_sky = 1. - len( idx_mask[0] ) / float(Defaults.NPIXEL)
        if self.countsmap is not None:
            countsmap = self.countsmap_fullsky.copy() + 0. # +0. to convert to float array
            for i in range(Defaults.NEbin):
                countsmap[i][idx_mask] = hp.UNSEEN
            self.countsmap = hp.ma(countsmap)

        fluxmap = self.fluxmap_fullsky.copy()
        self.fluxmap = hp.ma(fluxmap)
        for i in range(Defaults.NEbin):
            fluxmap[i][idx_mask] = hp.UNSEEN
        self.fluxmap = hp.ma(fluxmap)

    def getEventCounts(self):
        return self.countsmap.sum(axis=1)


    def getIntensity(self, dt_years):
        """Compute the intensity / energy flux of the neutirno sample"""

        assert self.fluxmap is not None, 'NeutrinoSample: fluxmap uninitialized'
        intensity = self.fluxmap.sum(axis=1) / (10.**Defaults.map_logE_center * np.log(10.) * Defaults.map_dlogE) / (dt_years * Defaults.DT_SECONDS) / (4 * np.pi * f_sky)  / 1e4 ## exposure map in m^2
        return intensity

    def getOverdensity(self):
        overdensity = np.zeros((Defaults.NEbin, Defaults.NPIXEL))
        for i in range(Defaults.NEbin):
            overdensity[i] = self.fluxmap[i] / self.fluxmap[i].mean() - 1.
        return overdensity

    def getPowerSpectrum(self):
        """Compute the power spectrum of the neutirno sample"""

        assert self.fluxmap is not None, 'NeutrinoSample: fluxmap uninitialized'
        w_auto = np.zeros((Defaults.NEbin, Defaults.NCL))
        for i in range(Defaults.NEbin):
            overdensity = self.fluxmap[i] / self.fluxmap[i].mean() - 1.
            w_auto[i] = hp.sphtfunc.anafast(overdensity)
        return w_auto


    def getCrossCorrelation(self, overdensityMap_g):
        """Compute the cross correlation between the overdensity map and a counts map"""

        assert self.fluxmap is not None, 'NeutrinoSample: fluxmap uninitialized'
        w_cross = np.zeros((Defaults.NEbin, Defaults.NCL))
        for i in range(Defaults.NEbin):
            overdensity = self.fluxmap[i] / self.fluxmap[i].mean() - 1.
            w_cross[i] = hp.sphtfunc.anafast(overdensity, overdensityMap_g)
        return w_cross



    def plotFluxmap(self, testfigpath):
        figs = FigureDict()
        figs.mollview_maps('fluxmap', self.fluxmap)
        figs.save_all(testfigpath, 'pdf')

    def plotCountsmap(self, testfigpath):
        figs = FigureDict()
        figs.mollview_maps('countsmap', self.countsmap)
        figs.save_all(testfigpath, 'pdf')




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
