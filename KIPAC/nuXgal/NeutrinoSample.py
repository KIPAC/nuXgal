"""Analysis class"""

import numpy as np

import healpy as hp

from . import Defaults

from . import file_utils

from .Exposure import ICECUBE_EXPOSURE_LIBRARY

from .plot_utils import FigureDict


class NeutrinoSample():
    """Neutrino class"""

    def __init__(self):
        """C'tor"""
        self.countsmap = None
        self.idx_mask = None
        self.f_sky = 1.
        self.countsmap_fullsky = None


    def inputCountsmap(self, countsmap):
        """Set the counts map

        Parameters
        ----------
        countsmap : `np.ndarray`
            The input map
        """
        self.countsmap = countsmap
        self.countsmap_fullsky = countsmap


    def inputData(self, countsmappath):
        """Set the counts map for a filename

        Parameters
        ----------
        countsmappath : `str`
            Path to the counts map
        """
        self.countsmap = file_utils.read_maps_from_fits(countsmappath, Defaults.NEbin)
        self.countsmap_fullsky = self.countsmap

    def updateMask(self, idx_mask):
        """Set the mask used to analyze the data

        Parameters
        ----------
        idx_mask : `np.ndarray`
            Masks
        """
        self.idx_mask = idx_mask
        self.f_sky = 1. - len(idx_mask[0]) / float(Defaults.NPIXEL)
        countsmap = self.countsmap_fullsky.copy() + 0. # +0. to convert to float array
        for i in range(Defaults.NEbin):
            countsmap[i][idx_mask] = hp.UNSEEN
        self.countsmap = hp.ma(countsmap)

    def getEventCounts(self):
        """Return the number of counts in each energy bin"""
        return self.countsmap.sum(axis=1)


    def getIntensity(self, dt_years, spectralIndex=3.7, year='IC86-2012'):
        """Compute the intensity / energy flux of the neutirno sample"""
        exposuremap = ICECUBE_EXPOSURE_LIBRARY.get_exposure(year, spectralIndex)
        fluxmap = np.divide(self.countsmap, exposuremap,
                            out=np.zeros_like(self.countsmap),
                            where=exposuremap != 0)
        intensity = fluxmap.sum(axis=1) / (10.**Defaults.map_logE_center * np.log(10.) *\
                                               Defaults.map_dlogE) / (dt_years * Defaults.DT_SECONDS) /\
                                               (4 * np.pi * self.f_sky)  / 1e4 ## exposure map in m^2
        return intensity

    def getOverdensity(self):
        """Compute and return the overdensity maps"""
        overdensity = [self.countsmap[i] / self.countsmap[i].mean() - 1. for i in range(Defaults.NEbin)]
        return overdensity

    def getAlm(self):
        """Compute and return the alms"""
        overdensity = self.getOverdensity()
        alm = [hp.sphtfunc.map2alm(overdensity[i]) for i in range(Defaults.NEbin)]
        return alm

    def getPowerSpectrum(self):
        """Compute and return the power spectrum of the neutirno sample"""
        overdensity = self.getOverdensity()
        w_auto = [hp.sphtfunc.anafast(overdensity[i]) / self.f_sky for i in range(Defaults.NEbin)]
        return w_auto


    #def getCrossCorrelation(self, overdensityMap_g):
    #    """Compute the cross correlation between the overdensity map and a counts map"""
    #    overdensity = self.getOverdensity()
    #    w_cross = [hp.sphtfunc.anafast(overdensity[i], overdensityMap_g) / self.f_sky for i in range(Defaults.NEbin)]
    #    return w_cross

    def getCrossCorrelation(self, alm_g):
        """Compute and return cross correlation between the overdensity map and a counts map

        Parameters
        ----------
        alm_g : `np.ndarray`
            The alm for the sample are correlating against

        Returns
        -------
        w_cross : `np.ndarray`
            The cross correlation
        """
        overdensity = self.getOverdensity()
        alm_nu = [hp.sphtfunc.map2alm(overdensity[i]) for i in range(Defaults.NEbin)]
        w_cross = [hp.sphtfunc.alm2cl(alm_nu[i], alm_g) / self.f_sky for i in range(Defaults.NEbin)]
        return w_cross


    def plotCountsmap(self, testfigpath):
        """Plot and save the maps"""
        figs = FigureDict()
        figs.mollview_maps('countsmap', self.countsmap)
        figs.save_all(testfigpath, 'pdf')



    #def getCrossCorrelation_countsmap(self, countsmap, overdensityMap_g, idx_mask):
    #    """Compute the cross correlation between the overdensity map and an atm counts map"""
    #    w_cross = np.zeros((Defaults.NEbin, Defaults.NCL))
    #    for i in range(Defaults.NEbin):
    #        overdensitymap_nu = Utilityfunc.overdensityMap_mask(countsmap[i], idx_mask)
    #        overdensitymap_nu[idx_mask] = hp.UNSEEN
    #        w_cross[i] = hp.sphtfunc.anafast(overdensitymap_nu, overdensityMap_g)
    #    return w_cross
