"""Analysis class"""

import os
import numpy as np

import healpy as hp

from .EventGenerator import EventGenerator

from . import Defaults

from . import file_utils

from . import hp_utils


aeffpath = os.path.join(Defaults.NUXGAL_IRF_DIR, 'Aeff{i}.fits')
cl_galaxy_path = os.path.join(Defaults.NUXGAL_ANCIL_DIR, 'Cl_ggRM.dat')
sample_galaxy_path = os.path.join(Defaults.NUXGAL_ANCIL_DIR, 'galaxySampleOverdensity.fits')

class Analyze():
    """Analysis class"""

    aeff_factor = Defaults.DT_SECONDS / (4 * np.pi)  / Defaults.M2_TO_CM2

    def __init__(self):
        """C'tor"""
        # exposure map
        self.exposuremap = file_utils.read_maps_from_fits(aeffpath, Defaults.NEbin)

        # generate galaxy samples
        self.l_cl = np.arange(1, 3 * Defaults.NSIDE + 1)
        self.l = np.linspace(1, 500, 500)
        self.cl_galaxy = file_utils.read_cls_from_txt(cl_galaxy_path)[0]
        self.overdensityMap_g = file_utils.read_maps_from_fits(sample_galaxy_path, 1)[0]
        self.eg = None


    def getIntensity(self, countsmap, dt_days=Defaults.DT_DAYS):
        intensitymap = np.divide(countsmap, self.exposuremap,
                                 out=np.zeros_like(countsmap), where=self.exposuremap != 0)
        intensity = np.zeros(Defaults.NEbin)
        for i in np.arange(Defaults.NEbin):
            intensity[i] = np.sum(intensitymap[i]) / (10.**Defaults.map_logE_center[i] * np.log(10.) * Defaults.dlogE) / (dt_days * 24 * 3600) / (4 * np.pi)  / 1e4 ## exposure map in m^2
        return intensity


    def getIntensityMap(self, countsmap, mask):
        intensitymap = np.divide(countsmap, self.exposuremap,
                                 out=np.zeros_like(countsmap), where=self.exposuremap != 0)

        maskmap = hp.pixelfunc.ma(mask)
        intensitymap = intensitymap + maskmap
        return intensitymap


    def powerSpectrum(self, intensitymap):
        overdensitymap = hp_utils.vector_overdensity_from_intensity(intensitymap)
        return hp_utils.vector_cl_from_overdensity(overdensitymap, Defaults.NCL)

    def powerSpectrumFromCountsmap(self, countsmap):
        intensitymap = hp_utils.vector_intensity_from_counts_and_exposure(countsmap, self.exposuremap)
        return self.powerSpectrum(intensitymap)

    def crossCorrelationFromCountsmap(self, countsmap):
        intensitymap = hp_utils.vector_intensity_from_counts_and_exposure(countsmap, self.exposuremap)
        odmap_2d = hp_utils.reshape_array_to_2d(self.overdensityMap_g)
        return hp_utils.vector_cross_correlate_maps(intensitymap, odmap_2d, Defaults.NCL)

    def crossCorrelation_atm_std(self, N_re=100):

        if self.eg is None:
            self.eg = EventGenerator()
        w_cross = np.zeros((N_re, Defaults.NEbin, 3 * Defaults.NSIDE))

        for iteration in np.arange(N_re):
            print("iter ", iteration)
            eventmap_atm = self.eg.atmEvent(1.)
            w_cross[iteration] = self.crossCorrelationFromCountsmap(eventmap_atm)

        return np.mean(w_cross, axis=0), np.std(w_cross, axis=0)
