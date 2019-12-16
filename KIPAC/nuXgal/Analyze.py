"""Analysis class"""

import os
import numpy as np

import healpy as hp

from . import Defaults

from . import file_utils

from . import hp_utils

from . import Utilityfunc

aeffpath = os.path.join(Defaults.NUXGAL_IRF_DIR, 'Aeff{i}.fits')

class Analyze():
    """Analysis class"""


    def __init__(self):
        """C'tor"""
        self.exposuremap = file_utils.read_maps_from_fits(aeffpath, Defaults.NEbin)


    def getIntensity(self, countsmap, dt_days=Defaults.DT_DAYS):
        """Convert a countsmap to total intensity using the exposure map"""
        intensitymap = np.divide(countsmap, self.exposuremap,
                                 out=np.zeros_like(countsmap), where=self.exposuremap != 0)
        intensity = np.zeros(Defaults.NEbin)
        for i in np.arange(Defaults.NEbin):
            intensity[i] = np.sum(intensitymap[i]) / (10.**Defaults.map_logE_center[i] * np.log(10.) * Defaults.dlogE) / (dt_days * 24 * 3600) / (4 * np.pi)  / 1e4 ## exposure map in m^2
        return intensity



    def powerSpectrum(self, intensitymap):
        """Build a power spectrum from an intensity map"""
        overdensitymap = hp_utils.vector_overdensity_from_intensity(intensitymap)
        return hp_utils.vector_cl_from_overdensity(overdensitymap, Defaults.NCL)

    def powerSpectrumFromCountsmap(self, countsmap):
        """Build a power spectrum from a counts map"""
        intensitymap = hp_utils.vector_intensity_from_counts_and_exposure(countsmap, self.exposuremap)
        return self.powerSpectrum(intensitymap)

    def crossCorrelationFromCountsmap(self, countsmap, overdensityMap_g):
        """Comput the cross correlation between the overdensity map and a counts map"""
        intensitymap = hp_utils.vector_intensity_from_counts_and_exposure(countsmap, self.exposuremap)
        overdensitymap = hp_utils.vector_overdensity_from_intensity(intensitymap)
        odmap_2d = hp_utils.reshape_array_to_2d(overdensityMap_g)
        return hp_utils.vector_cross_correlate_maps(overdensitymap, odmap_2d, Defaults.NCL)


    def crossCorrelationFromCountsmap_mask(self, countsmap, overdensityMap_g, idx_mask):
        intensitymap = hp_utils.vector_intensity_from_counts_and_exposure(countsmap, self.exposuremap)
        w_cross = np.zeros((Defaults.NEbin, Defaults.NCL))
        for i in range(Defaults.NEbin):
            overdensitymap_nu = Utilityfunc.overdensityMap_mask (countsmap[i], idx_mask)
            overdensitymap_nu[idx_mask] = hp.UNSEEN
            w_cross[i] = hp.sphtfunc.anafast(overdensitymap_nu, overdensityMap_g)
        return w_cross
