"""Analysis class"""

import os
import numpy as np

import healpy as hp

from . import Defaults

from . import file_utils

from . import hp_utils

from . import Utilityfunc

aeffpath = os.path.join(Defaults.NUXGAL_IRF_DIR, 'IC86-2012-TabulatedAeff.txt')

class NeutrinoSample():
    """Neutrino class"""

    def __init__(self):
        """C'tor"""
        Aeff_file = np.loadtxt(aeffpath)
        # effective area, 200 in cos zenith, 70 in E
        self.Aeff_table = Aeff_file[:, 4]
        Emin_eff = np.reshape(Aeff_file[:, 0], (70, 200))[:, 0]
        self.Emin_eff = Emin_eff
        self.Ec_eff_len = len(Emin_eff)
        logE_eff = np.log10(Emin_eff)
        self.dlogE_eff = np.mean(logE_eff[1:] - logE_eff[0:-1])
        self.Ec_eff = Emin_eff * 10. ** (0.5 * self.dlogE_eff)
        self.cosZenith_min = np.reshape(Aeff_file[:, 2], (70, 200))[0]
        exposuremap_costheta = np.cos(np.pi - Defaults.exposuremap_theta) # converting to South pole view
        self.index_coszenith = np.searchsorted(self.cosZenith_min, exposuremap_costheta) - 1
        self.exposuremap_atm = self.weightedAeff(3.7)
        self.exposuremap_astro = self.weightedAeff(2.2)


    def weightedAeff(self, spectralIndex = 3.7):
        weightedAeff = np.zeros((Defaults.NEbin, Defaults.NPIXEL))
        jend = 0
        for i in range(Defaults.NEbin):
            jstart = jend
            jend = np.searchsorted(self.Emin_eff, Defaults.map_E_edge[i+1])
            factor_i = (np.power(Defaults.map_E_edge[i+1], 1 - spectralIndex) - np.power(Defaults.map_E_edge[i], 1 - spectralIndex)) / (1 - spectralIndex) / (np.log(10.) * self.dlogE_eff)
            print (i, jstart, jend, self.Emin_eff[jstart], self.Emin_eff[jend-1], Defaults.map_E_edge[i], Defaults.map_E_edge[i+1], factor_i)
            for j in np.arange(jstart, jend):
                weightedAeff[i] += np.power(self.Ec_eff[j], 1 - spectralIndex) / self.Aeff_table[j * 200 + self.index_coszenith]
            weightedAeff[i] = factor_i / weightedAeff[i]
        return weightedAeff


    def getCrossCorrelation_fluxmap(self, fluxmap, overdensityMap_g, idx_mask):
        w_cross = np.zeros((Defaults.NEbin, Defaults.NCL))
        for i in range(Defaults.NEbin):
            overdensitymap_nu = Utilityfunc.overdensityMap_mask(fluxmap[i], idx_mask)
            overdensitymap_nu[idx_mask] = hp.UNSEEN
            w_cross[i] = hp.sphtfunc.anafast(overdensitymap_nu, overdensityMap_g)
        return w_cross


    def getCrossCorrelation_countsmap_atm(self, countsmap_atm, overdensityMap_g, idx_mask):
        fluxmap = hp_utils.vector_intensity_from_counts_and_exposure(countsmap_atm, self.exposuremap_atm)
        return self.getCrossCorrelation_fluxmap(fluxmap, overdensityMap_g, idx_mask)

    def getCrossCorrelation_countsmap_mix(self, countsmap_atm, countsmap_astro, overdensityMap_g, idx_mask):
        fluxmap_atm = hp_utils.vector_intensity_from_counts_and_exposure(countsmap_atm, self.exposuremap_atm)
        fluxmap_astro = hp_utils.vector_intensity_from_counts_and_exposure(countsmap_astro, self.exposuremap_astro)
        fluxmap = fluxmap_atm + fluxmap_astro
        return self.getCrossCorrelation_fluxmap(fluxmap, overdensityMap_g, idx_mask)



    def getIntensity(self, countsmap, dt_days, spectralIndex):
        """Convert a countsmap to total intensity using the exposure map"""
        if spectralIndex == 3.7:
            fluxmap = hp_utils.vector_intensity_from_counts_and_exposure(countsmap, self.exposuremap_atm)
        elif spectralIndex == 2.2:
            fluxmap = hp_utils.vector_intensity_from_counts_and_exposure(countsmap, self.exposuremap_astro)
        else:
            exposuremap = self.weightedAeff(spectralIndex)
            fluxmap = hp_utils.vector_intensity_from_counts_and_exposure(countsmap, exposuremap)

        for i in np.arange(Defaults.NEbin):
            intensity[i] = np.sum(fluxmap[i]) / (10.**Defaults.map_logE_center[i] * np.log(10.) * Defaults.dlogE) / (dt_days * Defaults.DT_SECONDS) / (4 * np.pi)  / 1e4 ## exposure map in m^2
        return intensity
