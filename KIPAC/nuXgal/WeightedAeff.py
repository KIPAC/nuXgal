"""Analysis class"""

import os

import numpy as np

from . import Defaults

from . import file_utils




aeffpath = os.path.join(Defaults.NUXGAL_IRF_DIR, 'IC86-2012-TabulatedAeff.txt')

class WeightedAeff():
    """Weighted Effective Area class"""

    def __init__(self, computeTables=False):
        """C'tor"""

        aeff_atm_path = os.path.join(Defaults.NUXGAL_IRF_DIR, 'WeightedAeff_atm{i}.fits')
        aeff_astro_path = os.path.join(Defaults.NUXGAL_IRF_DIR, 'WeightedAeff_astro{i}.fits')

        if computeTables:
            self.exposuremap_atm = self.computeWeightedAeff(3.7)
            self.exposuremap_astro = self.computeWeightedAeff(2.28)
            file_utils.write_maps_to_fits(self.exposuremap_atm, aeff_atm_path)
            file_utils.write_maps_to_fits(self.exposuremap_astro, aeff_astro_path)

        else:
            self.exposuremap_atm = file_utils.read_maps_from_fits(aeff_atm_path, Defaults.NEbin)
            self.exposuremap_astro = file_utils.read_maps_from_fits(aeff_astro_path, Defaults.NEbin)




    def readTables(self):
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


    def computeWeightedAeff(self, spectralIndex = 3.7):
        self.readTables()
        weightedAeff = np.zeros((Defaults.NEbin, Defaults.NPIXEL))
        jend = 0
        for i in range(Defaults.NEbin):
            jstart = jend
            jend = np.searchsorted(self.Emin_eff, Defaults.map_E_edge[i+1])
            factor_i = (np.power(Defaults.map_E_edge[i+1], 1 - spectralIndex) - np.power(Defaults.map_E_edge[i], 1 - spectralIndex)) / (1 - spectralIndex) / (np.log(10.) * self.dlogE_eff)
            #print (i, jstart, jend, self.Emin_eff[jstart], self.Emin_eff[jend-1], Defaults.map_E_edge[i], Defaults.map_E_edge[i+1], factor_i)
            for j in np.arange(jstart, jend):
                weightedAeff[i] += np.power(self.Ec_eff[j], 1 - spectralIndex) / self.Aeff_table[j * 200 + self.index_coszenith]
            weightedAeff[i] = factor_i / weightedAeff[i]
        return weightedAeff
