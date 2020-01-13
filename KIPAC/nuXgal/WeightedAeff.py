"""Analysis class"""

import os

import numpy as np

from . import Defaults

from . import file_utils





class WeightedAeff():
    """Weighted Effective Area class"""

    def __init__(self, year='IC86-2012', spectralIndex=3.7):
        """C'tor"""

        self.year = year
        self.spectralIndex = spectralIndex
        aeff_path = os.path.join(Defaults.NUXGAL_IRF_DIR, 'WeightedAeff_' + year + '_' + str(spectralIndex) + '_' + '{i}.fits')
        aeff_path_0 = os.path.join(Defaults.NUXGAL_IRF_DIR, 'WeightedAeff_' + year + '_' + str(spectralIndex) + '_' + '0.fits')

        if os.path.exists(aeff_path_0):
            self.exposuremap = file_utils.read_maps_from_fits(aeff_path, Defaults.NEbin)

        else:
            print (aeff_path, 'does not exist. Compute effective area with spectralIndex =', spectralIndex)
            self.exposuremap = self.computeWeightedAeff(spectralIndex)
            file_utils.write_maps_to_fits(self.exposuremap, aeff_path)





    def readTables(self):
        Aeff_file = np.loadtxt(os.path.join(Defaults.NUXGAL_IRF_DIR, self.year + '-TabulatedAeff.txt'))
        # effective area, 200 in cos zenith, 70 in E
        self.Aeff_table = Aeff_file[:, 4]
        Emin_eff = np.reshape(Aeff_file[:, 0], (70, 200))[:, 0]
        Emax_eff = np.reshape(Aeff_file[:, 1], (70, 200))[:, 0]

        self.Emin_eff = Emin_eff
        self.Emax_eff = Emax_eff
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
            #factor_i = (np.power(Defaults.map_E_edge[i+1], 1 - spectralIndex) - np.power(Defaults.map_E_edge[i], 1 - spectralIndex)) / (1 - spectralIndex) / (np.log(10.) * self.dlogE_eff)
            #print (i, jstart, jend, self.Emin_eff[jstart], self.Emin_eff[jend-1], Defaults.map_E_edge[i], Defaults.map_E_edge[i+1], factor_i)
            #for j in np.arange(jstart, jend):
            #    weightedAeff[i] += np.power(self.Ec_eff[j], 1 - spectralIndex) / self.Aeff_table[j * 200 + self.index_coszenith]
            #weightedAeff[i] = factor_i / weightedAeff[i]

            factor_i = 1. / (np.power(Defaults.map_E_edge[i+1], 1 - spectralIndex) - np.power(Defaults.map_E_edge[i], 1 - spectralIndex))
            for j in np.arange(jstart, jend):
                weightedAeff[i] += self.Aeff_table[j * 200 + self.index_coszenith] * (np.power(self.Emax_eff[j], 1 - spectralIndex) - np.power(self.Emin_eff[j], 1 - spectralIndex))
            weightedAeff[i] *= factor_i

            #weightedAeff[i] = self.Aeff_table[jstart * 200 + self.index_coszenith]
        return weightedAeff
