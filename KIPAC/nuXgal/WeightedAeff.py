"""A class to computed the spectrum-weighed effective area for IceCube Neutrino events"""

import os

import numpy as np

from . import Defaults

from . import file_utils

class WeightedAeff():
    """Weighted Effective Area class

    This computed and stores the weighted effective area for a single year
    """
    def __init__(self, year='IC86-2012', spectralIndex=3.7):
        """C'tor"""
        self.year = year
        self.spectralIndex = spectralIndex
        self.Aeff_table = None
        self.Emin_eff = None
        self.Ec_eff = None
        self.Ec_eff_len = None
        self.dlogE_eff = None
        self.cosZenith_min = None
        self.index_coszenith = None

        aeff_path = Defaults.WEIGHTED_AEFF_FORMAT.format(year=year,
                                                         specIndex=str(spectralIndex),
                                                         ebin='{i}')
        aeff_path_0 = aeff_path.format(i='0')

        if os.path.exists(aeff_path_0):
            self.exposuremap = file_utils.read_maps_from_fits(aeff_path, Defaults.NEbin)
        else:
            print(aeff_path, 'does not exist. Compute effective area with spectralIndex =', spectralIndex)
            self.exposuremap = self.computeWeightedAeff(spectralIndex)
            file_utils.write_maps_to_fits(self.exposuremap, aeff_path)


    def readTables(self):
        """Read the Effective area tabels for a particular year"""
        Aeff_file = np.loadtxt(TABULATED_AEFF_FORMAT.format(year=self.year))
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


    def computeWeightedAeff(self, spectralIndex=3.7):
        """Compute the weighed effective area

        Parameters
        ----------
        spectralIndex : `float`
            The spectral index to use for the weighting
        """
        self.readTables()
        weightedAeff = np.zeros((Defaults.NEbin, Defaults.NPIXEL))
        jend = 0
        for i in range(Defaults.NEbin):
            jstart = jend
            jend = np.searchsorted(self.Emin_eff, Defaults.map_E_edge[i+1])
            factor_i = (np.power(Defaults.map_E_edge[i+1],
                                 1 - spectralIndex) - np.power(Defaults.map_E_edge[i], 1 - spectralIndex)) /\
                                 (1 - spectralIndex) / (np.log(10.) * self.dlogE_eff)
            #print (i, jstart, jend, self.Emin_eff[jstart], self.Emin_eff[jend-1],
            #                     Defaults.map_E_edge[i], Defaults.map_E_edge[i+1], factor_i)
            for j in np.arange(jstart, jend):
                weightedAeff[i] += np.power(self.Ec_eff[j], 1 - spectralIndex) /\
                    self.Aeff_table[j * 200 + self.index_coszenith]
            weightedAeff[i] = factor_i / weightedAeff[i]
        return weightedAeff



class ExposureLibrary:
    """Library of exposure maps"""

    def __init__(self):
        """C'tor"""
        self._exposure_dict = {}
        
    def keys(self):
        """Return the names of exposure maps"""
        return self._exposure_dict.keys()

    def values(self):
        """Returns the exposure maps"""
        return self._exposure_dict.values()

    def items(self):
        """Return the name : map pairs"""
        return self._exposure_dict.items()

    def __getitem__(self, key):
        """Return a particular exposure map by name"""
        return self._exposure_dict[key]
    
    def get_exposure(self, year='IC86-2012', spectralIndex=3.7):
        """Get the spectrum weighted exposure map.
        This will read it from the IRF area if it exists and create it otherwise.

        Parameters
        ----------
        year : `str`:
            The year we want the exposure map for
        spectralIndex : `float`
            The spectral index to use for the weighting

        Returns
        -------
        exposure_map : `np.ndarray`
            The exposure map in question
        """
        key = "%s_%s" % (year, str(spectralIndex))
        if key in self._exposure_dict:
            return self._exposure_dict[key]
        weff = WeightedAeff(year, spectralIndex)
        exposuremap = weff.exposuremap
        self._exposure_dict[key] = exposuremap
        return exposuremap


ICECUBE_EXPOSURE_LIBRARY = ExposureLibrary()
