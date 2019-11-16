
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate as integrate
import healpy as hp

from .Utilityfunc import *
from .EventGenerator import *

from . import Defaults

class Analyze():

    aeff_factor = Defaults.DT_SECONDS / (4 * np.pi)  / Defaults.M2_TO_CM2



    def __init__(self):

        # exposure map
        self.exposuremap = np.zeros((Defaults.NEbin, Defaults.NPIXEL))
        for i in np.arange(Defaults.NEbin):
            #self.exposuremap[i] = hp.fitsfunc.read_map('../syntheticData/Aeff' + str(i)+'.fits', verbose=False)
            self.exposuremap[i] = hp.fitsfunc.read_map(os.path.join(Defaults.NUXGAL_IRF_DIR,
                                                                    'Aeff' + str(i)+'.fits'), verbose=False)

        # generate galaxy samples
        self.l_cl = np.arange(1, 3 * Defaults.NSIDE + 1)
        self.l = np.linspace(1, 500, 500)
        cl_galaxy_file = np.loadtxt(os.path.join(Defaults.NUXGAL_ANCIL_DIR, 'Cl_ggRM.dat'))

        self.cl_galaxy = cl_galaxy_file[:500]
        self.overdensityMap_g = hp.fitsfunc.read_map(os.path.join(Defaults.NUXGAL_ANCIL_DIR,
                                                                  'galaxySampleOverdensity.fits'), verbose=False)



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
        cl_nu = np.zeros(Defaults.NEbin, 3 * Defaults.NSIDE)
        for i in range(Defaults.NEbin):
            overdensityMap_nu = overdensityMap(intensitymap[i])
            cl_nu[i] = hp.sphtfunc.anafast(overdensityMap_nu)
        return cl_nu


    def powerSpectrumFromCountsmap(self, countsmap):
        cl_nu = np.zeros((Defaults.NEbin, 3 * Defaults.NSIDE))
        intensitymap = np.divide(countsmap, self.exposuremap,
                                 out=np.zeros_like(countsmap), where=self.exposuremap != 0)

        for i in range(Defaults.NEbin):
            overdensityMap_nu = overdensityMap(intensitymap[i])
            cl_nu[i] = hp.sphtfunc.anafast(overdensityMap_nu)
        return cl_nu


    def crossCorrelationFromCountsmap(self, countsmap):
        intensitymap = np.divide(countsmap, self.exposuremap,
                                 out=np.zeros_like(countsmap), where=self.exposuremap != 0)

        w_cross = np.zeros((Defaults.NEbin, 3 * Defaults.NSIDE))
        for i in range(Defaults.NEbin):
            overdensityMap_nu = overdensityMap(intensitymap[i])
            w_cross[i] = hp.sphtfunc.anafast(overdensityMap_nu, self.overdensityMap_g)
        return w_cross


    def crossCorrelation_atm_std(self, N_re=100):
        eg = EventGenerator()
        w_cross = np.zeros((N_re, Defaults.NEbin, 3 * Defaults.NSIDE))

        for iteration in np.arange(N_re):
            print("iter ", iteration)
            eventmap_atm = eg.atmEvent(1.)
            w_cross[iteration] = self.crossCorrelationFromCountsmap(eventmap_atm)

        return np.mean(w_cross, axis=0), np.std(w_cross, axis=0)
