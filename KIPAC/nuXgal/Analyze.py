import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate as integrate
import healpy as hp

from .Utilityfunc import *
from .EventGenerator import *

class Analyze():
    def __init__(self):
        self.NSIDE = 128
        self.NPIXEL = hp.pixelfunc.nside2npix(self.NSIDE)

        self.map_logE_edge = np.linspace(2, 9, 8)
        self.map_logE_center = (self.map_logE_edge[0:-1] + self.map_logE_edge[1:]) / 2.
        self.dlogE = np.mean(self.map_logE_edge[1:] - self.map_logE_edge[0:-1])
        self.NEbin = len(self.map_logE_center)

        # exposure map
        self.exposuremap = np.zeros((self.NEbin, self.NPIXEL))
        for i in np.arange(self.NEbin):
            self.exposuremap[i] = hp.fitsfunc.read_map('../syntheticData/Aeff' + str(i)+'.fits', verbose=False)

        # generate galaxy samples
        self.l_cl = np.arange(1, 3 * self.NSIDE + 1)
        self.l = np.linspace(1, 500, 500)
        cl_galaxy_file = np.loadtxt('../data/Cl_ggRM.dat')
        self.cl_galaxy = cl_galaxy_file[:500]
        self.overdensityMap_g = hp.fitsfunc.read_map('../syntheticData/galaxySampleOverdensity.fits', verbose=False)




    def getIntensity(self, countsmap, dt_days=333):
        intensitymap = np.divide(countsmap, self.exposuremap,
                                 out=np.zeros_like(countsmap), where=self.exposuremap != 0)
        intensity = np.zeros(self.NEbin)
        for i in np.arange(self.NEbin):
            intensity[i] = np.sum(intensitymap[i]) / (10.**self.map_logE_center[i] * np.log(10.) * self.dlogE) / (dt_days * 24 * 3600) / (4 * np.pi)  / 1e4 ## exposure map in m^2
        return intensity


    def getIntensityMap(self, countsmap, mask):
        intensitymap = np.divide(countsmap, self.exposuremap,
                                 out=np.zeros_like(countsmap), where=self.exposuremap != 0)

        maskmap = hp.pixelfunc.ma(mask)
        intensitymap = intensitymap + maskmap
        return intensitymap


    def powerSpectrum(self, intensitymap):
        cl_nu = np.zeros(self.NEbin, 3 * self.NSIDE)
        for i in range(self.NEbin):
            overdensityMap_nu = overdensityMap(intensitymap[i])
            cl_nu[i] = hp.sphtfunc.anafast(overdensityMap_nu)
        return cl_nu


    def powerSpectrumFromCountsmap(self, countsmap):
        cl_nu = np.zeros((self.NEbin, 3 * self.NSIDE))
        intensitymap = np.divide(countsmap, self.exposuremap,
                                 out=np.zeros_like(countsmap), where=self.exposuremap != 0)

        for i in range(self.NEbin):
            overdensityMap_nu = overdensityMap(intensitymap[i])
            cl_nu[i] = hp.sphtfunc.anafast(overdensityMap_nu)
        return cl_nu


    def crossCorrelationFromCountsmap(self, countsmap):
        intensitymap = np.divide(countsmap, self.exposuremap,
                                 out=np.zeros_like(countsmap), where=self.exposuremap != 0)

        w_cross = np.zeros((self.NEbin, 3 * self.NSIDE))
        for i in range(self.NEbin):
            overdensityMap_nu = overdensityMap(intensitymap[i])
            w_cross[i] = hp.sphtfunc.anafast(overdensityMap_nu, self.overdensityMap_g)
        return w_cross


    def crossCorrelation_atm_std(self, N_re=100):
        eg = EventGenerator()
        w_cross = np.zeros((N_re, self.NEbin, 3 * self.NSIDE))

        for iteration in np.arange(N_re):
            eventmap_atm = eg.atmEvent(1.)
            w_cross[iteration] = self.crossCorrelationFromCountsmap(eventmap_atm)

        return np.mean(w_cross, axis=0), np.std(w_cross, axis=0)
