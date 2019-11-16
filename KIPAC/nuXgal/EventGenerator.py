
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate as integrate
import healpy as hp
from scipy.interpolate import CubicSpline, interp1d, UnivariateSpline

from .Utilityfunc import *
from . import Defaults

class EventGenerator():
    def __init__(self):

        self.Aeff_max = np.zeros(Defaults.NEbin) # m^2


        self.prob_reject = np.zeros((Defaults.NEbin, Defaults.NPIXEL))
        for i in np.arange(Defaults.NEbin):
            self.prob_reject[i] = hp.fitsfunc.read_map(os.path.join(Defaults.NUXGAL_IRF_DIR,
                                                                    'Aeff' + str(i)+'.fits'), verbose=False)
            self.Aeff_max[i] = np.max(self.prob_reject[i])
            self.prob_reject[i] = self.prob_reject[i] / self.Aeff_max[i]

            #fig = plt.figure(figsize=(8,6))
            #hp.mollview(prob_reject[i])
            #plt.savefig('syntheticEventmap/prob_reject' + str(i) + '.pdf')

        self.meanEventnumber_year = np.loadtxt(os.path.join(Defaults.NUXGAL_IRF_DIR,
                                                            'eventNumber_Ebin_perIC86year.txt'))



    def astroEvent_galaxy(self, density, intrinsicCounts, verbose=True):
        # we have assumed that density, intrinsicCounts matches shapes of NPIXEL and NEbin, respectively
        eventmap = np.zeros((Defaults.NEbin, Defaults.NPIXEL))
        for i in range(Defaults.NEbin):
            # generate event following galaxy-like distribution
            eventmap[i] = poisson_sampling(density, intrinsicCounts[i])
            if verbose:
                print(i)
                print(np.sum(eventmap[i]))
            # cut events according to exposure
            for jpixel in np.where(eventmap[i] != 0)[0]:
                eventmap[i][jpixel] = np.sum(np.random.binomial(1, self.prob_reject[i][jpixel],
                                                                int(eventmap[i][jpixel])))
            if verbose:
                print(np.sum(eventmap[i]))
        return eventmap

    # dN/dE \propto E^alpha
    def randPowerLaw(self, alpha, Ntotal, emin, emax):
        if alpha == -1:
            part1 = np.log(emax)
            part2 = np.log(emin)
            return np.exp((part1 - part2) * np.random.rand(Ntotal) + part2)
        else:
            part1 = np.power(emax, alpha + 1)
            part2 = np.power(emin, alpha + 1)
            return np.power((part1 - part2) * np.random.rand(Ntotal) + part2, 1./(alpha + 1))

    def astroEvent_galaxy_powerlaw(self, density, Ntotal, alpha, emin=1e2, emax=1e9, verbose=True):
        energy = self.randPowerLaw(alpha, Ntotal, emin, emax)
        intrinsicCounts = np.histogram(np.log10(energy), Defaults.map_logE_edge)
        return self.astroEvent_galaxy(density, intrinsicCounts, verbose=True)


    def atmBG_coszenith(self, eventNumber, energyBin):
        N_coszenith = np.loadtxt(os.path.join(Defaults.NUXGAL_IRF_DIR, 'N_coszenith'+str(energyBin)+'.txt'))
        N_coszenith_spline = interp1d(N_coszenith[:, 0], N_coszenith[:, 1], bounds_error=False, fill_value=0.)
        grid_cdf = np.linspace(-1, 1, 300)
        dgrid_cdf = np.mean(grid_cdf[1:] - grid_cdf[0:-1])
        cdf = np.zeros(len(grid_cdf))

        for i in range(len(cdf)):
            cdf[i] = np.trapz(N_coszenith_spline(grid_cdf[:i]), grid_cdf[:i], dgrid_cdf)

        cdf /= cdf[-1]
        index_non_zero = np.where(cdf == 0)[0][-1]
        index_non_one = np.where(1 - cdf < 8e-5)[0][0] # to avoid peak in energy bin 3
        coin_toss = np.random.rand(eventNumber)
        #print(energyBin, cdf[index_non_zero:index_non_one],grid_cdf[index_non_zero:index_non_one])
        cdfFunc = interp1d(cdf[index_non_zero:index_non_one],
                           grid_cdf[index_non_zero:index_non_one],
                           bounds_error=False, fill_value="extrapolate")
        #print(energyBin,
        #      coin_toss[(coin_toss < cdf[index_non_zero])],
        #      coin_toss[(coin_toss > cdf[index_non_one-1])])

        return cdfFunc(coin_toss)

    def atmEvent_powerlaw(self, eventNumber, index):
        eventmap = np.zeros((Defaults.NEbin, Defaults.NPIXEL))
        event_energy = self.randPowerLaw(index, eventNumber,
                                         10.**Defaults.map_logE_center[0],
                                         10.**Defaults.map_logE_center[-1])
        eventnumber_Ebin = np.histogram(np.log10(event_energy), Defaults.map_logE_edge)[0]
        print(eventnumber_Ebin)
        for i in range(Defaults.NEbin): # note bin 5 and 6 have no event in 2012 data
            if eventnumber_Ebin[i] > 0:
                coszenith = self.atmBG_coszenith(eventnumber_Ebin[i], i)
                phi = np.random.rand(eventnumber_Ebin[i]) * 2 * np.pi
                indexPixel = hp.pixelfunc.ang2pix(Defaults.NSIDE, np.arccos(coszenith), phi)
                for _indexPixel in indexPixel:
                    eventmap[i][_indexPixel] += 1
        return eventmap

    def atmEvent(self, duration_year):
        eventmap = np.zeros((Defaults.NEbin, Defaults.NPIXEL))
        eventnumber_Ebin = np.random.poisson(self.meanEventnumber_year * duration_year)
        for i in range(Defaults.NEbin): # note bin 5 and 6 have no event in 2012 data
            if eventnumber_Ebin[i] > 0:
                coszenith = self.atmBG_coszenith(eventnumber_Ebin[i], i)
                if len(coszenith[np.abs(coszenith) > 1]):
                    print(coszenith[np.abs(coszenith) > 1])
                phi = np.random.rand(eventnumber_Ebin[i]) * 2 * np.pi
                indexPixel = hp.pixelfunc.ang2pix(Defaults.NSIDE, np.arccos(coszenith), phi)
                for _indexPixel in indexPixel:
                    eventmap[i][_indexPixel] += 1
        return eventmap
