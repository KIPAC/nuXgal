"""Top level class to generate synthetic events"""

import os
import numpy as np

from . import Defaults
from . import file_utils
from .Generator import AtmGenerator, AstroGenerator_v2
from .file_utils import write_maps_to_fits, read_maps_from_fits
from .WeightedAeff import WeightedAeff


# dN/dE \propto E^alpha
def randPowerLaw(alpha, Ntotal, emin, emax):
    """Generate a number of events from a power-law distribution"""
    if alpha == -1:
        part1 = np.log(emax)
        part2 = np.log(emin)
        return np.exp((part1 - part2) * np.random.rand(Ntotal) + part2)
    part1 = np.power(emax, alpha + 1)
    part2 = np.power(emin, alpha + 1)
    return np.power((part1 - part2) * np.random.rand(Ntotal) + part2, 1./(alpha + 1))


class EventGenerator():
    """Class to generate synthetic IceCube events

    This can generate both atmospheric and astrophysical events
    """
    def __init__(self):
        """C'tor
        """
        coszenith_path = os.path.join(Defaults.NUXGAL_IRF_DIR, 'N_coszenith{i}.txt')
        nevents_path = os.path.join(Defaults.NUXGAL_IRF_DIR, 'eventNumber_Ebin_perIC86year.txt')
        nnu_path = os.path.join(Defaults.NUXGAL_IRF_DIR, 'neutrinoNumber_Ebin_3yr.txt')
        #aeff_path = os.path.join(Defaults.NUXGAL_IRF_DIR, 'Aeff{i}.fits')
        #aeff = file_utils.read_maps_from_fits(aeff_path, Defaults.NEbin)

        aeff = WeightedAeff().exposuremap_astro
        cosz = file_utils.read_cosz_from_txt(coszenith_path, Defaults.NEbin)

        self.nevts = np.loadtxt(nevents_path)
        self.nnus = np.loadtxt(nnu_path)
        self._atm_gen = AtmGenerator(Defaults.NEbin, coszenith=cosz, nevents_expected=self.nevts)
        self._astro_gen = AstroGenerator_v2(Defaults.NEbin, aeff=aeff)
        self.Aeff_max = aeff.max(1)

        # calculate expected event number using IceCube diffuse neutrino flux
        # in GeV^-1 cm^-2 s^-1 sr^-1, muon neutrino
        dN_dE_astro = lambda E_GeV: 1.44E-18 * (E_GeV / 100e3)**(-2.28)
        # total expected number of events before cut, for one year data
        self.Nastro_1yr_Aeffmax = np.zeros(Defaults.NEbin)
        for i in np.arange(Defaults.NEbin):
            self.Nastro_1yr_Aeffmax[i] = dN_dE_astro(10.**Defaults.map_logE_center[i]) *\
                (10. ** Defaults.map_logE_center[i] * np.log(10.) * Defaults.map_dlogE) *\
                (self.Aeff_max[i] * 1E4) * (333 * 24. * 3600) * 4 * np.pi


    @property
    def atm_gen(self):
        """Astrospheric event generator"""
        return self._atm_gen

    @property
    def astro_gen(self):
        """Astrophysical event generator"""
        return self._astro_gen

    def astroEvent_galaxy(self, intrinsicCounts, normalized_counts_map):
        """Generate astrophysical event maps from a galaxy
        distribution and a number of intrinsice events

        Parameters
        ----------
        density : `np.ndarray`
            Galaxy density map, used as a pdf
        intrinsicCounts : `np.ndarray`
            True number of events, without accounting for Aeff variation

        Returns
        -------
        counts_map : `np.ndarray`
            Maps of simulated events
        """
        #pdf = density / density.mean()
        #self._astro_gen.pdf.set_value(pdf, clear_parent=False)
        self._astro_gen.normalized_counts_map = normalized_counts_map
        self._astro_gen.nevents_expected.set_value(intrinsicCounts, clear_parent=False)
        return self._astro_gen.generate_event_maps(1)[0]



    def astroEvent_galaxy_powerlaw(self, Ntotal, normalized_counts_map, alpha, emin=1e2, emax=1e9):
        """Generate astrophysical events from a power law
        distribution

        Parameters
        ----------
        density : `np.ndarray`
            Galaxy density map, used as a pdf
        Ntotal : `np.ndarray`
            Total number of events
        alpha : `float`
            Power law index
        emin : `float`
        emin : `float`


        Returns
        -------
        counts_map : `np.ndarray`
            Maps of simulated events
        """
        energy = randPowerLaw(alpha, Ntotal, emin, emax)
        intrinsicCounts = np.histogram(np.log10(energy), Defaults.map_logE_edge)
        return self.astroEvent_galaxy(intrinsicCounts, normalized_counts_map)


    def atmBG_coszenith(self, eventNumber, energyBin):
        """Generate atmospheric background cos(zenith) distributions

        Parameters
        ----------
        eventNumber : `int`
            Number of events to generate
        energyBin : `int`
            Energy bin to consider

        Returns
        -------
        cos_z : `np.ndarray`
            Array of synthetic cos(zenith) values
        """
        return self._atm_gen.cosz_cdf()[energyBin](np.random.rand(eventNumber))



    def atmEvent(self, duration_year):
        """Generate atmosphere event maps from expected rates per year

        Parameters
        ----------
        duration_year : `float`
            Number of eyars to generate

        Returns
        -------
        counts_map : `np.ndarray`
            Maps of simulated events
        """
        eventnumber_Ebin = np.random.poisson(self._atm_gen.nevents_expected() * duration_year)
        self._atm_gen.nevents_expected.set_value(eventnumber_Ebin, clear_parent=False)
        return self._atm_gen.generate_event_maps(1)[0]



    def SyntheticData(self, N_yr, f_diff, density_nu=None):
        """ f_diff = 1 means injecting astro events that sum up to 100% of diffuse muon neutrino flux """
        if f_diff == 0.:
            Natm = np.random.poisson(self.nevts * N_yr)
            self._atm_gen.nevents_expected.set_value(Natm, clear_parent=False)
            countsmap = self._atm_gen.generate_event_maps(1)[0]
            return countsmap, None

        else:
            f_atm = np.zeros(Defaults.NEbin)
            f_diff = 1.
            for i in range(Defaults.NEbin):
                if self.nnus[i] != 0.:
                    f_atm[i] = 1. - self.Nastro_1yr_Aeffmax[i] * f_diff / self.nnus[i]

            #print ('fraction of atm events:', f_atm)


            # generate atmospheric eventmaps
            Natm = np.random.poisson(self.nevts * N_yr * f_atm)
            self._atm_gen.nevents_expected.set_value(Natm, clear_parent=False)
            atm_map = self._atm_gen.generate_event_maps(1)[0]

            # generate astro maps
            Nastro = np.random.poisson(self.Nastro_1yr_Aeffmax * N_yr * f_diff)
            astro_map = self.astroEvent_galaxy(Nastro, density_nu)

            return atm_map, astro_map
