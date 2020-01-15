"""Top level class to generate synthetic events"""

import os
import numpy as np

from . import Defaults
from . import file_utils
from .Generator import AtmGenerator, AstroGenerator_v2, get_dnde_astro
from .Exposure import ICECUBE_EXPOSURE_LIBRARY


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
    def __init__(self, year='IC86-2012', galaxySample=None, astroModel=None):
        """C'tor
        """
        self.year = year

        coszenith_path = Defaults.NCOSTHETA_FORMAT.format(year=year, ebin='{i}')
        cosz = file_utils.read_cosz_from_txt(coszenith_path, Defaults.NEbin)
        self.nevts = np.sum(cosz[:, :, 1], axis=1)
        self._atm_gen = AtmGenerator(Defaults.NEbin, coszenith=cosz, nevents_expected=self.nevts)
        self.astroModel = None
        self.gs = None
        self.Aeff_max = None
        self._astro_gen = None
        if astroModel is not None and galaxySample is not None:
            self.initializeAstro(astroModel, galaxySample)


    def initializeAstro(self, astroModel, galaxySample):
        """Initialize the event generate for a particular astrophysical model

        Parameters
        ----------
        astroModel : `str`
            The astrophysical model we are using
        """
        self.astroModel = astroModel
        self.gs = galaxySample

        pars, dN_dE_astro = get_dnde_astro(self.astroModel)
        spectralIndex = pars['spectralIndex']

        aeff = ICECUBE_EXPOSURE_LIBRARY.get_exposure(self.year, spectralIndex)
        self.Aeff_max = aeff.max(1)

        self.Nastro_1yr_Aeffmax = dN_dE_astro(10.**Defaults.map_logE_center) *\
            (10. ** Defaults.map_logE_center * np.log(10.) * Defaults.map_dlogE) *\
            (self.Aeff_max * 1E4) * (333 * 24. * 3600) * 4 * np.pi

        self._astro_gen = AstroGenerator_v2(Defaults.NEbin, aeff=aeff, pdf=self.gs.density)


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

        self._astro_gen.normalized_counts_map = normalized_counts_map
        self._astro_gen.nevents_expected.set_value(intrinsicCounts, clear_parent=False)
        return self._astro_gen.generate_event_maps(1)[0]



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
        """Generate Synthetic Data

        Parameters
        ----------
        N_yr : `float`
            Number of years of data to generate
        f_diff : `float`
            Fraction of astro events w.r.t. diffuse muon neutrino flux,
            f_diff = 1 means injecting astro events that sum up to 100% of diffuse muon neutrino flux
        density_nu : `np.ndarray`
            Background neutrino density map

        Returns
        -----
        counts_map : `np.ndarray`
            Maps of simulated events
        """
        if f_diff == 0.:
            Natm = np.random.poisson(self.nevts * N_yr)
            self._atm_gen.nevents_expected.set_value(Natm, clear_parent=False)
            countsmap = self._atm_gen.generate_event_maps(1)[0]
            return countsmap

        assert (self.astroModel is not None), "EventGenerator: no astrophysical model"
        # generate astro maps
        Nastro = np.random.poisson(self.Nastro_1yr_Aeffmax * N_yr * f_diff)
        astro_map = self.astroEvent_galaxy(Nastro, density_nu)

        # generate atmospheric eventmaps
        Natm = np.random.poisson(self.nevts * N_yr) - Nastro
        Natm[np.where(Natm < 0)] = 0.
        self._atm_gen.nevents_expected.set_value(Natm, clear_parent=False)
        atm_map = self._atm_gen.generate_event_maps(1)[0]

        return atm_map + astro_map
