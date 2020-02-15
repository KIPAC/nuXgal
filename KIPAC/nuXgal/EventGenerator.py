"""Top level class to generate synthetic events"""

import os
import numpy as np

from . import Defaults
from . import file_utils
from .Generator import AtmGenerator, AstroGenerator_v2, get_dnde_astro
from .Exposure import ICECUBE_EXPOSURE_LIBRARY


class EventGenerator():
    """Class to generate synthetic IceCube events

    This can generate both atmospheric and astrophysical events
    """
    def __init__(self, year='IC86-2012', astroModel=None):
        """C'tor
        """
        self.year = year

        coszenith_path = Defaults.NCOSTHETA_FORMAT.format(year=year, ebin='{i}')
        cosz = file_utils.read_cosz_from_txt(coszenith_path, Defaults.NEbin)

        # calculate number of events in the northern sky
        for i in range(Defaults.NEbin):
            for j in range(len(cosz[i])):
                if cosz[i][j][0] < np.cos(Defaults.theta_north):
                    cosz[i][j][1] = 0.

        self.nevts = np.sum(cosz[:, :, 1], axis=1)
        self._atm_gen = AtmGenerator(Defaults.NEbin, coszenith=cosz, nevents_expected=self.nevts)


        self.astroModel = None
        self.Aeff_max = None
        self._astro_gen = None
        if astroModel is not None:
            self.initializeAstro(astroModel)


    def initializeAstro(self, astroModel):
        """Initialize the event generate for a particular astrophysical model

        Parameters
        ----------
        astroModel : `str`
            The astrophysical model we are using
        """
        self.astroModel = astroModel

        assert (astroModel == 'observed_numu_fraction'), "EventGenerator: incorrect astroModel"

        # Fig 3 of 1908.09551
        self.f_astro_north_truth = np.array([0, 0.00221405, 0.01216614, 0.15222642, 0., 0., 0.]) * 2.
        spectralIndex = 2.28

        aeff = ICECUBE_EXPOSURE_LIBRARY.get_exposure(self.year, spectralIndex)
        self.Aeff_max = aeff.max(1)
        self._astro_gen = AstroGenerator_v2(Defaults.NEbin, aeff=aeff)


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


        assert (self.astroModel == 'observed_numu_fraction'), "EventGenerator: incorrect astrophysical model"
        density_nu = density_nu.copy()
        density_nu[Defaults.idx_muon] = 0. # since we do not know the fraction of numu in southern sky
        density_nu = density_nu / density_nu.sum()

        N_astro_north_obs = np.random.poisson(self.nevts * N_yr * self.f_astro_north_truth)
        N_astro_north_exp = [N_astro_north_obs[i] / np.sum(self._astro_gen.prob_reject()[i] * density_nu) for i in range(Defaults.NEbin)]
        astro_map = self.astroEvent_galaxy(np.array(N_astro_north_exp), density_nu)

        Natm = np.random.poisson(self.nevts * N_yr * (1-self.f_astro_north_truth))
        self._atm_gen.nevents_expected.set_value(Natm, clear_parent=False)
        atm_map = self._atm_gen.generate_event_maps(1)[0]

        return (atm_map + astro_map)
