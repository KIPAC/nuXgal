"""Top level class to generate synthetic events"""

import os
import numpy as np

from . import Defaults
from . import file_utils
from .Generator import AtmGenerator, AstroGenerator_v2
from .WeightedAeff import ICECUBE_EXPOSURE_LIBRARY


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
    def __init__(self, year='IC86-2012', astroModel=None):
        """C'tor
        """
        self.year = year
        coszenith_path = os.path.join(Defaults.NUXGAL_IRF_DIR, 'Ncos_theta_'+year+'_'+'{i}.txt')
        cosz = file_utils.read_cosz_from_txt(coszenith_path, Defaults.NEbin)
        self.nevts = np.sum(cosz[:, :, 1], axis=1)
        self._atm_gen = AtmGenerator(Defaults.NEbin, coszenith=cosz, nevents_expected=self.nevts)
        self.astroModel = None
        self._astro_gen = None
        self.Aeff_max = None
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

        if astroModel == 'numu':
            # calculate expected event number using IceCube diffuse neutrino flux
            # in GeV^-1 cm^-2 s^-1 sr^-1, muon neutrino, 1908.09551
            dN_dE_astro = lambda E_GeV: 1.44E-18 * (E_GeV / 100e3)**(-2.28)
            spectralIndex = 2.28

        elif astroModel == 'hese':
            # calculate expected event number using IceCube HESE flux
            # in GeV^-1 cm^-2 s^-1 sr^-1, three flavors / 3 for muon neutrinos, 1907.11266
            dN_dE_astro = lambda E_GeV: 6.45E-18 * (E_GeV / 100e3)**(-2.89) / 3.
            spectralIndex = 2.89

        else:
            print("Unknown astro model. Use 'numu' as default value")
            dN_dE_astro = lambda E_GeV: 1.44E-18 * (E_GeV / 100e3)**(-2.28)
            spectralIndex = 2.28

        aeff = ICECUBE_EXPOSURE_LIBRARY.get_exposure(self.year, spectralIndex)
        self._astro_gen = AstroGenerator_v2(Defaults.NEbin, aeff=aeff)
        self.Aeff_max = aeff.max(1)


        # total expected number of events before cut, for one year data
        self.Nastro_1yr_Aeffmax = np.zeros(Defaults.NEbin)
        for i in np.arange(Defaults.NEbin):
            self.Nastro_1yr_Aeffmax[i] = dN_dE_astro(10.**Defaults.map_logE_center[i]) *\
                (10. ** Defaults.map_logE_center[i] * np.log(10.) * Defaults.map_dlogE) *\
                (self.Aeff_max[i] * 1E4) * (333 * 24. * 3600) * 4 * np.pi



        #self.nnus = np.zeros(Defaults.NEbin)
        #index_north = np.where(cosz[0][:,0] > 0)
        #for i in range(Defaults.NEbin):
        #    self.nnus[i] = np.sum(cosz[i][index_north][:,1])



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
        """Generate Synthetic Data

        Parameters
        ----------
        N_yr : `float`
            Number of years of data to generate
        f_diff : `float`
            Fraction of astro events w.r.t. diffuse muon neutrino flux,
            f_diff = 1 means injecting astro events that sum up to 100% of diffuse muon neutrino flux
        density_nu : `float`
            Background neutrino density (FIXME, check this)

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
