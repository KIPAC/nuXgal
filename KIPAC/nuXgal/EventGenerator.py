"""Top level class to generate synthetic events"""

import os
import numpy as np

import healpy as hp

from . import Defaults
from . import file_utils

from .Generator import AtmGenerator, AstroGenerator


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
        aeff_path = os.path.join(Defaults.NUXGAL_IRF_DIR, 'Aeff{i}.fits')
        nevents_path = os.path.join(Defaults.NUXGAL_IRF_DIR, 'eventNumber_Ebin_perIC86year.txt')
        gg_sample_path = os.path.join(Defaults.NUXGAL_ANCIL_DIR, 'galaxySampleOverdensity.fits')

        aeff = file_utils.read_maps_from_fits(aeff_path, Defaults.NEbin)
        cosz = file_utils.read_cosz_from_txt(coszenith_path, Defaults.NEbin)
        nevts = np.loadtxt(nevents_path)
        nastro = 0.003 * nevts

        gg_overdensity = hp.fitsfunc.read_map(gg_sample_path)
        gg_pdf = 1. + gg_overdensity
        gg_pdf /= gg_pdf.sum()

        self._atm_gen = AtmGenerator(Defaults.NEbin, coszenith=cosz, nevents_expected=nevts)
        self._astro_gen = AstroGenerator(Defaults.NEbin, aeff=aeff, nevents_expected=nastro, pdf=gg_pdf)
        self.Aeff_max = aeff.max(1)

    @property
    def atm_gen(self):
        """Astrospheric event generator"""
        return self._atm_gen

    @property
    def astro_gen(self):
        """Astrophysical event generator"""
        return self._astro_gen

    def astroEvent_galaxy(self, density, intrinsicCounts):
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
        pdf = density / density.mean()
        self._astro_gen.pdf.set_value(pdf, clear_parent=False)
        self._astro_gen.nevents_expected.set_value(intrinsicCounts, clear_parent=False)

        return self._astro_gen.generate_event_maps(1)[0]


    def astroEvent_galaxy_powerlaw(self, density, Ntotal, alpha, emin=1e2, emax=1e9):
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
        return self.astroEvent_galaxy(density, intrinsicCounts)


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


    def atmEvent_powerlaw(self, eventNumber, index):
        """Generate atmosphere event maps from a powerlaw and a number of input events

        Parameters
        ----------
        eventNumber : `int`
            Number of events to generate
        index : `float`
            Power law index

        Returns
        -------
        counts_map : `np.ndarray`
            Maps of simulated events
        """
        event_energy = randPowerLaw(index, eventNumber,
                                    Defaults.map_E_center[0],
                                    Defaults.map_E_center[-1])
        eventnumber_Ebin = np.histogram(np.log10(event_energy), Defaults.map_logE_edge)[0]
        self._atm_gen.nevents_expected.set_value(eventnumber_Ebin, clear_parent=False)
        return self._atm_gen.generate_event_maps(1)[0]


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
