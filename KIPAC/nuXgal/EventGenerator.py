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
    def __init__(self, year='IC86-2012', astroModel=None):
        """C'tor
        """
        self.year = year
        coszenith_path = os.path.join(Defaults.NUXGAL_IRF_DIR, 'Ncos_theta_'+year+'_'+'{i}.txt')
        cosz = file_utils.read_cosz_from_txt(coszenith_path, Defaults.NEbin)
        self.nevts = np.sum(cosz[:, :, 1], axis=1)
        self._atm_gen = AtmGenerator(Defaults.NEbin, coszenith=cosz, nevents_expected=self.nevts)

        # manually set all southern sky events to be 0
        #index_south = np.where(cosz[0][:,0] < np.cos(np.radians(95.)))

        for i in range(Defaults.NEbin):
            for j in range(len(cosz[i])):
                if cosz[i][j][0] < np.cos(Defaults.theta_north):
                    cosz[i][j][1] = 0.



            #for idx in index_south:
            #    cosz[i][idx] = 0.

        self.nnus = np.sum(cosz[:, :, 1], axis=1)


        # number of neutrinos in northern sky
        #self.nnus = np.zeros(Defaults.NEbin)
        #index_north = np.where(cosz[0][:,0] > 0)
        #for i in range(Defaults.NEbin):
        #    self.nnus[i] = np.sum(cosz[i][index_north][:,1])



        if astroModel is not None:
            self.initializeAstro(astroModel)


    def initializeAstro(self, astroModel):



        if astroModel == 'observed_numu_fraction':
            # Fig 3 of 1908.09551
            #self.f_astro_north_truth = np.array([0, 0.00391027, 0.04331592, 0.45597574, 1., 0., 0.])
            self.f_astro_north_truth = np.array([0, 0.00391027, 0.04331592, 0.45597574, 0., 0., 0.]) * 2.
            spectralIndex = 2.28

        else:

            if astroModel == 'numu':
                # calculate expected event number using IceCube diffuse neutrino flux
                # in GeV^-1 cm^-2 s^-1 sr^-1, muon neutrino, 1908.09551
                #dN_dE_astro = lambda E_GeV: 1.44E-18 * (E_GeV / 100e3)**(-2.28)
                dN_dE_astro_norm = 1.44E-18
                spectralIndex = 2.28

            elif astroModel == 'hese':
                # calculate expected event number using IceCube HESE flux
                # in GeV^-1 cm^-2 s^-1 sr^-1, three flavors / 3 for muon neutrinos, 1907.11266
                #dN_dE_astro = lambda E_GeV: 6.45E-18 * (E_GeV / 100e3)**(-2.89) / 3.
                dN_dE_astro_norm = 6.45E-18 / 3.
                spectralIndex = 2.89

            else:
                print ("Unknown astro model. Use 'numu' as default value")
                #dN_dE_astro = lambda E_GeV: 1.44E-18 * (E_GeV / 100e3)**(-2.28)
                spectralIndex = 2.28
                dN_dE_astro_norm = 1.44E-18


        self.astroModel = astroModel
        aeff = WeightedAeff(self.year, spectralIndex).exposuremap
        self._astro_gen = AstroGenerator_v2(Defaults.NEbin, aeff=aeff)
        self.Aeff_max = aeff.max(1)


        # total expected number of events before cut, for one year data
        self.Nastro_1yr_Aeffmax = np.zeros(Defaults.NEbin)

        if astroModel != 'observed_numu_fraction':
            for i in np.arange(Defaults.NEbin):
                self.Nastro_1yr_Aeffmax[i] = dN_dE_astro_norm * 1e5 / (1 - spectralIndex) *\
                (np.power(Defaults.map_E_edge[i+1] / 1e5, 1 - spectralIndex) - np.power(Defaults.map_E_edge[i] / 1e5, 1 - spectralIndex)) *\
                self.Aeff_max[i] * Defaults.M2_TO_CM2 * Defaults.DT_SECONDS * 4 * np.pi


            #dN_dE_astro(10.**Defaults.map_logE_center[i]) *\
            #    (10. ** Defaults.map_logE_center[i] * np.log(10.) * Defaults.map_dlogE) *\
            #    (self.Aeff_max[i] * 1E4) * (333 * 24. * 3600) * 4 * np.pi





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
        """ f_diff = 1 means injecting astro events that sum up to 100% of diffuse muon neutrino flux """
        if f_diff == 0.:
            #Natm = np.random.poisson(self.nevts * N_yr)
            Natm = np.random.poisson(self.nnus * N_yr)
            self._atm_gen.nevents_expected.set_value(Natm, clear_parent=False)
            countsmap = self._atm_gen.generate_event_maps(1)[0]
            return countsmap

        else:
            assert (self.astroModel is not None), "EventGenerator: no astrophysical model"
            if self.astroModel != 'observed_numu_fraction':
                Nastro = np.random.poisson(self.Nastro_1yr_Aeffmax * N_yr * f_diff)
                Nastro_generated = astro_map.sum(axis = 1)
                Natm = np.random.poisson(self.nevts * N_yr)
                Natm = Natm - Nastro_generated
                #print ('Nastro_generated=', Nastro_generated)
                #print ('Natm =', Natm)
                #print ('fastro=', Nastro_generated / float(Natm + Nastro_generated))
                Natm[np.where(Natm < 0)] = 0.
                astro_map = self.astroEvent_galaxy(Nastro, density_nu)
                self._atm_gen.nevents_expected.set_value(Natm, clear_parent=False)
                atm_map = self._atm_gen.generate_event_maps(1)[0]

            else:
                density_nu[Defaults.idx_muon] = 0. # since we do not know the fraction of numu in southern sky
                density_nu = density_nu / density_nu.sum()
                N_astro_north_obs = self.nnus * N_yr * self.f_astro_north_truth
                N_astro_north_exp = [N_astro_north_obs[i] / np.sum(self._astro_gen.prob_reject()[i] * density_nu) for i in range(Defaults.NEbin)]
                astro_map = self.astroEvent_galaxy(np.random.poisson(N_astro_north_exp), density_nu)

                Natm = np.random.poisson(self.nnus * N_yr * (1-self.f_astro_north_truth))
                self._atm_gen.nevents_expected.set_value(Natm, clear_parent=False)
                atm_map = self._atm_gen.generate_event_maps(1)[0]

                #print (np.sum(astro_map+atm_map, axis=1))


        return (atm_map + astro_map)
