"""Defintion of Likelihood function"""

import os
import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
from scipy.optimize import minimize


#from .Utilityfunc import
from .EventGenerator import EventGenerator
from .Analyze import Analyze
from . import Defaults
from .GalaxySample import GalaxySample
from .hp_utils import vector_apply_mask_hp, vector_apply_mask


class Likelihood():
    """Class to evaluate the likelihood for a particular model of neutrino galaxy correlation"""
    def __init__(self, N_yr, computeATM, computeASTRO, galaxyName, N_re=40):
        """C'tor

        Parameters
        ----------
        N_yr : `float`
            Number of years to simulate if computing the models
        computeATM : `bool`
            If true, compute the atmosphere correlation model
        computeASTRO : `bool`
            If true, compute the astrophysical correlation model
        galaxyName : `str`
            Name of the Galaxy sample
        N_re : `int`
           Number of realizations to use to compute the models
        """
        self.eg = EventGenerator()
        self.gs = GalaxySample()
        self.galaxyName = galaxyName
        self.cf = Analyze(self.gs.getOverdensity(galaxyName))
        self.anafastMask()


        # compute or load w_atm distribution
        if computeATM:
            self.computeAtmophericEventDistribution(N_yr, N_re, True)
        else:
            w_atm_mean_file = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR,
                                                      'w_atm_mean.txt'))
            self.w_atm_mean = w_atm_mean_file.reshape((Defaults.NEbin, Defaults.NCL))
            w_atm_std_file = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR,
                                                     'w_atm_std.txt'))
            self.w_atm_std = w_atm_std_file.reshape((Defaults.NEbin, Defaults.NCL))
            self.w_atm_std_square = self.w_atm_std ** 2
            self.Ncount_atm = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR,
                                                      'Ncount_atm_after_masking.txt'))

        """
        #compute or load w_astro distribution
        if computeASTRO:
            self.computeAstrophysicalEventDistribution(N_yr, N_re, True)
        else:
            w_astro_mean_file = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR,
                                                        'w_astro_mean.txt'))
            self.w_astro_mean = w_astro_mean_file.reshape((Defaults.NEbin, Defaults.NCL))
            w_astro_std_file = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR,
                                                       'w_astro_std.txt'))
            self.w_astro_std = w_astro_std_file.reshape((Defaults.NEbin, Defaults.NCL))
            self.w_astro_std_square = self.w_astro_std ** 2
            self.Ncount_astro = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'Ncount_astro_after_masking.txt'))
        """

        # scaled mean and std
        self.get_w_mean()

        self.w_std_square0 = np.zeros((Defaults.NEbin, Defaults.NCL))
        for i in range(3):
            self.w_std_square0[i] = self.w_atm_std_square[0] * self.Ncount_atm[0]
        for i in [3, 4]:
            self.w_std_square0[i] = self.w_atm_std_square[3] * self.Ncount_atm[3]


    def anafastMask(self):

        hpMask = hp.pixelfunc.ma(np.ones(Defaults.NPIXEL))

        mask_nu = np.zeros(Defaults.NPIXEL, dtype=np.bool)
        mask_nu[Defaults.idx_muon] = 1.

        if self.gs.getMask(self.galaxyName).any() != None:
            hpMask.mask = mask_nu + self.gs.getMask(self.galaxyName)

        else:
            hpMask.mask = mask_nu

        self.hpMask = hpMask


    def get_w_mean(self):
        overdensity = self.gs.getOverdensity(self.galaxyName)
        w_mean = hp.anafast(hp_utils.vector_apply_mask_hp(overdensity, self.hpMask))

        self.w_model_f1 = np.zeros((Defaults.NEbin, Defaults.NCL))
        for i in range(Defaults.NEbin):
                self.w_model_f1[i] = w_mean




    def computeAtmophericEventDistribution(self, N_yr, N_re, writeMap):
        """Compute the cross correlation distribution for Atmopheric event

        Parameters
        ----------
        N_yr : `float`
            Number of years to simulate if computing the models
        N_re : `int`
           Number of realizations to use to compute the models
        writeMap : `bool`
           If true, save the distributions
        """

        w_cross = np.zeros((N_re, Defaults.NEbin, 3 * Defaults.NSIDE))
        Ncount = np.zeros(Defaults.NEbin)

        eventnumber_Ebin = np.random.poisson(self.eg.nevts * N_yr)
        self.eg.atm_gen.nevents_expected.set_value(eventnumber_Ebin, clear_parent=False)
        eventmaps = self.eg.atm_gen.generate_event_maps(N_re)


        for iteration in np.arange(N_re):
            print("iter ", iteration)
            eventmap_atm = eventmaps[iteration]
            # first mask makes counts in masked region zero, for correct counting of event number. Second mask applies to healpy cross correlation calculation.
            eventmap_atm = vector_apply_mask(eventmap_atm, Defaults.idx_muon, copy=False)
            w_cross[iteration] = self.cf.crossCorrelationFromCountsmap(vector_apply_mask_hp(eventmap_atm, self.hpMask))
            Ncount = Ncount + np.sum(eventmap_atm, axis=1)

        self.w_atm_mean = np.mean(w_cross, axis=0)
        self.w_atm_std = np.std(w_cross, axis=0)
        self.Ncount_atm = Ncount / float(N_re)


        self.w_atm_std_square = self.w_atm_std ** 2
        if writeMap:
            np.savetxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'w_atm_mean.txt'),
                       self.w_atm_mean)
            np.savetxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'w_atm_std.txt'),
                       self.w_atm_std)
            np.savetxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'Ncount_atm_after_masking.txt'),
                       self.Ncount_atm)


    def computeAstrophysicalEventDistribution(self, N_yr, N_re, writeMap):
        """Compute the cross correlation distribution for Atmopheric event

        Parameters
        ----------
        N_yr : `float`
            Number of years to simulate if computing the models
        N_re : `int`
           Number of realizations to use to compute the models
        writeMap : `bool`
           If true, save the distributions
        """
        density_nu = self.gs.getDensity(self.galaxyName)
        Ncount_astro = np.zeros(Defaults.NEbin)
        w_cross_array = np.zeros((N_re, Defaults.NEbin, Defaults.NCL))
        for i in range(N_re):
            if i % 1 == 0:
                print(i)
            countsmap = self.eg.astroEvent_galaxy(self.eg.Nastro_1yr_Aeffmax * N_yr, density_nu)
            countsmap = vector_apply_mask(countsmap, Defaults.idx_muon, copy=False)
            w_cross_array[i] = self.cf.crossCorrelationFromCountsmap(vector_apply_mask_hp(countsmap, self.hpMask))
            Ncount_astro += np.sum(countsmap, axis=1)

        self.w_astro_mean = np.mean(w_cross_array, axis=0)
        self.w_astro_std = np.std(w_cross_array, axis=0)
        self.w_astro_std_square = self.w_astro_std ** 2
        self.Ncount_astro = Ncount_astro / N_re

        if writeMap:
            np.savetxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'w_astro_mean.txt'),
                       self.w_astro_mean)
            np.savetxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'w_astro_std.txt'),
                       self.w_astro_std)
            np.savetxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'Ncount_astro_after_masking.txt'),
                       self.Ncount_astro)





    def log_likelihood(self, f, w_data, Ncount, lmin, Ebinmin, Ebinmax):
        """Compute the log of the likelihood for a particular model

        Parameters
        ----------
        f : `float`
            The fraction of neutrino events correlated with the Galaxy sample
        w_data : `np.array`
            The cross-correlation coefficients
        Ncount : `np.array`
        lmin : `int`
            Minimum l to use in computing likelihood
        Ebinmin : `int`
        Ebinmax : `int`

        Returns
        -------
        logL : `float`
            The log likelihood, computed as sum_l (data_l - f * model_mean_l) /  model_std_l
        """
        w_model_mean = (self.w_model_f1[Ebinmin : Ebinmax].T * f).T
        w_model_std_square = (self.w_std_square0[Ebinmin : Ebinmax].T / Ncount[Ebinmin : Ebinmax]).T
        lnL_le = - (w_data[Ebinmin : Ebinmax] - w_model_mean) ** 2 / w_model_std_square
        return np.sum(lnL_le[:, lmin:])

    def minimize__lnL(self, w_data, Ncount, lmin, Ebinmin, Ebinmax):
        """Minimize the log-likelihood

        Parameters
        ----------
        f : `float`
            The fraction of neutrino events correlated with the Galaxy sample
        w_data : `np.array`
            The cross-correlation coefficients
        Ncount : `np.array`
        lmin : `int`
            Minimum l to use in computing likelihood
        Ebinmin : `int`
        Ebinmax : `int`

        Returns
        -------
        x : `array`
            The parameters that minimize the log-likelihood
        TS : `float`
            The Test Statistic, computed as 2 * logL_x - logL_0
        """
        len_f = Ebinmax - Ebinmin
        nll = lambda *args: -self.log_likelihood(*args)
        initial = 0.5 + 0.1 * np.random.randn(len_f)
        soln = minimize(nll, initial, args=(w_data, Ncount, lmin, Ebinmin, Ebinmax), bounds=[(0, 4)] * (len_f))
        return soln.x, (self.log_likelihood(soln.x, w_data, Ncount, lmin, Ebinmin, Ebinmax) -\
                            self.log_likelihood(np.zeros(len_f), w_data, Ncount, lmin, Ebinmin, Ebinmax)) * 2

    def TS_distribution(self, N_re, N_yr, lmin, f_diff, galaxyName, writeData=True):
        """Generate a Test Statistic distribution for simulated trials

        Parameters
        ----------
        N_re : `int`
           Number of realizations to use
        N_yr : `float`
            Number of years to simulate
        lmin : `int`
            Minimum l to use in computing likelihood
        f_diff : `float`
            Input value for signal fraction
        galaxyName : `str`
            Name of Galaxy sample
        writeData : `bool`
            Write the TS distribution to a text file

        Returns
        -------
        TS_array : `np.array`
            The array of TS values
        """
        TS_array = np.zeros(N_re)
        for i in range(N_re):
            #if i % 10 == 0:
            #    print (i)
            datamap = self.eg.SyntheticData(N_yr, f_diff=f_diff, density_nu=self.gs.getDensity(galaxyName))
            datamap = vector_apply_mask(datamap, Defaults.idx_muon, copy=False)
            w_data = self.cf.crossCorrelationFromCountsmap(vector_apply_mask_hp(datamap, self.hpMask))
            Ncount = np.sum(datamap, axis=1)
            Ebinmax = np.min([np.where(Ncount != 0)[0][-1]+1, 5])
            minimizeResult = (self.minimize__lnL(w_data, Ncount, lmin, 0, Ebinmax))
            print(i, Ncount, minimizeResult[0], minimizeResult[-1])
            TS_array[i] = minimizeResult[-1] #(self.minimize__lnL(w_data, Ncount, lmin, 0, Ebinmax))[-1]
        if writeData:
            np.savetxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'TS_'+str(f_diff)+'_'+galaxyName+'.txt'), TS_array)
        return TS_array


    def log_prior(self, f):
        """Compute log of the prior on a f, implemented as a flat prior between 0 and 2

        Parameters
        ----------
        f : `float`
            The signal fraction

        Returns
        -------
        value : `float`
            The log of the prior
        """
        if np.min(f) > 0. and np.max(f) < 2.:
            return 0.
        return -np.inf


    def log_probability(self, f, w_data, Ncount, lmin, Ebinmin, Ebinmax):
        """Compute log of the probablity of f, given some data

        Parameters
        ----------
        f : `float`
            The signal fraction
        w_data : `np.array`
            The cross-correlation coefficients
        Ncount : `np.array`
        lmin : `int`
            Minimum l to use in computing likelihood
        Ebinmin : `int`
        Ebinmax : `int`

        Returns
        -------
        value : `float`
            The log of the probability, defined as log_prior + log_likelihood
        """
        lp = self.log_prior(f)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(f, w_data, Ncount, lmin, Ebinmin, Ebinmax)



    def runMCMC(self, w_data, Ncount, lmin, Ebinmin, Ebinmax, Nwalker, Nstep=500):
        """Run a Markov Chain Monte Carlo

        Parameters
        ----------
        w_data : `np.array`
            The cross-correlation coefficients
        Ncount : `np.array`
        lmin : `int`
            Minimum l to use in computing likelihood
        Ebinmin : `int`
        Ebinmax : `int`
        Nwalker : `int`
        Nstep : `int`
        """

        ndim = Ebinmax - Ebinmin
        pos = 0.3 + np.random.randn(Nwalker, ndim) * 0.1
        nwalkers, ndim = pos.shape
        backend = emcee.backends.HDFBackend(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'test.h5'))
        backend.reset(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability,
                                        args=(w_data, Ncount, lmin, Ebinmin, Ebinmax), backend=backend)
        sampler.run_mcmc(pos, Nstep, progress=True)



    def plotMCMCchain(self, ndim, labels, truths):
        """Plot the results of a Markov Chain Monte Carlo

        Parameters
        ----------
        ndim : `int`
            The number of variables
        labels : `array`
            Labels for the variables
        truths : `array`
            The MC truth values
        """

        reader = emcee.backends.HDFBackend(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'test.h5'))
        fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
        samples = reader.get_chain()

        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")
        fig.savefig(os.path.join(Defaults.NUXGAL_PLOT_DIR, 'MCMCchain.pdf'))

        flat_samples = reader.get_chain(discard=100, thin=15, flat=True)
        #print(flat_samples.shape)
        fig = corner.corner(flat_samples, labels=labels, truths=truths)
        fig.savefig(os.path.join(Defaults.NUXGAL_PLOT_DIR, 'MCMCcorner.pdf'))
