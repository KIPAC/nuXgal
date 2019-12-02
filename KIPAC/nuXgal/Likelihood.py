"""Defintion of Likelihood function"""

import os
import numpy as np
import healpy as np
import emcee
import corner
import matplotlib.pyplot as plt
from scipy.optimize import minimize


from .Utilityfunc import *
from .EventGenerator import *
from .Analyze import *
from . import Defaults
from .plot_utils import FigureDict


class Likelihood():
    def __init__(self, N_yr=1., computeATM = False, computeASTRO = False, N_re=40):
        self.cf = Analyze()
        self.eg = EventGenerator()

        # calculate expected event number using IceCube diffuse neutrino flux
        dN_dE_astro = lambda E_GeV: 1.44E-18 * (E_GeV / 100e3)**(-2.28) # GeV^-1 cm^-2 s^-1 sr^-1, muon neutrino
        # total expected number of events before cut, for one year data
        self.N_2012_Aeffmax = np.zeros(Defaults.NEbin)
        for i in np.arange(Defaults.NEbin):
            self.N_2012_Aeffmax[i] = dN_dE_astro(10.**Defaults.map_logE_center[i]) * (10. ** Defaults.map_logE_center[i] * np.log(10.) * Defaults.dlogE) * (self.eg.Aeff_max[i] * 1E4) * (333 * 24. * 3600) * 4 * np.pi


        # compute or load w_atm distribution
        if computeATM:
            self.computeAtmophericEventDistribution(N_yr, N_re, True)
        else:
            w_atm_mean_file = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'w_atm_mean.txt'))
            self.w_atm_mean  = w_atm_mean_file.reshape((Defaults.NEbin, Defaults.NCL))
            w_atm_std_file = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'w_atm_std.txt'))
            self.w_atm_std  = w_atm_std_file.reshape((Defaults.NEbin, Defaults.NCL))
            self.w_atm_std_square  =  self.w_atm_std ** 2
            self.Ncount_atm = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'Ncount_atm_after_masking.txt'))


        # compute or load w_astro distribution
        if computeASTRO:
            self.computeAstrophysicalEventDistribution(N_yr, N_re, True)
        else:
            #w_astro_mean_file = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'w_astro_mean.txt'))
            #self.w_astro_mean = w_astro_mean_file.reshape((Defaults.NEbin, Defaults.NCL))
            self.w_astro_mean = np.zeros((Defaults.NEbin, Defaults.NCL))
            for i in range(Defaults.NEbin):
                self.w_astro_mean[i] = self.cf.cl_galaxy[0:Defaults.NCL]
            w_astro_std_file = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'w_astro_std.txt'))
            self.w_astro_std = w_astro_std_file.reshape((Defaults.NEbin, Defaults.NCL))
            self.w_astro_std_square = self.w_astro_std ** 2
            self.Ncount_astro = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'Ncount_astro_after_masking.txt'))

		# expected fraction of astrophysical events in total counts, assuming f_diff = 1
        self.ratio_atm_astro = self.Ncount_atm / self.Ncount_astro


    def computeAtmophericEventDistribution(self, N_yr, N_re, writeMap):
        self.w_atm_mean, self.w_atm_std, self.Ncount_atm = self.cf.crossCorrelation_atm_std(N_yr, N_re)
        self.w_atm_std_square = self.w_atm_std ** 2
        if writeMap:
            np.savetxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'w_atm_mean.txt'), self.w_atm_mean)
            np.savetxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'w_atm_std.txt'), self.w_atm_std)
            np.savetxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'Ncount_atm_after_masking.txt'), self.Ncount_atm)


    def computeAstrophysicalEventDistribution(self, N_yr, N_re, writeMap):

        np.random.seed(Defaults.randomseed_galaxy)
        density_nu = hp.sphtfunc.synfast(self.cf.cl_galaxy, Defaults.NSIDE)
        density_nu = np.exp(density_nu)
        density_nu /= density_nu.sum() # a unique neutrino source distribution that shares the randomness of density_g
        Ncount_astro = np.zeros(Defaults.NEbin)
        w_cross_array = np.zeros((N_re, Defaults.NEbin, Defaults.NCL))
        for i in range(N_re):
            if i % 1 == 0:
                print(i)
            countsmap = self.eg.astroEvent_galaxy(self.N_2012_Aeffmax * N_yr, density_nu)
            countsmap = hp_utils.vector_apply_mask(countsmap, Defaults.mask_muon, copy=False)
            w_cross_array[i] = self.cf.crossCorrelationFromCountsmap(countsmap)
            Ncount_astro += np.sum(countsmap, axis=1)

        self.w_astro_mean = np.mean(w_cross_array, axis=0)
        #self.w_astro_mean = np.zeros((Defaults.NEbin, Defaults.NCL))
        #for i in range(Defaults.NEbin):
        #    self.w_astro_mean[i] = self.cf.cl_galaxy[0:Defaults.NCL]

        self.w_astro_std = np.std(w_cross_array, axis=0)
        self.w_astro_std_square = self.w_astro_std ** 2
        self.Ncount_astro = Ncount_astro / N_re

        if writeMap:
            np.savetxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'w_astro_mean.txt'), self.w_astro_mean)
            np.savetxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'w_astro_std.txt'), self.w_astro_std)
            np.savetxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'Ncount_astro_after_masking.txt'), self.Ncount_astro)





    def log_likelihood(self, f, datamap, lmin, energyBin=2, MAKE_TEST_PLOTS=False):

        f_diff, f_gal = f
        #f_gal = 0.6
        w_data = self.cf.crossCorrelationFromCountsmap(datamap)
        w_astro_mean = self.w_astro_mean * f_gal ** 0.5
        w_astro_std_square = self.w_astro_std_square #/ f_diff * f_gal # ?????
        fraction_count_astro = 1. / (1. + self.ratio_atm_astro / f_diff)
        fraction_count_atm = 1. - fraction_count_astro

        #print (f_diff, fraction_count_astro)
        w_model_mean = (w_astro_mean.T * fraction_count_astro).T #+ (self.w_atm_mean.T * fraction_count_atm).T
        w_model_std_square = self.w_atm_std_square
        #w_model_std_square = (w_astro_std_square.T * fraction_count_astro ** 2).T + (self.w_atm_std_square.T * fraction_count_atm ** 2).T
        #print ((w_model_std_square_old-w_model_std_square)[3]/w_model_std_square[3])
        #print (self.w_atm_std[3]/self.w_astro_std[3])
        lnL_array = -(w_data - w_model_mean) ** 2 / w_model_std_square


        if MAKE_TEST_PLOTS:
            testfigpath = os.path.join(Defaults.NUXGAL_PLOT_DIR, 'test')
            figs = FigureDict()
            color = ['r', 'orange', 'limegreen', 'skyblue', 'mediumslateblue', 'purple', 'grey']
            o_dict = figs.setup_figure('lnLtest', xlabel="l", ylabel=r'$w$', figsize=(6, 8))
            axes = o_dict['axes']
            w_model_std = w_model_std_square ** 0.5
            for i in range(Defaults.NEbin):
                axes.plot(self.cf.l, self.cf.cl_galaxy * 10 ** (i*2) * f_gal ** 0.5, color='grey', lw=1)
                w_model_mean[i] *= 10 ** (i * 2)
                w_model_std[i] *= 10 ** (i * 2)
                w_data[i] *= 10 ** (i * 2)
            figs.plot_cl('lnLtest', self.cf.l_cl, np.abs(w_model_mean), xlabel="l", ylabel=r'$C_{l}$', colors=color, ymin=1e-7, ymax=1e10, lw=2)
            figs.plot_cl('lnLtest', self.cf.l_cl, np.abs(w_model_std), xlabel="l", ylabel=r'$C_{l}$', colors=color, ymin=1e-7, ymax=1e10, lw=1)
            #figs.plot_cl('lnLtest', self.cf.l_cl, np.abs(w_model_mean - w_data), xlabel="l", ylabel=r'$C_{l}$', colors=color, ymin=1e-7, ymax=1e10, lw=2)
            figs.plot_cl('lnLtest', self.cf.l_cl, np.abs(w_data), colors=color, alpha=0.1, ymin=1e-7, ymax=1e10, lw=3)
            figs.save_all(testfigpath, 'pdf')


        return np.sum(lnL_array[:,lmin:], axis=1)[energyBin]

    def log_likelihood_nullHypo(self, datamap, lmin):
        w_data = self.cf.crossCorrelationFromCountsmap(datamap)
        w_model_mean = self.w_atm_mean * 0.
        w_model_std_square = self.w_atm_std_square
        lnL_array = -(w_data - w_model_mean) ** 2 / w_model_std_square
        return np.sum(lnL_array[:, lmin:], axis=1)

    def TS(self, f, datamap, lmin):
        return 2 * (self.log_likelihood(f, datamap, lmin) - self.log_likelihood_nullHypo(datamap, lmin))


    def minimize__lnL(self, datamap, lmin, energyBin):
        nll = lambda *args: -self.log_likelihood(*args)
        initial = np.array([1, 0.6 ]) + 0.1 * np.random.randn(2)
        soln = minimize(nll, initial, args=(datamap, lmin, energyBin))
        fd_ml, fgal_ml = soln.x
        return fd_ml, fgal_ml







    def log_prior(self, f):
        f_diff, f_gal = f
        if 0. < f_diff < 2. and 0. < f_gal < 1.:
            return 0.
        return -np.inf

    def log_probability(self, f, datamap, energyBin, lmin):
        lp = self.log_prior(f)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(f, datamap, energyBin, lmin)



    def runMCMC(self, datamap, energyBin, Nwalker, Nstep=500, lmin=10):
        ndim = 2
        pos = np.array([1., 0.6]) + np.random.randn(Nwalker, ndim) * 1e-2
        nwalkers, ndim = pos.shape
        backend =  emcee.backends.HDFBackend(os.path.join(Defaults.NUXGAL_PLOT_DIR, 'test.h5'))
        backend.reset(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability, args=(datamap, energyBin, lmin), backend=backend)
        sampler.run_mcmc(pos, Nstep, progress=True);



    def plotMCMCchain(self, ndim, labels, truths):
        reader = emcee.backends.HDFBackend(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'test.h5'))
        fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
        samples = reader.get_chain()

        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number");
        fig.savefig(os.path.join(Defaults.NUXGAL_PLOT_DIR,'MCMCchain.pdf'))


        flat_samples = reader.get_chain(discard=100, thin=15, flat=True)
        #print(flat_samples.shape)
        fig = corner.corner(flat_samples, labels=labels, truths=truths)
        fig.savefig(os.path.join(Defaults.NUXGAL_PLOT_DIR, 'MCMCcorner.pdf'))
