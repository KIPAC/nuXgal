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
from .GalaxySample import GalaxySample
from .hp_utils import vector_apply_mask_hp, vector_apply_mask


class Likelihood():
    def __init__(self, N_yr, galaxyName, computeSTD, N_re=40):
        self.eg = EventGenerator()
        self.gs = GalaxySample(galaxyName)
        self.cf = Analyze()
        self.anafastMask()

        # scaled mean and std
        self.calculate_w_mean()


        # compute or load w_atm distribution
        if computeSTD:
            self.computeAtmophericEventDistribution(N_yr, N_re, True)
        else:
            w_atm_mean_file = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'w_atm_mean.txt'))
            self.w_atm_mean  = w_atm_mean_file.reshape((Defaults.NEbin, Defaults.NCL))
            w_atm_std_file = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'w_atm_std.txt'))
            self.w_atm_std  = w_atm_std_file.reshape((Defaults.NEbin, Defaults.NCL))
            self.w_atm_std_square  =  self.w_atm_std ** 2
            self.Ncount_atm = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'Ncount_atm_after_masking.txt'))

        self.w_std_square0 = np.zeros((Defaults.NEbin, Defaults.NCL))
        for i in range(3):
            self.w_std_square0[i] = self.w_atm_std_square[0] * self.Ncount_atm[0]
        for i in [3, 4]:
            self.w_std_square0[i] = self.w_atm_std_square[3] * self.Ncount_atm[3]

    def anafastMask(self):

        """ mask Southern sky to avoid muons """
        mask_nu = np.zeros(Defaults.NPIXEL, dtype=np.bool)
        mask_nu[Defaults.idx_muon] = 1.
        """ add the mask of galaxy sample """
        mask_nu[self.gs.idx_galaxymask] = 1.
        self.idx_mask = np.where(mask_nu != 0)


    def calculate_w_mean(self):
        overdensity_g = self.gs.overdensity.copy()
        overdensity_g[self.idx_mask] = hp.UNSEEN
        w_mean = hp.anafast(overdensity_g)
        self.w_model_f1 = np.zeros((Defaults.NEbin, Defaults.NCL))
        for i in range(Defaults.NEbin):
                self.w_model_f1[i] = w_mean


    def computeAtmophericEventDistribution(self, N_yr, N_re, writeMap):

        w_cross = np.zeros((N_re, Defaults.NEbin, 3 * Defaults.NSIDE))
        Ncount = np.zeros(Defaults.NEbin)

        for iteration in np.arange(N_re):
            print("iter ", iteration)
            eventnumber_Ebin = np.random.poisson(self.eg.nevts * N_yr)
            self.eg._atm_gen.nevents_expected.set_value(eventnumber_Ebin, clear_parent=False)
            eventmap_atm = self.eg._atm_gen.generate_event_maps(1)[0]
            # first mask makes counts in masked region zero, for correct counting of event number. Second mask applies to healpy cross correlation calculation.
            eventmap_atm = hp_utils.vector_apply_mask(eventmap_atm, self.idx_mask, copy=False)
            w_cross[iteration] = self.cf.crossCorrelationFromCountsmap_mask( eventmap_atm, self.gs.overdensity, self.idx_mask )
            Ncount = Ncount + np.sum(eventmap_atm, axis=1)

        self.w_atm_mean = np.mean(w_cross, axis=0)
        self.w_atm_std = np.std(w_cross, axis=0)
        self.Ncount_atm = Ncount / float(N_re)


        self.w_atm_std_square = self.w_atm_std ** 2
        if writeMap:
            np.savetxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'w_atm_mean.txt'), self.w_atm_mean)
            np.savetxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'w_atm_std.txt'), self.w_atm_std)
            np.savetxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'Ncount_atm_after_masking.txt'), self.Ncount_atm)




    def log_likelihood(self, f, w_data, Ncount, lmin, Ebinmin, Ebinmax):
        w_model_mean = (self.w_model_f1[Ebinmin : Ebinmax].T * f).T
        w_model_std_square = (self.w_std_square0[Ebinmin : Ebinmax].T / Ncount[Ebinmin : Ebinmax]).T

        lnL_le = - (w_data[Ebinmin : Ebinmax] - w_model_mean) ** 2 / w_model_std_square / 2.
        return np.sum(lnL_le[:, lmin:])

    def minimize__lnL(self, w_data, Ncount, lmin, Ebinmin, Ebinmax):
        len_f = Ebinmax - Ebinmin
        nll = lambda *args: -self.log_likelihood(*args)
        initial = 0.5 + 0.1 * np.random.randn(len_f)
        soln = minimize(nll, initial, args=(w_data, Ncount, lmin, Ebinmin, Ebinmax), bounds=[(0, 4)] * (len_f))
        return soln.x, (self.log_likelihood(soln.x, w_data, Ncount, lmin, Ebinmin, Ebinmax) - self.log_likelihood(np.zeros(len_f), w_data, Ncount, lmin, Ebinmin, Ebinmax)) * 2

    def TS_distribution(self, N_re, N_yr, lmin, f_diff, galaxyName, writeData=True):
        TS_array = np.zeros(N_re)
        for i in range(N_re):
            datamap = self.eg.SyntheticData(N_yr, f_diff=f_diff, density_nu=self.gs.density)
            datamap = vector_apply_mask(datamap, self.idx_mask, copy=False)
            w_data = self.cf.crossCorrelationFromCountsmap_mask( datamap, self.gs.overdensity, self.idx_mask )
            Ncount = np.sum(datamap, axis=1)
            Ebinmax = np.min([np.where(Ncount != 0)[0][-1]+1, 5])
            minimizeResult =  (self.minimize__lnL(w_data, Ncount, lmin, 0, Ebinmax))
            print (i, Ncount, minimizeResult[0], minimizeResult[-1])
            TS_array[i] = minimizeResult[-1]
        if writeData:
            np.savetxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR,'TS_'+str(f_diff)+'_'+galaxyName+'.txt'), TS_array)
        return TS_array


    def log_prior(self, f):
        if np.min(f) > 0. and np.max(f) < 1.5:
            return 0.
        return -np.inf


    def log_probability(self, f, w_data, Ncount, lmin, Ebinmin, Ebinmax):
        lp = self.log_prior(f)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(f, w_data, Ncount, lmin, Ebinmin, Ebinmax)



    def runMCMC(self, w_data, Ncount, lmin, Ebinmin, Ebinmax, Nwalker, Nstep=500):
        ndim = Ebinmax - Ebinmin
        pos = 0.3 + np.random.randn(Nwalker, ndim) * 0.1
        nwalkers, ndim = pos.shape
        backend =  emcee.backends.HDFBackend(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'test.h5'))
        backend.reset(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability, args=(w_data, Ncount, lmin, Ebinmin, Ebinmax), backend=backend)
        sampler.run_mcmc(pos, Nstep, progress=True)



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
