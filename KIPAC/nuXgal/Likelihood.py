
import os
import numpy as np
import healpy as np
import emcee
import corner
import matplotlib.pyplot as plt

from .Utilityfunc import *
from .EventGenerator import *
from .Analyze import *
from . import Defaults


class Likelihood():
    def __init__(self, computeStdATM = False, Nreal=100):
        self.cf = Analyze()
        self.eg = EventGenerator()
        # compute or load w_atm distribution
        if computeStdATM:
            self.computeAtmophericEventDistribution(Nreal)
        else:
            w_atm_file = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'w_atm.txt'))
            self.w_atm_std  = w_atm_file.reshape((Defaults.NEbin, 384))
            
        
        # calculate expected event number using IceCube diffuse neutrino flux
        dN_dE_astro = lambda E_GeV: 1.44E-18 * (E_GeV / 100e3)**(-2.28) # GeV^-1 cm^-2 s^-1 sr^-1, muon neutrino
        # total expected number of events before cut, for one year data
        self.N_2012_Aeffmax = np.zeros(Defaults.NEbin)
        for i in np.arange(Defaults.NEbin):
            self.N_2012_Aeffmax[i] = dN_dE_astro(10.**Defaults.map_logE_center[i]) * (10. ** Defaults.map_logE_center[i] * np.log(10.) * Defaults.dlogE) * (self.eg.Aeff_max[i] * 1E4) * (333 * 24. * 3600) * 4 * np.pi

        
    def computeAtmophericEventDistribution(self, Nreal, writeMap = True):
        w_atm_mean, self.w_atm_std = self.cf.crossCorrelation_atm_std(Nreal)
        if writeMap:
            np.savetxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'w_atm.txt'), self.w_atm_std)
           
           
       

    def log_likelihood(self, f, datamap, energyBin, Nyr, lmin, Nreal):
        f_diff, f_gal = f
        seed_g = 42
        density_nu = density_cl(self.cf.cl_galaxy * f_gal, Defaults.NSIDE, seed_g)
        density_nu = np.exp(density_nu) - 1.0
        w_astro_N = np.zeros((Nreal, Defaults.NEbin, 3 * Defaults.NSIDE))
        for iter in np.arange(Nreal):
            astromap = self.eg.astroEvent_galaxy(density_nu, self.N_2012_Aeffmax * f_diff, False) + self.eg.atmEvent(Nyr)
            w_astro_N[iter] = self.cf.crossCorrelationFromCountsmap(astromap)
        w_astro_mean = np.mean(w_astro_N, axis=0)
        w_data = self.cf.crossCorrelationFromCountsmap(datamap)
        lnL_array = (w_data - w_astro_mean) ** 2 / self.w_atm_std ** 2
        #print (lnL_array[4][lmin:])
        lnL = 0.
        for i in energyBin:
            lnL += np.sum( lnL_array[i][lmin:] )
        return lnL
        


    def log_prior(self, f):
        f_diff, f_gal = f
        if 0. < f_diff < 2. and 0. < f_gal < 1.:
            return 0.
        return -np.inf
    
    def log_probability(self, f, datamap, energyBin, Nyr, lmin, Nreal):
        lp = self.log_prior(f)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(f, datamap, energyBin, Nyr, lmin, Nreal)
        
        
    """
    def runMCMC(self, datamap, energyBin, Nyr, Nwalker, Nstep=500, lmin=10, Nreal=20):
        ndim = 2
        pos = np.array([1., 0.6]) + np.random.randn(Nwalker, ndim) * 1e-2
        nwalkers, ndim = pos.shape
        backend =  emcee.backends.HDFBackend(os.path.join(NUXGAL_PLOT_DIR, 'test.h5'))
        backend.reset(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability, args=(datamap, energyBin, Nyr, lmin, Nreal), backend=backend)
        sampler.run_mcmc(pos, Nstep, progress=True);
        
        
        
    def plotMCMCchain(self):
        reader = emcee.backends.HDFBackend(os.path.join(NUXGAL_PLOT_DIR, 'test.h5'))
        fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
        samples = reader.get_chain()
        labels = ["f_diff", "f_gal"]
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number");
        fig.savefig('check.pdf')


        flat_samples = reader.get_chain(discard=100, thin=15, flat=True)
        #print(flat_samples.shape)
        fig = corner.corner(flat_samples, labels=['f_diff', 'f_gal'], truths=[1.0, 0.6])
        fig.savefig('corner.pdf')

    """




        
