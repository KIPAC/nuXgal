import os

import numpy as np

import healpy as hp
import emcee
import corner
import cProfile

import matplotlib.pyplot as plt

from KIPAC.nuXgal import Defaults

from KIPAC.nuXgal.Utilityfunc import significance

from KIPAC.nuXgal.Analyze import Analyze

from KIPAC.nuXgal.Utilityfunc import *

from KIPAC.nuXgal.Likelihood import log_prior

from KIPAC.nuXgal.EventGenerator import EventGenerator

cf = Analyze()
w_cross_mean, w_cross_std = cf.crossCorrelation_atm_std(10)

eg = EventGenerator()
seed_g = 42

cl_galaxy_file = np.loadtxt(os.path.join(Defaults.NUXGAL_ANCIL_DIR, 'Cl_ggRM.dat'))
cl_galaxy = cl_galaxy_file[:500]

# calculate expected event number using IceCube diffuse neutrino flux
dN_dE_astro = lambda E_GeV: 1.44E-18 * (E_GeV / 100e3)**(-2.28) # GeV^-1 cm^-2 s^-1 sr^-1, muon neutrino
# total expected number of events before cut, for one year data
N_2012_Aeffmax = np.zeros(Defaults.NEbin)
for i in np.arange(Defaults.NEbin):
    N_2012_Aeffmax[i] = dN_dE_astro(10.**Defaults.map_logE_center[i]) * (10. ** Defaults.map_logE_center[i] * np.log(10.) * Defaults.dlogE) * (eg.Aeff_max[i] * 1E4) * (333 * 24. * 3600) * 4 * np.pi

datamap = np.zeros((Defaults.NEbin, Defaults.NPIXEL))
for i in np.arange(Defaults.NEbin):
    datamap[i] = hp.fitsfunc.read_map(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'eventmap_astro' + str(i)+'.fits'), verbose=False)
datamap = datamap + eg.atmEvent(1.-0.003)

w_data = cf.crossCorrelationFromCountsmap(datamap)


lmin = 10
energyBin = 4

f_gal = 0.6
density_g = density_cl(cl_galaxy * f_gal, Defaults.NSIDE, seed_g)
density_g = np.exp(density_g) - 1.0



def log_likelihood(f, w_data_i, w_error_i):
    f_diff = f
    
    
    """
    N_real = 5
    w_astro_N = np.zeros((N_real, cf.NEbin, 3 * cf.NSIDE))
    for i in range(N_real):
        astromap = eg.astroEvent_galaxy(density_g, N_2012_Aeffmax * f_diff, False) + eg.atmEvent(1.-0.003)
        w_astro_N[i] = cf.crossCorrelationFromCountsmap(astromap)
    w_astro_mean = np.mean(w_astro_N, axis=0)
    """
    astromap = eg.astroEvent_galaxy(density_g, N_2012_Aeffmax * f_diff, False) + eg.atmEvent(1.-0.003)
    w_astro_mean = cf.crossCorrelationFromCountsmap(astromap)
    
    #w_astro_std = np.std(w_astro_N, axis=0)
    #w_astro = cf.crossCorrelationFromCountsmap(astromap)
    lnLi =  (w_data_i - w_astro_mean[energyBin]) ** 2 / w_error_i ** 2
    lnLi = np.sum(lnLi[lmin:])
    
    #lnLi = np.sum( (w_data_4[lmin:] - np.array(w_astro_mean[4])[lmin:]) ** 2 / np.array(w_cross_std[4])[lmin:] ** 2)
    return lnLi

def log_probability(f, w_data_i, w_error_i):
    lp = log_prior(f)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(f, w_data_i, w_error_i)
    
    

def test_likelihood():




    ndim = 1


    pos = np.array([1.]) + np.random.randn(2, 1) * 1e-2
    nwalkers, ndim = pos.shape


    filename = 'test.h5'

    backend =  emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(w_data[energyBin], w_cross_std[energyBin]),backend=backend)
    sampler.run_mcmc(pos, 100, progress=True);

    reader = emcee.backends.HDFBackend(filename)

    fig, axes = plt.subplots(1, figsize=(10, 7), sharex=True)
    samples = reader.get_chain()
    labels = ["f_diff", "f_gal"]

    ax = axes
    ax.plot(samples[:, :, 0], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[0])
    ax.yaxis.set_label_coords(-0.1, 0.5)

    axes.set_xlabel("step number");
    fig.savefig(os.path.join(Defaults.NUXGAL_PLOT_DIR, 'check.pdf'))


#tau = sampler.get_autocorr_time()
#print(tau)


    flat_samples = reader.get_chain(discard=100, thin=15, flat=True)
    print(flat_samples.shape)




    fig = corner.corner(
        flat_samples, labels=['f_diff', 'f_gal'], truths=[1.0, 0.6]
        )
    fig.savefig(os.path.join(Defaults.NUXGAL_PLOT_DIR, 'corner.pdf'))



cProfile.run('test_likelihood()', 'fit_profile.pstats')
