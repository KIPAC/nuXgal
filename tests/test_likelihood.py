import os

import cProfile

from KIPAC.nuXgal.Likelihood import Likelihood

import matplotlib.pyplot as plt

from KIPAC.nuXgal import Defaults

from KIPAC.nuXgal.EventGenerator import EventGenerator

from KIPAC.nuXgal.Analyze import Analyze

from KIPAC.nuXgal.Utilityfunc import density_cl

import numpy as np

import healpy as hp

font = {'family':'Arial', 'weight':'normal', 'size': 20}
legendfont = {'fontsize':18, 'frameon':False}



llh = Likelihood(True, 100)
#llh.plotMCMCchain()

seed_g = 42
density_nu = density_cl(Analyze().cl_galaxy * 0.6, Defaults.NSIDE, seed_g)
density_nu = np.exp(density_nu) - 1.0
eg = EventGenerator()
# calculate expected event number using IceCube diffuse neutrino flux
dN_dE_astro = lambda E_GeV: 1.44E-18 * (E_GeV / 100e3)**(-2.28) # GeV^-1 cm^-2 s^-1 sr^-1, muon neutrino
# total expected number of events before cut, for one year data
N_2012_Aeffmax = np.zeros(Defaults.NEbin)
for i in np.arange(Defaults.NEbin):
    N_2012_Aeffmax[i] = dN_dE_astro(10.**Defaults.map_logE_center[i]) * (10. ** Defaults.map_logE_center[i] * np.log(10.) * Defaults.dlogE) * (eg.Aeff_max[i] * 1E4) * (333 * 24. * 3600) * 4 * np.pi


datamap = eg.astroEvent_galaxy(density_nu, N_2012_Aeffmax * 1., False) + eg.atmEvent(1.)

def testLnLDistribution_fdiff():
    """
    astromap = np.zeros((Defaults.NEbin, Defaults.NPIXEL))
    for i in np.arange(Defaults.NEbin):
        astromap[i] = hp.fitsfunc.read_map(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR,
                                                        'eventmap_astro' + str(i)+'.fits'),
                                                        verbose=False)
                                                        
    
    datamap = astromap + EventGenerator().atmEvent(1)
    """
    f_diff = np.linspace(0, 2, 100)
    lnL = np.zeros_like(f_diff)
    for i, _f_diff in enumerate(f_diff):
        lnL[i] = llh.log_likelihood([_f_diff, 0.6], datamap, [4], 1., 10, 30)
        
    
    fig = plt.figure(figsize=(8, 6))
    plt.rc('font', **font)
    plt.rc('legend', **legendfont)
    plt.plot(f_diff, lnL, lw=2)
    plt.ylabel(r'$\log L$')
    plt.xlabel('$f_{\mathrm{diff}}$')
    plt.legend()
    plt.savefig(os.path.join(Defaults.NUXGAL_PLOT_DIR, 'llh_f_diff.pdf'))


def testLnLDistribution_fgal():
    """
    astromap = np.zeros((Defaults.NEbin, Defaults.NPIXEL))
    for i in np.arange(Defaults.NEbin):
        astromap[i] = hp.fitsfunc.read_map(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR,
                                                    'eventmap_astro' + str(i)+'.fits'), verbose=False)
    datamap = astromap + EventGenerator().atmEvent(1)
    """
    f_gal = np.linspace(-0.8, 1.3, 100)
    lnL = np.zeros_like(f_gal)
    for i, _f_gal in enumerate(f_gal):
        lnL[i] = llh.log_likelihood([1., _f_gal], datamap, [4], 1., 10, 30)
        
    fig = plt.figure(figsize=(8, 6))
    plt.rc('font', **font)
    plt.rc('legend', **legendfont)
    plt.plot(f_gal, lnL, lw=2)
    plt.ylabel(r'$\log L$')
    plt.xlabel('$f_{\mathrm{gal}}$')
    plt.legend()
    plt.savefig(os.path.join(Defaults.NUXGAL_PLOT_DIR, 'llh_f_gal.pdf'))



testLnLDistribution_fdiff()
testLnLDistribution_fgal()

