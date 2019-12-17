"""Test utility for Likelihood fitting"""

import os

import numpy as np

import healpy as hp

from scipy.optimize import minimize

import emcee

from KIPAC.nuXgal import Defaults

from KIPAC.nuXgal.EventGenerator import EventGenerator

from KIPAC.nuXgal.Analyze import Analyze

from KIPAC.nuXgal.Likelihood import Likelihood

from KIPAC.nuXgal.file_utils import read_maps_from_fits, write_maps_to_fits

from KIPAC.nuXgal.hp_utils import vector_apply_mask, vector_apply_mask_hp

from KIPAC.nuXgal.plot_utils import FigureDict

from KIPAC.nuXgal.GalaxySample import GalaxySample

try:
    from Utils import MAKE_TEST_PLOTS
except ImportError:
    from .Utils import MAKE_TEST_PLOTS


from scipy import stats

testfigpath = os.path.join(Defaults.NUXGAL_PLOT_DIR, 'test')
N_yr = 10

#gs = GalaxySample(galaxyName='analy', computeGalaxy=True)
#gs.generateGalaxy(N_g = 2000000)

galaxyName = 'WISE'
llh = Likelihood(N_yr=N_yr,  galaxyName=galaxyName, computeSTD=False, N_re=100)

datamap = llh.eg.SyntheticData(N_yr, f_diff=0., density_nu = llh.gs.getDensity('WISE'))
datamap = vector_apply_mask(datamap, Defaults.idx_muon, copy=False)




def plotLnL(w_data, Ncount, lmin, energyBin):
    figs = FigureDict()
    o_dict = figs.setup_figure('test__lnL', xlabel='$l$', ylabel='$C_l$', figsize=(8, 6))
    fig = o_dict['fig']
    axes = o_dict['axes']
    ftest = np.linspace(0, 1, 50)
    lnL_f = []
    for _f in ftest:
        lnL_f.append(llh.lnL([_f], w_data, Ncount, lmin, energyBin, energyBin+1))
    axes.plot(ftest, lnL_f)
    figs.save_all(testfigpath, 'pdf')


def test_STDdependence(energyBin=2, energyBin2=0):
    figs = FigureDict()
    o_dict = figs.setup_figure('compare_std', xlabel='$l$', ylabel='$C_l$', figsize=(8, 6))
    fig = o_dict['fig']
    axes = o_dict['axes']
    axes.set_yscale('log')
    axes.set_ylim(1e-8, 1e-1)

    w_atm_std_file = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'w_atm_std.txt'))
    #w_atm_std_file2 = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, '10yr/w_atm_std.txt'))
    axes.plot(Defaults.ell, w_atm_std_file[0], label='new')
    #axes.plot(Defaults.ell, w_atm_std_file2[0], label='old')

    fig.legend()
    figs.save_all(testfigpath, 'pdf')




def test_TS_distribution(readfile = False):
    lmin = 50
    N_re = 200
    if not readfile:
        #llh.TS_distribution(N_re, N_yr, lmin, f_diff=0, galaxyName=galaxyName)
        llh.TS_distribution(N_re, N_yr, lmin, f_diff=1, galaxyName=galaxyName)

    TS_atm = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR,'TS_0_'+galaxyName+'.txt'))
    TS_Gal = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR,'TS_1_'+galaxyName+'.txt'))

    TS_bins = np.linspace(0, 50, 101)
    TS_bins_c = (TS_bins[0:-1] + TS_bins[1:]) / 2.
    dTS_bins = np.mean(TS_bins[1:] - TS_bins[0:-1])

    p_atm = np.histogram(TS_atm, TS_bins)[0] / float(len(TS_atm))
    #p_atm = np.histogram(TS_atm, TS_bins)[0] / float(N_re) / dTS_bins
    p_Gal = np.histogram(TS_Gal, TS_bins)[0] / float(len(TS_Gal))



    figs = FigureDict()
    o_dict = figs.setup_figure('TS_distribution', xlabel='TS', ylabel='cumulative probability', figsize=(8, 6))
    fig = o_dict['fig']
    axes = o_dict['axes']
    axes.set_xlim(-5, 30)
    axes.set_yscale('log')
    #axes.set_ylim(1e-4, 1.2)

    #axes.plot(TS_bins_c,  p_atm, lw=3, label='atm')
    #axes.plot(TS_bins_c, stats.chi2.pdf(TS_bins_c, 2),'--', lw=2, label='chi2, dof=2')

    axes.plot(TS_bins_c,  np.cumsum(p_atm), lw=2, label='atm')
    axes.plot(TS_bins_c,  np.cumsum(p_Gal), lw=2, label='Gal')
    axes.plot(TS_bins_c, 0.5+ stats.chi2.cdf(TS_bins_c, 4)/2,'--', lw=2, label='chi2, dof=4')
    #axes.plot(TS_bins_c,  stats.chi2.cdf(TS_bins_c, 2),'--', lw=2, label='chi2, dof=2')


    fig.legend()
    figs.save_all(testfigpath, 'pdf')




def testMCMC(datamap Ncount_astro):
    w_data = llh.cf.crossCorrelationFromCountsmap(datamap)
    #datamap = vector_apply_mask(datamap, llh.idx_mask, copy=False)
    lmin = 50
    w_data = llh.cf.crossCorrelationFromCountsmap_mask(datamap, llh.gs.overdensity, llh.idx_mask)
    Ncount = np.sum(datamap, axis=1)
    Ebinmax = np.min([np.where(Ncount != 0)[0][-1]+1, 5])
    print (Ncount)
    print ((llh.minimize__lnL(w_data, Ncount, lmin=lmin, Ebinmin=0, Ebinmax=Ebinmax))[0])
    print ((llh.minimize__lnL(w_data, Ncount, lmin=lmin, Ebinmin=0, Ebinmax=Ebinmax))[-1])


    llh.runMCMC(w_data, Ncount, lmin=lmin, Ebinmin=0, Ebinmax=Ebinmax, Nwalker=640, Nstep=500)


    f_astro = Ncount_astro / Ncount

    print (f_astro)
    ndim = Ebinmax
    labels = []
    truths = []
    for i in range(ndim):
        labels.append('f' + str(i))
        truths.append(f_astro[i])
    llh.plotMCMCchain(ndim, labels, truths)



if __name__ == '__main__':

    #test_STDdependence(2, 0)

    #test_TS_distribution(False)

    #datamap = llh.eg.SyntheticData(N_yr, f_diff=1., density_nu = llh.gs.density)
    f_atm = np.zeros(Defaults.NEbin)
    f_diff = 1.
    for i in range(Defaults.NEbin):
        if llh.eg.nevts[i] != 0.:
            f_atm[i] = 1. - llh.eg.Nastro_1yr_Aeffmax[i] * f_diff / llh.eg.nevts[i]
    # generate atmospheric eventmaps
    Natm = np.random.poisson(llh.eg.nevts * N_yr * f_atm)
    llh.eg._atm_gen.nevents_expected.set_value(Natm, clear_parent=False)
    atm_map = llh.eg._atm_gen.generate_event_maps(1)[0]

    # generate astro maps
    Nastro = np.random.poisson(llh.eg.Nastro_1yr_Aeffmax * N_yr * f_diff)
    astro_map = llh.eg.astroEvent_galaxy(Nastro, llh.gs.density)

    datamap = atm_map + astro_map
    astro_ = vector_apply_mask(astro_map, llh.idx_mask, copy=False)
    Ncount_astro = np.sum(astro_, axis=1)


    testMCMC(datamap, Ncount_astro)
    #showDataModel(datamap, energyBin=3)
