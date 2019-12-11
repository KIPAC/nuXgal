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

from KIPAC.nuXgal.hp_utils import vector_apply_mask

from KIPAC.nuXgal.plot_utils import FigureDict

from KIPAC.nuXgal.GalaxySample import GalaxySample

from Utils import MAKE_TEST_PLOTS

from scipy import stats

testfigpath = os.path.join(Defaults.NUXGAL_PLOT_DIR, 'test')
N_yr = 3

llh = Likelihood(N_yr=N_yr, computeATM=True, computeASTRO =True, galaxyName='analy', N_re=100)
#llh = Likelihood(N_yr=N_yr, computeATM=False, computeASTRO =False, galaxyName='analy', N_re=100)




def showDataModel(datamap, energyBin):
    datamap = vector_apply_mask(datamap, Defaults.mask_muon, copy=False)
    w_data = llh.cf.crossCorrelationFromCountsmap(datamap)

    if MAKE_TEST_PLOTS:
        figs = FigureDict()
        o_dict = figs.setup_figure('test_w_CL', xlabel='$l$', ylabel='$C_l$', figsize=(8, 6))
        fig = o_dict['fig']
        axes = o_dict['axes']
        axes.set_ylim(-3e-4, 3e-4)
        axes.set_xlim(0, 400)

        axes.plot(Defaults.ell, llh.gs.analyCL[0:Defaults.NCL] * 0.6 ** 0.5, 'r')
        axes.errorbar(Defaults.ell,  w_data[energyBin], yerr=llh.w_atm_std[0] * (llh.Ncount_atm[0] / np.sum(datamap[energyBin]))**0.5, markersize=4, color='grey', fmt='s', capthick=2)

        figs.save_all(testfigpath, 'pdf')




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


def test_STDdependence(energyBin, energyBin2):
    figs = FigureDict()
    o_dict = figs.setup_figure('compare_std', xlabel='$l$', ylabel='$C_l$', figsize=(8, 6))
    fig = o_dict['fig']
    axes = o_dict['axes']
    axes.set_yscale('log')
    axes.set_ylim(1e-8, 1e-1)

    #w_atm_std_file = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'w_atm_std.txt'))
    #w_atm_std_file2 = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, '10yr/w_atm_std.txt'))
    #axes.plot(Defaults.ell, w_atm_std_file[0], label='new')
    #axes.plot(Defaults.ell, w_atm_std_file2[0], label='old')

    axes.plot(Defaults.ell, llh.w_astro_std[energyBin], label='actual')
    axes.plot(Defaults.ell, llh.w_atm_std[energyBin2] * (llh.Ncount_atm[energyBin2] / llh.Ncount_astro[energyBin])**0.5, label='est')

    axes.plot(Defaults.ell, np.abs(llh.w_astro_mean[energyBin]), label='astro mean')
    axes.plot(Defaults.ell, np.abs(llh.w_atm_mean[energyBin]), label='atm mean')
    axes.plot(Defaults.ell, llh.gs.analyCL[0:Defaults.NCL] / 0.3, label='analy')
    axes.plot(Defaults.ell, llh.gs.WISE_galaxymap_overdensity_cl, label='WISE cl')

    fig.legend()
    figs.save_all(testfigpath, 'pdf')




def test_TS_distribution(readfile = True):
    lmin = 50
    N_re = 200
    if not readfile:
        llh.TS_distribution(N_re, N_yr, lmin, f_diff=0, galaxyName='analy')
        llh.TS_distribution(N_re, N_yr, lmin, f_diff=1, galaxyName='analy')
        #llh.TS_distribution(N_re, N_yr, lmin, 1, 'nonGal')

    TS_atm = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR,'TS_0_analy.txt'))
    #TS_nonGal = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR,'TS_1_nonGal.txt'))
    TS_Gal = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR,'TS_1_analy.txt'))

    TS_bins = np.linspace(0, 100, 101)
    TS_bins_c = (TS_bins[0:-1] + TS_bins[1:]) / 2.
    p_atm, _ = np.histogram(TS_atm, TS_bins, density=True)
    #p_nonGal, _ = np.histogram(TS_nonGal, TS_bins, density=True)
    p_Gal, _ = np.histogram(TS_Gal, TS_bins, density=True)


    figs = FigureDict()
    o_dict = figs.setup_figure('TS_distribution', xlabel='TS', ylabel='cumulative probability', figsize=(8, 6))
    fig = o_dict['fig']
    axes = o_dict['axes']
    axes.set_xlim(-5, 40)

    axes.plot(TS_bins_c, np.cumsum(p_atm), lw=2, label='atm')
    #axes.plot(TS_bins_c, 1 - np.cumsum(p_atm), lw=2, label='atm')
    #axes.plot(TS_bins_c, 1 - np.cumsum(p_nonGal), lw=2, label='nonGal')
    axes.plot(TS_bins_c, np.cumsum(p_Gal), lw=3, label='Gal')

    axes.plot(TS_bins_c, stats.chi2.cdf(TS_bins_c, 3), lw=2, label='chi2, dof=3')
    axes.plot(TS_bins_c, stats.chi2.cdf(TS_bins_c, 4), lw=2, label='chi2, dof=4')

    fig.legend()
    figs.save_all(testfigpath, 'pdf')




def testMCMC(datamap):
    w_data = llh.cf.crossCorrelationFromCountsmap(datamap)
    Ncount = np.sum(datamap, axis=1)
    Ebinmax = np.min([np.where(Ncount != 0)[0][-1]+1, 5])
    print (Ncount)
    print ((llh.minimize__lnL(w_data, Ncount, lmin=20, Ebinmin=0, Ebinmax=Ebinmax))[0])
    print ((llh.minimize__lnL(w_data, Ncount, lmin=20, Ebinmin=0, Ebinmax=Ebinmax))[-1])


    llh.runMCMC(w_data, Ncount, lmin=20, Ebinmin=0, Ebinmax=Ebinmax, Nwalker=640, Nstep=500)

    ndim = Ebinmax
    labels = []
    truths = []
    for i in range(ndim):
        labels.append('f' + str(i))
        truths.append(1)
    llh.plotMCMCchain(ndim, labels, truths)



if __name__ == '__main__':


    #datamap = llh.eg.SyntheticData(N_yr, f_diff=0., density_nu = llh.gs.getDensity('WISE'))
    #datamap = vector_apply_mask(datamap, Defaults.mask_muon, copy=False)

    #test_STDdependence(2, 0)
    #testMCMC(datamap)
    #showDataModel(datamap, energyBin=3)

    test_TS_distribution(False)
