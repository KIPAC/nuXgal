"""Test utility for Likelihood fitting"""

import os

import numpy as np

import healpy as hp

from scipy.optimize import minimize

import emcee

from KIPAC.nuXgal import Defaults

from KIPAC.nuXgal.EventGenerator import EventGenerator

from KIPAC.nuXgal.Likelihood import Likelihood

from KIPAC.nuXgal.file_utils import read_maps_from_fits, write_maps_to_fits

from KIPAC.nuXgal.hp_utils import vector_apply_mask, vector_apply_mask_hp

from KIPAC.nuXgal.plot_utils import FigureDict

from KIPAC.nuXgal.GalaxySample import GalaxySample
from KIPAC.nuXgal.NeutrinoSample import NeutrinoSample

from KIPAC.nuXgal.WeightedAeff import WeightedAeff

try:
    from Utils import MAKE_TEST_PLOTS
except ImportError:
    from .Utils import MAKE_TEST_PLOTS


from scipy import stats

#WeightedAeff(year='IC86-2012', computeTables=True)
#WeightedAeff(year='IC86-2011', computeTables=True)
#WeightedAeff(year='IC79-2010', computeTables=True)

#gs = GalaxySample(galaxyName='analy', computeGalaxy=True)
#gs.generateGalaxy(N_g = 2000000)

testfigpath = os.path.join(Defaults.NUXGAL_PLOT_DIR, 'test')
N_yr = 1

testfigpath = os.path.join(Defaults.NUXGAL_PLOT_DIR, 'Fig_')
countsmappath = os.path.join(Defaults.NUXGAL_DATA_DIR, 'IceCube3yr_countsmap{i}.fits')
IC3yr = NeutrinoSample()
IC3yr.inputData(countsmappath)
IC3yr.updateMask(Defaults.idx_muon)


galaxyName = 'WISE'
llh = Likelihood(N_yr=N_yr,  galaxyName=galaxyName, computeSTD=False, N_re=300)

#datamap = llh.eg.SyntheticData(N_yr, f_diff=0., density_nu = llh.gs.getDensity('WISE'))
#datamap = vector_apply_mask(datamap, Defaults.idx_muon, copy=False)




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
    axes.set_ylim(1e-6, 1e-1)

    w_atm_std_file = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'w_atm_std.txt'))
    w_astro_std_file = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'w_astro_std.txt'))

    Natm = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'Ncount_atm_after_masking.txt'))
    Nastro = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'Ncount_astro_after_masking.txt'))


    axes.plot(Defaults.ell, w_atm_std_file[3], lw=2, label='atm-3')
    axes.plot(Defaults.ell, w_atm_std_file[2], label='atm-2')
    axes.plot(Defaults.ell, w_astro_std_file[2], label='astro-2')
    axes.plot(Defaults.ell, w_astro_std_file[3], label='astro-3')
    axes.plot(Defaults.ell, w_atm_std_file[1] * (Natm[1] / Natm[2])**0.5, label='atm-2 from atm 1')
    axes.plot(Defaults.ell, w_atm_std_file[1] * (Natm[1] / Natm[3])**0.5, lw=1, label='atm-3 from atm 1')
    axes.plot(Defaults.ell, w_atm_std_file[1] * (Natm[1] / Nastro[2])**0.5, label='astro-2 from atm 1')
    axes.plot(Defaults.ell, w_atm_std_file[1] * (Natm[1] / Nastro[3])**0.5, label='astro-3 from atm 1')


    axes.plot(Defaults.ell, llh.w_model_f1[0], label='galaxy')

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

    TS_bins = np.linspace(0, 100, 201)
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
    axes.plot(TS_bins_c, 0.5+ stats.chi2.cdf(TS_bins_c, 1)/2,'--', lw=2, label='chi2, dof=3')
    #axes.plot(TS_bins_c,  stats.chi2.cdf(TS_bins_c, 2),'--', lw=2, label='chi2, dof=2')


    fig.legend()
    figs.save_all(testfigpath, 'pdf')




def testMCMC(datamap, Ncount_astro):
    #w_data = llh.cf.crossCorrelationFromCountsmap(datamap)
    datamap = vector_apply_mask(datamap, llh.idx_mask, copy=False)
    lmin = 50
    ns = NeutrinoSample()
    w_data = ns.getCrossCorrelation_countsmap(datamap, llh.gs.overdensity, llh.idx_mask)
    Ncount = np.sum(datamap, axis=1)
    Ebinmin = 1
    Ebinmax = 4 # np.min([np.where(Ncount != 0)[0][-1]+1, 5])
    print (Ncount)
    print ((llh.minimize__lnL(w_data, Ncount, lmin=lmin, Ebinmin=Ebinmin, Ebinmax=Ebinmax))[0])
    print ((llh.minimize__lnL(w_data, Ncount, lmin=lmin, Ebinmin=Ebinmin, Ebinmax=Ebinmax))[-1])


    llh.runMCMC(w_data, Ncount, lmin=lmin, Ebinmin=Ebinmin, Ebinmax=Ebinmax, Nwalker=640, Nstep=500)



    ndim = Ebinmax - Ebinmin
    labels = []
    truths = []
    for i in range(ndim):
        labels.append('f' + str(i))
        truths.append(Ncount_astro[i])
    llh.plotMCMCchain(ndim, labels, truths)



if __name__ == '__main__':

    #test_STDdependence(2, 0)

    #test_TS_distribution(False)
    #exit(0)
    #datamap = EventGenerator().SyntheticData(N_yr, f_diff=1., density_nu = llh.gs.density)

    f_atm = np.zeros(Defaults.NEbin)
    f_diff = 1.0
    eg = EventGenerator()
    for i in range(Defaults.NEbin):
        if eg.nevts[i] != 0.:
            f_atm[i] = 1. - eg.Nastro_1yr_Aeffmax[i] * f_diff / eg.nevts[i]
            #print (eg.Nastro_1yr_Aeffmax[i] , eg.nevts[i])
    # generate atmospheric eventmaps
    Natm = np.random.poisson(eg.nevts * N_yr * f_atm)
    eg._atm_gen.nevents_expected.set_value(Natm, clear_parent=False)
    atm_map = eg._atm_gen.generate_event_maps(1)[0]

    # generate astro maps
    Nastro = np.random.poisson(eg.Nastro_1yr_Aeffmax * N_yr * f_diff)
    astro_map = eg.astroEvent_galaxy(Nastro, llh.gs.density)

    #datamap = atm_map + astro_map
    astro_ = vector_apply_mask(astro_map, llh.idx_mask, copy=False)
    Ncount_astro = np.sum(astro_, axis=1)

    datamap = IC3yr.countsmap

    testMCMC(datamap, Ncount_astro)
    #showDataModel(datamap, energyBin=3)
