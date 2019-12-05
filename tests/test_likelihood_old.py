"""Test utility for Likelihood fitting"""

import os

import numpy as np

import healpy as hp

from KIPAC.nuXgal import Defaults

from KIPAC.nuXgal.EventGenerator import EventGenerator

from KIPAC.nuXgal.Analyze import Analyze

from KIPAC.nuXgal.Likelihood import Likelihood

from KIPAC.nuXgal.file_utils import read_maps_from_fits, write_maps_to_fits

from KIPAC.nuXgal.hp_utils import vector_apply_mask

from KIPAC.nuXgal.plot_utils import FigureDict

try:
    from .Utils import MAKE_TEST_PLOTS
except:
    from Utils import MAKE_TEST_PLOTS

testfigpath = os.path.join(Defaults.NUXGAL_PLOT_DIR, 'test')
N_yr = 1.

llh = Likelihood(N_yr=N_yr)#, computeATM=False, computeASTRO =True, N_re=20)


def generateData(f_diff, f_gal, N_yr, fromGalaxy, writeMap, basekey='syntheticData'):
    cf = Analyze()
    eg = EventGenerator()
    if fromGalaxy:
        np.random.seed(Defaults.randomseed_galaxy)
    else:
        np.random.seed(Defaults.randomseed_galaxy + 102)


    density_nu = hp.sphtfunc.synfast(cf.cl_galaxy * f_gal, Defaults.NSIDE)
    density_nu = np.exp(density_nu)
    density_nu /= density_nu.sum() # a unique neutrino source distribution that shares the randomness of density_g
    #countsmap = eg.atmEvent(N_yr)
    countsmap = eg.astroEvent_galaxy(llh.N_2012_Aeffmax * N_yr * f_diff, density_nu) + eg.atmEvent(N_yr)
    counts_map = vector_apply_mask(countsmap, Defaults.mask_muon, copy=False)

    if writeMap:
        filename_format = os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, basekey + '{i}.fits')
        write_maps_to_fits(countsmap, filename_format)
    return countsmap


def readData(basekey='syntheticData'):
    filename_format = os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, basekey + '{i}.fits')
    return read_maps_from_fits(filename_format, Defaults.NEbin)



def testLnLDistribution_f(testfdiff = True, Ntest = 50, lmin=0, energyBin=2):
    f_gal_data, f_diff_data = 0.6, 1.
    #datamap = generateData(f_gal_data, f_diff_data, N_yr, True, True)
    datamap = readData()

    lnL = np.zeros(Ntest)

    if testfdiff:
        f_test = np.linspace(1e-3, 2, Ntest)
        for i, _f_diff in enumerate(f_test):
            lnL[i] = llh.log_likelihood([_f_diff, f_gal_data], datamap, lmin, energyBin)
        figkey, xlabel = 'testLnL_fdiff', '$f_{\mathrm{diff}}$'

    else:
        f_test = np.linspace(1e-3, 1, Ntest)
        for i, _f_gal in enumerate(f_test):
            lnL[i] = llh.log_likelihood([f_diff_data, _f_gal], datamap,  lmin,energyBin,)
        figkey, xlabel = 'testLnL_fgal', '$f_{\mathrm{gal}}$'


    if MAKE_TEST_PLOTS:
        figs = FigureDict()
        o_dict = figs.setup_figure(figkey, xlabel=xlabel, ylabel='$\ln\,L$', figsize=(8, 6))
        fig = o_dict['fig']
        axes = o_dict['axes']
        axes.set_xscale('log')
        axes.set_yscale('log')
        axes.plot(f_test, -lnL, lw=2)
        figs.save_all(testfigpath, 'pdf')


def testModels(datamap=generateData(1., 0.6, N_yr, True, False)):
    TS_model1 = llh.TS([1, 0.6], datamap, 30)
    TS_model2 = llh.TS([0, 0.6], datamap, 30)
    TS_model3 = llh.TS([10, 0.6], datamap, 30)

    if MAKE_TEST_PLOTS:
            color = ['r', 'orange', 'limegreen', 'skyblue', 'mediumslateblue', 'purple', 'grey']

            testfigpath = os.path.join(Defaults.NUXGAL_PLOT_DIR, 'test')
            figs = FigureDict()
            o_dict = figs.setup_figure('lnL_E', xlabel="E bin", ylabel='TS', figsize=(8, 6))
            axes = o_dict['axes']
            #axes.set_yscale('log')
            #axes.set_ylim(-10, 10)
            Ebin = np.arange(Defaults.NEbin)
            for i in range(Defaults.NEbin):
                axes.scatter(Ebin[i], TS_model1[i], color=color[i], marker='*')
                axes.scatter(Ebin[i], TS_model2[i], color=color[i], marker='^')
                axes.scatter(Ebin[i], TS_model3[i], color=color[i], marker='s')
            figs.save_all(testfigpath, 'pdf')



def testWcrossEnergybin(energyBin=4, N_re=40, readfile=True):

    if readfile:
        w_astro_mean_file = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'w_astro_mean.txt'))
        w_astro_mean = w_astro_mean_file.reshape((Defaults.NEbin, Defaults.NCL))
        w_astro_std_file = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'w_astro_std.txt'))
        w_astro_std = w_astro_std_file.reshape((Defaults.NEbin, Defaults.NCL))

    else:
        np.random.seed(Defaults.randomseed_galaxy)
        density_nu = hp.sphtfunc.synfast(llh.cf.cl_galaxy * 0.6, Defaults.NSIDE)
        density_nu = np.exp(density_nu)
        density_nu /= density_nu.sum() # a unique neutrino source distribution that shares the randomness of density_g
        w_cross_array = np.zeros((N_re, Defaults.NEbin, Defaults.NCL))
        for i in range(N_re):
            if i % 1 == 0:
                print(i)
            countsmap = llh.eg.astroEvent_galaxy(llh.N_2012_Aeffmax , density_nu)
            countsmap = vector_apply_mask(countsmap, Defaults.mask_muon, copy=False)
            w_cross_array[i] = llh.cf.crossCorrelationFromCountsmap(countsmap)

        w_astro_mean = np.mean(w_cross_array, axis=0)
        w_astro_std = np.std(w_cross_array, axis=0)
        np.savetxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'w_astro_mean.txt'), w_astro_mean)
        np.savetxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'w_astro_std.txt'), w_astro_std)


    w_atm_mean_file = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'w_atm_mean.txt'))
    w_atm_mean  = w_atm_mean_file.reshape((Defaults.NEbin, Defaults.NCL))
    w_atm_std_file = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'w_atm_std.txt'))
    w_atm_std  = w_atm_std_file.reshape((Defaults.NEbin, Defaults.NCL))

    powerSpectrum_g = hp.sphtfunc.anafast(llh.cf.overdensityMap_g)

    if MAKE_TEST_PLOTS:
        figs = FigureDict()
        o_dict = figs.setup_figure('test_w_CL', xlabel='$l$', ylabel='$C_l$', figsize=(8, 6))
        fig = o_dict['fig']
        axes = o_dict['axes']
        #axes.set_xscale('log')
        axes.set_ylim(-3e-4, 3e-4)
        axes.set_xlim(0, 400)

        axes.plot(llh.cf.l, llh.cf.cl_galaxy * 0.6 ** 0.5, 'k-')
        axes.plot(llh.cf.l_cl,  powerSpectrum_g, lw=2, label='g', color='orange')
        #axes.plot(llh.cf.l_cl,  w_astro_mean[energyBin], lw=2, label='nuXg, same random seed')
        #axes.errorbar(llh.cf.l_cl,  w_astro_mean[energyBin], yerr=w_astro_std[energyBin], fmt='o', capthick=2, label='nuXg, same random seed')
        #axes.plot(llh.cf.l_cl, w_atm_mean[energyBin], label='atm')
        axes.errorbar(llh.cf.l_cl,  w_atm_mean[energyBin], yerr=w_atm_std[energyBin], color='grey', fmt='s', capthick=2, label='atm')
        axes.scatter(llh.cf.l_cl,  w_astro_mean[energyBin],marker='o', s=4, color='b', label='nuXg, same random seed',zorder=10)

        fig.legend()
        figs.save_all(testfigpath, 'pdf')

        figs = FigureDict()
        o_dict = figs.setup_figure('test_w_CL_log', xlabel='$l$', ylabel='$C_l$', figsize=(8, 6))
        fig = o_dict['fig']
        axes = o_dict['axes']
        axes.set_yscale('log')
        axes.set_xscale('log')
        axes.set_ylim(3e-6, 3e-4)
        axes.set_xlim(1, 400)

        axes.plot(llh.cf.l, llh.cf.cl_galaxy * 0.6 ** 0.5, 'k-')
        axes.plot(llh.cf.l_cl,  powerSpectrum_g, lw=2, label='g', color='orange')
        #axes.plot(llh.cf.l_cl,  w_astro_mean[energyBin], lw=2, label='nuXg, same random seed')
        #axes.errorbar(llh.cf.l_cl,  w_astro_mean[energyBin], yerr=w_astro_std[energyBin], fmt='o', capthick=2, label='nuXg, same random seed')
        axes.scatter(llh.cf.l_cl,  w_astro_mean[energyBin],marker='o', color='b', label='nuXg, same random seed')
        #axes.plot(llh.cf.l_cl, w_atm_mean[energyBin], label='atm')
        axes.errorbar(llh.cf.l_cl,  w_atm_mean[energyBin], yerr=w_atm_std[energyBin], color='grey', fmt='s', capthick=2, label='atm')
        fig.legend()
        figs.save_all(testfigpath, 'pdf')



        figs = FigureDict()
        o_dict = figs.setup_figure('test_w_CL_lnL', xlabel='$l$', ylabel='TS', figsize=(8, 6))
        fig = o_dict['fig']
        axes = o_dict['axes']
        #axes.set_xscale('log')
        #axes.set_ylim(-10, 400)
        axes.set_xlim(0, 400)

        axes.scatter(llh.cf.l_cl, 2 * (w_astro_mean[energyBin] * 0.1)**2  / (w_atm_std[energyBin] )**2,marker='o', color='b', label='nuXg, same random seed')

        fig.legend()
        figs.save_all(testfigpath, 'pdf')




if __name__ == '__main__':
    #datamap = readData()
    #testWcrossEnergybin(2, N_re = 30)#, readfile=False)


    datamap = generateData(1., 0.6, N_yr, True, False)
    #testModels(datamap)
    #print (llh.minimize__lnL(datamap, 30, 3))

    #datamap = llh.eg.atmEvent(1)
    #figs = FigureDict()
    #figs.mollview_maps('eventmap_atm', datamap)
    #figs.save_all(testfigpath, 'pdf')


    #llh.log_likelihood([1, 0.6], datamap, [2], 0, MAKE_TEST_PLOTS=True)
    testLnLDistribution_f(True)

    #import cProfile
    #cProfile.run('test_likelihood()', 'fit_profile.pstats')
