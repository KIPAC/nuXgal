"""Figures"""

import os
import numpy as np
import healpy as hp
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import emcee

from KIPAC.nuXgal import Defaults
from KIPAC.nuXgal.file_utils import read_maps_from_fits, write_maps_to_fits
from KIPAC.nuXgal.EventGenerator import EventGenerator
from KIPAC.nuXgal.Likelihood import Likelihood, significance
from KIPAC.nuXgal.NeutrinoSample import NeutrinoSample
from KIPAC.nuXgal.Exposure import ICECUBE_EXPOSURE_LIBRARY
from KIPAC.nuXgal.GalaxySample import GALAXY_LIBRARY


font = { 'family': 'Arial', 'weight' : 'normal', 'size'   : 21}
legendfont = {'fontsize' : 21, 'frameon' : False}

testfigpath = os.path.join(Defaults.NUXGAL_PLOT_DIR, 'Fig_')



countsmappath = os.path.join(Defaults.NUXGAL_DATA_DIR, 'IceCube3yr_countsmap{i}.fits')
IC3yr = NeutrinoSample()
IC3yr.inputData(countsmappath)


def CompareNeutrinoMaps(energyBin=2, plotcount=False, plotoverdensity=False, plotpowerspectrum=False, plotcostheta=True):
    IC3yr.updateMask(Defaults.idx_muon)

    # generate synthetic data with astrophysical events from galaxies
    gs = GALAXY_LIBRARY.get_sample('analy')
    # generate data year by year as effective areas are different
    countsmap = np.zeros((Defaults.NEbin, Defaults.NPIXEL))

    for year in ['IC79-2010', 'IC86-2011', 'IC86-2012']:
        print (year)
        eg = EventGenerator(year, 'observed_numu_fraction')
        counts = eg.SyntheticData(N_yr=1., f_diff=1., density_nu=gs.density)
        SyntheticData = NeutrinoSample()
        SyntheticData.inputCountsmap(counts)
        SyntheticData.updateMask(Defaults.idx_muon)
        print (SyntheticData.getEventCounts())
        print ('-------------------')
        countsmap = countsmap + SyntheticData.countsmap_fullsky


    SyntheticData3yr = NeutrinoSample()
    SyntheticData3yr.inputCountsmap(countsmap)
    SyntheticData3yr.updateMask(Defaults.idx_muon)
    print ('IceCube 3yr:', IC3yr.getEventCounts())
    print ('Synthetic data 3yr:', SyntheticData3yr.getEventCounts())

    color = [ 'r', 'orange', 'limegreen', 'dodgerblue', 'mediumslateblue', 'purple', 'grey']
    labels = ['$10^{1.5}-10^{2.5}$ GeV', '$10^{2.5}-10^{3.5}$ GeV', '$10^{3.5}-10^{4.5}$ GeV','$10^{4.5}-10^{5.5}$ GeV','$10^{5.5}-10^{6.5}$ GeV','$10^{6.5}-10^{7.5}$ GeV','$10^{7.5}-10^{8.5}$ GeV']

    if plotcount:
        matplotlib.rc('font', **font)
        matplotlib.rc('legend', **legendfont)
        matplotlib.rcParams.update({'font.size':18})
        matplotlib.rc("text", usetex=True)

        plt.figure(figsize = (8,6))
        hp.mollview(IC3yr.countsmap[energyBin], title='IceCube 3 year 3-30 TeV')
        plt.savefig(testfigpath+'CompareNeutrinoCountsmaps_IC.pdf')
        plt.figure(figsize = (8,6))
        hp.mollview(SyntheticData3yr.countsmap[energyBin], title='Synthetic data')
        plt.savefig(testfigpath+'CompareNeutrinoCountsmaps_synthetic.pdf')


    if plotoverdensity:
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize = (20,6))
        matplotlib.rc('font', **font)
        matplotlib.rc('legend', **legendfont)
        matplotlib.rcParams.update({'font.size':18})
        matplotlib.rc("text", usetex=True)

        plt.axes(ax1)
        hp.mollview(IC3yr.getOverdensity()[energyBin], title='IceCube 3 year', hold=True)
        plt.axes(ax2)
        hp.mollview(SyntheticData3yr.getOverdensity()[energyBin], title='Synthetic data', hold=True)
        plt.savefig(testfigpath+'CompareNeutrinoOverdensitymaps.pdf')

    if plotpowerspectrum:
        plt.figure(figsize = (8,6))
        matplotlib.rc('font', **font)
        matplotlib.rc('legend', **legendfont)
        matplotlib.rc("text", usetex=True)

        w_auto_IC3yr = IC3yr.getPowerSpectrum()
        w_auto_SyntheticData3yr = SyntheticData3yr.getPowerSpectrum()

        # compare to IceCube exposure
        wAeff = ICECUBE_EXPOSURE_LIBRARY.get_exposure()
        wAeff_i = wAeff[2] / wAeff[2].mean() - 1.
        plt.plot(Defaults.ell, hp.anafast(wAeff_i), lw=2, color='grey', label='IceCube effective area')

        for i in [0, 1, 2, 3, 4]:
            plt.plot(Defaults.ell, w_auto_IC3yr[i], lw=3, color=color[i], label=labels[i])
            plt.plot(Defaults.ell, w_auto_SyntheticData3yr[i], lw=1, color=color[i])
        plt.yscale('log')
        plt.ylim(1e-9, 3e0)
        plt.ylabel('$C_\ell$')
        plt.xlabel('$\ell$')
        plt.grid()
        plt.legend(numpoints=1, scatterpoints=1, frameon=True,fontsize=14, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.19))
        plt.subplots_adjust(left=0.14, bottom=0.14, top=0.85)
        plt.savefig(testfigpath+'CompareNeutrinoPowerSpectrum.pdf')

    if plotcostheta:
        plt.figure(figsize = (8,6))
        matplotlib.rc('font', **font)
        matplotlib.rc('legend', **legendfont)
        matplotlib.rc("text", usetex=True)

        cosThetaBin = np.linspace(-1, 1, 50)
        cos_exposuremap_theta = np.cos(Defaults.exposuremap_theta)
        for i in range(Defaults.NEbin-2):
            n_cost_IC3yr, _ = np.histogram(cos_exposuremap_theta, bins=cosThetaBin, weights=IC3yr.countsmap_fullsky[i])
            plt.step(cosThetaBin[:-1], n_cost_IC3yr, lw=3, color=color[i], label=labels[i])
            n_cost_SyntheticData3yr, _ = np.histogram(cos_exposuremap_theta, bins=cosThetaBin, weights=SyntheticData3yr.countsmap_fullsky[i])
            plt.step(cosThetaBin[:-1], n_cost_SyntheticData3yr, lw=1, color=color[i])
        plt.ylabel('Number of events')
        plt.xlabel(r'$\cos\theta$')
        plt.yscale('log')
        plt.ylim(3e-1, 3e4)
        plt.legend(numpoints=1, scatterpoints=1, frameon=True,fontsize=14, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.19))
        plt.subplots_adjust(left=0.14, bottom=0.14, top=0.85)
        plt.savefig(testfigpath+'CompareNeutrinoCosTheta.pdf')


def GalaxySampleCharacters(plotWISEmap=True, plotpowerspectrum=True):

    gs_WISE = GALAXY_LIBRARY.get_sample('WISE')

    if plotWISEmap:
        plt.figure(figsize = (8,6))
        matplotlib.rc('font', **font)
        matplotlib.rc('legend', **legendfont)
        matplotlib.rc("text", usetex=True)
        map = gs_WISE.galaxymap.copy()
        map[gs_WISE.idx_galaxymask] = hp.UNSEEN
        hp.mollview(map, title='',  max=70, margins=[0,0,0,0.9])
        plt.savefig(testfigpath+'WISE_galaxymap.pdf')

    if plotpowerspectrum:
        gs_analy = GALAXY_LIBRARY.get_sample('analy')

        plt.figure(figsize = (8,6))
        matplotlib.rc('font', **font)
        matplotlib.rc('legend', **legendfont)
        matplotlib.rc("text", usetex=True)

        N_tot_analy = float(gs_WISE.galaxymap.sum())
        shortNoiseMap = np.random.poisson(N_tot_analy / Defaults.NPIXEL, Defaults.NPIXEL)
        shortNoiseMap_Cl_mean = np.mean(hp.anafast(shortNoiseMap / shortNoiseMap.mean() - 1.))

        plt.plot(Defaults.ell, gs_analy.analyCL[0:Defaults.NCL], '--', lw=4, color='grey', label='Analytical power spectrum')
        plt.plot(Defaults.ell, hp.sphtfunc.alm2cl(gs_analy.overdensityalm) / gs_analy.f_sky, lw=1, color='silver', label='Simulated galaxy sample')
        plt.plot(Defaults.ell, hp.sphtfunc.alm2cl(gs_WISE.overdensityalm) / gs_WISE.f_sky, lw=2, color='k', label='WISE-2MASS galaxy sample')
        plt.yscale('log')
        plt.ylim(1e-7, 2e-2)
        plt.ylabel('$C_\ell$')
        plt.xlabel('$\ell$')
        plt.legend(numpoints=1, scatterpoints=1, frameon=True,fontsize=16, loc=0)
        plt.subplots_adjust(left=0.14, bottom=0.14)
        plt.savefig(testfigpath+'GalaxySamplePowerSpectrum.pdf')


def TS_distribution_calculate(plotN_yr, galaxyName, computeSTD, Ebinmin, Ebinmax, lmin, N_re):

    llh = Likelihood(N_yr=plotN_yr,  galaxyName=galaxyName, computeSTD=computeSTD, Ebinmin=Ebinmin, Ebinmax=Ebinmax, lmin=lmin)
    llh.TS_distribution(N_re, f_diff=0)
    llh.TS_distribution(N_re, f_diff=1,  astroModel='observed_numu_fraction')


def TS_distributionPlot(galaxyName, lmin,  pdf=False, N_re=10000):

    TS_bins = np.linspace(-1, 200, 4001)
    TS_bins_c = (TS_bins[0:-1] + TS_bins[1:]) / 2.

    plt.figure(figsize = (8,6))
    matplotlib.rc('font', **font)
    matplotlib.rc('legend', **legendfont)
    matplotlib.rc("text", usetex=True)

    plt.xlabel('Test Statistic')
    plt.xlim(0, 30)
    plt.yscale('log')
    plt.ylabel('1 - Cumulative Probability')
    plt.ylim(1e-4, 3)

    colors_atm = 'k'
    colors_astro = ['royalblue', 'deepskyblue']

    plt.plot([], [], colors_atm[0], lw=2, label='Atm. only')
    plt.plot([], [], colors_astro[0], lw=3, label='Atm. + Astro. 3 yr' )
    plt.plot([], [], colors_astro[1], lw=6, label='Atm. + Astro. 10 yr' )


    # 10yr
    TS_atm = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR,'TS_0_'+galaxyName+'_10.txt'))
    TS_astro1 = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR,'TS_1_'+galaxyName+'_10_observed_numu_fraction1.txt'))
    TS_astro2 = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR,'TS_1_'+galaxyName+'_10_observed_numu_fraction2.txt'))

    # 3yr
    TS_astro1_3yr = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR,'TS_1_'+galaxyName+'_3_observed_numu_fraction1.txt'))

    TS2p = lambda TS: np.histogram(TS, TS_bins)[0] / float(len(TS))

    plt.plot(TS_bins_c, 1 - (stats.chi2.cdf(TS_bins_c, 3)),'--', color='silver', lw=6, label=r'$\chi^2$ (dof=3)')
    plt.step(TS_bins[:-1],  1 - np.cumsum(TS2p(TS_atm)), lw=3, color=colors_atm,  where='post')

    plt.step(TS_bins[:-1],  1 - np.cumsum(TS2p(TS_astro1_3yr)), lw=4, color=colors_astro[0],  where='post')


    plt.fill_between(TS_bins[:-1],  1 - np.cumsum(TS2p(TS_astro1)), 1 - np.cumsum(TS2p(TS_astro2)), step='post', alpha=0.4, color=colors_astro[1])
    print (np.median (np.sort(TS_astro2)))

    plt.plot([-10, 100], [0.5, 0.5], 'r--', lw=1)
    plt.legend(numpoints=1, scatterpoints=1,ncol=1,  frameon=True,fontsize=16, loc=3)
    plt.subplots_adjust(left=0.14, bottom=0.14)
    plt.savefig(testfigpath+galaxyName+'_'+'TS_distribution.pdf')





def BestfitModel(ns, N_yr=1, galaxyName='WISE',Ebinmin=1, Ebinmax=4, lmin=50, plotMCMC=False, plotSED=False):

    llh = Likelihood(N_yr=N_yr,  galaxyName=galaxyName, computeSTD=False, Ebinmin=Ebinmin, Ebinmax=Ebinmax, lmin=lmin)
    llh.inputData(ns)
    ns.updateMask(llh.idx_mask)
    ndim = Ebinmax - Ebinmin
    minimizeResult = llh.minimize__lnL()
    print (llh.Ncount)
    print (minimizeResult[0])
    print (minimizeResult[-1], significance(minimizeResult[-1], 3))

    if plotMCMC:
        llh.runMCMC(Nwalker=640, Nstep=500)
        labels = [ '$f_{\mathrm{astro},\,1}$', '$f_{\mathrm{astro},\,2}$', '$f_{\mathrm{astro},\,3}$']
        truths = np.array([0.00221405, 0.01216614, 0.15222642, 0., 0., 0.]) * 2.
        llh.plotMCMCchain(ndim, labels, truths, plotChain=True)

    if plotSED:
        llh.plotCastro()

def Projected10yr(readdata=True, plotMCMC=False):
    N_yr = 10
    ns = NeutrinoSample()
    astropath = os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'eventmap_10yr_{i}.fits')

    if readdata:
        ns.inputData(astropath)
    else:
        gs_WISE = GALAXY_LIBRARY.get_sample('WISE')
        eg_2010 = EventGenerator('IC79-2010',   astroModel='observed_numu_fraction')
        eg_2011 = EventGenerator('IC86-2011',   astroModel='observed_numu_fraction')
        eg_2012 = EventGenerator('IC86-2012',   astroModel='observed_numu_fraction')

        countsmap = eg_2010.SyntheticData(1., f_diff=1, density_nu=gs_WISE.density) +\
        eg_2011.SyntheticData(4.5, f_diff=1, density_nu=gs_WISE.density) +\
        eg_2012.SyntheticData(4.5, f_diff=1, density_nu=gs_WISE.density)
        ns.inputCountsmap(countsmap)
        write_maps_to_fits(countsmap, astropath)

    BestfitModel(ns=ns, N_yr=N_yr, galaxyName='WISE', lmin=50, Ebinmin=1, Ebinmax=4, plotMCMC=plotMCMC, plotSED=True)


def SED_3yr(plotMCMC=False):
    ns = IC3yr
    N_yr = 3
    BestfitModel(ns=ns, N_yr=N_yr, galaxyName='WISE', lmin=50, Ebinmin=1, Ebinmax=4, plotMCMC=plotMCMC, plotSED=True)


if __name__ == '__main__':
    CompareNeutrinoMaps(energyBin=2, plotcount=True, plotoverdensity=True, plotpowerspectrum=True, plotcostheta=True)
    GalaxySampleCharacters(plotWISEmap=True, plotpowerspectrum=True)
    #TS_distribution_calculate(3, galaxyName='WISE', computeSTD=computeSTD, Ebinmin=1, Ebinmax=4, lmin=lmin, N_re = N_re)
    #TS_distribution_calculate(10, galaxyName='WISE', computeSTD=False, Ebinmin=1, Ebinmax=4, lmin=50, N_re = 200)
    TS_distributionPlot(galaxyName='WISE', lmin=50, pdf=False)
    SED_3yr(plotMCMC=False)
    Projected10yr(readdata=True, plotMCMC=False)
