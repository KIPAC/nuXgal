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
from KIPAC.nuXgal.Likelihood import Likelihood
from KIPAC.nuXgal.GalaxySample import GalaxySample
from KIPAC.nuXgal.NeutrinoSample import NeutrinoSample






font = { 'family': 'Arial', 'weight' : 'normal', 'size'   : 18}
legendfont = {'fontsize' : 18, 'frameon' : False}

testfigpath = os.path.join(Defaults.NUXGAL_PLOT_DIR, 'Fig_')
countsmappath = os.path.join(Defaults.NUXGAL_DATA_DIR, 'IceCube3yr_countsmap{i}.fits')
IC3yr = NeutrinoSample()
IC3yr.inputData(countsmappath)


def CompareNeutrinoMaps(energyBin=2, plotcount=False, plotoverdensity=False, plotpowerspectrum=False, plotcostheta=True):
    #IC3yr.updateMask(Defaults.idx_muon)

    # generate synthetic data with astrophysical events from galaxies
    gs = GalaxySample('analy')
    # generate data year by year as effective areas are different
    countsmap = np.zeros((Defaults.NEbin, Defaults.NPIXEL))

    for year in ['IC79-2010', 'IC86-2011', 'IC86-2012']:
        print (year)
        eg = EventGenerator(year, 'numu')
        counts = eg.SyntheticData(N_yr=1., f_diff=1., density_nu=gs.density)
        SyntheticData = NeutrinoSample()
        SyntheticData.inputCountsmap(counts)
        #SyntheticData.updateMask(Defaults.idx_muon)
        print (SyntheticData.getEventCounts())
        print ('-------------------')
        countsmap = countsmap + SyntheticData.countsmap_fullsky


    SyntheticData3yr = NeutrinoSample()
    SyntheticData3yr.inputCountsmap(countsmap)
    #SyntheticData3yr.updateMask(Defaults.idx_muon)
    print ('IceCube 3yr:', IC3yr.getEventCounts())
    print ('Synthetic data 3yr:', SyntheticData3yr.getEventCounts())

    color = [ 'r', 'orange', 'limegreen', 'dodgerblue', 'mediumslateblue', 'purple', 'grey']
    labels = ['$10^2-10^3$ GeV', '$10^3-10^4$ GeV','$10^4-10^5$ GeV','$10^5-10^6$ GeV','$10^6-10^7$ GeV','$10^7-10^8$ GeV', '$10^8-10^9$ GeV']

    if plotcount:
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize = (20,6))
        matplotlib.rc('font', **font)
        matplotlib.rc('legend', **legendfont)
        matplotlib.rcParams.update({'font.size':16})

        plt.axes(ax1)
        hp.mollview(IC3yr.countsmap[energyBin], title='IceCube 3 year', hold=True)
        plt.axes(ax2)
        hp.mollview(SyntheticData3yr.countsmap[energyBin], title='Synthetic data', hold=True)
        plt.savefig(testfigpath+'CompareNeutrinoCountsmaps.pdf')

    if plotoverdensity:
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize = (20,6))
        matplotlib.rc('font', **font)
        matplotlib.rc('legend', **legendfont)
        matplotlib.rcParams.update({'font.size':16})

        plt.axes(ax1)
        hp.mollview(IC3yr.getOverdensity()[energyBin], title='IceCube 3 year', hold=True)
        plt.axes(ax2)
        hp.mollview(SyntheticData3yr.getOverdensity()[energyBin], title='Synthetic data', hold=True)
        plt.savefig(testfigpath+'CompareNeutrinoOverdensitymaps.pdf')

    if plotpowerspectrum:
        plt.figure(figsize = (8,6))
        matplotlib.rc('font', **font)
        matplotlib.rc('legend', **legendfont)
        w_auto_IC3yr = IC3yr.getPowerSpectrum()
        w_auto_SyntheticData3yr = SyntheticData3yr.getPowerSpectrum()

        for i in [0, 1, 2, 3, 4]:
            plt.plot(Defaults.ell, w_auto_IC3yr[i], lw=3, color=color[i], label=labels[i])
            plt.plot(Defaults.ell, w_auto_SyntheticData3yr[i], lw=1, color=color[i])
        plt.yscale('log')
        plt.ylim(1e-5, 1e0)
        plt.ylabel('$C_\ell$')
        plt.xlabel('$\ell$')
        plt.legend(numpoints=1, scatterpoints=1, frameon=True,fontsize=16, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.15))
        plt.subplots_adjust(left=0.14, bottom=0.14)
        plt.savefig(testfigpath+'CompareNeutrinoPowerSpectrum.pdf')

    if plotcostheta:
        plt.figure(figsize = (8,6))
        matplotlib.rc('font', **font)
        matplotlib.rc('legend', **legendfont)
        cosThetaBin = np.linspace(-1, 1, 50)
        cos_exposuremap_theta = np.cos(Defaults.exposuremap_theta)
        for i in range(Defaults.NEbin-2):
            n_cost_IC3yr, _ = np.histogram(cos_exposuremap_theta, bins=cosThetaBin, weights=IC3yr.countsmap_fullsky[i])
            plt.step(cosThetaBin[:-1], n_cost_IC3yr, lw=3, color=color[i], label=labels[i])
            n_cost_SyntheticData3yr, _ = np.histogram(cos_exposuremap_theta, bins=cosThetaBin, weights=SyntheticData3yr.countsmap_fullsky[i])
            plt.step(cosThetaBin[:-1], n_cost_SyntheticData3yr, lw=1, color=color[i])
        plt.ylabel(r'$N_\theta$')
        plt.xlabel(r'$\theta$')
        plt.yscale('log')
        plt.ylim(3e-1, 3e4)
        plt.legend(numpoints=1, scatterpoints=1, frameon=True,fontsize=16, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.25))
        plt.subplots_adjust(left=0.14, bottom=0.14, top=0.8)
        plt.savefig(testfigpath+'CompareNeutrinoCosTheta.pdf')


def GalaxySampleCharacters(plotWISEmap=True, plotpowerspectrum=True):
    gs_WISE = GalaxySample('WISE')

    if plotWISEmap:
        plt.figure(figsize = (8,6))
        matplotlib.rc('font', **font)
        matplotlib.rc('legend', **legendfont)
        matplotlib.rcParams.update({'font.size':16})
        map = gs_WISE.galaxymap.copy()
        map[gs_WISE.idx_galaxymask] = hp.UNSEEN
        hp.mollview(map, title='WISE-2MASS Galaxy Distribution', max=100, margins=[0,0,0,0.9])
        plt.savefig(testfigpath+'WISE_galaxymap.pdf')

    if plotpowerspectrum:
        gs_analy = GalaxySample('analy')
        plt.figure(figsize = (8,6))
        matplotlib.rc('font', **font)
        matplotlib.rc('legend', **legendfont)

        N_tot_analy = float(gs_analy.galaxymap.sum())
        shortNoiseMap = np.random.poisson(N_tot_analy / Defaults.NPIXEL, Defaults.NPIXEL)
        shortNoiseMap_Cl_mean = np.mean(hp.anafast(shortNoiseMap / shortNoiseMap.mean() - 1.))

        plt.plot(Defaults.ell, gs_analy.analyCL[0:Defaults.NCL], lw=3, color='k', label='Analytical power spectrum')
        plt.plot(Defaults.ell, gs_analy.analyCL[0:Defaults.NCL]+shortNoiseMap_Cl_mean, lw=2, color='k', label='Analytical power spectrum + short noise')
        plt.plot(Defaults.ell, hp.anafast(gs_analy.overdensity) / gs_analy.f_sky, lw=2, color='grey', label='Simulated galaxy sample')
        plt.plot(Defaults.ell, hp.anafast(gs_WISE.overdensity) / gs_WISE.f_sky, lw=4, color='mediumslateblue', label='WISE-2MASS galaxy sample')
        plt.yscale('log')
        plt.ylim(1e-7, 2e-2)
        plt.ylabel('$C_\ell$')
        plt.xlabel('$\ell$')
        plt.legend(numpoints=1, scatterpoints=1, frameon=True,fontsize=16, loc=0)
        plt.subplots_adjust(left=0.14, bottom=0.14)
        plt.savefig(testfigpath+'GalaxySamplePowerSpectrum.pdf')


def TS_distribution(readfile = False, galaxyName='WISE', computeSTD=False, Ebinmin=1, Ebinmax=4, lmin=50, N_re = 200):
    plotN_yr = [3, 10]

    TS_bins = np.linspace(0, 100, 201)
    TS_bins_c = (TS_bins[0:-1] + TS_bins[1:]) / 2.

    plt.figure(figsize = (8,6))
    matplotlib.rc('font', **font)
    matplotlib.rc('legend', **legendfont)
    plt.ylabel('1 - Cumulative Probability')
    plt.xlabel('Test Statistics')
    plt.xlim(-5, 50)
    plt.ylim(1e-2, 2)
    plt.yscale('log')

    colors_atm = ['b', 'b']
    colors_astro = ['r', 'r']
    lw = [2, 4]

    for idx_N_yr, N_yr in enumerate(plotN_yr):

        if not readfile:
            llh = Likelihood(N_yr=N_yr,  galaxyName=galaxyName, computeSTD=computeSTD, Ebinmin=Ebinmin, Ebinmax=Ebinmax, lmin=lmin)
            llh.TS_distribution(N_re, f_diff=0)
            llh.TS_distribution(N_re, f_diff=1)
            #llh.TS_distribution(N_re, f_diff=1, astroModel='hese')


        TS_atm = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR,'TS_0_'+galaxyName+'_'+str(N_yr)+'.txt'))
        TS_numu = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR,'TS_1_'+galaxyName+'_'+str(N_yr)+'_numu.txt'))
        #TS_hese = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR,'TS_1_'+galaxyName+'_'+str(N_yr)+'_hese.txt'))

        p_atm = np.histogram(TS_atm, TS_bins)[0] / float(len(TS_atm))
        p_numu = np.histogram(TS_numu, TS_bins)[0] / float(len(TS_numu))
        #p_hese = np.histogram(TS_hese, TS_bins)[0] / float(len(TS_hese))

        #dTS_bins = np.mean(TS_bins[1:] - TS_bins[0:-1])
        #p_atm = np.histogram(TS_atm, TS_bins)[0] / float(N_re) / dTS_bins
        #axes.plot(TS_bins_c,  p_atm, lw=3, label='atm')
        #axes.plot(TS_bins_c, stats.chi2.pdf(TS_bins_c, 2),'--', lw=2, label='chi2, dof=2')


        plt.step(TS_bins[:-1],  1 - np.cumsum(p_atm), lw=lw[idx_N_yr], color=colors_atm[idx_N_yr],  where='post')
        plt.step(TS_bins[:-1],  1 - np.cumsum(p_numu), lw=lw[idx_N_yr], color=colors_astro[idx_N_yr], where='post')
        #plt.step(TS_bins[:-1],  1 - np.cumsum(p_hese), lw=3, color='orange', label=r'Atmospheric + Astrophysical HESE flux', where='post')

    plot_lines = []
    a, = plt.plot([], [], 'k', lw=lw[1])
    b, = plt.plot([], [], 'k', lw=lw[0])
    plot_lines.append([a, b])

    c = plt.plot([], [], 'r', lw=2, label='Atm. + Astro.' )
    d = plt.plot([], [], 'b', lw=2, label='Atm. only')
    plt.plot(TS_bins_c, 1 - (0.5 + stats.chi2.cdf(TS_bins_c, Ebinmax-Ebinmin)/2),'--', color='grey', lw=2, label=r'$\chi^2$ distribution (dof=3)')

    legend1 = plt.legend(plot_lines[0], ["10 yr", "3 yr"], loc=4)
    plt.legend(numpoints=1, scatterpoints=1, frameon=True,fontsize=16, loc=1)
    plt.gca().add_artist(legend1)
    plt.savefig(testfigpath+'TS_distribution.pdf')


def BestfitModel(ns, N_yr=1, galaxyName='WISE',Ebinmin=1, Ebinmax=4, lmin=50, plotMCMC=False, plotSED=False):

    llh = Likelihood(N_yr=N_yr,  galaxyName=galaxyName, computeSTD=False, Ebinmin=Ebinmin, Ebinmax=Ebinmax, lmin=lmin)
    llh.inputData(ns)
    ns.updateMask(llh.idx_mask)
    ndim = Ebinmax - Ebinmin
    minimizeResult = llh.minimize__lnL()
    print (llh.Ncount)
    print (minimizeResult[0])
    print (minimizeResult[-1])

    if plotMCMC:
        llh.runMCMC(Nwalker=640, Nstep=500)
        labels = [ '$f_{\mathrm{astro},\,1}$', '$f_{\mathrm{astro},\,2}$', '$f_{\mathrm{astro},\,3}$']
        truths = [ 1., 1., 1.]
        llh.plotMCMCchain(ndim, labels, truths)

    if plotSED:
        llh.plotCastro()


if __name__ == '__main__':
    CompareNeutrinoMaps(energyBin=2, plotcount=True, plotoverdensity=True, plotpowerspectrum=True, plotcostheta=True)
    GalaxySampleCharacters(plotWISEmap=True, plotpowerspectrum=True)
    #TS_distribution(readfile = True, galaxyName='WISE', computeSTD=False, Ebinmin=1, Ebinmax=4, lmin=50, N_re=1000)

    ns = IC3yr
    N_yr = 3
    #ns = NeutrinoSample()
    #ns.inputCountsmap(EventGenerator('IC86-2012', 'numu').SyntheticData(10., 1., density_nu=GalaxySample('WISE').density))
    BestfitModel(ns=ns, N_yr=N_yr, galaxyName='WISE', lmin=50, plotMCMC=False, plotSED=True)
