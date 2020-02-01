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
from KIPAC.nuXgal.NeutrinoSample import NeutrinoSample
from KIPAC.nuXgal.Exposure import ICECUBE_EXPOSURE_LIBRARY
from KIPAC.nuXgal.GalaxySample import GALAXY_LIBRARY


font = { 'family': 'Arial', 'weight' : 'normal', 'size'   : 21}
legendfont = {'fontsize' : 21, 'frameon' : False}

testfigpath = os.path.join(Defaults.NUXGAL_PLOT_DIR, 'Fig_')


def test_STDdependence():
    plt.figure(figsize = (8,6))
    axes = plt.axes()
    axes.set_yscale('log')
    axes.set_ylim(1e-6, 1e-1)

    w_atm_mean_file = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'w_atm_mean_30.txt'))
    w_atm_std_file = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'w_atm_std_WISE_30.txt'))

    Natm = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'Ncount_atm_after_masking_WISE_30.txt'))

    axes.plot(Defaults.ell, np.abs(w_atm_mean_file[3]), lw=2, label='atm-mean-3')

    axes.plot(Defaults.ell, w_atm_std_file[3], lw=2, label='atm-3')
    axes.plot(Defaults.ell, w_atm_std_file[2], label='atm-2')

    axes.plot(Defaults.ell, w_atm_std_file[1] * (Natm[1] / Natm[2])**0.5, label='atm-2 from atm 1')
    axes.plot(Defaults.ell, w_atm_std_file[1] * (Natm[1] / Natm[3])**0.5, lw=1, label='atm-3 from atm 1')


    plt.legend()
    plt.savefig(testfigpath+'wstd.pdf')




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
        #fig, (ax1, ax2) = plt.subplots(ncols=2, figsize = (20,6))
        matplotlib.rc('font', **font)
        matplotlib.rc('legend', **legendfont)
        matplotlib.rcParams.update({'font.size':18})
        matplotlib.rc("text", usetex=True)

        #plt.axes(ax1)
        #hp.mollview(IC3yr.countsmap[energyBin], title='IceCube 3 year', hold=True)
        #plt.axes(ax2)
        #hp.mollview(SyntheticData3yr.countsmap[energyBin], title='Synthetic data', hold=True)
        #plt.savefig(testfigpath+'CompareNeutrinoCountsmaps.pdf')

        plt.figure(figsize = (8,6))
        hp.mollview(IC3yr.countsmap[energyBin], title='IceCube 3 year')
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
        #plt.xlim(0, 150)
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
        #plt.plot(Defaults.ell, (gs_analy.analyCL[0:Defaults.NCL]+shortNoiseMap_Cl_mean) / 0.4, lw=2, color='k', label='Analytical power spectrum + short noise')
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
    #llh.TS_distribution(N_re, f_diff=1,  astroModel='observed_numu_fraction')


def TS_distribution_expected(dof = 3):
    TS_array = np.linspace(0, 50, 10000)
    cdf_TS = stats.chi2.cdf(TS_array, 1) / 2 + 0.5
    from scipy.interpolate import interp1d
    f = interp1d(cdf_TS, TS_array, bounds_error=False, fill_value=0.)
    N_TS = 200000
    TS_sum = np.zeros(N_TS)
    for i in np.arange(N_TS):
        for j in range(dof):
            TS_sum[i] += f(np.random.rand())
    p_TS = np.histogram(TS_sum, bins=TS_array)[0] / float(N_TS)
    TS_c = (TS_array[0:-1] + TS_array[1:])/2.
    np.savetxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'TS_compare'+str(dof)+'.txt'), np.c_[TS_c, 1-np.cumsum(p_TS)])
    return TS_c, 1-np.cumsum(p_TS)



def TS_distribution_byBin():
    TS_bins = np.linspace(-1, 200, 4001)
    TS_bins_c = (TS_bins[0:-1] + TS_bins[1:]) / 2.

    plt.figure(figsize = (8,6))
    matplotlib.rc('font', **font)
    matplotlib.rc('legend', **legendfont)
    matplotlib.rc("text", usetex=True)
    color = [ 'r', 'orange', 'limegreen', 'dodgerblue', 'mediumslateblue', 'purple', 'grey']

    plt.xlabel('Test Statistics')
    plt.xlim(0, 25)
    plt.yscale('log')
    plt.ylim(1e-4, 2)
    plt.ylabel('1 - Cumulative Probability')

    """
    for i in [1, 2, 3]:
        TS_atm = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR,'TS_0_WISE_10_' + str(i) + '-' + str(i+1) +'.txt'))
        p_atm = np.histogram(TS_atm, TS_bins)[0] / float(len(TS_atm))
        plt.step(TS_bins[:-1],  1 - np.cumsum(p_atm), lw=i * 1, color=color[i],  where='post', label='bin'+str(i))
    """

    #TS_atm = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR,'TS_0_WISE_10_1-4.txt'))
    TS_atm = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR,'TS_0_WISE_5_bin1_lmin50.txt'))
    p_atm = np.histogram(TS_atm, TS_bins)[0] / float(len(TS_atm))
    plt.step(TS_bins[:-1],  1 - np.cumsum(p_atm), lw=1, color='k',  where='post', label='lmin=50')

    TS_atm = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR,'TS_0_WISE_5_bin1_lmin100.txt'))
    p_atm = np.histogram(TS_atm, TS_bins)[0] / float(len(TS_atm))
    plt.step(TS_bins[:-1],  1 - np.cumsum(p_atm), lw=2, color='g',  where='post', label='lmin=100')

    TS_atm = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR,'TS_0_WISE_5_bin1_lmin200.txt'))
    p_atm = np.histogram(TS_atm, TS_bins)[0] / float(len(TS_atm))
    plt.step(TS_bins[:-1],  1 - np.cumsum(p_atm), lw=4, color='b',  where='post', label='lmin=200')

    TS_atm = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR,'TS_0_WISE_5_bin1_lmin20.txt'))
    p_atm = np.histogram(TS_atm, TS_bins)[0] / float(len(TS_atm))
    plt.step(TS_bins[:-1],  1 - np.cumsum(p_atm), lw=1, color='hotpink',  where='post', label='lmin=20')

    TS_atm = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR,'TS_0_WISE_5_bin1_lmin20b.txt'))
    p_atm = np.histogram(TS_atm, TS_bins)[0] / float(len(TS_atm))
    plt.step(TS_bins[:-1],  1 - np.cumsum(p_atm), lw=1, color='dodgerblue',  where='post', label='lmin=20b')


    plt.plot(TS_bins_c, 1 - (0.5 + stats.chi2.cdf(TS_bins_c, 1) / 2.),'--', color='grey', lw=2, label=r'$\chi^2$ (dof=1)')
    #plt.plot(TS_bins_c, 1 - ( stats.chi2.cdf(TS_bins_c, 1) ),'-.', color='grey', lw=2, label=r'$\chi^2$ (dof=1)')
    #plt.plot(TS_bins_c, 1 - (0.5 + stats.chi2.cdf(TS_bins_c, 3) / 2. ),'--', color='grey', lw=3, label=r'$\chi^2$ (dof=3)')

    """
    dof = 3
    #x, y = TS_distribution_expected(dof)
    TS_distribution_expected_file = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'TS_compare'+str(dof)+'.txt'))
    x, y = TS_distribution_expected_file[:, 0], TS_distribution_expected_file[:,1]
    plt.plot(x, y, '--', color='grey', lw=4, label=r'one-sided test $(\rm dof = 3)$')
    plt.title('$l_{\mathrm{min}}=100$')
    """



    plt.legend(numpoints=1, scatterpoints=1,  frameon=True,fontsize=16, loc=1)
    #plt.gca().add_artist(legend1)
    plt.subplots_adjust(left=0.14, bottom=0.14)
    plt.savefig(testfigpath+'WISE_'+'TS_distribution_byBin.pdf')


def TS_distribution(readfile, galaxyName, computeSTD, lmin, N_re, pdf=False):

    if not readfile:
        #TS_distribution_calculate(3, galaxyName='WISE', computeSTD=computeSTD, Ebinmin=1, Ebinmax=4, lmin=lmin, N_re = N_re)
        TS_distribution_calculate(5, galaxyName='WISE', computeSTD=computeSTD, Ebinmin=1, Ebinmax=2, lmin=lmin, N_re = N_re)


    plotN_yr = [10] # [3, 10]

    TS_bins = np.linspace(-1, 200, 4001)
    #TS_bins = np.linspace(-1, 200, 2001)
    TS_bins_c = (TS_bins[0:-1] + TS_bins[1:]) / 2.

    plt.figure(figsize = (8,6))
    matplotlib.rc('font', **font)
    matplotlib.rc('legend', **legendfont)
    matplotlib.rc("text", usetex=True)

    plt.xlabel('Test Statistics')
    plt.xlim(0, 40)
    plt.yscale('log')

    colors_atm = ['blue', 'silver']
    colors_astro = ['royalblue', 'deepskyblue']
    colors_astro2 = colors_astro
    colors_fill = colors_astro
    lw = [4, 2]

    plot_lines = []
    a, = plt.plot([], [], 'k', lw=lw[1])
    b, = plt.plot([], [], 'k', lw=lw[0])
    plot_lines.append([a, b])
    plt.plot([], [], colors_atm[0], lw=2, label='Atm. only')

    for idx_N_yr, N_yr in enumerate(plotN_yr):

        TS_atm = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR,'TS_0_'+galaxyName+'_'+str(N_yr)+'.txt'))
        #TS_astro1 = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR,'TS_1_'+galaxyName+'_'+str(N_yr)+'_observed_numu_fraction1.txt'))
        print (len(np.where(TS_atm == 0.)[0]))

        if pdf:
            dTS_bins = np.mean(TS_bins[1:] - TS_bins[0:-1])
            p_atm = np.histogram(TS_atm, TS_bins)[0] / float(N_re) / dTS_bins
            plt.plot(TS_bins_c, p_atm, lw=lw[idx_N_yr], color=colors_atm[idx_N_yr])
            plt.plot(TS_bins_c, stats.chi2.pdf(TS_bins_c, 3),'r--', label='chi2, dof=3')
            plt.ylabel('Probability distribution function')
            plt.ylim(1e-5, 2)

        else:
            p_atm = np.histogram(TS_atm, TS_bins)[0] / float(len(TS_atm))
            """
            p_astro1 = np.histogram(TS_astro1, TS_bins)[0] / float(len(TS_astro1))
            if N_yr == 3:
                plt.step(TS_bins[:-1],  1 - np.cumsum(p_astro1), lw=lw[idx_N_yr], color=colors_fill[idx_N_yr],  where='post')

            if N_yr == 10:
                TS_astro2 = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR,'TS_1_'+galaxyName+'_'+str(N_yr)+'_observed_numu_fraction2.txt'))
                p_astro2 = np.histogram(TS_astro2, TS_bins)[0] / float(len(TS_astro2))
                plt.fill_between(TS_bins[:-1],  1 - np.cumsum(p_astro1), 1 - np.cumsum(p_astro2), step='post', alpha=0.4, color=colors_fill[idx_N_yr])
                #print (np.median (np.sort(TS_astro2)))
            """
            #print ( np.where(TS_atm == 0) , stats.chi2.cdf(0., 4))
            plt.step(TS_bins[:-1],  1 - np.cumsum(p_atm), lw=lw[idx_N_yr], color=colors_atm[idx_N_yr],  where='post')
            #plt.plot([-10, 1000], [0.5, 0.5], 'r-.', lw=2)
            plt.ylabel('1 - Cumulative Probability')
            plt.ylim(1e-4, 2)

    if not pdf:
        #c = plt.plot([], [], colors_astro[0], lw=2, label='Atm. + Astro. 3 yr' )
        #e = plt.plot([], [], colors_astro[1], lw=6, label='Atm. + Astro. 10 yr' )
        plt.plot(TS_bins_c, 1 - (  stats.chi2.cdf(TS_bins_c, 1) ),'-', color='k', lw=3, label=r'$\chi^2$ (dof=1)')
        #plt.plot(TS_bins_c, 1 - (0.5 + stats.chi2.cdf(TS_bins_c, 3)/2),'--', color='r', lw=2, label=r'$\chi^2\,(\rm dof = 3)$')
        #x, y = TS_distribution_expected()
        #TS_distribution_expected_file = np.loadtxt(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'TS_compare.txt'))
        #x, y = TS_distribution_expected_file[:, 0], TS_distribution_expected_file[:,1]
        #plt.plot(x, y, '--', color='r', lw=2, label=r'one-sided test $(\rm dof = 3)$')


    #legend1 = plt.legend(plot_lines[0], ["10 yr", "3 yr"], loc=1)
    plt.legend(numpoints=1, scatterpoints=1,  frameon=True,fontsize=16, loc=1)
    #plt.gca().add_artist(legend1)
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
    print (minimizeResult[-1])

    if plotMCMC:
        llh.runMCMC(Nwalker=640, Nstep=500)
        labels = [ '$f_{\mathrm{astro},\,1}$', '$f_{\mathrm{astro},\,2}$', '$f_{\mathrm{astro},\,3}$']
        truths = [ 1., 1., 1.]
        llh.plotMCMCchain(ndim, labels, truths, plotChain=True)

    if plotSED:
        llh.plotCastro()

def Projected10yr():
    N_yr = 10
    ns = NeutrinoSample()
    gs_WISE = GALAXY_LIBRARY.get_sample('WISE')
    ns.inputCountsmap(EventGenerator('IC86-2012', 'observed_numu_fraction').SyntheticData(N_yr, 1., density_nu=gs_WISE.density))
    BestfitModel(ns=ns, N_yr=N_yr, galaxyName='WISE', lmin=50, Ebinmin=1, Ebinmax=4, plotMCMC=True, plotSED=True)


def SED_3yr():
    ns = IC3yr
    N_yr = 3
    #print (np.sum(ns.getEventCounts()[1:]) / 3.)
    BestfitModel(ns=ns, N_yr=N_yr, galaxyName='WISE', lmin=50, Ebinmin=1, Ebinmax=4, plotMCMC=True, plotSED=True)


if __name__ == '__main__':
    TS_distribution_byBin()
    #test_STDdependence()
    #CompareNeutrinoMaps(energyBin=2, plotcount=True, plotoverdensity=True, plotpowerspectrum=True, plotcostheta=True)
    #GalaxySampleCharacters(plotWISEmap=True, plotpowerspectrum=True)
    #TS_distribution(readfile = False, galaxyName='WISE', computeSTD=False, lmin=20, N_re=500)
    #TS_distribution(readfile = True, galaxyName='WISE', computeSTD=False, lmin=50, N_re=35000, pdf=False)
    #SED_3yr()
    #Projected10yr()
