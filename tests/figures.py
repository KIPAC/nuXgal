"""Test utility for Likelihood fitting"""

import os

import numpy as np

import healpy as hp

from KIPAC.nuXgal import Defaults

from KIPAC.nuXgal.EventGenerator import EventGenerator

from KIPAC.nuXgal.Likelihood import Likelihood

from KIPAC.nuXgal.file_utils import read_maps_from_fits, write_maps_to_fits

from KIPAC.nuXgal.hp_utils import vector_apply_mask, vector_apply_mask_hp

from KIPAC.nuXgal.plot_utils import FigureDict

from KIPAC.nuXgal.GalaxySample import GalaxySample

from KIPAC.nuXgal.NeutrinoSample import NeutrinoSample

from KIPAC.nuXgal import file_utils

from KIPAC.nuXgal import hp_utils

import matplotlib
import matplotlib.pyplot as plt

try:
    from Utils import MAKE_TEST_PLOTS
except ImportError:
    from .Utils import MAKE_TEST_PLOTS

font = { 'family': 'Arial', 'weight' : 'normal', 'size'   : 20}
legendfont = {'fontsize' : 18, 'frameon' : False}

testfigpath = os.path.join(Defaults.NUXGAL_PLOT_DIR, 'Fig_')
fluxmappath = os.path.join(Defaults.NUXGAL_DATA_DIR, 'IceCube3yr_fluxmap{i}.fits')
countsmappath = os.path.join(Defaults.NUXGAL_DATA_DIR, 'IceCube3yr_countsmap{i}.fits')
IC3yr = NeutrinoSample()
IC3yr.inputData(countsmappath, fluxmappath)



def CompareMaps(energyBin=2, plotflux=True, plotcount=False, plotoverdensity=True, plotpowerspectrum=True):
    IC3yr.updateMask(Defaults.idx_muon)

    # generate synthetic data with astrophysical events from galaxies
    gs = GalaxySample('analy')
    # generate data year by year as effective areas are different
    fluxmap = np.zeros((Defaults.NEbin, Defaults.NPIXEL))
    countsmap = np.zeros((Defaults.NEbin, Defaults.NPIXEL))

    for year in ['IC79-2010', 'IC86-2011', 'IC86-2012']:
        print (year)
        eg = EventGenerator(year)
        counts_atm, counts_astro = eg.SyntheticData(N_yr=1., f_diff=1., density_nu=gs.density)
        SyntheticData = NeutrinoSample(year)
        SyntheticData.inputCountsmapMix(counts_atm, counts_astro)
        #SyntheticData.inputCountsmap(counts_atm, spectralIndex=3.7)
        SyntheticData.updateMask(Defaults.idx_muon)
        print (SyntheticData.getEventCounts())
        print ('-------------------')
        fluxmap = fluxmap + SyntheticData.fluxmap_fullsky
        countsmap = countsmap + SyntheticData.countsmap_fullsky

    SyntheticData3yr = NeutrinoSample()
    SyntheticData3yr.inputFluxmap(fluxmap)
    SyntheticData3yr.inputCountsmap(countsmap)
    SyntheticData3yr.updateMask(Defaults.idx_muon)
    print ('IceCube 3yr:', IC3yr.getEventCounts())
    print ('Synthetic data 3yr:', SyntheticData3yr.getEventCounts())

    if plotcount:
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize = (20,6))
        matplotlib.rc('font', **font)
        matplotlib.rc('legend', **legendfont)
        matplotlib.rcParams.update({'font.size':16})

        plt.axes(ax1)
        hp.mollview(IC3yr.countsmap[energyBin], title='IceCube 3 year', hold=True)
        plt.axes(ax2)
        hp.mollview(SyntheticData3yr.countsmap[energyBin], title='Synthetic data', hold=True)
        plt.savefig(testfigpath+'CompareCountsmaps.pdf')

    if plotflux:
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize = (20,6))
        matplotlib.rc('font', **font)
        matplotlib.rc('legend', **legendfont)
        matplotlib.rcParams.update({'font.size':16})

        plt.axes(ax1)
        hp.mollview(IC3yr.fluxmap[energyBin], title='IceCube 3 year', hold=True)
        plt.axes(ax2)
        hp.mollview(SyntheticData3yr.fluxmap[energyBin], title='Synthetic data', hold=True)
        plt.savefig(testfigpath+'CompareFluxmaps.pdf')

    if plotoverdensity:
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize = (20,6))
        matplotlib.rc('font', **font)
        matplotlib.rc('legend', **legendfont)
        matplotlib.rcParams.update({'font.size':16})

        plt.axes(ax1)
        hp.mollview(IC3yr.getOverdensity()[energyBin], title='IceCube 3 year', hold=True)
        plt.axes(ax2)
        hp.mollview(SyntheticData3yr.getOverdensity()[energyBin], title='Synthetic data', hold=True)
        plt.savefig(testfigpath+'CompareOverdensitymaps.pdf')

    if plotpowerspectrum:
        plt.figure(figsize = (8,6))
        matplotlib.rc('font', **font)
        matplotlib.rc('legend', **legendfont)
        w_auto_IC3yr = IC3yr.getPowerSpectrum()
        w_auto_SyntheticData3yr = SyntheticData3yr.getPowerSpectrum()
        color = [ 'r', 'orange', 'limegreen', 'dodgerblue', 'mediumslateblue', 'purple', 'grey']
        labels = ['$10^2-10^3$ GeV', '$10^3-10^4$ GeV','$10^4-10^5$ GeV','$10^5-10^6$ GeV','$10^6-10^7$ GeV','$10^7-10^8$ GeV', '$10^8-10^9$ GeV']
        for i in [1, 2, 3]:
            plt.plot(Defaults.ell, w_auto_IC3yr[i], lw=3, color=color[i], label=labels[i])
            plt.plot(Defaults.ell, w_auto_SyntheticData3yr[i], lw=1, color=color[i])
        plt.yscale('log')
        plt.ylim(1e-5, 1e0)
        plt.ylabel('$C_\ell$')
        plt.xlabel('$\ell$')
        plt.legend(numpoints=1, scatterpoints=1, frameon=True,fontsize=16, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.15))
        plt.subplots_adjust(left=0.14, bottom=0.14)

        #plt.legend(loc=9, ncol=3)
        plt.savefig(testfigpath+'ComparePowerSpectrum.pdf')


if __name__ == '__main__':
    CompareMaps(energyBin=3)
