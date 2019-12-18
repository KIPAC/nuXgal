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

N_yr = 2 + 79./86


def CompareMaps(plotflux=True, plotcount=True, energyBin=2):
    IC3yr.updateMask(Defaults.idx_muon)

    # generate synthetic data with astrophysical events from galaxies
    eg = EventGenerator()
    gs = GalaxySample('analy')
    counts_atm, counts_astro = eg.SyntheticData(N_yr, f_diff=0., density_nu=gs.density)
    SyntheticData = NeutrinoSample()
    #SyntheticData.inputCountsmapMix(counts_atm, counts_astro)
    SyntheticData.inputCountsmap(counts_atm, spectralIndex=3.7)
    SyntheticData.updateMask(Defaults.idx_muon)
    print (IC3yr.getEventCounts())
    print (SyntheticData.getEventCounts())

    if plotcount:
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize = (20,6))
        matplotlib.rc('font', **font)
        matplotlib.rc('legend', **legendfont)
        matplotlib.rcParams.update({'font.size':16})

        plt.axes(ax1)
        hp.mollview(IC3yr.countsmap[energyBin], title='IceCube 3 year', hold=True)
        plt.axes(ax2)
        hp.mollview(SyntheticData.countsmap[energyBin], title='Synthetic data', hold=True)
        plt.savefig(testfigpath+'CompareCountsmaps.pdf')

    if plotflux:
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize = (20,6))
        matplotlib.rc('font', **font)
        matplotlib.rc('legend', **legendfont)
        matplotlib.rcParams.update({'font.size':16})

        plt.axes(ax1)
        hp.mollview(IC3yr.fluxmap[energyBin], title='IceCube 3 year', hold=True)
        plt.axes(ax2)
        hp.mollview(SyntheticData.fluxmap[energyBin], title='Synthetic data', hold=True)
        plt.savefig(testfigpath+'CompareFluxmaps.pdf')







if __name__ == '__main__':
    CompareMaps(energyBin=1)
