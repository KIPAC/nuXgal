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
from KIPAC.nuXgal.NeutrinoSample import NeutrinoSample
from KIPAC.nuXgal import file_utils
from KIPAC.nuXgal import hp_utils

try:
    from Utils import MAKE_TEST_PLOTS
except ImportError:
    from .Utils import MAKE_TEST_PLOTS


testfigpath = os.path.join(Defaults.NUXGAL_PLOT_DIR, 'test')
fluxmappath = os.path.join(Defaults.NUXGAL_DATA_DIR, 'IceCube3yr_fluxmap{i}.fits')

def test_energySpectrum():

    ns = NeutrinoSample()

    bgmap = file_utils.read_maps_from_fits(fluxmappath, Defaults.NEbin)

    # southern sky mask
    bgmap_nu = hp_utils.vector_apply_mask(bgmap, Defaults.idx_muon, copy=True)
    intensity_atm_nu = ns.getIntensityFromFluxmap(bgmap_nu, 3.)


    print ( intensity_atm_nu * Defaults.map_E_center**2)

    if MAKE_TEST_PLOTS:

        figs = FigureDict()

        intensities = [intensity_atm_nu]

        plot_dict = dict(colors=['b', 'orange', 'k'],
                         markers=['o', '^', 's'])

        figs.plot_intesity_E2('SED', Defaults.map_E_center, intensities, **plot_dict)


        atm_nu_mu = np.loadtxt(os.path.join(Defaults.NUXGAL_ANCIL_DIR, 'atm_nu_mu.txt'))
        figs.plot('SED', 10.** atm_nu_mu[:, 0], atm_nu_mu[:, 1], lw=2, label=r'atm $\nu_\mu$', color='orange')

        ene = np.logspace(1, 7, 40)
        figs.plot('SED', ene, 1.44e-18 * (ene/100e3)**(-2.28) * ene**2 * 1, lw=2, label=r'IceCube $\nu_\mu$', color='b')

        figs.save_all(testfigpath, 'pdf')

def test_fluxmap(N_yr = 3):
    ns = NeutrinoSample()

    eg = EventGenerator()
    atmCounts = eg.atmEvent(N_yr)



    fluxmap = ns.getFluxmap(atmCounts, N_yr, 3.7)

    for i in range(Defaults.NEbin):
        fluxmap[i][Defaults.idx_muon] = hp.UNSEEN


    figs = FigureDict()
    figs.mollview_maps('fluxmap', fluxmap)
    figs.save_all(testfigpath, 'pdf')

    mask = np.zeros(Defaults.NPIXEL)
    mask[Defaults.idx_muon] = 1.
    for i in range(Defaults.NEbin):
        test = np.ma.masked_array(atmCounts[i], mask = mask)
        print (i, test.sum())


def test_autoCorrelation(N_yr = 3):
    ns = NeutrinoSample()
    eg = EventGenerator()
    countsmap = eg.atmEvent(N_yr)
    w_auto = ns.getPowerSpectrumFromCountsmap(countsmap, spectralIndex=3.7, idx_mask=Defaults.idx_muon)


    fluxmap = file_utils.read_maps_from_fits(fluxmappath, Defaults.NEbin)
    w_auto_IC3yr = ns.getPowerSpectrumFromFluxmap(fluxmap, idx_mask=Defaults.idx_muon)




    if MAKE_TEST_PLOTS:
        figs = FigureDict()
        o_dict = figs.setup_figure('PowerSpectrum_atm', xlabel=r'$\ell$', ylabel='auto correlation', figsize=(8, 6))
        fig = o_dict['fig']
        axes = o_dict['axes']
        axes.set_yscale('log')
        axes.set_xscale('log')
        axes.set_ylim(3e-6, 3)
        color = ['r', 'orange', 'limegreen', 'skyblue', 'mediumslateblue', 'purple', 'grey']

        for i in range(Defaults.NEbin):
            axes.plot(Defaults.ell, w_auto_IC3yr[i], color=color[i], lw=2, label=str(i))

        fig.legend()
        figs.plot_cl('PowerSpectrum_atm', Defaults.ell, w_auto, xlabel=r'$\ell$', ylabel='auto correlation', colors=color, ymin=1e-6, ymax=10, lw=3)
        figs.save_all(testfigpath, 'pdf')




if __name__ == '__main__':
    #test_energySpectrum()
    #test_fluxmap()
    test_autoCorrelation()
