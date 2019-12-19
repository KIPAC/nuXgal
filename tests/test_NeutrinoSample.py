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
from KIPAC.nuXgal.WeightedAeff import WeightedAeff



try:
    from Utils import MAKE_TEST_PLOTS
except ImportError:
    from .Utils import MAKE_TEST_PLOTS


testfigpath = os.path.join(Defaults.NUXGAL_PLOT_DIR, 'test')
fluxmappath = os.path.join(Defaults.NUXGAL_DATA_DIR, 'IceCube3yr_fluxmap{i}.fits')
IC3yrfluxmap = file_utils.read_maps_from_fits(fluxmappath, Defaults.NEbin)
N_yr = 2 + 79./86

def test_energySpectrum():

    ns = NeutrinoSample()
    ns.inputFluxmap(IC3yrfluxmap)
    intensity_atm_nu = ns.getIntensity(dt_years=N_yr, idx_mask=Defaults.idx_muon)
    #print ( intensity_atm_nu * Defaults.map_E_center**2)

    eg = EventGenerator()
    gs = GalaxySample('analy')
    f_atm = np.zeros(Defaults.NEbin)
    f_diff = 1.
    for i in range(Defaults.NEbin):
        if eg.nnus[i] != 0.:
            f_atm[i] = 1. - eg.Nastro_1yr_Aeffmax[i] * f_diff / eg.nnus[i]

    print ('fraction of atm events:', f_atm)

    # generate atmospheric eventmaps
    Natm = np.random.poisson(eg.nevts * N_yr * f_atm)
    eg._atm_gen.nevents_expected.set_value(Natm, clear_parent=False)
    atm_map = eg._atm_gen.generate_event_maps(1)[0]

    # generate astro maps
    Nastro = np.random.poisson(eg.Nastro_1yr_Aeffmax * N_yr * f_diff)
    astro_map = eg.astroEvent_galaxy(Nastro, gs.density)

    ns.inputCountsmap(atm_map, spectralIndex=3.7)
    intensity_atm_nu_syn = ns.getIntensity(dt_years=N_yr, idx_mask=Defaults.idx_muon)

    ns.inputCountsmap(astro_map, spectralIndex=2.28)
    intensity_astro_nu = ns.getIntensity(dt_years=N_yr, idx_mask=Defaults.idx_muon)


    if MAKE_TEST_PLOTS:
        figs = FigureDict()
        intensities = [intensity_astro_nu, intensity_atm_nu_syn, intensity_atm_nu]
        plot_dict = dict(colors=['b', 'orange', 'k'],
                         markers=['o', '^', 's'])
        figs.plot_intesity_E2('SED', Defaults.map_E_center, intensities, **plot_dict)
        atm_nu_mu = np.loadtxt(os.path.join(Defaults.NUXGAL_ANCIL_DIR, 'atm_nu_mu.txt'))
        figs.plot('SED', 10.** atm_nu_mu[:, 0], atm_nu_mu[:, 1], lw=2, label=r'atm $\nu_\mu$', color='orange')
        ene = np.logspace(1, 7, 40)
        figs.plot('SED', ene, 1.44e-18 * (ene/100e3)**(-2.28) * ene**2 * 1, lw=2, label=r'IceCube $\nu_\mu$', color='b')
        figs.save_all(testfigpath, 'pdf')




def test_fluxmap():
    ns = NeutrinoSample()
    ns.inputFluxmap(IC3yrfluxmap)
    ns.plotFluxmap(testfigpath+'IC', Defaults.idx_muon)

    eg = EventGenerator()
    atmCounts = eg.atmEvent(N_yr)
    ns.inputCountsmap(atmCounts, 3.7)
    ns.plotFluxmap(testfigpath, Defaults.idx_muon)

    mask = np.zeros(Defaults.NPIXEL)
    mask[Defaults.idx_muon] = 1.
    for i in range(Defaults.NEbin):
        test = np.ma.masked_array(atmCounts[i], mask = mask)
        print (i, test.sum())


def test_autoCorrelation():
    ns = NeutrinoSample()
    ns.inputFluxmap(IC3yrfluxmap)
    w_auto_IC3yr = ns.getPowerSpectrum(idx_mask=Defaults.idx_muon)

    eg = EventGenerator()
    N_yr = 2 + 79./86
    countsmap = eg.atmEvent(N_yr)
    ns.inputCountsmap(countsmap, spectralIndex=3.7)
    w_auto = ns.getPowerSpectrum(idx_mask=Defaults.idx_muon)


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
    test_energySpectrum()
    #test_fluxmap()
    #test_autoCorrelation()
