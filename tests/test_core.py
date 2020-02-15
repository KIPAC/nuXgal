import os

import numpy as np

import pytest

import healpy as hp

from scipy import integrate

from KIPAC.nuXgal import EventGenerator

from KIPAC.nuXgal import Defaults

from KIPAC.nuXgal import file_utils

from KIPAC.nuXgal import hp_utils

from KIPAC.nuXgal import FigureDict

from KIPAC.nuXgal.GalaxySample import GALAXY_LIBRARY

from KIPAC.nuXgal.NeutrinoSample import NeutrinoSample

try:
    from Utils import MAKE_TEST_PLOTS
except ImportError:
    from .Utils import MAKE_TEST_PLOTS

astropath = os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'eventmap_astro{i}.fits')
bgpath = os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'eventmap_atm{i}.fits')
testfigpath = os.path.join(Defaults.NUXGAL_PLOT_DIR, 'test')

for dirname in [Defaults.NUXGAL_SYNTHETICDATA_DIR, Defaults.NUXGAL_PLOT_DIR]:
    try:
        os.makedirs(dirname)
    except OSError:
        pass





# --- EventGenerator tests ---
def astroEvent_galaxy(f_diff = 1.):
    gs = GALAXY_LIBRARY.get_sample('analy')
    eg = EventGenerator(year='IC86-2012', astroModel='observed_numu_fraction')
    N_astro_north_obs = np.random.poisson(eg.nevts * 1 * eg.f_astro_north_truth)
    N_astro_north_exp = [N_astro_north_obs[i] / np.sum(eg._astro_gen.prob_reject()[i] * gs.density) for i in range(Defaults.NEbin)]
    astro_map = eg.astroEvent_galaxy(np.array(N_astro_north_exp), gs.density)
    file_utils.write_maps_to_fits(astro_map, astropath)

    if MAKE_TEST_PLOTS:
        figs = FigureDict()
        figs.mollview_maps('astro', astro_map)
        figs.save_all(testfigpath, 'pdf')




def atmBG_coszenith(energyBin=0):

    eg = EventGenerator()
    N_coszenith = eg.atm_gen.coszenith()[energyBin]
    recovered_values = eg.atmBG_coszenith(int(np.sum(N_coszenith[:, 1])), energyBin)

    index = np.where(np.abs(recovered_values) > 1)
    if len(index) > 1:
        print(index, recovered_values[index])

    if MAKE_TEST_PLOTS:
        figs = FigureDict()
        figkey = 'N_coszenith'+str(energyBin)
        o_dict = figs.setup_figure(figkey, xlabel=r'$\cos\,\theta$', ylabel='Number of counts', figsize=(8, 6))
        fig = o_dict['fig']
        axes = o_dict['axes']
        axes.plot(N_coszenith[:, 0], N_coszenith[:, 1], lw=2, label='data')
        axes.hist(recovered_values, N_coszenith[:, 0], label='mock')
        fig.legend()
        figs.save_all(testfigpath, 'pdf')



def atmBG():
    eg = EventGenerator()
    eventmap = eg.atmEvent(1.)
    eventmap2 = np.zeros((Defaults.NEbin, Defaults.NPIXEL))

    file_utils.write_maps_to_fits(eventmap, bgpath)
    for i in range(Defaults.NEbin):
        eventmap2[i] = eventmap[i]
        eventmap2[i][Defaults.idx_muon] = hp.UNSEEN


    mask = np.zeros(Defaults.NPIXEL)
    mask[Defaults.idx_muon] = 1.
    for i in range(Defaults.NEbin):
        test = np.ma.masked_array(eventmap[i], mask = mask)
        print (test.sum())


    if MAKE_TEST_PLOTS:
        figs = FigureDict()
        figs.mollview_maps('eventmap_atm', eventmap2)
        figs.save_all(testfigpath, 'pdf')


def test_EventGenerator():
    for i in range(0, 5):
        atmBG_coszenith(i)

    astroEvent_galaxy()
    atmBG()



def test_PowerSpectrum():

    ns_atm = NeutrinoSample()
    ns_atm.inputData(bgpath)

    gs = GALAXY_LIBRARY.get_sample('analy')


    if MAKE_TEST_PLOTS:

        figs = FigureDict()
        color = ['r', 'orange', 'limegreen', 'skyblue', 'mediumslateblue', 'purple', 'grey']

        figs.plot_cl('PowerSpectrum_atm', Defaults.ell, ns_atm.getPowerSpectrum(),
                     xlabel="l", ylavel=r'$C_{l}$', figsize=(8, 6),
                     colors=color)
        figs.plot('PowerSpectrum_atm', Defaults.ell, gs.analyCL[0:3*Defaults.NSIDE], color='k', linestyle='--', lw=2)
        figs.save_all(testfigpath, 'pdf')


def test_CrossCorrelation():
    ns_astro = NeutrinoSample()
    ns_astro.inputData(astropath)



    gs = GALAXY_LIBRARY.get_sample('analy')
    w_cross = ns_astro.getCrossCorrelation(gs.overdensityalm)

    if MAKE_TEST_PLOTS:
        figs = FigureDict()
        color = ['r', 'orange', 'limegreen', 'skyblue', 'mediumslateblue', 'purple', 'grey']

        o_dict = figs.setup_figure('Wcross', xlabel="l", ylabel=r'$w$', figsize=(6, 8))
        axes = o_dict['axes']
        for i in range(Defaults.NEbin):
            axes.plot(Defaults.ell, gs.analyCL[0:3*Defaults.NSIDE] * 10 ** (i * 2), color='k', lw=2)
            w_cross[i] *= 10 ** (i*2)


        figs.plot_cl('Wcross', Defaults.ell, np.abs(w_cross),
                     xlabel="l", ylabel=r'$C_{l}$',
                     colors=color, ymin=1e-7, ymax=1e10, lw=3)

        figs.save_all(testfigpath, 'pdf')





if __name__ == '__main__':

    test_EventGenerator()
    test_PowerSpectrum()
    test_CrossCorrelation()
