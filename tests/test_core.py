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
    eg = EventGenerator(year='IC86-2012', galaxySample=gs, astroModel='numu')
    Nastro = np.random.poisson(eg.Nastro_1yr_Aeffmax * f_diff)
    print (Nastro)
    eventmap = eg.astroEvent_galaxy(Nastro, gs.density)
    file_utils.write_maps_to_fits(eventmap, astropath)

    if MAKE_TEST_PLOTS:
        figs = FigureDict()
        figs.mollview_maps('astro', eventmap)
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



def test_energySpectrum():

    ns_astro = NeutrinoSample()
    ns_astro.inputData(astropath)

    ns_atm = NeutrinoSample()
    ns_atm.inputData(bgpath)

    # apply southern sky mask
    ns_atm.updateMask(Defaults.idx_muon)
    intensity_atm_nu = ns_atm.getIntensity(1., spectralIndex=3.7, year='IC86-2012')

    ns_astro.updateMask(Defaults.idx_muon)
    # assuming an E^-2.28 spectrum inside energy bins for estimation of effective area
    intensity_astro = ns_astro.getIntensity(1., spectralIndex=2.28, year='IC86-2012')


    #print (intensity_astro * Defaults.map_E_center**2 , intensity_atm_nu * Defaults.map_E_center**2)

    if MAKE_TEST_PLOTS:

        figs = FigureDict()

        intensities = [intensity_astro, intensity_atm_nu]

        plot_dict = dict(colors=['b', 'orange', 'k'],
                         markers=['o', '^', 's'])

        figs.plot_intesity_E2('SED', Defaults.map_E_center, intensities, **plot_dict)


        atm_nu_mu = np.loadtxt(os.path.join(Defaults.NUXGAL_ANCIL_DIR, 'atm_nu_mu.txt'))
        figs.plot('SED', 10.** atm_nu_mu[:, 0], atm_nu_mu[:, 1], lw=2, label=r'atm $\nu_\mu$', color='orange')

        ene = np.logspace(1, 7, 40)

        # Fig 24 of 1506.07981
        #plt.plot(ene, 50 / ((10.**4.6)**3.7) * (ene / 10.**4.6)**(-3.78) * ene **2, lw=2, label=r'atm $\mu$', color=color_atm_mu)

        figs.plot('SED', ene, 1.44e-18 * (ene/100e3)**(-2.28) * ene**2 * 1, lw=2, label=r'IceCube $\nu_\mu$', color='b')
        figs.save_all(testfigpath, 'pdf')


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




def test_Demonstration():
    """
    Demonstration of concept
    """
    randomSeed = 45
    gs = GALAXY_LIBRARY.get_sample('analy')
    cl_galaxy = gs.analyCL
    cf = Analyze(gs.getOverdensity('analy'))

    # source_1
    np.random.seed(randomSeed)
    density_g = hp.sphtfunc.synfast(cl_galaxy, Defaults.NSIDE)
    density_g = np.exp(density_g)
    density_g /= density_g.sum()
    N_g = 20000000
    events_map_g = np.random.poisson(density_g * N_g)
    overdensityMap_g = Utilityfunc.overdensityMap(events_map_g)
    powerSpectrum_g = hp.sphtfunc.anafast(overdensityMap_g)


    # source_2, with same randomness, N_realization
    N_realization = 200
    N_nu = 100000

    w_cross_array = np.zeros((N_realization, Defaults.NCL))
    powerSpectrum_nu_array = np.zeros((N_realization, Defaults.NCL))

    np.random.seed(randomSeed)
    density_nu = hp.sphtfunc.synfast(cl_galaxy * 0.6, Defaults.NSIDE)
    density_nu = np.exp(density_nu)
    density_nu /= density_nu.sum()
    #print (np.where((density_nu - density_g) != 0))
    #density_nu = density_g

    for i in range(N_realization):
        if i % 100 == 0:
            print (i)
        events_map_nu = np.random.poisson(density_nu * N_nu)
        overdensityMap_nu = Utilityfunc.overdensityMap(events_map_nu)
        powerSpectrum_nu_array[i] = hp.sphtfunc.anafast(overdensityMap_nu)
        w_cross_array[i] = hp.sphtfunc.anafast(overdensityMap_g, overdensityMap_nu)

    powerSpectrum_nu_mean = np.mean(powerSpectrum_nu_array, axis=0)
    w_cross_mean = np.mean(w_cross_array, axis=0)


    # source_3, with a different pattern of randomness, N_realization
    w_cross_array3 = np.zeros((N_realization, Defaults.NCL))
    np.random.seed(randomSeed+10)
    density_nu3 = hp.sphtfunc.synfast(cl_galaxy * 0.6, Defaults.NSIDE)
    density_nu3 = np.exp(density_nu3)
    density_nu3 /= density_nu3.sum()


    for i in range(N_realization):
        if i % 100 == 0:
            print (i)
        events_map_nu = np.random.poisson(density_nu3 * N_nu)
        overdensityMap_nu = Utilityfunc.overdensityMap(events_map_nu)
        w_cross_array3[i] = hp.sphtfunc.anafast(overdensityMap_g, overdensityMap_nu)

    w_cross_mean3 = np.mean(w_cross_array3, axis=0)



    if MAKE_TEST_PLOTS:
        figs = FigureDict()
        o_dict = figs.setup_figure('Demonstration', xlabel='$l$', ylabel='$C_l$', figsize=(8, 6))
        fig = o_dict['fig']
        axes = o_dict['axes']
        axes.set_xscale('log')
        axes.plot(Defaults.ell, cl_galaxy, 'k-')
        axes.plot(Defaults.ell,  powerSpectrum_g, lw=2, label='g')
        axes.plot(Defaults.ell,  w_cross_mean, lw=2, label='nuXg, same random seed')
        axes.plot(Defaults.ell,  w_cross_mean3, lw=2, label='nuXg, different random seed')
        fig.legend()
        figs.save_all(testfigpath, 'pdf')

        figs = FigureDict()
        o_dict = figs.setup_figure('Demonstration', xlabel='$l$', ylabel='$C_l$', figsize=(8, 6))
        fig = o_dict['fig']
        axes = o_dict['axes']
        axes.set_yscale('log')
        axes.set_xscale('log')
        axes.set_ylim(3e-6, 3e-4)
        axes.plot(Defaults.ell, cl_galaxy, 'k-')
        axes.plot(Defaults.ell, cl_galaxy * 0.6, 'k-')
        axes.plot(Defaults.ell, cl_galaxy * 0.6 ** 0.5, 'k-')
        axes.plot(Defaults.ell,  powerSpectrum_g, lw=2, label='g')
        axes.plot(Defaults.ell,  powerSpectrum_nu_mean, lw=2, label='nu')
        axes.plot(Defaults.ell,  w_cross_mean, lw=2, label='nuXg, same random seed')
        axes.plot(Defaults.ell,  w_cross_mean3, lw=2, label='nuXg, different random seed')
        fig.legend()
        figs.save_all(testfigpath, 'pdf')




if __name__ == '__main__':
    #test_Demonstration()

    test_EventGenerator()
    test_PowerSpectrum()
    test_CrossCorrelation()
    test_energySpectrum()
