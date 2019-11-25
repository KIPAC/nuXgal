import os

import numpy as np

import pytest

import healpy as hp

from scipy import integrate

from KIPAC.nuXgal import Analyze

from KIPAC.nuXgal import EventGenerator

from KIPAC.nuXgal import Defaults

from KIPAC.nuXgal import file_utils

from KIPAC.nuXgal import hp_utils

from KIPAC.nuXgal import FigureDict

from KIPAC.nuXgal import Utilityfunc

from .Utils import MAKE_TEST_PLOTS

astropath = os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'eventmap_astro{i}.fits')
bgpath = os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'eventmap_atm{i}.fits')
ggclpath = os.path.join(Defaults.NUXGAL_ANCIL_DIR, 'Cl_ggRM.dat')
ggsamplepath = os.path.join(Defaults.NUXGAL_ANCIL_DIR, 'galaxySampleOverdensity.fits')
testfigpath = os.path.join(Defaults.NUXGAL_PLOT_DIR, 'test')

for dirname in [Defaults.NUXGAL_SYNTHETICDATA_DIR, Defaults.NUXGAL_PLOT_DIR]:
    try:
        os.makedirs(dirname) 
    except OSError:
        pass


# --- EventGenerator tests ---
def astroEvent_galaxy(seed_g=42):
    eg = EventGenerator()
    # generate density from galaxy cl
    cl_galaxy = file_utils.read_cls_from_txt(ggclpath)[0]
    density_g = Utilityfunc.density_cl(cl_galaxy * 0.6, Defaults.NSIDE, seed_g)
    density_g = np.exp(density_g) - 1.0

    # calculate expected event number using IceCube diffuse neutrino flux
    dN_dE_astro = lambda E_GeV: 1.44E-18 * (E_GeV / 100e3)**(-2.28) # GeV^-1 cm^-2 s^-1 sr^-1, muon neutrino
    # total expected number of events before cut
    N_2012_Aeffmax = np.zeros(Defaults.NEbin)
    for i in np.arange(Defaults.NEbin):
        N_2012_Aeffmax[i] = integrate.quad(dN_dE_astro, Defaults.map_E_edge[i],
                                           Defaults.map_E_edge[i+1])[0] * (eg.Aeff_max[i] * 1E4) * (333 * 24. * 3600) * 4 * np.pi
        #N_2012_Aeffmax[i] = dN_dE_astro(10.**map_logE_center[i]) * (Aeff_max[i] * 1E4) * (333 * 24. * 3600) * 4 * np.pi * (10.**map_logE_center[i] * np.log(10.) * dlogE) * 1

    eventmap = eg.astroEvent_galaxy(density_g, N_2012_Aeffmax)
    if seed_g == 42:
        basekey = 'eventmap_astro'
    else:
        basekey = 'eventmap_astro_nonGalaxy'

    filename_format = os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, basekey + '{i}.fits')
    file_utils.write_maps_from_fits(eventmap, filename_format)

    if MAKE_TEST_PLOTS:
        figs = FigureDict()
        figs.mollview_maps(basekey, eventmap)
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

    file_utils.write_maps_from_fits(eventmap, bgpath)

    if MAKE_TEST_PLOTS:
        figs = FigureDict()
        figs.mollview_maps('eventmap_atm', eventmap)
        figs.save_all(testfigpath, 'pdf')


def test_EventGenerator():
    for i in range(0, 5):
        atmBG_coszenith(i)

    astroEvent_galaxy()
    atmBG()



def test_SyntheticData():

    cf = Analyze()

    # --- event -

    countsmap = file_utils.read_maps_from_fits(astropath, Defaults.NEbin)
    intensity_astro = cf.getIntensity(countsmap)

    # --- atm -
    exposuremap_theta, _ = hp.pixelfunc.pix2ang(Defaults.NSIDE,
                                                np.arange(hp.pixelfunc.nside2npix(Defaults.NSIDE)))

    bgmap = file_utils.read_maps_from_fits(bgpath, Defaults.NEbin)

    # southern sky mask
    mask_muon = np.where(exposuremap_theta > 85. / 180 * np.pi)
    bgmap_nu = hp_utils.vector_apply_mask(bgmap, mask_muon, copy=True)
    intensity_atm_nu = cf.getIntensity(bgmap_nu)

    # northern sky mask
    mask_north = np.where(exposuremap_theta < 85. / 180 * np.pi)
    bgmap_mu = hp_utils.vector_apply_mask(bgmap, mask_north, copy=True)
    intensity_atm_mu = cf.getIntensity(bgmap_mu)

    if MAKE_TEST_PLOTS:

        figs = FigureDict()

        intensities = [intensity_astro, intensity_atm_nu, intensity_atm_mu]

        plot_dict = dict(colors=['b', 'orange', 'k'],
                         markers=['o', '^', 's'])

        figs.plot_intesity_E2('SED', Defaults.map_E_center, intensities, **plot_dict)


        #atm_nu_mu = np.loadtxt(os.path.join(Defaults.NUXGAL_ANCIL_DIR, 'atm_nu_mu.txt'))
        #plt.plot(10.** atm_nu_mu[:, 0], atm_nu_mu[:, 1], lw=2, label=r'atm $\nu_\mu$', color=color_atm_nu)

        #ene = np.logspace(1, 7, 40)

        # Fig 24 of 1506.07981
        #plt.plot(ene, 50 / ((10.**4.6)**3.7) * (ene / 10.**4.6)**(-3.78) * ene **2, lw=2, label=r'atm $\mu$', color=color_atm_mu)

        #plt.plot(ene, 1.44e-18 * (ene/100e3)**(-2.28) * ene**2 * 1, lw=2, label=r'IceCube $\nu_\mu$', color=color_astro)
        #plt.legend()
        figs.save_all(testfigpath, 'pdf')


def test_PowerSpectrum():

    cf = Analyze()

    bgmap = file_utils.read_maps_from_fits(bgpath, Defaults.NEbin)
    cl_nu = cf.powerSpectrumFromCountsmap(bgmap)

    if MAKE_TEST_PLOTS:

        figs = FigureDict()
        color = ['r', 'orange', 'limegreen', 'skyblue', 'mediumslateblue', 'purple', 'grey']

        figs.plot_cl('PowerSpectrum_atm', cf.l_cl, cl_nu,
                     xlabel="l", ylavel=r'$C_{l}$', figsize=(8, 6),
                     colors=color)
        figs.save_all(testfigpath, 'pdf')


def test_CrossCorrelation():
    cf = Analyze()
    eg = EventGenerator()
    bgmap = eg.atmEvent(1.)

    astromap = file_utils.read_maps_from_fits(astropath, Defaults.NEbin)
    countsmap = bgmap + astromap
    w_cross = cf.crossCorrelationFromCountsmap(countsmap)

    if MAKE_TEST_PLOTS:

        figs = FigureDict()
        color = ['r', 'orange', 'limegreen', 'skyblue', 'mediumslateblue', 'purple', 'grey']

        o_dict = figs.setup_figure('Wcross', xlabel="l", ylavel=r'$C_{l}$', figsize=(8, 6))
        axes = o_dict['axes']
        axes.plot(cf.l, cf.cl_galaxy, color='k', label='galaxy cl', lw=2)
        figs.plot_cl('Wcross', cf.l_cl, np.abs(w_cross),
                     xlabel="l", ylavel=r'$C_{l}$', figsize=(8, 6),
                     colors=color)
        figs.save_all(testfigpath, 'pdf')




# --- Calculate cross correlation ---

# generate galaxy sample
def generateGalaxy():

    randomSeed = 42

    cl_galaxy = file_utils.read_cls_from_txt(ggclpath)[0]

    density_g = Utilityfunc.density_cl(cl_galaxy, Defaults.NSIDE, randomSeed)
    density_g = np.exp(density_g) - 1.0
    N_g = 2000000
    events_map_g = Utilityfunc.poisson_sampling(density_g, N_g)
    overdensityMap_g = Utilityfunc.overdensityMap(events_map_g)
    hp.fitsfunc.write_map(ggsamplepath, overdensityMap_g, overwrite=True)



def test_w_cross_plot():
    cf = Analyze()

    astromap = file_utils.read_maps_from_fits(astropath, Defaults.NEbin)

    eg = EventGenerator()
    bgmap = eg.atmEvent(1.-0.003) # astro event is about 0.3% of total event
    countsmap = bgmap + astromap
    #print(np.sum(bgmap), np.sum(astromap))

    w_cross = cf.crossCorrelationFromCountsmap(countsmap)

    # get standard derivation
    w_cross_mean, w_cross_std = cf.crossCorrelation_atm_std(50)

    chi_square_index = 1
    chi_square = 0

    print('------------------')
    print('energy bin', 'chi2', 'dof', 'sigma')

    for i in np.arange(Defaults.NEbin-2):
        chi_square_i = np.sum((w_cross[i][chi_square_index:] - w_cross_mean[i][chi_square_index:]) ** 2 / w_cross_std[i][chi_square_index:]**2)
        chi_square += chi_square_i
        print(i, chi_square_i, len(w_cross[0][chi_square_index:]),
              Utilityfunc.significance(chi_square_i, len(w_cross[0][chi_square_index:])))


    print('total', chi_square, (Defaults.NEbin - 2) * len(w_cross[0][chi_square_index:]),
          Utilityfunc.significance(chi_square, (Defaults.NEbin - 2) * len(w_cross[0][chi_square_index:])))
    print('------------------')


    #astromap = np.zeros((Defaults.NEbin, Defaults.NPIXEL))
    #for i in np.arange(Defaults.NEbin):
    #     astromap[i] = hp.fitsfunc.read_map(os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR,
    #                                                     'eventmap_astro_nonGalaxy' + str(i)+'.fits'), verbose=False)
    countsmap = bgmap #+ astromap
    #print(np.sum(bgmap), np.sum(astromap))
    chi_square = 0
    w_cross = cf.crossCorrelationFromCountsmap(countsmap)
    print('------------------')
    print('energy bin', 'chi2', 'dof', 'sigma')

    for i in np.arange(Defaults.NEbin-2):
        chi_square_i = np.sum((w_cross[i][chi_square_index:] - w_cross_mean[i][chi_square_index:]) ** 2 / w_cross_std[i][chi_square_index:]**2)
        chi_square += chi_square_i
        print(i, chi_square_i,
              len(w_cross[0][chi_square_index:]),
              Utilityfunc.significance(chi_square_i, len(w_cross[0][chi_square_index:])))


    print('total', chi_square, (Defaults.NEbin - 2) * len(w_cross[0][chi_square_index:]),
          Utilityfunc.significance(chi_square, (Defaults.NEbin - 2) * len(w_cross[0][chi_square_index:])))
    print('------------------')

    if MAKE_TEST_PLOTS:

        figs = FigureDict()
        color = ['r', 'orange', 'limegreen', 'skyblue', 'mediumslateblue', 'purple', 'grey']

        figs.plot_cl('w_cross', cf.l_cl, w_cross,
                     xlabel="l", ylavel=r'$C_{l}$', figsize=(8, 6),
                     colors=color, yerrs=[w_cross_mean, w_cross_std])

        figs.save_all(testfigpath, 'pdf')


def test_w_cross_sigma():
    cf = Analyze()
    # get standard derivation
    w_cross_mean, w_cross_std = cf.crossCorrelation_atm_std(50)

    eg = EventGenerator()
    chi_square_index = 1
    chi_square_Ebin = np.zeros(Defaults.NEbin)
    N_realization = 10
    for _ in range(N_realization):
        countsmap = eg.atmEvent(1.)
        w_cross = cf.crossCorrelationFromCountsmap(countsmap)
        for i in np.arange(Defaults.NEbin-2):
            chi_square_i = np.sum((w_cross[i][chi_square_index:] - w_cross_mean[i][chi_square_index:]) ** 2 / w_cross_std[i][chi_square_index:]**2)
            chi_square_Ebin[i] += chi_square_i
            #print(i, chi_square_i)
        #print('--------')
    #print(chi_square_Ebin / N_realization)
    sigma_Ebin = np.zeros(Defaults.NEbin)
    for i in np.arange(Defaults.NEbin - 2):
        sigma_Ebin[i] = Utilityfunc.significance(chi_square_Ebin[i] / N_realization,
                                                 len(w_cross[0][chi_square_index:]))
        #print(sigma_Ebin[i], significance(chi_square_Ebin[i] / N_realization, len(w_cross[0][chi_square_index:])))


    astromap = file_utils.read_maps_from_fits(astropath, Defaults.NEbin)

    chi_square_Ebin = np.zeros(Defaults.NEbin)
    N_realization = 10
    for _ in range(N_realization):
        countsmap = eg.atmEvent(1.-0.003) + astromap
        w_cross = cf.crossCorrelationFromCountsmap(countsmap)
        for i in np.arange(Defaults.NEbin-2):
            chi_square_i = np.sum((w_cross[i][chi_square_index:] - w_cross_mean[i][chi_square_index:]) ** 2 / w_cross_std[i][chi_square_index:]**2)
            chi_square_Ebin[i] += chi_square_i
            #print(i, chi_square_i)
        #print('--------')
    #print(chi_square_Ebin / N_realization)
    sigma_Ebin_signal = np.zeros(Defaults.NEbin)
    for i in np.arange(Defaults.NEbin - 2):
        sigma_Ebin_signal[i] = Utilityfunc.significance(chi_square_Ebin[i] / N_realization,
                                                        len(w_cross[0][chi_square_index:]))


    if MAKE_TEST_PLOTS:

        figs = FigureDict()

        o_dict = figs.setup_figure('sigma_E',
                                   xlabel=r'$\log (E / {\rm GeV})$',
                                   ylabel='Significance', figsize=(8, 6))
        fig = o_dict['fig']
        axes = o_dict['axes']

        axes.set_xlim(1.5, 7.5)
        axes.scatter(Defaults.map_logE_center, sigma_Ebin, marker='o', c='k', label='atm')
        axes.scatter(Defaults.map_logE_center, sigma_Ebin_signal, marker='x', c='r', label='atm+astro')
        fig.legend()
        figs.save_all(testfigpath, 'pdf')


if __name__ == '__main__':
    test_EventGenerator()
    test_SyntheticData()
    test_PowerSpectrum()
    test_CrossCorrelation()
    test_w_cross_plot()
    test_w_cross_sigma()
