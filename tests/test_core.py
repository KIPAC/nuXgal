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

from Utils import MAKE_TEST_PLOTS

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

randomSeed_g = 42


# generate galaxy sample
def generateGalaxy():

    randomSeed = randomSeed_g

    cl_galaxy = file_utils.read_cls_from_txt(ggclpath)[0]
    np.random.seed(randomSeed)
    density_g = hp.sphtfunc.synfast(cl_galaxy, Defaults.NSIDE)
    #density_g = Utilityfunc.density_cl(cl_galaxy, Defaults.NSIDE, randomSeed)
    
    density_g = np.exp(density_g) #- 1.0
    #print (np.where(density_g < 1))
    density_g /= density_g.sum()
    N_g = 2000000
    expected_counts_map = density_g * N_g
    events_map_g = np.random.poisson(expected_counts_map)
    #events_map_g = Utilityfunc.poisson_sampling(density_g, N_g)
    overdensityMap_g = Utilityfunc.overdensityMap(events_map_g)
    hp.fitsfunc.write_map(ggsamplepath, overdensityMap_g, overwrite=True)



# --- EventGenerator tests ---
def astroEvent_galaxy(seed_g=42):
    eg = EventGenerator()
    
    # calculate expected event number using IceCube diffuse neutrino flux
    dN_dE_astro = lambda E_GeV: 1.44E-18 * (E_GeV / 100e3)**(-2.28) # GeV^-1 cm^-2 s^-1 sr^-1, muon neutrino
    # total expected number of events before cut
    N_2012_Aeffmax = np.zeros(Defaults.NEbin)
    for i in np.arange(Defaults.NEbin):
        N_2012_Aeffmax[i] = integrate.quad(dN_dE_astro, Defaults.map_E_edge[i],
                                           Defaults.map_E_edge[i+1])[0] * (eg.Aeff_max[i] * 1E4) * (333 * 24. * 3600) * 4 * np.pi
        #N_2012_Aeffmax[i] = dN_dE_astro(10.**map_logE_center[i]) * (Aeff_max[i] * 1E4) * (333 * 24. * 3600) * 4 * np.pi * (10.**map_logE_center[i] * np.log(10.) * dlogE) * 1

    eventmap = eg.astroEvent_galaxy(0.6, N_2012_Aeffmax)
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

    cf = Analyze()

    bgmap = file_utils.read_maps_from_fits(bgpath, Defaults.NEbin)
    cl_nu = cf.powerSpectrumFromCountsmap(bgmap)

    if MAKE_TEST_PLOTS:

        figs = FigureDict()
        color = ['r', 'orange', 'limegreen', 'skyblue', 'mediumslateblue', 'purple', 'grey']

        figs.plot_cl('PowerSpectrum_atm', cf.l_cl, cl_nu,
                     xlabel="l", ylavel=r'$C_{l}$', figsize=(8, 6),
                     colors=color)
        figs.plot('PowerSpectrum_atm', cf.l, cf.cl_galaxy, color='k', linestyle='--', lw=2)
        figs.save_all(testfigpath, 'pdf')


def test_CrossCorrelation():
    cf = Analyze()
    eg = EventGenerator()
    bgmap = file_utils.read_maps_from_fits(bgpath, Defaults.NEbin)

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

def getEventNumberEbin(f_diff):
    """
    Compute event number using nu_mu diffuse flux. Return in the form of (NEbin)
    """
    eg = EventGenerator()

    # calculate expected event number using IceCube diffuse neutrino flux
    dN_dE_astro = lambda E_GeV: 1.44E-18 * (E_GeV / 100e3)**(-2.28) # GeV^-1 cm^-2 s^-1 sr^-1, muon neutrino
    # total expected number of events before cut
    N_2012_Aeffmax = np.zeros(Defaults.NEbin)
    for i in np.arange(Defaults.NEbin):
        N_2012_Aeffmax[i] = integrate.quad(dN_dE_astro, Defaults.map_E_edge[i],
                                           Defaults.map_E_edge[i+1])[0] * (eg.Aeff_max[i] * 1E4) * (333 * 24. * 3600) * 4 * np.pi * f_diff
    return N_2012_Aeffmax



def test_MeanCrossCorrelation(N_realization = 50, f_gal = 0.6, f_diff=1):
    """
    Cross correlation of astro events generated using IceCube effective area
    """
    cf = Analyze()
    eg = EventGenerator()
    N_2012_Aeffmax = getEventNumberEbin(f_diff)
    
    np.random.seed(randomSeed_g)
    density_nu = hp.sphtfunc.synfast(cf.cl_galaxy * 0.6, Defaults.NSIDE)
    density_nu = np.exp(density_nu)
    density_nu /= density_nu.sum() # a unique neutrino source distribution that shares the randomness of density_g
    
    
    # southern sky mask
    exposuremap_theta, _ = hp.pixelfunc.pix2ang(Defaults.NSIDE, np.arange(hp.pixelfunc.nside2npix(Defaults.NSIDE)))
    mask_muon = np.where(exposuremap_theta > 85. / 180 * np.pi)

    w_cross_array = np.zeros((N_realization, Defaults.NEbin, Defaults.NCL))
    for i in range(N_realization):
        if i % 100 == 0:
            print(i)
        countsmap = eg.astroEvent_galaxy(f_gal, N_2012_Aeffmax, density_nu)
        countsmap = hp_utils.vector_apply_mask(countsmap, mask_muon, copy=False)

        for j in range(Defaults.NEbin):
            overdensityMap = Utilityfunc.overdensityMap(countsmap[j])
            w_cross_array[i][j] = hp.anafast(cf.overdensityMap_g, overdensityMap)
            #w_cross_array[i][j] = hp.anafast(overdensityMap)

    w_cross_mean = np.mean(w_cross_array, axis=0)
    w_cross_std = np.std(w_cross_array, axis=0)

    cl_galaxy_sample = hp.sphtfunc.anafast(cf.overdensityMap_g)
    
 
    if MAKE_TEST_PLOTS:
        figs = FigureDict()
        color = ['r', 'orange', 'limegreen', 'skyblue', 'mediumslateblue', 'purple', 'grey']

        o_dict = figs.setup_figure('WcrossMean', xlabel="l", ylabel=r'$w$', figsize=(6, 8))
        axes = o_dict['axes']
        for i in range(Defaults.NEbin):
            axes.plot(cf.l, cf.cl_galaxy * 10 ** (i*2), color='grey', lw=1)
            axes.plot(cf.l_cl, cl_galaxy_sample * 10 ** (i*2), color='k', lw=2)

            w_cross_mean[i] *= 10 ** (i*2)
            w_cross_std[i] *= 10** (i * 2)
        figs.plot_cl('WcrossMean', cf.l_cl, np.abs(w_cross_mean),
                     xlabel="l", ylabel=r'$C_{l}$',
                     colors=color, ymin=1e-7, ymax=1e10, lw=3)
        figs.plot_cl('WcrossMean', cf.l_cl, np.abs(w_cross_std), linestyle='-.', colors=color,  ymin=1e-7, ymax=1e10, lw=1)
        
        figs.save_all(testfigpath, 'pdf')



def test_MeanCrossCorrelation_UniformAeff(N_realization = 50, f_gal = 0.6, f_diff=1):
    """
    Cross correlation of astro events generated assuming uniform exposure
    """

    cf = Analyze()
    eg = EventGenerator()
    w_cross_array = np.zeros((N_realization, Defaults.NEbin, Defaults.NCL))
    N_2012_Aeffmax = getEventNumberEbin(f_diff)
 
    np.random.seed(randomSeed_g)
    density_nu = hp.sphtfunc.synfast(cf.cl_galaxy * 0.6, Defaults.NSIDE)
    density_nu = np.exp(density_nu)
    density_nu /= density_nu.sum() # a unique neutrino source distribution that shares the randomness of density_g
    
    eg.astro_gen.nevents_expected.set_value(N_2012_Aeffmax, clear_parent=False)
    eg.astro_gen.normalized_counts_map = density_nu
    
    # southern sky mask
    exposuremap_theta, _ = hp.pixelfunc.pix2ang(Defaults.NSIDE, np.arange(hp.pixelfunc.nside2npix(Defaults.NSIDE)))
    mask_muon = np.where(exposuremap_theta > 85. / 180 * np.pi)



    for i in range(N_realization):
        #countsmap = eg.astroEvent_galaxy(f_gal, N_2012_Aeffmax, density_nu, **kwargs)
        countsmap = eg.astro_gen.generate_event_maps_NoReject(1)[0]
        #countsmap = hp_utils.vector_apply_mask(countsmap, mask_muon, copy=True)

        for j in range(Defaults.NEbin):
            overdensityMap = Utilityfunc.overdensityMap(countsmap[j])

            w_cross_array[i][j] = hp.anafast(cf.overdensityMap_g, overdensityMap)
            #w_cross_array[i][j] = hp.anafast(overdensityMap)

    w_cross_mean = np.mean(w_cross_array, axis=0)
    w_cross_std = np.std(w_cross_array, axis=0)
    
    cl_galaxy_sample = hp.sphtfunc.anafast(cf.overdensityMap_g)
    if MAKE_TEST_PLOTS:
        figs = FigureDict()
        color = ['r', 'orange', 'limegreen', 'skyblue', 'mediumslateblue', 'purple', 'grey']

        o_dict = figs.setup_figure('WcrossMean_uniformAeff', xlabel="l", ylabel=r'$w$', figsize=(6, 8))
        axes = o_dict['axes']
        for i in range(Defaults.NEbin):
            axes.plot(cf.l, cf.cl_galaxy * 10 ** (i*2), color='grey', lw=1)
            axes.plot(cf.l_cl, cl_galaxy_sample * 10 ** (i*2), color='k', lw=2)

            w_cross_mean[i] *= 10 ** (i*2)
            w_cross_std[i] *= 10** (i * 2)
        figs.plot_cl('WcrossMean_uniformAeff', cf.l_cl, np.abs(w_cross_mean),
                     xlabel="l", ylabel=r'$C_{l}$',
                     colors=color, ymin=1e-7, ymax=1e10, lw=3)
        figs.plot_cl('WcrossMean_uniformAeff', cf.l_cl, np.abs(w_cross_std), linestyle='-.', colors=color,  ymin=1e-7, ymax=1e10, lw=1)
        
        figs.save_all(testfigpath, 'pdf')



def test_Demonstration():
    """
    Demonstration of concept
    """
    randomSeed = 45
    cf = Analyze()
    cl_galaxy = file_utils.read_cls_from_txt(ggclpath)[0]

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
        o_dict = figs.setup_figure('test_w_CL', xlabel='$l$', ylabel='$C_l$', figsize=(8, 6))
        fig = o_dict['fig']
        axes = o_dict['axes']
        axes.set_xscale('log')
        axes.plot(cf.l, cl_galaxy, 'k-')
        axes.plot(cf.l_cl,  powerSpectrum_g, lw=2, label='g')
        axes.plot(cf.l_cl,  w_cross_mean, lw=2, label='nuXg, same random seed')
        axes.plot(cf.l_cl,  w_cross_mean3, lw=2, label='nuXg, different random seed')
        fig.legend()
        figs.save_all(testfigpath, 'pdf')

        figs = FigureDict()
        o_dict = figs.setup_figure('test_w_CL_log', xlabel='$l$', ylabel='$C_l$', figsize=(8, 6))
        fig = o_dict['fig']
        axes = o_dict['axes']
        axes.set_yscale('log')
        axes.set_xscale('log')
        axes.set_ylim(3e-6, 3e-4)
        axes.plot(cf.l, cl_galaxy, 'k-')
        axes.plot(cf.l, cl_galaxy * 0.6, 'k-')
        axes.plot(cf.l, cl_galaxy * 0.6 ** 0.5, 'k-')
        axes.plot(cf.l_cl,  powerSpectrum_g, lw=2, label='g')
        axes.plot(cf.l_cl,  powerSpectrum_nu_mean, lw=2, label='nu')
        axes.plot(cf.l_cl,  w_cross_mean, lw=2, label='nuXg, same random seed')
        axes.plot(cf.l_cl,  w_cross_mean3, lw=2, label='nuXg, different random seed')
        fig.legend()
        figs.save_all(testfigpath, 'pdf')




if __name__ == '__main__':

    ## -- tests of functions of the code
    #generateGalaxy()
    #test_EventGenerator()
    #test_SyntheticData()
    #test_PowerSpectrum()
    #test_CrossCorrelation()
    #test_w_cross_plot()
    #test_w_cross_sigma()

    ## -- tests of efficiency of the method
    test_Demonstration()
    
    #test_MeanCrossCorrelation_UniformAeff(50, f_diff=10000)
    #test_MeanCrossCorrelation(50, f_diff=1) # f_diff=100000

