from Analyze import *


font = { 'family': 'Arial',  'weight' : 'normal',   'size'   : 20}
legendfont = {'fontsize' : 18, 'frameon' : False}

def checkSyntheticData():

    cf = Analyze()
    
    # --- event -
    countsmap = np.zeros((cf.NEbin, cf.NPIXEL))
    for i in np.arange(cf.NEbin):
        countsmap[i] = hp.fitsfunc.read_map('../syntheticData/eventmap_astro' + str(i)+'.fits', verbose=False)
    intensity_astro = cf.getIntensity(countsmap)
    
    
    # --- atm -
    exposuremap_theta, exposuremap_phi = hp.pixelfunc.pix2ang(cf.NSIDE, np.arange(hp.pixelfunc.nside2npix(cf.NSIDE)))

    bgmap = np.zeros((cf.NEbin, cf.NPIXEL))
    for i in np.arange(cf.NEbin):
        bgmap[i] = hp.fitsfunc.read_map('../syntheticData/eventmap_atm' + str(i)+'.fits', verbose=False)
        #bgmap[i] = hp.fitsfunc.read_map('../syntheticData/counts_atm' + str(i)+'.fits', verbose=False)


    # southern sky mask
    mask_muon = np.where(exposuremap_theta > 85. / 180 * np.pi)
    bgmap_nu = bgmap
    for i in np.arange(cf.NEbin):
        bgmap_nu[i][mask_muon] = 0.
    intensity_atm_nu = cf.getIntensity(bgmap_nu)
    
    
    bgmap = np.zeros((cf.NEbin, cf.NPIXEL))
    for i in np.arange(cf.NEbin):
        bgmap[i] = hp.fitsfunc.read_map('../syntheticData/eventmap_atm' + str(i)+'.fits', verbose=False)
        

    bgmap_mu = bgmap
    # northern sky mask
    mask_north = np.where(exposuremap_theta < 85. / 180 * np.pi)
    for i in np.arange(cf.NEbin):
        bgmap_mu[i][mask_north] = 0.
    intensity_atm_mu = cf.getIntensity(bgmap_mu)

    

    color_astro, color_atm_nu, color_atm_mu = 'b', 'orange', 'k'
    fig = plt.figure(figsize=(8,6))
    plt.rc('font', **font)
    plt.rc('legend', **legendfont)

    plt.xscale('log')
    plt.yscale('log')
    plt.scatter(10.**cf.map_logE_center, intensity_astro * (10.**cf.map_logE_center)**2, marker='o', c=color_astro)
    plt.scatter(10.**cf.map_logE_center, intensity_atm_nu * (10.**cf.map_logE_center)**2, marker='^', c=color_atm_nu)
    plt.scatter(10.**cf.map_logE_center, intensity_atm_mu * (10.**cf.map_logE_center)**2, marker='s', c=color_atm_mu)


    atm_nu_mu = np.loadtxt('../data/atm_nu_mu.txt')
    plt.plot(10.** atm_nu_mu[:,0], atm_nu_mu[:,1], lw=2, label = r'atm $\nu_\mu$',color=color_atm_nu)

    ene = np.logspace(1, 7, 40)

    # Fig 24 of 1506.07981
    plt.plot(ene, 50 / ((10.**4.6)**3.7) * (ene / 10.**4.6)**(-3.78) * ene **2, lw=2, label=r'atm $\mu$',color=color_atm_mu)

    plt.plot(ene, 1.44e-18 * (ene/100e3)**(-2.28) * ene**2 * 1, lw=2, label=r'IceCube $\nu_\mu$',color=color_astro)
    plt.xlabel('E [GeV]')
    plt.ylabel(r'$E^2 \Phi\,[GeV\,cm^{-2}\,s^{-1}\,sr^{-1}]$')
    plt.ylim(1e-9, 1e-2)
    plt.xlim(1e2, 1e8)
    plt.gcf().subplots_adjust(left=0.18, bottom=0.2, right=0.9)
    plt.legend()
    plt.savefig('../plots/testSED.pdf')



def checkPowerSpectrum():
    cf = Analyze()
    bgmap = np.zeros((cf.NEbin, cf.NPIXEL))
    for i in np.arange(cf.NEbin):
        bgmap[i] = hp.fitsfunc.read_map('../syntheticData/eventmap_atm' + str(i)+'.fits', verbose=False)
    cl_nu = cf.powerSpectrumFromCountsmap(bgmap)
    
    fig = plt.figure(figsize=(8,6))
    plt.rc('font', **font)
    plt.rc('legend', **legendfont)

    color = ['r', 'orange', 'limegreen', 'skyblue', 'mediumslateblue', 'purple','grey']

    for i in np.arange(cf.NEbin):
        plt.plot(cf.l_cl, cl_nu[i], label=str(i), color=color[i])

    plt.plot(cf.l, cf.cl_galaxy, 'k--', lw=2)
    plt.xlabel('l')
    plt.ylabel('Cl')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim(1e-8, 10)
    plt.gcf().subplots_adjust(left=0.18, top=0.9, right=0.9)
    plt.legend(ncol=2)
    plt.savefig('../plots/testPowerSpectrum_atm.pdf')

 


def checkCrossCorrelation():
    cf = Analyze()
    eg = EventGenerator()
    bgmap = eg.atmEvent(1.)
    
    #bgmap = np.zeros((cf.NEbin, cf.NPIXEL))
    #for i in np.arange(cf.NEbin):
    #    bgmap[i] = hp.fitsfunc.read_map('../syntheticData/eventmap_atm' + str(i)+'.fits', verbose=False)
    astromap = np.zeros((cf.NEbin, cf.NPIXEL))
    for i in np.arange(cf.NEbin):
         astromap[i] = hp.fitsfunc.read_map('../syntheticData/eventmap_astro' + str(i)+'.fits', verbose=False)
    countsmap = bgmap + astromap
    w_cross = cf.crossCorrelationFromCountsmap(countsmap)

     
     
    fig = plt.figure(figsize=(8,6))
    plt.rc('font', **font)
    plt.rc('legend', **legendfont)

    plt.xscale('log')
    plt.yscale('log')
    plt.plot(cf.l, cf.cl_galaxy, color='k', label='galaxy cl', lw=2)
    color = ['r', 'orange', 'limegreen', 'skyblue', 'mediumslateblue', 'purple','grey']

    for i in np.arange(cf.NEbin):
        plt.plot(cf.l_cl, np.abs(w_cross[i]), label=str(i), color=color[i])

    plt.xlabel('l')
    plt.ylabel('Cl')
    plt.ylim(1e-8, 10)
    plt.gcf().subplots_adjust(left=0.18, top=0.9, right=0.9)
    plt.legend(ncol=2)
    plt.savefig('../plots/testWcross.pdf')



# --- EventGenerator tests ---


def astroEvent_galaxyTest(seed_g = 42):
    eg = EventGenerator()
    # generate density from galaxy cl
    cl_galaxy_file = np.loadtxt('../data/Cl_ggRM.dat')
    cl_galaxy = cl_galaxy_file[:500]
    density_g = density_cl(cl_galaxy * 0.6, eg.NSIDE, seed_g)
    density_g = np.exp(density_g) - 1.0
    
    # calculate expected event number using IceCube diffuse neutrino flux
    dN_dE_astro = lambda E_GeV: 1.44E-18 * (E_GeV / 100e3)**(-2.28) # GeV^-1 cm^-2 s^-1 sr^-1, muon neutrino
    # total expected number of events before cut
    N_2012_Aeffmax = np.zeros(eg.NEbin)
    for i in np.arange(eg.NEbin):
        N_2012_Aeffmax[i] = integrate.quad(dN_dE_astro, 10.**eg.map_logE_edge[i], 10.**eg.map_logE_edge[i+1])[0] * (eg.Aeff_max[i] * 1E4) * (333 * 24. * 3600) * 4 * np.pi
        #N_2012_Aeffmax[i] = dN_dE_astro(10.**map_logE_center[i]) * (Aeff_max[i] * 1E4) * (333 * 24. * 3600) * 4 * np.pi * (10.**map_logE_center[i] * np.log(10.) * dlogE) * 1
    eventmap = eg.astroEvent_galaxy(density_g, N_2012_Aeffmax, True)
            
    if seed_g == 42:
        filename = '../syntheticData/eventmap_astro'
    else:
        filename = '../syntheticData/eventmap_astro_nonGalaxy'
    for i in np.arange(eg.NEbin):
        fig = plt.figure(figsize=(8,6))
        hp.mollview(eventmap[i])
        plt.savefig(filename + str(i) + '.pdf')
        hp.fitsfunc.write_map(filename + str(i)+'.fits', eventmap[i])


def atmBG_coszenith_test(energyBin = 0):
    eg = EventGenerator()
    N_coszenith = np.loadtxt('../syntheticData/N_coszenith'+str(energyBin)+'.txt')
    recovered_values = eg.atmBG_coszenith(int(np.sum(N_coszenith[:,1])), energyBin)

    index = np.where(np.abs(recovered_values) > 1)
    if len(index) > 1:
        print index, recovered_values[index]
    

    fig = plt.figure(figsize=(8,6))
    plt.plot(N_coszenith[:,0], N_coszenith[:,1], lw=2, label='data')
    plt.hist(recovered_values, N_coszenith[:,0], label='mock')

    plt.xlabel(r'$\cos\,\theta$')
    plt.ylabel('Number of counts')
    plt.legend()
    plt.savefig('../plots/N_coszenith_test'+str(energyBin)+'.pdf')


def atmBGtest():
    eg = EventGenerator()
    eventmap = eg.atmEvent(1.)
             
    for i in np.arange(eg.NEbin):
        fig = plt.figure(figsize=(8,6))
        hp.mollview(eventmap[i])
        plt.savefig('../syntheticData/eventmap_atm' + str(i) + '.pdf')
        hp.fitsfunc.write_map('../syntheticData/eventmap_atm' + str(i)+'.fits', eventmap[i])





def checkEventGenerator():
    for i in range(0,5):
        atmBG_coszenith_test(i)

    astroEvent_galaxyTest()
    atmBGtest()

#astroEvent_galaxyTest(59)


#checkEventGenerator()
#checkSyntheticData()
#checkPowerSpectrum()
#checkCrossCorrelation()
 
# --- Calculate cross correlation ---

# generate galaxy sample
def generateGalaxy():
    NSIDE = 128
    randomSeed = 42
    l_cl = np.arange(1,3 * NSIDE + 1)
    cl_galaxy_file = np.loadtxt('../data/Cl_ggRM.dat')
    cl_galaxy = cl_galaxy_file[:500]

    density_g = density_cl(cl_galaxy, NSIDE, randomSeed)
    density_g = np.exp(density_g) - 1.0
    N_g = 2000000
    events_map_g = poisson_sampling(density_g, N_g)
    overdensityMap_g = overdensityMap(events_map_g)
    hp.fitsfunc.write_map('../syntheticData/galaxySampleOverdensity.fits', overdensityMap_g)

#generateGalaxy()




def w_cross_plot():
    cf = Analyze()
    
    astromap = np.zeros((cf.NEbin, cf.NPIXEL))
    for i in np.arange(cf.NEbin):
         astromap[i] = hp.fitsfunc.read_map('../syntheticData/eventmap_astro' + str(i)+'.fits', verbose=False)
    
    eg = EventGenerator()
    bgmap = eg.atmEvent(1.-0.003) # astro event is about 0.3% of total event
    countsmap = bgmap + astromap
    #print np.sum(bgmap), np.sum(astromap)
     
    w_cross = cf.crossCorrelationFromCountsmap(countsmap)

    # get standard derivation
    w_cross_mean, w_cross_std = cf.crossCorrelation_atm_std(50)
        
    chi_square_index = 1
    chi_square = 0
    
    print '------------------'
    print 'energy bin', 'chi2', 'dof', 'sigma'
    
    for i in np.arange(cf.NEbin-2):
        chi_square_i = np.sum( (w_cross[i][chi_square_index:] - w_cross_mean[i][chi_square_index:]) ** 2 / w_cross_std[i][chi_square_index:]**2)
        chi_square += chi_square_i
        print i, chi_square_i, len(w_cross[0][chi_square_index:]),significance(chi_square_i, len(w_cross[0][chi_square_index:]))
  
            
    print 'total', chi_square, (cf.NEbin - 2) * len(w_cross[0][chi_square_index:]), significance(chi_square, (cf.NEbin - 2) * len(w_cross[0][chi_square_index:]))
    print '------------------'

     
    """
    astromap = np.zeros((cf.NEbin, cf.NPIXEL))
    for i in np.arange(cf.NEbin):
         astromap[i] = hp.fitsfunc.read_map('../syntheticData/eventmap_astro_nonGalaxy' + str(i)+'.fits', verbose=False)
    """
    countsmap = bgmap #+ astromap
    #print np.sum(bgmap), np.sum(astromap)
    chi_square = 0
    w_cross = cf.crossCorrelationFromCountsmap(countsmap)
    print '------------------'
    print 'energy bin', 'chi2', 'dof', 'sigma'
      
    for i in np.arange(cf.NEbin-2):
        chi_square_i = np.sum( (w_cross[i][chi_square_index:] - w_cross_mean[i][chi_square_index:]) ** 2 / w_cross_std[i][chi_square_index:]**2)
        chi_square += chi_square_i
        print i, chi_square_i, len(w_cross[0][chi_square_index:]),significance(chi_square_i, len(w_cross[0][chi_square_index:]))
    
              
    print 'total', chi_square, (cf.NEbin - 2) * len(w_cross[0][chi_square_index:]), significance(chi_square, (cf.NEbin - 2) * len(w_cross[0][chi_square_index:]))
    print '------------------'



    color = ['r', 'orange', 'limegreen', 'skyblue', 'mediumslateblue', 'purple','grey']
    for i in np.arange(cf.NEbin):
        fig = plt.figure(figsize=(8,6))
        plt.rc('font', **font)
        plt.rc('legend', **legendfont)
        plt.xscale('log')
        #plt.yscale('log')
        
        plt.plot(cf.l_cl,   w_cross[i], label=str(i), color=color[i])
        plt.errorbar(cf.l_cl, w_cross_mean[i], yerr=w_cross_std[i], color='grey')

        plt.xlabel('l')
        plt.ylabel('Cl')
        #plt.ylim(1e-8, 10)
        plt.gcf().subplots_adjust(left=0.18, top=0.9, right=0.9)
        plt.legend(ncol=2)
        plt.savefig('../plots/w_cross'+str(i)+'.pdf')


#w_cross_plot()


def w_cross_sigma():
    cf = Analyze()
    # get standard derivation
    w_cross_mean, w_cross_std = cf.crossCorrelation_atm_std(50)

    eg = EventGenerator()
    chi_square_index = 1
    chi_square_Ebin = np.zeros(cf.NEbin)
    N_realization = 10
    for realization in range(N_realization):
        countsmap = eg.atmEvent(1.)
        w_cross = cf.crossCorrelationFromCountsmap(countsmap)
        for i in np.arange(cf.NEbin-2):
            chi_square_i = np.sum( (w_cross[i][chi_square_index:] - w_cross_mean[i][chi_square_index:]) ** 2 / w_cross_std[i][chi_square_index:]**2)
            chi_square_Ebin[i] += chi_square_i
            #print i, chi_square_i
        #print '--------'
    #print chi_square_Ebin / N_realization
    sigma_Ebin = np.zeros(cf.NEbin)
    for i in np.arange(cf.NEbin - 2):
        sigma_Ebin[i] = significance(chi_square_Ebin[i] / N_realization,  len(w_cross[0][chi_square_index:]))
        #print sigma_Ebin[i], significance(chi_square_Ebin[i] / N_realization, len(w_cross[0][chi_square_index:]))
    
    
    astromap = np.zeros((cf.NEbin, cf.NPIXEL))
    for i in np.arange(cf.NEbin):
         astromap[i] = hp.fitsfunc.read_map('../syntheticData/eventmap_astro' + str(i)+'.fits', verbose=False)
    chi_square_Ebin = np.zeros(cf.NEbin)
    N_realization = 10
    for realization in range(N_realization):
        countsmap = eg.atmEvent(1.-0.003) + astromap
        w_cross = cf.crossCorrelationFromCountsmap(countsmap)
        for i in np.arange(cf.NEbin-2):
            chi_square_i = np.sum( (w_cross[i][chi_square_index:] - w_cross_mean[i][chi_square_index:]) ** 2 / w_cross_std[i][chi_square_index:]**2)
            chi_square_Ebin[i] += chi_square_i
            #print i, chi_square_i
        #print '--------'
    #print chi_square_Ebin / N_realization
    sigma_Ebin_signal = np.zeros(cf.NEbin)
    for i in np.arange(cf.NEbin - 2):
        sigma_Ebin_signal[i] = significance(chi_square_Ebin[i] / N_realization,  len(w_cross[0][chi_square_index:]))
    
    
    
    
    
    fig = plt.figure(figsize=(8,6))
    plt.rc('font', **font)
    plt.rc('legend', **legendfont)
    plt.scatter(cf.map_logE_center, sigma_Ebin, marker='o', c='k', label='atm')
    plt.scatter(cf.map_logE_center, sigma_Ebin_signal, marker='x', c='r', label='atm+astro')
    plt.xlim(1.5, 7.5)
    plt.xlabel(r'$\log (E / {\rm GeV})$')
    plt.ylabel('Significance')
    plt.legend()
    plt.savefig('../plots/sigma_E.pdf')
    

w_cross_sigma()
