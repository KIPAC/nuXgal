from .Analyze import *

cf = Analyze()
w_cross_mean, w_cross_std = cf.crossCorrelation_atm_std(50)

eg = EventGenerator()
seed_g = 42

cl_galaxy_file = np.loadtxt('../data/Cl_ggRM.dat')
cl_galaxy = cl_galaxy_file[:500]

# calculate expected event number using IceCube diffuse neutrino flux
dN_dE_astro = lambda E_GeV: 1.44E-18 * (E_GeV / 100e3)**(-2.28) # GeV^-1 cm^-2 s^-1 sr^-1, muon neutrino
# total expected number of events before cut, for one year data
N_2012_Aeffmax = np.zeros(cf.NEbin)
for i in np.arange(cf.NEbin):
    N_2012_Aeffmax[i] = dN_dE_astro(10.**eg.map_logE_center[i]) * (10. ** eg.map_logE_center[i] * np.log(10.) * cf.dlogE) * (eg.Aeff_max[i] * 1E4) * (333 * 24. * 3600) * 4 * np.pi


if __name__ == "__main__":
    datamap = np.zeros((cf.NEbin, cf.NPIXEL))
    for i in np.arange(cf.NEbin):
        datamap[i] = hp.fitsfunc.read_map('../syntheticData/eventmap_astro' + str(i)+'.fits', verbose=False)
    datamap = datamap + eg.atmEvent(1.-0.003)
    lnLi = LogLikelihood(1., 0.6, datamap)
    
    for i, _lnLi in enumerate(lnLi):
        print(i, _lnLi, significance(_lnLi,  3 * cf.NSIDE - 1))
    
    


