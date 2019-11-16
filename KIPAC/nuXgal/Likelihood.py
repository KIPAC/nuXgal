from Analyze import *
import numpy as np


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

datamap = np.zeros((cf.NEbin, cf.NPIXEL))
for i in np.arange(cf.NEbin):
     datamap[i] = hp.fitsfunc.read_map('../syntheticData/eventmap_astro' + str(i)+'.fits', verbose=False)
datamap = datamap + eg.atmEvent(1.-0.003)

w_data = cf.crossCorrelationFromCountsmap(datamap)

 
lmin = 10
energyBin = 4

f_gal = 0.6
density_g = density_cl(cl_galaxy * f_gal, cf.NSIDE, seed_g)
density_g = np.exp(density_g) - 1.0


def log_likelihood(f, w_data_i, w_error_i):
    f_diff = f
    
    
    """
    N_real = 5
    w_astro_N = np.zeros((N_real, cf.NEbin, 3 * cf.NSIDE))
    for i in range(N_real):
        astromap = eg.astroEvent_galaxy(density_g, N_2012_Aeffmax * f_diff, False) + eg.atmEvent(1.-0.003)
        w_astro_N[i] = cf.crossCorrelationFromCountsmap(astromap)
    w_astro_mean = np.mean(w_astro_N, axis=0)
    """
    astromap = eg.astroEvent_galaxy(density_g, N_2012_Aeffmax * f_diff, False) + eg.atmEvent(1.-0.003)
    w_astro_mean = cf.crossCorrelationFromCountsmap(astromap)
    
    #w_astro_std = np.std(w_astro_N, axis=0)
    #w_astro = cf.crossCorrelationFromCountsmap(astromap)
    lnLi =  (w_data_i - w_astro_mean[energyBin]) ** 2 / w_error_i ** 2
    lnLi = np.sum(lnLi[lmin:])
    
    #lnLi = np.sum( (w_data_4[lmin:] - np.array(w_astro_mean[4])[lmin:]) ** 2 / np.array(w_cross_std[4])[lmin:] ** 2)
    return lnLi
    
    """
    lnLi = np.zeros(cf.NEbin)
    for i in range(cf.NEbin):
        #lnLi[i] = np.sum( (w_data[i][lmin:] - w_astro[i][lmin:]) ** 2 / w_cross_std[i][lmin:] ** 2)
        #lnLi[i] = np.sum( (w_data[i][lmin:] - w_cross_mean[i][lmin:]) ** 2 / w_cross_std[i][lmin:] ** 2)
        lnLi[i] = np.sum( (w_data[i][lmin:] - w_astro_mean[i][lmin:]) ** 2 / w_cross_std[i][lmin:] ** 2)
    #lnLi = np.sum( (w_data - w_cross_mean) ** 2 / w_cross_std ** 2, axis = 1)
    return lnLi
    """
    

#print log_likelihood(np.array([1, 0.6]), w_data[4], w_cross_std[4])




def log_prior(f):
    f_diff = f
    if 0. < f_diff < 2.:
        return 0.
    return -np.inf
    
def log_probability(f, w_data_i, w_error_i):
    lp = log_prior(f)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(f, w_data_i, w_error_i)
    
    
import emcee
ndim = 1


pos = np.array([1.]) + np.random.randn(4, 1) * 1e-2
nwalkers, ndim = pos.shape


filename = 'test.h5'

backend =  emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(w_data[energyBin], w_cross_std[energyBin]),backend=backend)
sampler.run_mcmc(pos, 500, progress=True);




reader = emcee.backends.HDFBackend(filename)

fig, axes = plt.subplots(1, figsize=(10, 7), sharex=True)
samples = reader.get_chain()
labels = ["f_diff", "f_gal"]

ax = axes
ax.plot(samples[:, :, 0], "k", alpha=0.3)
ax.set_xlim(0, len(samples))
ax.set_ylabel(labels[0])
ax.yaxis.set_label_coords(-0.1, 0.5)

axes.set_xlabel("step number");
fig.savefig('check.pdf')


#tau = sampler.get_autocorr_time()
#print(tau)


flat_samples = reader.get_chain(discard=100, thin=15, flat=True)
print(flat_samples.shape)



import corner

fig = corner.corner(
    flat_samples, labels=['f_diff', 'f_gal'], truths=[1.0, 0.6]
)
fig.savefig('corner.pdf')


