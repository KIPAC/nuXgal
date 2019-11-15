from .Analyze import *


def LogLikelihood(f_diff, f_gal, datamap, lmin = 10):
    density_g = density_cl(cl_galaxy * f_gal, cf.NSIDE, seed_g)
    density_g = np.exp(density_g) - 1.0
    astromap = datamap * 0.

    w_data = cf.crossCorrelationFromCountsmap(datamap)

    N_real = 20
    w_astro_N = np.zeros((N_real, cf.NEbin, 3 * cf.NSIDE))
    for i in range(N_real):
        astromap = eg.astroEvent_galaxy(density_g, N_2012_Aeffmax * f_diff, False) + eg.atmEvent(1.-0.003)
        w_astro_N[i] = cf.crossCorrelationFromCountsmap(astromap)

    w_astro_mean, w_astro_std = np.mean(w_astro_N, axis=0), np.std(w_astro_N, axis=0)
    #w_astro = cf.crossCorrelationFromCountsmap(astromap)

    lnLi = np.zeros(cf.NEbin)
    for i in range(cf.NEbin):
        #lnLi[i] = np.sum( (w_data[i][lmin:] - w_astro[i][lmin:]) ** 2 / w_cross_std[i][lmin:] ** 2)
        #lnLi[i] = np.sum( (w_data[i][lmin:] - w_cross_mean[i][lmin:]) ** 2 / w_cross_std[i][lmin:] ** 2)
        lnLi[i] = np.sum( (w_data[i][lmin:] - w_astro_mean[i][lmin:]) ** 2 / w_cross_std[i][lmin:] ** 2)
    #lnLi = np.sum( (w_data - w_cross_mean) ** 2 / w_cross_std ** 2, axis = 1)
    return lnLi
