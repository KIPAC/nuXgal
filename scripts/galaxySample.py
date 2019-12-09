import os

import numpy as np

import healpy as hp

from KIPAC.nuXgal import Defaults

from KIPAC.nuXgal import file_utils

from KIPAC.nuXgal import FigureDict

import matplotlib.pyplot as plt

from KIPAC.nuXgal import Utilityfunc

def computeGalaxymap():

    galaxySampleFile = np.loadtxt('scripts/irsa_catalog_search_results.txt')

    # from dec, RA to healpy coordinates: theta = pi / 2 - dec, phi = 2 * pi - RA 
    index_map_pixel = hp.pixelfunc.ang2pix(Defaults.NSIDE, np.radians(90. - galaxySampleFile[:, 1]), np.radians(360. - galaxySampleFile[:, 0]))

    galaxymap = np.zeros(hp.pixelfunc.nside2npix(Defaults.NSIDE))

    for i in index_map_pixel:
        galaxymap[i] += 1.

    hp.fitsfunc.write_map('galaxymap.fits', galaxymap, overwrite=True)


def plotGalaxymap():

    galaxymap = hp.fitsfunc.read_map('galaxymap.fits')
    hp.mollview(galaxymap, title="Mollview image RING", max=100)
    plt.savefig('galaxymap.pdf')


def getCl():
    galaxymap = hp.fitsfunc.read_map('galaxymap.fits')
    cl = hp.anafast(Utilityfunc.overdensityMap(galaxymap))
    ell = np.arange(len(cl))

    plt.figure(figsize=(8, 6))
    plt.yscale('log')
    plt.xscale('log')
    analyCL = np.loadtxt('data/ancil/Cl_ggRM.dat')
    plt.plot(np.arange(500), analyCL[0:500], label='analytical')
    #plt.plot(ell, cl, label='WISE-2MASS galaxies')
    plt.plot(ell, cl * 0.3, label='WISE-2MASS galaxies x 0.3')

    plt.xlabel("$\ell$")
    plt.ylabel("$C_{\ell}$")
    plt.grid()
    plt.legend()
    plt.ylim(1e-7, 1e-1)
    plt.savefig('galaxy_cl.pdf')
    #hp.write_cl("cl.fits", cl, overwrite=True)

if __name__ == '__main__':
    computeGalaxymap()
    plotGalaxymap()
    getCl()
