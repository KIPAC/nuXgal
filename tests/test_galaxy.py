import os

import numpy as np

import healpy as hp

from KIPAC.nuXgal import Defaults

from KIPAC.nuXgal import file_utils

from KIPAC.nuXgal import FigureDict

import matplotlib.pyplot as plt

from KIPAC.nuXgal import Utilityfunc

galaxymap_path = os.path.join(Defaults.NUXGAL_ANCIL_DIR,'WISE_galaxymap.fits')
testfigpath = Defaults.NUXGAL_PLOT_DIR


def plotGalaxymap():

    galaxymap = hp.fitsfunc.read_map(galaxymap_path)
    hp.mollview(galaxymap, title="Mollview image RING", max=100)
    testfigfile = os.path.join(testfigpath, 'WISE_galaxymap.pdf')
    plt.savefig(testfigfile)


def getCl():
    galaxymap = hp.fitsfunc.read_map(galaxymap_path)
    cl = hp.anafast(Utilityfunc.overdensityMap(galaxymap))
    ell = np.arange(len(cl))

    plt.figure(figsize=(8, 6))
    plt.yscale('log')
    plt.xscale('log')
    analyCL = np.loadtxt('data/ancil/Cl_ggRM.dat')
    plt.plot(np.arange(500), analyCL[0:500], label='analytical')
    plt.plot(ell, cl * 0.3, label='WISE-2MASS galaxies x 0.3')

    plt.xlabel("$\ell$")
    plt.ylabel("$C_{\ell}$")
    plt.grid()
    plt.legend()
    plt.ylim(1e-7, 1e-1)
    testfigfile = os.path.join(testfigpath, 'WISE_galaxy_cl.pdf')
    plt.savefig(testfigfile)

def getalm():
    galaxymap = hp.fitsfunc.read_map(galaxymap_path)
    alm = hp.sphtfunc.map2alm(Utilityfunc.overdensityMap(galaxymap))
    print (alm)

if __name__ == '__main__':
    #plotGalaxymap()
    #getCl()
    getalm()
