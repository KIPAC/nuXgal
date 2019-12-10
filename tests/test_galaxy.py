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

from KIPAC.nuXgal import GalaxySample

import matplotlib.pyplot as plt


for dirname in [Defaults.NUXGAL_SYNTHETICDATA_DIR, Defaults.NUXGAL_PLOT_DIR]:
    try:
        os.makedirs(dirname)
    except OSError:
        pass


def testGalaxy():
    gs = GalaxySample.GalaxySample()
    gs.plotGalaxymap('WISE')
    gs.plotCL()



def testOverdensity():
    gs = GalaxySample.GalaxySample(computeGalaxy = True)
    galaxymap = gs.generateGalaxy(N_g = 5000000, write_map = False)
    analy_galaxymap_overdensity = Utilityfunc.overdensityMap(galaxymap)
    alm = hp.sphtfunc.map2alm(analy_galaxymap_overdensity)
    map = hp.sphtfunc.alm2map(alm, Defaults.NSIDE)

    galaxymap_normalized = galaxymap / np.sum(galaxymap)

    hp.mollview(analy_galaxymap_overdensity, title="analyoverdensity")
    plt.savefig('plots/analyoverdensity.pdf')

    hp.mollview(map, title="map from alm")
    plt.savefig('plots/mapFromALM.pdf')

    hp.mollview(galaxymap_normalized, title="galaxymap normalized")
    plt.savefig('plots/galaxymap_normalized.pdf')


    np.random.seed(Defaults.randomseed_galaxy)
    density_g = hp.sphtfunc.synfast(gs.analyCL, Defaults.NSIDE)
    density_g = np.exp(density_g)
    density_g /= density_g.sum()
    hp.mollview(density_g, title='density_g')
    plt.savefig('plots/density_g.pdf')


if __name__ == '__main__':

    #testGalaxy()
    testOverdensity()
