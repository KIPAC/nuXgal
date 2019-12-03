
import os
import numpy as np
import healpy as hp

from KIPAC import nuXgal
from KIPAC.nuXgal import Defaults
from KIPAC.nuXgal import FigureDict
from KIPAC.nuXgal import hp_utils
from KIPAC.nuXgal import Utilityfunc

import matplotlib.pyplot as plt


def hist_density_complex(complex_vect, axis_bins=None):
    if axis_bins is None:
        axis_bins = np.linspace(-3., 3., 61)

    hist = np.histogram2d(np.real(complex_vect.flat), np.imag(complex_vect.flat), axis_bins)
    return hist
    

def plot_density_complex(hist):
    
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
    axes.set_xlabel(r'$\mathcal{R}$')
    axes.set_ylabel(r'$\mathcal{I}$')
    extent = (hist[1][0], hist[1][-1], hist[2][0], hist[2][-1])
    img = axes.imshow(hist[0], extent=extent, interpolation='none')
    cbar = plt.colorbar(img)

    return dict(fig=fig, axes=axes, img=img, cbar=cbar)

    
figs = FigureDict()

galaxy_od_path = os.path.join(Defaults.NUXGAL_ANCIL_DIR, 'galaxySampleOverdensity.fits')
galaxy_galaxy_cl_path = os.path.join(Defaults.NUXGAL_ANCIL_DIR,'Cl_ggRM.dat')
aeff_path = os.path.join(Defaults.NUXGAL_IRF_DIR, 'Aeff{i}.fits')
syn_astro_counts_path = os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'eventmap_astro{i}.fits')
syn_atm_counts_path = os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'eventmap_atm{i}.fits')

#syn_map_astr = nuXgal.Map.create_from_counts_and_exposure_maps(syn_astro_counts_path, aeff_path, Defaults.NEbin)
#syn_map_atm = nuXgal.Map.create_from_counts_and_exposure_maps(syn_atm_counts_path, aeff_path, Defaults.NEbin)

gg_od_map = nuXgal.Map.create_from_overdensity_maps(galaxy_od_path)

gg_cl_map = nuXgal.Map.create_from_cl(galaxy_galaxy_cl_path)


mean_density = 1.

gg_overdensity = gg_od_map.overdensity()[0]
gg_density = mean_density*(gg_overdensity + 1)

npix = 12*128*128
iso_pdf = np.ones((npix))
iso_pdf /= iso_pdf.sum()

pdf = gg_density
nevt = 1e3
l = np.arange(1, Defaults.NCL + 1)

gal_alms = hp_utils.vector_generate_alm_from_cl(gg_cl_map.cl()[0,0:384], int(Defaults.NALM), int(500))
cls = gg_cl_map.cl()[0,0:384]

l0_idx = hp_utils.get_alm_idxs_for_l(0, Defaults.MAX_L)
m0_idx = hp_utils.get_alm_idxs_for_m(0, Defaults.MAX_L)
l1_idx = hp_utils.get_alm_idxs_for_l(1, Defaults.MAX_L)
m1_idx = hp_utils.get_alm_idxs_for_m(1, Defaults.MAX_L)
l50_idx = hp_utils.get_alm_idxs_for_l(50, Defaults.MAX_L)
m50_idx = hp_utils.get_alm_idxs_for_m(50, Defaults.MAX_L)

l380_idx = hp_utils.get_alm_idxs_for_l(380, Defaults.MAX_L)

l50_vals = gal_alms[:,l50_idx[1:]] / np.sqrt(cls[1])

hist_50 = hist_density_complex(l50_vals)

f_50 = plot_density_complex(hist_50)

testfigpath = os.path.join(Defaults.NUXGAL_PLOT_DIR, 'test')  
testfigfile = os.path.join(testfigpath, 'syn_alm_complex')
f_50['fig'].savefig(testfigfile + '.pdf')
