
import os
import numpy as np
import healpy as hp

from KIPAC import nuXgal
from KIPAC.nuXgal import Defaults
from KIPAC.nuXgal import FigureDict
from KIPAC.nuXgal import Utilityfunc

import matplotlib.pyplot as plt


figs = FigureDict()

galaxy_od_path = os.path.join(Defaults.NUXGAL_ANCIL_DIR, 'galaxySampleOverdensity.fits')
galaxy_galaxy_cl_path = os.path.join(Defaults.NUXGAL_ANCIL_DIR,'Cl_ggRM.dat')
aeff_path = os.path.join(Defaults.NUXGAL_IRF_DIR, 'Aeff{i}.fits')
syn_astro_counts_path = os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'eventmap_astro{i}.fits')
syn_atm_counts_path = os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, 'eventmap_atm{i}.fits')

syn_map_astr = nuXgal.Map.create_from_counts_and_exposure_maps(syn_astro_counts_path, aeff_path, Defaults.NEbin)
syn_map_atm = nuXgal.Map.create_from_counts_and_exposure_maps(syn_atm_counts_path, aeff_path, Defaults.NEbin)

gg_od_map = nuXgal.Map.create_from_overdensity_maps(galaxy_od_path)
gg_cl_map = nuXgal.Map.create_from_cl(galaxy_galaxy_cl_path)


mean_density = 1.

gg_overdensity = gg_od_map.overdensity()[0]
gg_density = mean_density*(gg_overdensity + 1)

figs.mollview('overdensity', gg_overdensity)
figs.mollview('density', gg_density)

cl_overdensity = hp.sphtfunc.anafast(gg_overdensity)
cl_density = hp.sphtfunc.anafast(gg_density)/ (mean_density*mean_density)

n_cl = Defaults.NCL
cl_xvals = np.linspace(1, n_cl+1, n_cl)
cl_data = [cl_overdensity, cl_density]
cl_labels = ['overdensity', 'density']

figs.plot_cl('cl', cl_xvals, cl_data, xlabel='l', ylabel=r'$c_{l}$', labels=cl_labels)


testfigpath = os.path.join(Defaults.NUXGAL_PLOT_DIR, 'test')  
testfigfile = os.path.join(testfigpath, 'density_v_overdensity')
Utilityfunc.makedir_safe(testfigfile)

figs.save_all(testfigfile, 'pdf')
