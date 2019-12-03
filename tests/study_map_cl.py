
import os
import numpy as np
import healpy as hp

from KIPAC import nuXgal
from KIPAC.nuXgal import Defaults
from KIPAC.nuXgal import FigureDict
from KIPAC.nuXgal import Utilityfunc

import matplotlib.pyplot as plt

eg = nuXgal.EventGenerator()

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

syn_map_astr_cl = syn_map_astr.cl()
syn_map_atm_cl = syn_map_atm.cl()
gg_od_map_cl = gg_od_map.cl()
gg_cl_map_cl = gg_cl_map.cl()                
                              
n_cl = Defaults.NCL

cl_xvals = np.linspace(1, n_cl+1, n_cl)
cl_vals = [syn_map_astr_cl[0], syn_map_atm_cl[0],
           gg_od_map_cl[0], gg_cl_map_cl[0,:n_cl]]
labels = ['astro nu', 'atm nu', 'gg overdensity', 'gg from cl']

figs.plot_cl('cl', cl_xvals, cl_vals, xlabel='l', ylabel=r'$c_{l}$', labels=labels)

#cl_syn_map_atm = syn_map_astr.cross_correlation(syn_map_atm)

testfigpath = os.path.join(Defaults.NUXGAL_PLOT_DIR, 'test')  
testfigfile = os.path.join(testfigpath, 'cl_distrib')
Utilityfunc.makedir_safe(testfigfile)

figs.save_all(testfigfile, 'pdf')
