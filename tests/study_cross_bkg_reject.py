
import os
import numpy as np
import healpy as hp

from KIPAC import nuXgal
from KIPAC.nuXgal import EventGenerator
from KIPAC.nuXgal import Defaults
from KIPAC.nuXgal import FigureDict
from KIPAC.nuXgal import hp_utils
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

eg = EventGenerator()
rej_map = eg.astro_gen.prob_reject()[0]

nevts_2000 = 2000.
nevts_10000 = 10000.

cls_od = gg_od_map.cl()[0]

gg_overdensity = gg_od_map.overdensity()[0]
gg_density = np.clip(gg_overdensity + 1, 0., np.inf)
gg_density /= gg_density.sum()

flat_map = np.ones(Defaults.NPIXEL) / float(Defaults.NPIXEL)
accept_map = flat_map*rej_map
accept_map /= accept_map.sum()

pdf_2000 = flat_map*rej_map
pdf_2000 *= nevts_2000 / pdf_2000.sum()
pdf_10000 = flat_map*rej_map
pdf_10000 *= nevts_2000 / pdf_10000.sum()

cls_pdf_2000 = hp.sphtfunc.anafast(pdf_2000)
cls_pdf_10000 = hp.sphtfunc.anafast(pdf_10000)
cls_gg = hp.sphtfunc.anafast(gg_density)

cls_pdf_2000[1::2] = cls_pdf_2000[0::2] 
cls_pdf_2000[1] = cls_pdf_2000[2]

cls_pdf_10000[1::2] = cls_pdf_10000[0::2] 
cls_pdf_10000[1] = cls_pdf_10000[2]

syn_maps_2000 = hp_utils.vector_generate_counts_from_pdf(accept_map, nevts_2000, 100)
syn_maps_10000 = hp_utils.vector_generate_counts_from_pdf(accept_map, nevts_10000, 100)

sigma_maps_2000 = syn_maps_2000.std(0)
sigma_maps_10000 = syn_maps_10000.std(0)

cls_sigma_maps_2000 = hp.sphtfunc.anafast(sigma_maps_2000)
cls_sigma_maps_10000 = hp.sphtfunc.anafast(sigma_maps_10000)

w_cross_2000 = hp_utils.vector_cross_correlate_maps(gg_density, syn_maps_2000, Defaults.NCL)
w_cross_10000 = hp_utils.vector_cross_correlate_maps(gg_density, syn_maps_10000, Defaults.NCL)

w_cross_2000_norm = w_cross_2000 / np.sqrt(cls_sigma_maps_2000 * cls_gg)
w_cross_10000_norm = w_cross_10000 / np.sqrt(cls_sigma_maps_10000 * cls_gg)

#w_cross_2000_norm = hp_utils.vector_cross_correlate_maps_normed(gg_density, syn_maps_2000, Defaults.NCL)
#w_cross_10000_norm = hp_utils.vector_cross_correlate_maps_normed(gg_density, syn_maps_10000, Defaults.NCL)

chi_2000 = w_cross_2000_norm/w_cross_2000_norm.std(0)
chi_10000 = w_cross_10000_norm/w_cross_10000_norm.std(0)

signif_2000 = Utilityfunc.significance_from_chi(chi_2000[:,5:].mean(0))
signif_10000 = Utilityfunc.significance_from_chi(chi_10000[:,5:].mean(0))


print(signif_2000, signif_10000)

cl_xvals = np.linspace(0, Defaults.NCL-1, Defaults.NCL)
cl_data = [w_cross_2000_norm.std(0), w_cross_10000_norm.std(0), w_cross_2000_norm.mean(0), w_cross_10000_norm.mean(0)]
cl_labels = [r'$\sigma_{2000}$', r'$\sigma_{10000}$', r'$\mu_{2000}$', r'$\mu_{10000}$']

figs.plot_w_cross_norm('cl', cl_xvals, cl_data, xlabel='l', ylabel=r'$c_{l}$', labels=cl_labels)


testfigpath = os.path.join(Defaults.NUXGAL_PLOT_DIR, 'test')  
testfigfile = os.path.join(testfigpath, 'gal_cross_bkg_reject')
Utilityfunc.makedir_safe(testfigfile)

figs.save_all(testfigfile, 'pdf')
