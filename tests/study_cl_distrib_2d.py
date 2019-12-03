
import os
import numpy as np
import healpy as hp

from scipy import stats

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

pdf_2000 = gg_density*rej_map
pdf_2000 *= nevts_2000 / pdf_2000.sum()
pdf_10000 = gg_density*rej_map
pdf_10000 *= nevts_2000 / pdf_10000.sum()

cls_pdf_2000 = hp.sphtfunc.anafast(pdf_2000)
cls_pdf_10000 = hp.sphtfunc.anafast(pdf_10000)
cls_gg = hp.sphtfunc.anafast(gg_density)

alm_gg = hp.sphtfunc.map2alm(gg_density)

n_trial = 500

syn_maps_2000 = hp_utils.vector_generate_counts_from_pdf(gg_density, nevts_2000, n_trial)
syn_maps_10000 = hp_utils.vector_generate_counts_from_pdf(gg_density, nevts_10000, n_trial)

alm_2000 = hp_utils.vector_alm_from_overdensity(syn_maps_2000, Defaults.NALM)
alm_10000 = hp_utils.vector_alm_from_overdensity(syn_maps_10000, Defaults.NALM)

syn_alm_2000 = hp_utils.vector_generate_alm_from_cl(cls_pdf_2000, Defaults.NALM, n_trial)
syn_alm_10000 = hp_utils.vector_generate_alm_from_cl(cls_pdf_10000, Defaults.NALM, n_trial)

w_cross_2000 = hp_utils.vector_cross_correlate_alms(alm_2000, alm_gg, Defaults.NCL)
w_cross_10000 = hp_utils.vector_cross_correlate_alms(alm_10000, alm_gg, Defaults.NCL)
w_cross_syn_2000 = hp_utils.vector_cross_correlate_alms(syn_alm_2000, alm_gg, Defaults.NCL)
w_cross_syn_10000 = hp_utils.vector_cross_correlate_alms(syn_alm_10000, alm_gg, Defaults.NCL)


vals = [w_cross_2000, w_cross_10000, w_cross_syn_2000, w_cross_syn_10000]
for vect in vals:
    vect /= vect.std(0)

lvals = np.arange(384)

l_bin_step = 10
n_bins_y = 100

bins_x = np.arange(0, 384, l_bin_step)
bins_y = np.linspace(-5., 5., n_bins_y + 1)
l_array = np.expand_dims(lvals, -1)*np.ones(n_trial)
bins = [bins_x, bins_y]

figs.plot_hist_verus_l("w_cross_2000", bins, l_array,  w_cross_2000, 
                       xlabel=r'$l$', ylabel=r'$w / \sigma_{w}')

figs.plot_hist_verus_l("w_cross_10000", bins, l_array,  w_cross_10000, 
                       xlabel=r'$l$', ylabel=r'$w / \sigma_{w}')

figs.plot_hist_verus_l("w_cross_syn_2000", bins, l_array,  w_cross_syn_2000, 
                       xlabel=r'$l$', ylabel=r'$w / \sigma_{w}')

figs.plot_hist_verus_l("w_cross_syn_10000", bins, l_array,  w_cross_syn_10000, 
                       xlabel=r'$l$', ylabel=r'$w / \sigma_{w}')


testfigpath = os.path.join(Defaults.NUXGAL_PLOT_DIR, 'test')  
testfigfile = os.path.join(testfigpath, 'cl_distrib')
Utilityfunc.makedir_safe(testfigfile)

figs.save_all(testfigfile, 'pdf')
