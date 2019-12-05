
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

syn_maps_2000 = hp_utils.vector_generate_counts_from_pdf(gg_density, nevts_2000, 100)
syn_maps_10000 = hp_utils.vector_generate_counts_from_pdf(gg_density, nevts_10000, 100)


alm_2000 = hp_utils.vector_alm_from_overdensity(syn_maps_2000, Defaults.NALM)
alm_10000 = hp_utils.vector_alm_from_overdensity(syn_maps_10000, Defaults.NALM)

syn_alm_2000 = hp_utils.vector_generate_alm_from_cl(cls_pdf_2000, Defaults.NALM, 100)
syn_alm_10000 = hp_utils.vector_generate_alm_from_cl(cls_pdf_10000, Defaults.NALM, 100)


# pick alm for l = 150
alm_idx = hp_utils.get_alm_idxs_for_l(150, Defaults.MAX_L)

re_alm_2000 = np.real(alm_2000[:,alm_idx])
re_alm_10000 = np.real(alm_10000[:,alm_idx])
re_syn_alm_2000 = np.real(syn_alm_2000[:,alm_idx])
re_syn_alm_10000 = np.real(syn_alm_10000[:,alm_idx])

im_alm_2000 = np.imag(alm_2000[:,alm_idx[1:]])
im_alm_10000 = np.imag(alm_10000[:,alm_idx[1:]])
im_syn_alm_2000 = np.imag(syn_alm_2000[:,alm_idx[1:]])
im_syn_alm_10000 = np.imag(syn_alm_10000[:,alm_idx[1:]])

vals = [re_alm_2000, re_alm_10000, re_syn_alm_2000, re_syn_alm_10000,
        im_alm_2000, im_alm_10000, im_syn_alm_2000, im_syn_alm_10000]
for vect in vals:
    vect /= vect.std()

labels = [r'$\mathcal{R}, n=2000$, Poisson', r'$\mathcal{R}, n=10000$, Poisson', 
          r'$\mathcal{R}, n=2000$, PDF', r'$\mathcal{R}, n=10000$, PDF',
          r'$\mathcal{I}, n=2000$, Poisson', r'$\mathcal{I}, n=10000$, Poisson', 
          r'$\mathcal{I}, n=2000$, PDF', r'$\mathcal{I}, n=10000$, PDF',]


#bins = np.linspace(-0.003, 0.003, 61)

bins = np.linspace(-5, 5, 101)
curve_pts = np.linspace(-5, 5, 1001)

gauss_vals = stats.norm.pdf(curve_pts)*re_alm_2000.size/10.
o_dict = figs.plot_hists("alm", bins, vals, labels=labels)
axes = o_dict['axes']
axes.plot(curve_pts, gauss_vals, 'b-', label="Unit Normal")


testfigpath = os.path.join(Defaults.NUXGAL_PLOT_DIR, 'test')  
testfigfile = os.path.join(testfigpath, 'alm_distrib')
Utilityfunc.makedir_safe(testfigfile)

figs.save_all(testfigfile, 'pdf')
