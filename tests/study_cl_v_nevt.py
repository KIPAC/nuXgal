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
galaxy_galaxy_cl_path = os.path.join(Defaults.NUXGAL_ANCIL_DIR,'Cl_ggRM.dat')
galaxy_od_path = os.path.join(Defaults.NUXGAL_ANCIL_DIR, 'galaxySampleOverdensity.fits')

gg_cl_map = nuXgal.Map.create_from_cl(galaxy_galaxy_cl_path)

cls_od = gg_cl_map.cl()[0]

gg_overdensity = hp.sphtfunc.synfast(cls_od, Defaults.NSIDE)
gg_density = np.clip(gg_overdensity + 1, 0., np.inf)
gg_density /= gg_density.sum()

cl_density = hp.sphtfunc.anafast(gg_density)

n_events = np.logspace(1, 7, 13)
n_trial = 20

syn_cls_means = []
syn_cls_stds = []
labels = []

lvals = np.arange(384)

for n_evt in n_events:
    syn_maps = hp_utils.vector_generate_counts_from_pdf(gg_density, n_evt, n_trial)
    pdf_map = gg_density*n_evt
    syn_cls = hp_utils.vector_cl_from_overdensity(syn_maps, Defaults.NCL)
    syn_cls_means.append(syn_cls.mean(0))
    syn_cls_stds.append(syn_cls.std(0))
    pdf_cls = hp.sphtfunc.anafast(pdf_map)
    syn_cls_means.append(pdf_cls)
    syn_cls_stds.append(pdf_cls*0.1)
    labels.append("Syn %i" % n_evt)
    labels.append("Pdf %i" % n_evt)

o_dict = figs.plot_cl("cl", lvals, syn_cls_means, yerr=syn_cls_stds, ymin=1e-20)

testfigpath = os.path.join(Defaults.NUXGAL_PLOT_DIR, 'test')  
testfigfile = os.path.join(testfigpath, 'cl_v_nevt')
Utilityfunc.makedir_safe(testfigfile)

figs.save_all(testfigfile, 'pdf')
