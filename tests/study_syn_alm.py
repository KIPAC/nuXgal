
import os
import numpy as np
import healpy as hp

from KIPAC import nuXgal
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

std_l0_real = np.real(gal_alms[:,l0_idx]).std(0) / np.sqrt(cls[0])
std_m0_real = np.real(gal_alms[:,m0_idx]).std(0) / np.sqrt(cls[0:Defaults.MAX_L])
std_l1_real = np.real(gal_alms[:,l1_idx]).std(0) / np.sqrt(cls[1])
std_l1_imag = np.imag(gal_alms[:,l1_idx]).std(0) / np.sqrt(cls[1])
std_m1_real = np.real(gal_alms[:,m1_idx]).std(0) / np.sqrt(cls[1:Defaults.MAX_L])
std_m1_imag = np.imag(gal_alms[:,m1_idx]).std(0) / np.sqrt(cls[1:Defaults.MAX_L])
std_l50_real = np.real(gal_alms[:,l50_idx]).std(0) / np.sqrt(cls[50])
std_l50_imag = np.imag(gal_alms[:,l50_idx]).std(0) / np.sqrt(cls[50])
std_m50_real = np.real(gal_alms[:,m50_idx]).std(0) / np.sqrt(cls[50:Defaults.MAX_L])
std_m50_imag = np.imag(gal_alms[:,m50_idx]).std(0) / np.sqrt(cls[50:Defaults.MAX_L])
std_l380_real = np.real(gal_alms[:,l380_idx]).std(0) / np.sqrt(cls[380])
std_l380_imag = np.imag(gal_alms[:,l380_idx]).std(0) / np.sqrt(cls[380])

mean_l0_real = np.real(gal_alms[:,l0_idx]).mean(0) / np.sqrt(cls[0])
mean_m0_real = np.real(gal_alms[:,m0_idx]).mean(0) / np.sqrt(cls[0:Defaults.MAX_L])
mean_l1_real = np.real(gal_alms[:,l1_idx]).mean(0) / np.sqrt(cls[1])
mean_l1_imag = np.imag(gal_alms[:,l1_idx]).mean(0) / np.sqrt(cls[1])
mean_m1_real = np.real(gal_alms[:,m1_idx]).mean(0) / np.sqrt(cls[1:Defaults.MAX_L])
mean_m1_imag = np.imag(gal_alms[:,m1_idx]).mean(0) / np.sqrt(cls[1:Defaults.MAX_L])
mean_l50_real = np.real(gal_alms[:,l50_idx]).mean(0) / np.sqrt(cls[50])
mean_l50_imag = np.imag(gal_alms[:,l50_idx]).mean(0) / np.sqrt(cls[50])
mean_m50_real = np.real(gal_alms[:,m50_idx]).mean(0) / np.sqrt(cls[50:Defaults.MAX_L])
mean_m50_imag = np.imag(gal_alms[:,m50_idx]).mean(0) / np.sqrt(cls[50:Defaults.MAX_L])


xvals_list = [np.arange(0, Defaults.MAX_L), 
              np.arange(1, Defaults.MAX_L), np.arange(50, Defaults.MAX_L),
              np.arange(1, Defaults.MAX_L), np.arange(50, Defaults.MAX_L)]
yvals_list = [std_m0_real, std_m1_real, std_m50_real,  std_m1_imag, std_m50_imag]
labels = ["m=0", r"$re(m=1)$", r"$re(m=50)$", r"$im(m=1)$",  r"$im(m=50)$"]

figs.plot_xyvals("alm_m_std", xvals_list, yvals_list, xlabel="l", ylabel="std of 100 trials", labels=labels)


xvals_list = [np.arange(0, 1), 
              np.arange(0, 2), np.arange(0, 51), 
              np.arange(0, 2), np.arange(0, 51)]
yvals_list = [std_l0_real, std_l1_real, std_l50_real, std_l1_imag, std_l50_imag]
labels = ["l=0", r"$re(l=1)$", r"$re(l=50)$", r"$im(l=1)$",  r"$im(l=50)$"]

figs.plot_xyvals("alm_l_std", xvals_list, yvals_list, xlabel="m", ylabel="std of 100 trials", labels=labels)


xvals_list = [np.arange(0, 1), 
              np.arange(0, 2), np.arange(0, 51),
              np.arange(0, 2), np.arange(0, 51)]
yvals_list = [mean_l0_real, mean_l1_real, mean_l50_real,  mean_l1_imag, mean_l50_imag]
labels = ["l=0", r"$re(l=1)$", r"$re(l=50)$", r"$im(l=1)$",  r"$im(l=50)$"]

figs.plot_xyvals("alm_l_mean", xvals_list, yvals_list, xlabel="m", ylabel="std of 100 trials", labels=labels)


        
testfigpath = os.path.join(Defaults.NUXGAL_PLOT_DIR, 'test')  
testfigfile = os.path.join(testfigpath, 'syn_alm')
Utilityfunc.makedir_safe(testfigfile)

figs.save_all(testfigfile, 'pdf')

#figs.plot("alm_std", np.arange(Defaults.NALM), np.real(gal_alms).std(0))



#iso_maps = hp_utils.vector_generate_counts_from_pdf(iso_pdf, nevt, 100)
#syn_cls = hp_utils.vector_cross_correlate_maps(iso_maps[0:50], iso_maps[50:], Defaults.NCL)



#nevt2 = 1e6
#iso_maps2 = hp_utils.vector_generate_counts_from_pdf(iso_pdf, nevt2, 100)
#syn_cls2 = hp_utils.vector_cross_correlate_maps(iso_maps2[0:50], iso_maps2[50:], Defaults.NCL)



