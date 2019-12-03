
import os
import numpy as np
import healpy as hp

from KIPAC import nuXgal
from KIPAC.nuXgal import Defaults
from KIPAC.nuXgal import FigureDict
from KIPAC.nuXgal import file_utils
from KIPAC.nuXgal import Utilityfunc

from KIPAC.nuXgal.Generator import AtmGenerator, AstroGenerator

import matplotlib.pyplot as plt

figs = FigureDict()


coszenith_path = os.path.join(Defaults.NUXGAL_IRF_DIR, 'N_coszenith{i}.txt')
aeff_path = os.path.join(Defaults.NUXGAL_IRF_DIR, 'Aeff{i}.fits')
nevents_path = os.path.join(Defaults.NUXGAL_IRF_DIR, 'eventNumber_Ebin_perIC86year.txt')
gg_sample_path = os.path.join(Defaults.NUXGAL_ANCIL_DIR, 'galaxySampleOverdensity.fits')


aeff = file_utils.read_maps_from_fits(aeff_path, Defaults.NEbin)
cosz = file_utils.read_cosz_from_txt(coszenith_path, Defaults.NEbin)
nevts = np.loadtxt(nevents_path)
gg_overdensity = hp.fitsfunc.read_map(gg_sample_path)
gg_density = 1. + gg_overdensity
gg_density /= gg_density.sum()

nastro = 0.003 * nevts

atm_gen = AtmGenerator(Defaults.NEbin, coszenith=cosz, nevents_expected=nevts)

astro_gen = AstroGenerator(Defaults.NEbin, aeff=aeff, nevents_expected=nastro, pdf=gg_density)

astro_maps = astro_gen.generate_event_maps(1)
atm_maps = atm_gen.generate_event_maps(1)

figs.mollview_maps("astro", astro_maps[0])
figs.mollview_maps("atm", atm_maps[0])

testfigpath = os.path.join(Defaults.NUXGAL_PLOT_DIR, 'test')  
testfigfile = os.path.join(testfigpath, 'gen')
Utilityfunc.makedir_safe(testfigfile)

figs.save_all(testfigfile, 'pdf')
