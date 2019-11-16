
import os

import numpy as np

import healpy as hp

NSIDE = 128

LOG_EMIN = 2. # 100 GeV
LOG_EMAX = 9. # 1e9 GeV
NEEdges = 8
NEbin = NEEdges-1

DT_DAYS = 333 # Length of run1
DT_SECONDS = 28771200 # 333 * 86400

M2_TO_CM2 = 1e4 # Conversion for effective area

if 'NUXGAL_DATA_DIR' in os.environ:
    NUXGAL_DATA_DIR = os.environ['NUXGAL_DATA_DIR']
else:
    NUXGAL_DATA_DIR = '..'
print("Using %s for NUXGAL_DATA_DIR" % NUXGAL_DATA_DIR)

#Derived quantities
NPIXEL = hp.pixelfunc.nside2npix(NSIDE)
map_logE_edge = np.linspace(LOG_EMIN, LOG_EMAX, NEEdges)
map_logE_center = (map_logE_edge[0:-1] + map_logE_edge[1:]) / 2.
dlogE = np.mean(map_logE_edge[1:] - map_logE_edge[0:-1])
map_E_center = np.power(10, map_logE_center)

