"""Default values for analysis parameters"""

import os

import numpy as np

import healpy as hp

VERBOSE = False

NSIDE = 128

LOG_EMIN = 2. # 100 GeV
LOG_EMAX = 9. # 1e9 GeV
NEEdges = 8
NEbin = NEEdges-1

DT_DAYS = 333 # Length of run1
DT_SECONDS = 28771200 # 333 * 86400

M2_TO_CM2 = 1e4 # Conversion for effective area

if 'NUXGAL_DIR' in os.environ:
    NUXGAL_DIR = os.environ['NUXGAL_DIR']
else:
    NUXGAL_DIR = os.path.dirname(__file__).replace('/KIPAC/nuXgal', '')
print("Using %s for NUXGAL_DIR" % NUXGAL_DIR)

NUXGAL_ANCIL_DIR = os.path.join(NUXGAL_DIR, 'data', 'ancil')
NUXGAL_IRF_DIR = os.path.join(NUXGAL_DIR, 'data', 'irfs')
NUXGAL_DATA_DIR = os.path.join(NUXGAL_DIR, 'data', 'data')
NUXGAL_SYNTHETICDATA_DIR = os.path.join(NUXGAL_DIR, 'syntheticData')
NUXGAL_PLOT_DIR = os.path.join(NUXGAL_DIR, 'plots')

#Derived quantities
NPIXEL = hp.pixelfunc.nside2npix(NSIDE)
map_logE_edge = np.linspace(LOG_EMIN, LOG_EMAX, NEEdges)
map_logE_center = (map_logE_edge[0:-1] + map_logE_edge[1:]) / 2.
dlogE = np.mean(map_logE_edge[1:] - map_logE_edge[0:-1])
map_E_edge = np.power(10, map_logE_edge)
map_E_center = np.power(10, map_logE_center)
map_E_center_sq = map_E_center * map_E_center

NCL = 3*NSIDE
NALM = int((NCL) * (NCL+1) / 2)
MAX_L = NCL - 1

randomseed_galaxy = 42

ell = np.arange(NSIDE * 3)

# southern sky mask
exposuremap_theta, exposuremap_phi = hp.pixelfunc.pix2ang(NSIDE, np.arange(NPIXEL))
mask_muon = np.where(exposuremap_theta > 85. / 180 * np.pi)
