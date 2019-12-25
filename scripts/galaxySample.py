import os
import argparse
import numpy as np
import healpy as hp

from KIPAC.nuXgal import Defaults


def computeGalaxymap():

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", default=None, required=True,
                        help="Directory with WISE-2MASS galaxy data")

    parser.add_argument("-o", "--output", default=Defaults.NUXGAL_DIR,
                        help="Output directory")

    args = parser.parse_args()

    WISE_data_dir = os.path.join(args.input, 'irsa_catalog_search_results.txt')

    galaxySampleFile = np.loadtxt(WISE_data_dir)

    # from dec, RA to healpy coordinates: theta = pi / 2 - dec, phi = RA
    index_map_pixel = hp.pixelfunc.ang2pix(Defaults.NSIDE, np.radians(90. - galaxySampleFile[:, 1]), np.radians(galaxySampleFile[:, 0]))

    galaxymap = np.zeros(hp.pixelfunc.nside2npix(Defaults.NSIDE))

    for i in index_map_pixel:
        galaxymap[i] += 1.

    fitspath = os.path.join(Defaults.NUXGAL_ANCIL_DIR, 'WISE_galaxymap.fits')

    hp.fitsfunc.write_map(fitspath, galaxymap, overwrite=True)


if __name__ == '__main__':
    computeGalaxymap()
