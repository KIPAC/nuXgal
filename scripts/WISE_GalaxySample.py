import os
import argparse
import numpy as np
import healpy as hp

from KIPAC.nuXgal import Defaults




def computeGalaxymap(galaxySampleFile):

    # from dec, RA to healpy coordinates: theta = pi / 2 - dec, phi = RA
    index_map_pixel = hp.pixelfunc.ang2pix(Defaults.NSIDE, np.radians(90. - galaxySampleFile[:, 1]), np.radians(galaxySampleFile[:, 0]))
    galaxymap = np.zeros(hp.pixelfunc.nside2npix(Defaults.NSIDE))
    for i in index_map_pixel:
        galaxymap[i] += 1.
    fitspath = os.path.join(Defaults.NUXGAL_ANCIL_DIR, 'WISE_galaxymap.fits')
    hp.fitsfunc.write_map(fitspath, galaxymap, overwrite=True)


def computeAlm(galaxySampleFile):

    # finer pixels for alm computation
    NSIDE_local = 1024
    index_map_pixel = hp.pixelfunc.ang2pix(NSIDE_local, np.radians(90. - galaxySampleFile[:, 1]), np.radians(galaxySampleFile[:, 0]))
    galaxymap = np.zeros(hp.pixelfunc.nside2npix(NSIDE_local))

    for i in index_map_pixel:
        galaxymap[i] += 1.


    from astropy import units as u
    from astropy.coordinates import SkyCoord
    exposuremap_theta, exposuremap_phi = hp.pixelfunc.pix2ang(NSIDE_local, np.arange(hp.pixelfunc.nside2npix(NSIDE_local)))
    c_icrs = SkyCoord(ra=exposuremap_phi * u.radian, dec=(np.pi/2 - exposuremap_theta)*u.radian, frame='icrs')
    idx_galaxymask = np.where(np.abs(c_icrs.galactic.b.degree) < 10)
    galaxymap[idx_galaxymask] = hp.UNSEEN
    galaxymap = hp.ma(galaxymap)
    galaxymap_od = galaxymap / galaxymap.mean() - 1.
    alm = hp.map2alm(galaxymap_od)
    fitspath = os.path.join(Defaults.NUXGAL_ANCIL_DIR, 'WISE_galaxyalm.fits')
    hp.fitsfunc.write_alm(fitspath, alm, lmax = Defaults.MAX_L, overwrite=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", default=None, required=True,
                            help="Directory with WISE-2MASS galaxy data")

    parser.add_argument("-o", "--output", default=Defaults.NUXGAL_DIR,
                            help="Output directory")

    args = parser.parse_args()

    WISE_data_dir = os.path.join(args.input, 'irsa_catalog_search_results.txt')

    galaxySampleFile = np.loadtxt(WISE_data_dir)

    computeGalaxymap(galaxySampleFile)

    computeAlm(galaxySampleFile)
