import os
import argparse

import numpy as np
import healpy as hp

from KIPAC.nuXgal import Defaults

from KIPAC.nuXgal import file_utils

from KIPAC.nuXgal import FigureDict




def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", default=None, required=True,
                        help="Directory with IceCube data")

    parser.add_argument("-o", "--output", default=Defaults.NUXGAL_DIR,
                        help="Output directory")

    args = parser.parse_args()

    icecube_data_dir = os.path.join(args.input, '3year-data-release')
    data_dir = os.path.join(args.output, 'data', 'data')
    irf_dir = os.path.join(args.output, 'data', 'irfs')
    plot_dir = os.path.join(args.output, 'plots')
    testfigpath = os.path.join(Defaults.NUXGAL_PLOT_DIR, 'test')

    fluxmap_format = os.path.join(data_dir, 'fluxmap_atm{i}.fits')

    figs = FigureDict()

    for dirname in [data_dir, irf_dir, plot_dir]:
        try:
            os.makedirs(dirname)
        except OSError:
            pass

    # counts map in selected energy bins

    Aeff_file = np.loadtxt(os.path.join(icecube_data_dir, 'IC86-2012-TabulatedAeff.txt'))
    AtmBG_file = np.loadtxt(os.path.join(icecube_data_dir, 'IC86-2012-events.txt'))

    # effective area, 200 in cos zenith, 70 in E
    Aeff_table = Aeff_file[:, 4] # np.reshape(Aeff_file[:, 4], (70, 200))
    Emin = np.reshape(Aeff_file[:, 0], (70, 200))[:, 0]
    #Emax = np.reshape(Aeff_file[:, 1], (70, 200))[:, 0]
    cosZenith_min = np.reshape(Aeff_file[:, 2], (70, 200))[0]
    #cosZenith_max = np.reshape(Aeff_file[:, 3], (70, 200))[0]
    logEmin = np.log10(Emin)

    index_E_Aeff_table = np.searchsorted(logEmin, AtmBG_file[:, 1]) - 1
    index_E_Aeff_table[index_E_Aeff_table == -1] = 0 # group logE < 2 events to bin 0
    index_coszenith_Aeff_table = np.searchsorted(cosZenith_min, np.cos(np.radians(AtmBG_file[:, 6]))) - 1
    Aeff_event = Aeff_table[index_E_Aeff_table * 200 + index_coszenith_Aeff_table]

    idx = 1
    print ('--- check event ', idx, '----')
    print ('energy: ', AtmBG_file[idx][1], index_E_Aeff_table[idx], logEmin[index_E_Aeff_table[idx]], logEmin[index_E_Aeff_table[idx] + 1])
    print ('coszenith: ',np.cos(np.radians(AtmBG_file[idx][6])), index_coszenith_Aeff_table[idx], cosZenith_min[index_coszenith_Aeff_table[idx]], cosZenith_min[index_coszenith_Aeff_table[idx]+1])
    print ('Aeff: ', Aeff_event[idx])


    # countsmap has the shape (number of energy bins, healpy map size)
    fluxmap = np.zeros((Defaults.NEbin, hp.pixelfunc.nside2npix(Defaults.NSIDE)))

    # get energy index of events
    _index_map_logE = np.searchsorted(Defaults.map_logE_edge, AtmBG_file[:, 1]) - 1
    _index_map_logE[_index_map_logE == -1] = 0 # group logE < 2 events to bin 0

    # convert event directions to pixel numbers
    # pi - zenith_south_pole = zenith_regular
    # assign random azimuthal angles
    _index_map_pixel = hp.pixelfunc.ang2pix(Defaults.NSIDE, np.radians(90. - AtmBG_file[:, 4]) , np.radians(360. - AtmBG_file[:, 3]))

    # put events into healpy maps
    for i, _ in enumerate(AtmBG_file):
        fluxmap[_index_map_logE[i]][_index_map_pixel[i]] += 1. / Aeff_event[i]


    file_utils.write_maps_to_fits(fluxmap, fluxmap_format)
    figs = FigureDict()
    figs.mollview_maps('fluxmap', fluxmap)
    figs.save_all(testfigpath, 'pdf')

if __name__ == '__main__':
    main()
