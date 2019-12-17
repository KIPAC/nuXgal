import os
import argparse

import numpy as np
import healpy as hp

from KIPAC.nuXgal import Defaults

from KIPAC.nuXgal import file_utils

from KIPAC.nuXgal import FigureDict

testfigpath = os.path.join(Defaults.NUXGAL_PLOT_DIR, 'IceCube3yr')
fluxmap3yr_format = os.path.join(Defaults.NUXGAL_DATA_DIR,  'IceCube3yr_fluxmap{i}.fits')
countsmap3yr_format = os.path.join(Defaults.NUXGAL_DATA_DIR,  'IceCube3yr_countsmap{i}.fits')



def getFlux_Countmaps(year, check=False):

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

    countsmap_format = os.path.join(data_dir, year+'_countsmap{i}.fits')


    figs = FigureDict()

    for dirname in [data_dir, irf_dir, plot_dir]:
        try:
            os.makedirs(dirname)
        except OSError:
            pass

    # counts map in selected energy bins

    Aeff_file = np.loadtxt(os.path.join(icecube_data_dir,  year+'-TabulatedAeff.txt'))
    AtmBG_file = np.loadtxt(os.path.join(icecube_data_dir, year+'-events.txt'))

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

    if check:
        idx = 1
        print ('--- check event ', idx, '----')
        print ('energy: ', AtmBG_file[idx][1], index_E_Aeff_table[idx], logEmin[index_E_Aeff_table[idx]], logEmin[index_E_Aeff_table[idx] + 1])
        print ('coszenith: ',np.cos(np.radians(AtmBG_file[idx][6])), index_coszenith_Aeff_table[idx], cosZenith_min[index_coszenith_Aeff_table[idx]], cosZenith_min[index_coszenith_Aeff_table[idx]+1])
        print ('Aeff: ', Aeff_event[idx])


    # countsmap has the shape (number of energy bins, healpy map size)
    fluxmap = np.zeros((Defaults.NEbin, hp.pixelfunc.nside2npix(Defaults.NSIDE)))
    countsmap = np.zeros((Defaults.NEbin, hp.pixelfunc.nside2npix(Defaults.NSIDE)))

    # get energy index of events
    _index_map_logE = np.searchsorted(Defaults.map_logE_edge, AtmBG_file[:, 1]) - 1
    _index_map_logE[_index_map_logE == -1] = 0 # group logE < 2 events to bin 0

    # convert event directions to pixel numbers
    # pi - zenith_south_pole = zenith_regular
    # assign random azimuthal angles
    _index_map_pixel = hp.pixelfunc.ang2pix(Defaults.NSIDE, np.radians(90. - AtmBG_file[:, 4]) , np.radians(360. - AtmBG_file[:, 3]))

    # put events into healpy maps
    for i, _ in enumerate(AtmBG_file):

        if Aeff_event[i] == 0 and AtmBG_file[i][6] > 90. :
            idx = i
            print ('--- check event ',year, ' ', idx, '----')
            print ('energy: ', AtmBG_file[idx][1], index_E_Aeff_table[idx], logEmin[index_E_Aeff_table[idx]], logEmin[index_E_Aeff_table[idx] + 1])
            print ('coszenith: ',np.cos(np.radians(AtmBG_file[idx][6])), index_coszenith_Aeff_table[idx], cosZenith_min[index_coszenith_Aeff_table[idx]], cosZenith_min[index_coszenith_Aeff_table[idx]+1])
            print ('Aeff: ', Aeff_event[idx])


        else:
            fluxmap[_index_map_logE[i]][_index_map_pixel[i]] += 1. / Aeff_event[i]
            countsmap[_index_map_logE[i]][_index_map_pixel[i]] += 1.


    file_utils.write_maps_to_fits(countsmap, countsmap_format)

    if check:
        mask = np.zeros(Defaults.NPIXEL)
        mask[Defaults.idx_muon] = 1.
        for i in range(Defaults.NEbin):
            test = np.ma.masked_array(countsmap[i], mask = mask)
            print (year, i, test.sum())

    return fluxmap, countsmap


if __name__ == '__main__':
    flux2010, counts2010 = getFlux_Countmaps('IC79-2010')
    flux2011, counts2011 = getFlux_Countmaps('IC86-2011')
    flux2012, counts2012 = getFlux_Countmaps('IC86-2012')

    fluxmap = flux2010 + flux2011 + flux2012
    countsmap = counts2010 + counts2011 + counts2012

    file_utils.write_maps_to_fits(fluxmap, fluxmap3yr_format)
    file_utils.write_maps_to_fits(countsmap, countsmap3yr_format)

    for i in range(Defaults.NEbin):
        fluxmap[i][Defaults.idx_muon] = hp.UNSEEN
        countsmap[i][Defaults.idx_muon] = hp.UNSEEN


    figs = FigureDict()
    figs.mollview_maps('fluxmap', fluxmap)
    figs.save_all(testfigpath, 'pdf')

    figs = FigureDict()
    figs.mollview_maps('countsmap', countsmap)
    figs.save_all(testfigpath, 'pdf')

    mask = np.zeros(Defaults.NPIXEL)
    mask[Defaults.idx_muon] = 1.
    for i in range(Defaults.NEbin):
        test = np.ma.masked_array(countsmap[i], mask = mask)
        print (i, test.sum())
