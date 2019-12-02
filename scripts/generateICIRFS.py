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

    icecube_data_dir = os.path.join(args.input, 'data/3year-data-release')
    data_dir = os.path.join(args.output, 'data', 'data')
    irf_dir = os.path.join(args.output, 'data', 'irfs')
    plot_dir = os.path.join(args.output, 'plots')

    counts_map_format = os.path.join(data_dir, 'counts_atm{i}.fits')
    aeff_map_format = os.path.join(irf_dir, 'Aeff{i}.fits')

    figs = FigureDict()

    for dirname in [data_dir, irf_dir, plot_dir]:
        try:
            os.makedirs(dirname)
        except OSError:
            pass

    # counts map in selected energy bins


    # -------------- counts map --------------

    AtmBG_file = np.loadtxt(os.path.join(icecube_data_dir, 'IC86-2012-events.txt'))


    # countsmap has the shape (number of energy bins, healpy map size)
    countsmap = np.zeros((Defaults.NEbin, hp.pixelfunc.nside2npix(Defaults.NSIDE)))

    # get energy index of events
    _index_map_logE = np.searchsorted(Defaults.map_logE_edge, AtmBG_file[:, 1]) - 1
    _index_map_logE[_index_map_logE == -1] = 0 # group logE < 2 events to bin 0

    # convert event directions to pixel numbers
    # pi - zenith_south_pole = zenith_regular
    # assign random azimuthal angles
    randomphi = np.random.random_sample(len(AtmBG_file)) * 2 * np.pi
    _index_map_pixel = hp.pixelfunc.ang2pix(Defaults.NSIDE, (180. - AtmBG_file[:, 6]) * np.pi / 180., randomphi)

    # put events into healpy maps
    for i, _ in enumerate(AtmBG_file):
        countsmap[_index_map_logE[i]][_index_map_pixel[i]] += 1


    file_utils.write_maps_from_fits(countsmap, counts_map_format)
    figs.mollview_maps('counts_atm', countsmap)


    # -------------- exposure map --------------
    Aeff_file = np.loadtxt(os.path.join(icecube_data_dir, 'IC86-2012-TabulatedAeff.txt'))

    # compute exposure map at the center of an energy bin
    exposuremap = np.zeros((Defaults.NEbin, hp.pixelfunc.nside2npix(Defaults.NSIDE)))

    # effective area, 200 in cos zenith, 70 in E
    Aeff_table = Aeff_file[:, 4] # np.reshape(Aeff_file[:, 4], (70, 200))
    Emin = np.reshape(Aeff_file[:, 0], (70, 200))[:, 0]
    #Emax = np.reshape(Aeff_file[:, 1], (70, 200))[:, 0]
    cosZenith_min = np.reshape(Aeff_file[:, 2], (70, 200))[0]
    #cosZenith_max = np.reshape(Aeff_file[:, 3], (70, 200))[0]

    exposuremap_theta, _ = hp.pixelfunc.pix2ang(Defaults.NSIDE, np.arange(hp.pixelfunc.nside2npix(Defaults.NSIDE)))
    exposuremap_costheta = np.cos(np.pi - exposuremap_theta) # converting to South pole view
    index_coszenith = np.searchsorted(cosZenith_min, exposuremap_costheta) - 1
    index_E = np.searchsorted(np.log10(Emin), Defaults.map_logE_center) - 1

    for i in np.arange(len(Defaults.map_logE_center)):
        exposuremap[i] = Aeff_table[index_E[i] * 200 + index_coszenith]

    file_utils.write_maps_from_fits(exposuremap, aeff_map_format)
    figs.mollview_maps('Aeff', exposuremap)


    # -------------- coszenith distribution --------------
    Nzenith_bin = 60
    N_coszenith = np.zeros((len(Defaults.map_logE_center), Nzenith_bin))

    for file in [os.path.join(icecube_data_dir, 'IC86-2012-events.txt'),
                 os.path.join(icecube_data_dir, 'IC86-2011-events.txt'),
                 os.path.join(icecube_data_dir, 'IC79-2010-events.txt')]:

        AtmBG_file = np.loadtxt(file)
        _index_map_logE = np.searchsorted(Defaults.map_logE_edge, AtmBG_file[:, 1]) - 1
        _index_map_logE[_index_map_logE == -1] = 0 # group logE < 2 events to bin 0

        for i in np.arange(len(Defaults.map_logE_center)):
            N_coszenith_i, cosZenithBinEdges =\
                np.histogram(np.cos(np.pi - AtmBG_file[:, 6][_index_map_logE == i] * np.pi / 180.), Nzenith_bin, (-1, 1))
            N_coszenith[i] += N_coszenith_i


    cosZenithBinCenters = (cosZenithBinEdges[0:-1] + cosZenithBinEdges[1:])/2.

    figs.plot_yvals('N_coszenith', cosZenithBinCenters, N_coszenith,
                    xlabel=r'$\cos\,\theta$',
                    ylabel='Number of counts in 2010-2012 data',
                    figsize=(8, 6))

    for i in np.arange(Defaults.NEbin):
        np.savetxt(os.path.join(irf_dir, 'N_coszenith'+str(i)+'.txt'),
                   np.column_stack((cosZenithBinCenters, N_coszenith[i])))


    # average count number per energy bin in IC86 one year data
    eventnumber_Ebin = np.zeros(len(Defaults.map_logE_center))
    eventnumber_Ebin2 = np.zeros(len(Defaults.map_logE_center))

    for file in [os.path.join(icecube_data_dir, 'IC86-2012-events.txt'),
                 os.path.join(icecube_data_dir, 'IC86-2011-events.txt')]:
        AtmBG_file = np.loadtxt(file)
        eventnumber_Ebin += np.histogram(AtmBG_file[:, 1], Defaults.map_logE_edge)[0]
        _index_map_logE = np.searchsorted(Defaults.map_logE_edge, AtmBG_file[:, 1]) - 1
        #_index_map_logE[_index_map_logE == -1] = 0 # group logE < 2 events to bin 0
        eventnumber_Ebin2 += np.histogram(_index_map_logE, range(len(Defaults.map_logE_edge)))[0]


    #print eventnumber_Ebin
    #print eventnumber_Ebin2
    np.savetxt(os.path.join(irf_dir, 'eventNumber_Ebin_perIC86year.txt'), eventnumber_Ebin / 2.)
    #print np.sum(eventnumber_Ebin),   np.sum(eventnumber_Ebin2)


    # ------------- check counts rate -----------
    bgmap = np.zeros((len(Defaults.map_logE_center), hp.pixelfunc.nside2npix(Defaults.NSIDE)))
    for i in np.arange(len(Defaults.map_logE_center)):
        bgmap[i] = hp.fitsfunc.read_map(os.path.join(data_dir, 'counts_atm' + str(i)+'.fits'), verbose=False)

    # check rate per cos zenith bin
    coszenith_bin = np.linspace(-1, 1, 20)
    coszenith_bin_IC = np.cos(np.pi - np.arccos(coszenith_bin))
    countsRate = np.zeros(len(coszenith_bin))
    for i in range(len(coszenith_bin)-1):
        index_coszenith = np.where((np.cos(exposuremap_theta) < coszenith_bin[i+1]) & (np.cos(exposuremap_theta) > coszenith_bin[i]))
        for iE in range(len(Defaults.map_logE_center)):
            countsRate[i] += np.sum(bgmap[iE][index_coszenith]) / (333 * 24 * 3600.)


    o_dict = figs.plot('countsRate', coszenith_bin_IC, countsRate,
                       xlabel=r"$\cos(\theta_{z})$",
                       ylabel="Count Rate")
    axes = o_dict['axes']
    axes.set_yscale('log')
    axes.set_ylim(1e-8, 1e-2)

    figs.save_all(os.path.join(plot_dir, 'IceCube'), 'pdf')


if __name__ == '__main__':
    main()
