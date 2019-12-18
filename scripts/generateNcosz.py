import os
import argparse

import numpy as np
import healpy as hp

from KIPAC.nuXgal import Defaults

from KIPAC.nuXgal import file_utils

from KIPAC.nuXgal import FigureDict


def geneateFile(year):

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
    testfigpath = os.path.join(Defaults.NUXGAL_PLOT_DIR, 'test')


    figs = FigureDict()

    for dirname in [data_dir, irf_dir, plot_dir]:
        try:
            os.makedirs(dirname)
        except OSError:
            pass



    # -------------- coszenith distribution --------------
    Nzenith_bin = 60
    N_coszenith = np.zeros((len(Defaults.map_logE_center), Nzenith_bin))

    for file in [os.path.join(icecube_data_dir, year + '-events.txt')]:

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
        np.savetxt(os.path.join(irf_dir, 'Ncosz'+year+'-'+str(i)+'.txt'),
                   np.column_stack((cosZenithBinCenters, N_coszenith[i])))




if __name__ == '__main__':
    geneateFile('IC79-2010')
    geneateFile('IC86-2011')
    geneateFile('IC86-2012')
