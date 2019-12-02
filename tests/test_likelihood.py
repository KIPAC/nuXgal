"""Test utility for Likelihood fitting"""

import os

import numpy as np

import healpy as hp

from scipy.optimize import minimize


from KIPAC.nuXgal import Defaults

from KIPAC.nuXgal.EventGenerator import EventGenerator

from KIPAC.nuXgal.Analyze import Analyze

from KIPAC.nuXgal.Likelihood import Likelihood

from KIPAC.nuXgal.file_utils import read_maps_from_fits, write_maps_to_fits

from KIPAC.nuXgal.hp_utils import vector_apply_mask

from KIPAC.nuXgal.plot_utils import FigureDict

from Utils import MAKE_TEST_PLOTS

testfigpath = os.path.join(Defaults.NUXGAL_PLOT_DIR, 'test')
N_yr = 10.

llh = Likelihood(N_yr=N_yr)#, computeATM=False, computeASTRO =True, N_re=50)


def generateData(f_diff, f_gal, N_yr, fromGalaxy, writeMap, basekey='syntheticData'):
    cf = Analyze()
    eg = EventGenerator()
    if fromGalaxy:
        np.random.seed(Defaults.randomseed_galaxy)
    else:
        np.random.seed(Defaults.randomseed_galaxy + 102)


    density_nu = hp.sphtfunc.synfast(cf.cl_galaxy * f_gal, Defaults.NSIDE)
    density_nu = np.exp(density_nu)
    density_nu /= density_nu.sum() # a unique neutrino source distribution that shares the randomness of density_g

    np.random.seed(1202)
    countsmap = eg.astroEvent_galaxy(llh.N_2012_Aeffmax * N_yr * f_diff, density_nu)
    counts_map = vector_apply_mask(countsmap, Defaults.mask_muon, copy=False)

    if writeMap:
        filename_format = os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, basekey + '{i}.fits')
        write_maps_to_fits(countsmap, filename_format)
    return countsmap

def generateDataATM(N_yr, writeMap=False, basekey='syntheticDataATM'):
    eg = EventGenerator()
    countsmap = eg.atmEvent(N_yr)
    counts_map = vector_apply_mask(countsmap, Defaults.mask_muon, copy=False)

    if writeMap:
        filename_format = os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, basekey + '{i}.fits')
        write_maps_to_fits(countsmap, filename_format)
    return countsmap


def readData(basekey='syntheticData'):
    filename_format = os.path.join(Defaults.NUXGAL_SYNTHETICDATA_DIR, basekey + '{i}.fits')
    return read_maps_from_fits(filename_format, Defaults.NEbin)





def showDataModel(datamap, energyBin, testATM):
    w_data = llh.cf.crossCorrelationFromCountsmap(datamap)
    if MAKE_TEST_PLOTS:
        figs = FigureDict()
        o_dict = figs.setup_figure('test_w_CL', xlabel='$l$', ylabel='$C_l$', figsize=(8, 6))
        fig = o_dict['fig']
        axes = o_dict['axes']
        #axes.set_xscale('log')
        axes.set_ylim(-3e-4, 3e-4)
        axes.set_xlim(0, 400)

        axes.plot(llh.cf.l, llh.cf.cl_galaxy * 0.6 ** 0.5, 'r')
        if testATM:
            axes.errorbar(llh.cf.l_cl,  w_data[energyBin], yerr=llh.w_atm_std[energyBin], markersize=4, color='grey', fmt='s', capthick=2)
        else:
            axes.errorbar(llh.cf.l_cl,  w_data[energyBin], yerr=llh.w_astro_std[energyBin], markersize=4, color='grey', fmt='^', capthick=2)
            #axes.scatter(llh.cf.l_cl,  w_data[energyBin],marker='o', s=4, color='b',zorder=10)

        fig.legend()
        figs.save_all(testfigpath, 'pdf')


def lnL_atm(f, w_data, lmin, energyBin):
    w_model = f * llh.cf.cl_galaxy[0:Defaults.NCL]
    lnL_l = -(w_data[energyBin] - w_model)**2 / llh.w_atm_std_square[energyBin]
    return np.sum(lnL_l[lmin:])

def lnL_astro(f, w_data, lmin, energyBin):
    w_model = f * llh.cf.cl_galaxy[0:Defaults.NCL]
    lnL_l = -(w_data[energyBin] - w_model)**2 / llh.w_astro_std_square[energyBin]
    return np.sum(lnL_l[lmin:])

def minimize__lnL(w_data, lmin, energyBin, lnL):
    nll = lambda *args: -lnL(*args)
    initial = 0.5 + 0.1 * np.random.randn()
    soln = minimize(nll, initial, args=(w_data, lmin, energyBin), bounds=[(0, 1)])
    return soln.x



w_model_f1 = np.zeros((Defaults.NEbin, Defaults.NCL))
for i in range(Defaults.NEbin):
    w_model_f1[i] = llh.cf.cl_galaxy[0:Defaults.NCL]


def lnL_all_atm(f, w_data, lmin):
    w_model = (w_model_f1[0:4].T * f).T
    lnL_le = -(w_data[0:4] - w_model) ** 2 / llh.w_atm_std_square[0:4]
    return np.sum(lnL_le[:, lmin:])


Ebinmax = 3
N_nonzero = 1
while (N_nonzero > 0) & (Ebinmax < Defaults.NEbin - 1) :
    Ebinmax += 1
    N_nonzero = len(np.where(llh.w_astro_std_square[Ebinmax] > 0)[0])

print (Ebinmax, N_nonzero)

def lnL_all_astro(f, w_data, lmin):
    w_model = (w_model_f1[0:Ebinmax].T * f).T
    lnL_le = -(w_data[0:Ebinmax] - w_model) ** 2 / llh.w_astro_std_square[0:Ebinmax]
    return np.sum(lnL_le[:, lmin:])

def minimize__lnL_all(w_data, lmin, lnL_all, len_f):
    nll = lambda *args: -lnL_all(*args)
    initial = 1 + 0.1 * np.random.randn(len_f)
    soln = minimize(nll, initial, args=(w_data, lmin), bounds=[(0, 1)] * (len_f))
    return soln.x, (lnL_all(soln.x, w_data, lmin) - lnL_all(np.zeros(len_f), w_data, lmin)) * 2




def plotLnL(w_data, lmin, energyBin, lnL):
    figs = FigureDict()
    o_dict = figs.setup_figure('test__lnL', xlabel='$l$', ylabel='$C_l$', figsize=(8, 6))
    fig = o_dict['fig']
    axes = o_dict['axes']
    ftest = np.linspace(0, 1, 50)
    lnL_f = []
    for _f in ftest:
        lnL_f.append(lnL(_f, w_data, lmin, energyBin))
    axes.plot(ftest, lnL_f)
    figs.save_all(testfigpath, 'pdf')


def test_STDdependence():
    energyBin = 4
    energyBin2 = 2
    figs = FigureDict()
    o_dict = figs.setup_figure('compare_std', xlabel='$l$', ylabel='$C_l$', figsize=(8, 6))
    fig = o_dict['fig']
    axes = o_dict['axes']
    axes.set_yscale('log')
    axes.set_ylim(1e-6, 1e-2)
    axes.plot(llh.cf.l_cl, llh.w_astro_std[energyBin], label='astro')
    axes.plot(llh.cf.l_cl, llh.w_astro_std[energyBin2] * (llh.Ncount_astro[energyBin2] / llh.Ncount_astro[energyBin])**0.5, label='astro2')

    #axes.plot(llh.cf.l_cl, llh.w_atm_std[energyBin], label='atm')
    #axes.plot(llh.cf.l_cl, llh.w_atm_std[energyBin2] * (llh.Ncount_atm[energyBin2] / llh.Ncount_atm[energyBin])**0.5, label='atm2')

    axes.plot(llh.cf.l_cl, llh.w_atm_std[energyBin2] * (llh.Ncount_atm[energyBin2] / llh.Ncount_astro[energyBin])**0.5, label='atm-astro2')

    fig.legend()
    figs.save_all(testfigpath, 'pdf')




if __name__ == '__main__':

     


    testATM = False

    if testATM:
        datamap = generateDataATM(N_yr)
        w_data = llh.cf.crossCorrelationFromCountsmap(datamap)

        lmin, energyBin = 2, 1
        showDataModel(datamap, energyBin, testATM)
        plotLnL(w_data, lmin, energyBin, lnL_atm)
        print (minimize__lnL_all(w_data, lmin, lnL_all_atm, 4))

    else:
        datamap = generateData(1.0, 0.6, N_yr, fromGalaxy=True, writeMap=False)
        w_data = llh.cf.crossCorrelationFromCountsmap(datamap)
        lmin, energyBin = 2, 3
        showDataModel(datamap, energyBin, testATM)
        print (minimize__lnL_all(w_data, lmin, lnL_all_astro, Ebinmax))
