"""Utility function for reading and writing files"""

import numpy as np

import healpy as hp

from . import Defaults

from .hp_utils import reshape_array_to_2d


def read_cls_from_txt(fileformat, nmap=1, ncl=500):
    """Read a set of cls or series of sets of cls from text files

    Parameters
    ----------
    fileformat : `str`
        For a single map it is just the file path,
        for multiple maps, it should include {i} to format the filepath
    nmap : `int`
        Number of maps to read

    Returns
    -------
    cls : `np.ndarray`
        A 2 dimensional array of cls.  In the case of a single map, the first dimension size is 1.
    """
    if nmap == 1:
        the_cls = np.loadtxt(fileformat)[:ncl]
        return reshape_array_to_2d(the_cls)
    cl_list = [np.loadtxt(fileformat.format(i=i))[:ncl] for i in range(nmap)]
    return np.vstack(cl_list)


def read_cosz_from_txt(fileformat, nmap=1, ncl=500):
    """Read a set of cls or series of sets of cls from text files

    Parameter
    ---------
    fileformat : `str`
        For a single map it is just the file path,
        for multiple maps, it should include {i} to format the filepath
    nmap : `int`
        Number of maps to read
    ncl : `int`
        Number of cl to reah

    Returns
    -------
    cls : `np.ndarray`
        A 2 dimensional array of cls.  In the case of a single map, the first dimension size is 1.
    """
    if nmap == 1:
        the_cosz = np.loadtxt(fileformat)[:ncl]
        return the_cosz
    cosz_list = [np.loadtxt(fileformat.format(i=i))[:ncl] for i in range(nmap)]
    return np.array(cosz_list)


def read_maps_from_fits(fileformat, nmap=1):
    """Read a map or series of maps from FITS files

    Parameter
    ---------
    fileformat : `str`
        For a single map it is just the file path,
        for multiple maps, it should include {i} to format the filepath
    nmap : `int`
        Number of maps to read

    Returns
    -------
    maps : `np.ndarray`
        A 2 dimensional array of maps.  In the case of a single map, the first dimension size is 1.
    """
    if nmap == 1:
        the_map = hp.fitsfunc.read_map(fileformat, verbose=Defaults.VERBOSE)
        return reshape_array_to_2d(the_map)
    map_list = [hp.fitsfunc.read_map(fileformat.format(i=i), verbose=Defaults.VERBOSE) for i in range(nmap)]
    return np.vstack(map_list)


def write_maps_from_fits(maps, fileformat):
    """Read a map or series of maps from FITS files

    Parameter
    ---------
    maps : `np.ndarray`
        Maps to write
    fileformat : `str`
        For a single map it is just the file path, for multiple maps,
        it should include {i} to format the filepath
    """
    maps_2d = reshape_array_to_2d(maps)
    for i, _map in enumerate(maps_2d):
        hp.fitsfunc.write_map(fileformat.format(i=i), _map, overwrite=True)
