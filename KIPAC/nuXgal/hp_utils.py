"""Utility function for healpy, mainly to vectorize calls to healpy.sphtfunc"""

import numpy as np

import healpy as hp

def cl_to_alm_no_phi(cl, nalm):
    """Convert cl to alm, assuming no phi dependence

    I.e., all the power goes to the m=0 component

    Parameters
    ----------
    cl : `np.array`
        Single array of cl
    nalm : `int`
        Number of output alm, could be computed from side of `cl`

    Returns
    -------
    alm : `np.array`
        Array of the alm values, only the m=0 elements are non-zero
    """
    out_alm = np.zeros((nalm), np.complex128)
    n_cl = cl.size
    out_alm[0:n_cl].real = cl
    return out_alm


def reshape_array_to_2d(in_array):
    """Reshape an N-dimensional array to 2d

    Adds a placeholder index to 1d arrays,
    Compactifies axes 0 to n-1 for 2>d arrays

    This is useful to vectorize calls to healpy.sphtfunc functions

    Parameters
    ----------
    in_array : `np.array`
        Input array

    Returns
    -------
    out_array : `np.array`
        2d array
    """
    orig_shape = in_array.shape
    n_dim = len(orig_shape)
    if n_dim == 1:
        return in_array.reshape((1, orig_shape[0]))
    if n_dim == 2:
        return in_array
    prod = np.product(orig_shape[0:-1])
    return in_array.reshape((prod, orig_shape[-1]))


def vectorize_hp_func(hp_func, in_array, *args, **kwargs):
    """Vectorize a call to a healpy function

    Parameters
    ----------
    hp_func : `func`
        The function to be called

    in_array : `np.array`
        Input array

    Keywords
    --------
    out_shape : `tuple` or `list`
        Shape of the output array

    Other keywords are passed to the calls to `hp_func`


    Returns
    -------
    out_array : `np.array`
        Output array
    """
    kwcopy = kwargs.copy()
    out_shape = kwcopy.pop('out_shape', in_array.shape)
    array_2d = reshape_array_to_2d(in_array)
    out_list = [hp_func(array_1d, *args, **kwcopy) for array_1d in array_2d]
    return np.vstack(out_list).reshape(out_shape)


def vectorize_gen_syn_data(gen_func, in_array, n_trials, *args, **kwargs):
    """Vectorize a call to generate synthetic data

    Parameters
    ----------
    gen_func : `func`
        The function to be called

    in_array : `np.array`
        Input array

    n_trials : `int`
        Number of realizations of synthetic data to generate

    Keywords
    --------
    out_shape : `tuple` or `list`
        Shape of the output array

    Other keywords are passed to the calls to `hp_func`

    Returns
    -------
    out_array : `np.array`
        Output array
    """
    kwcopy = kwargs.copy()
    out_shape = kwcopy.pop('out_shape', None)
    if out_shape is None:
        out_shape = [n_trials] + list(in_array.shape)

    out_list = [gen_func(in_array, *args, **kwcopy) for _ in range(n_trials)]
    return np.vstack(out_list).reshape(out_shape)


def vector_intensity_from_counts_and_exposure(counts, exposure):
    """Generate intensity maps from counts and exposure maps

    Parameters
    ----------
    counts : `np.array`
        Counts, i.e., number of galaxies, or number of neutrinos

    exposure : `np.array` or `None`
        Exposure, i.e,. intstrument or survey sensitivity

    Returns
    -------
    intensity : `np.array`
        Basically counts / exposure or just counts if exposure is `None`
    """
    if counts is None:
        return None
    if exposure is None:
        return counts
    return np.divide(counts, exposure, out=np.zeros_like(exposure), where=exposure != 0)


def vector_overdensity_from_intensity(intensity):
    """Convert intenstiy maps to overdensity maps

    Parameters
    ----------
    intensity : `np.array`
        Basically counts / exposure or just counts if exposure is `None`

    Returns
    -------
    overdensity : `np.array`
        Fractional overdensity per pixel
    """
    orig_shape = intensity.shape
    intensity_2d = reshape_array_to_2d(intensity)
    mean_intensity = intensity_2d.mean(-1)
    return np.nan_to_num(((intensity_2d.T - mean_intensity)/mean_intensity).T, 0.).reshape(orig_shape)


def vector_alm_from_overdensity(overdensity, n_alm, **kwargs):
    """Convert overdensity maps to alm coefficients

    Parameters
    ----------
    overdensity : `np.array`
        Fractional overdensity per pixel

    npix : `int`
        Number of alm coefficients, could be computed from map size

    Returns
    -------
    alm : `np.array`
        alm coefficients
    """
    out_shape = list(overdensity.shape)
    out_shape[-1] = n_alm
    overdensity_2d = reshape_array_to_2d(overdensity)
    return vectorize_hp_func(hp.sphtfunc.map2alm, overdensity_2d, out_shape=out_shape, **kwargs)


def vector_overdensity_from_alm(alm, npix, **kwargs):
    """Convert alm coefficients to overdensity maps

    Parameters
    ----------
    alm : `np.array`
        alm coefficients

    npix : `int`
        Number of npixels, could be computed from alm size

    Returns
    -------
    overdensity : `np.array`
        Fractional overdensity per pixel
    """
    out_shape = list(alm.shape)
    out_shape[-1] = npix
    alm_2d = reshape_array_to_2d(alm)
    return vectorize_hp_func(hp.sphtfunc.alm2map, alm_2d, out_shape=out_shape, **kwargs)


def vector_cl_from_overdensity(overdensity, n_cl, **kwargs):
    """Convert overdensity maps to cl coefficients

    Parameters
    ----------
    overdensity : `np.array`
        Fractional overdensity per pixel

    n_cl : `int`
        Number of cl coeffiecients, could be computed from alm size

    Returns
    -------
    cl : `np.array`
        cl coefficients
    """
    out_shape = list(overdensity.shape)
    out_shape[-1] = n_cl
    overdensity_2d = reshape_array_to_2d(overdensity)
    return vectorize_hp_func(hp.sphtfunc.anafast, overdensity_2d, out_shape=out_shape, **kwargs)


def vector_cl_from_alm(alm, n_cl, **kwargs):
    """Convert alm coefficients to cl coefficients,
    i.e., compute the power as a function of l

    Parameters
    ----------
    alm : `np.array`
        alm coefficients

    n_cl : `int`
        Number of cl coeffiecients, could be computed from alm size

    Returns
    -------
    cl : `np.array`
        cl coefficients
    """
    out_shape = list(alm.shape)
    out_shape[-1] = n_cl
    alm_2d = reshape_array_to_2d(alm)
    return vectorize_hp_func(hp.sphtfunc.alm2cl, alm_2d, out_shape=out_shape, **kwargs)


def vector_cl_to_alm_no_phi(cl, n_alm, **kwargs):
    """Convert cl to alm, assuming no phi dependence

    I.e., all the power goes to the m=0 component

    Parameters
    ----------
    cl : `np.array`
        Single array of cl
    n_alm : `int`
        Number of output alm, could be computed from size of `cl`

    Returns
    -------
    alm : `np.array`
        Array of the alm values, only the m=0 elements are non-zero
    """
    out_shape = list(cl.shape)
    out_shape[-1] = n_alm
    cl_2d = reshape_array_to_2d(cl)

    return vectorize_hp_func(cl_to_alm_no_phi, cl_2d, n_alm, out_shape=out_shape, **kwargs)


def vector_synalm_from_cl(cl, n_alm, **kwargs):
    """Generate synthetic alm coefficients from cl coefficients,

    Parameters
    ----------
    cl : `np.array`
        cl coefficients

    n_alm : `int`
        Number of output alm, could be computed from size of `cl`

    Returns
    -------
    alm : `np.array`
        Array of the generate alm values
    """
    out_shape = list(cl.shape)
    out_shape[-1] = n_alm
    cl_2d = reshape_array_to_2d(cl)
    return vectorize_hp_func(hp.sphtfunc.synalm, cl_2d, out_shape=out_shape, **kwargs)


def vector_synmap_from_cl(cl, nside, **kwargs):
    """Generate synthetic overdensity maps from cl coefficients,

    Parameters
    ----------
    cl : `np.array`
        cl coefficients

    nside : `int`
        `healpy` nside parameter

    Returns
    -------
    overdenstiy : `np.array`
        Synthetic overdensity maps
    """
    out_shape = list(cl.shape)
    npix = 12 * nside * nside
    out_shape[-1] = npix
    cl_2d = reshape_array_to_2d(cl)
    return vectorize_hp_func(hp.sphtfunc.synfast, cl_2d, nside, out_shape=out_shape, **kwargs)


def vector_generate_counts_from_pdf(pdf, norm, n_trials, **kwargs):
    """Generate synthetic maps from pdf map and a normalization,

    Parameters
    ----------
    pdf : `np.array`
        PDF map

    norm : `float`
        Normalization, total number of expected counts

    n_trials : `int`
        Number of realizations of synthetic data to generate

    Returns
    -------
    counts : `np.array`
        Synthetic counts maps
    """

    expected_counts = norm*pdf
    return vectorize_gen_syn_data(np.random.poisson, expected_counts, n_trials, **kwargs)


def vector_generate_alm_from_cl(cl, n_alm, n_trials, **kwargs):
    """Generate synthetic alm coefficients from cl coefficients,

    Parameters
    ----------
    cl : `np.array`
        cl coefficients

    n_alm : `int`
        Number of output alm, could be computed from size of `cl`

    n_trials : `int`
        Number of realizations of synthetic data to generate

    Returns
    -------
    alm : `np.array`
        Array of the generate alm values
    """

    out_shape = [n_trials] + list(cl.shape)
    out_shape[-1] = n_alm
    return vectorize_gen_syn_data(vector_synalm_from_cl, cl, n_trials, n_alm, out_shape=out_shape, **kwargs)


def vector_generate_overdensity_from_cl(cl, nside, n_trials, **kwargs):
    """Generate synthetic overdensity maps from cl coefficients,

    Parameters
    ----------
    cl : `np.array`
        cl coefficients

    nside : `int`
        `healpy` nside parameter, could be computed from size of cl

    n_trials : `int`
        Number of realizations of synthetic data to generate

    Returns
    -------
    overdenstiy : `np.array`
        Synthetic overdensity maps
    """

    out_shape = [n_trials] + list(cl.shape)
    out_shape[-1] = 12 * nside * nside
    return vectorize_gen_syn_data(vector_synmap_from_cl, cl, n_trials, nside, out_shape=out_shape, **kwargs)


def get_short_long_arrays(array1, array2):
    """Pick the longer and shorter of two arrays and
    make sure that that longer size is a multiple of the shorter size

    This is used to broadcast arrays to the `healpy` analysis functions,
    for example when we want to cross-correlated maps
    at many different energies with a single galaxy map

    Parameters
    ----------
    array1 : `np.ndarray`
        Array of maps or alms
    array2 : `np.ndarray`
        Array of maps or alms

    Returns
    -------
    array_long_2d : `np.ndarray`
        The longer array, reshaped to 2d
    array_short_2d : `np.ndarray`
        The shorter array, reshaped to 2d
    shape_full : `list`
        The shape of the longer array
    nshort : `int`
        The size of the first axis of the shorter array
    """
    na1 = np.sum(array1.shape[0:-1])
    na2 = np.sum(array2.shape[0:-1])

    if na1 >= na2:
        array_long = array1
        array_short = array2
        rem = na1 % na2
    else:
        array_long = array2
        array_short = array1
        rem = na2 % na1
    if rem:
        raise ValueError("Length of longer array not divisible by length of shorter array: %i %i"\
                             % (na1, na2))

    shape_full = list(array_long.shape)
    array_long_2d = reshape_array_to_2d(array_long)
    array_short_2d = reshape_array_to_2d(array_short)
    nshort = array_short_2d.shape[0]

    return array_long_2d, array_short_2d, shape_full, nshort



def vector_cross_correlate_alms(alms1, alms2, ncl, **kwargs):
    """Cross correlate two sets for alms,

    Parameters
    ----------
    alms1 : `np.array`
        alm coefficients

    alms1 : `np.array`
        alm coefficients

    ncl : `int`
        Number of Cl in output, could be computed from shapes of alms

    Returns
    -------
    cls : `np.array`
        Cross correleation power spectra
    """
    alms_long_2d, alms_short_2d, shape_full, nshort = get_short_long_arrays(alms1, alms2)
    shape_full[-1] = ncl
    cl_list = [hp.sphtfunc.alm2cl(alms_long, alms_short_2d[i % nshort], **kwargs)\
                   for i, alms_long in enumerate(alms_long_2d)]
    return np.vstack(cl_list).reshape(shape_full)


def vector_cross_correlate_maps(maps1, maps2, ncl, **kwargs):
    """Cross correlate two sets for alms,

    Parameters
    ----------
    maps1 : `np.array`
        Maps to cross-correlate

    masp2 : `np.array`
        Maps to cross-correlate

    ncl : `int`
        Number of Cl in output, could be computed from shapes of maps

    Returns
    -------
    cls : `np.array`
        Cross correleation power spectra
    """
    maps_long_2d, maps_short_2d, shape_full, nshort = get_short_long_arrays(maps1, maps2)
    shape_full[-1] = ncl
    cl_list = [hp.sphtfunc.anafast(maps_long, maps_short_2d[i % nshort], **kwargs)\
                   for i, maps_long in enumerate(maps_long_2d)]
    return np.vstack(cl_list).reshape(shape_full)


def vector_apply_mask(maps, mask, copy=True):
    """Apply a mask to a set of maps,

    Parameters
    ----------
    maps : `np.ndarray`
        The input maps
    mask : `np.ndarray`
        The mask
    copy : `bool`
        Returns a copy

    Returns
    -------
    out_maps : `np.ndarray`
        The masked arrays
    """
    orig_shape = maps.shape
    if copy:
        maps_2d = reshape_array_to_2d(maps.copy())
    else:
        maps_2d = reshape_array_to_2d(maps)

    for maps_1d in maps_2d:
        maps_1d[mask] = mask

    return maps_2d.reshape(orig_shape)
