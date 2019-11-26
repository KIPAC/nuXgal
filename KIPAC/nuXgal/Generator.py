"""Helper classes to generate synthetic events"""

import numpy as np

from scipy.interpolate import interp1d

import healpy as hp

from . import Defaults

from . import hp_utils

from .utilities import CachedObject, CachedArray, Cache


def build_cdf(pdf_vals, grid_vals, dgrid_vals):
    """Construct an interpolator that gives the values of a cumulative distribution function

    This is used to generate events according to a particular distribution

    Parameters
    ----------
    pdf_vals : `np.darray`
        2D Array of x and y values
    grid_vals : `np.ndarray`
        Grid used to compute the cdf
    dgrid_vals : `np.ndarray`
        Deltas of the grid used to compute the cdf

    Returns
    -------
    cdf : `scipy.interp1d`
        Functor that implements the inverse of the cdf
    """
    pdf_interp = interp1d(pdf_vals[:, 0], pdf_vals[:, 1], bounds_error=False, fill_value=0.)
    interp_vals = pdf_interp(grid_vals)
    trap_vals = (interp_vals[0:-1] + interp_vals[1:])*dgrid_vals
    cdf = np.cumsum(trap_vals)
    if cdf[-1] > 0:
        cdf /= cdf[-1]
    #index_non_zero = np.where(cdf == 0)[0][-1]
    #index_non_one = np.where(1 - cdf < 8e-5)[0][0] # to avoid peak in energy bin 3
    return interp1d(cdf,
                    grid_vals[1:],
                    bounds_error=False, fill_value="extrapolate")


def generate_coszenith(cosz_cdf, nevents):
    """Generate sets of events in cos(zentith)

    Note that the inputs should be iterables of the same length

    Parameters
    ----------
    cosz_cdf : `list`
        List of functors to generate events in cos(zenith)
    nevents : `np.dnarray`
        Numbers of events to geneate with each distribution

    Returns
    -------
    cos_zenith : `list`
        List of arrays of cos(zenith) values
    """
    return [_cosz_cdf(np.random.rand(_nevents)) for _cosz_cdf, _nevents in zip(cosz_cdf, nevents)]


def generate_phi(nevents):
    """Generate sets of events in phi

    Note that the inputs should be an iterable

    Parameters
    ----------
    nevents : `np.dnarray`
        Numbers of events to geneate with each distribution

    Returns
    -------
    phi : `list`
        List of arrays of phi values
    """
    return [np.random.rand(_nevents) * 2 * np.pi for _nevents in nevents]


def generate_hpmaps(cosz_cdf, nevents, nside):
    """Generate sets of `healpy` maps

    Note that the inputs should be an iterable

    Parameters
    ----------
    cosz_cdf : `list`
        List of functors to generate events in cos(zenith)
    nevents : `np.dnarray`
        Numbers of events to geneate with each distribution
    nsize : `int`
        `healpy` nside parameter for output maps

    Returns
    -------
    maps : `np.ndarray`
        `healpy` maps
    """

    coszenith = generate_coszenith(cosz_cdf, nevents)
    phi = generate_phi(nevents)
    npix = 12*nside*nside
    indexPixel = [hp.pixelfunc.ang2pix(nside, np.arccos(_coszenith), _phi)
                  for _coszenith, _phi in zip(coszenith, phi)]
    return np.array([np.bincount(_indexPixel, minlength=npix) for _indexPixel in indexPixel])


class AtmGenerator(Cache):
    """Class to generate atmospheric events

    The events are randomly distributed in phi and follow input distributions in cos(zenith)
    """
    def __init__(self, nmap, **kwargs):
        """C'tor

        Parameters
        ----------
        nmap : `int`
            Number of maps (i.e., energy bins) to consider
        """
        kwcopy = kwargs.copy()
        self._nmap = nmap
        self._nside = kwcopy.pop('nside', Defaults.NSIDE)
        self._npix = hp.pixelfunc.nside2npix(self._nside)

        self.nevents_expected = CachedArray(self, "_nevents", [self._nmap])
        self.coszenith = CachedArray(self, "_coszenith", (self._nmap, None, 2))
        self.cosz_cdf = CachedObject(self, self._cosz_cdf, list)
        Cache.__init__(self, **kwcopy)


    def _cosz_cdf(self):
        N_coszenith = self.coszenith()
        grid_cdf = np.linspace(-1, 1, 300)
        dgrid_cdf = grid_cdf[1:] - grid_cdf[0:-1]
        cdf_list = [build_cdf(N_coszenith[i], grid_cdf, dgrid_cdf) for i in range(self._nmap)]
        return cdf_list


    def generate_event_maps(self, nTrials):
        """Generate a set of `healpy` maps

        Parameters
        ----------
        nTrials : `int`
            Number of trials to generay

        Returns
        -------
        maps : `np.ndarray`
            An array of (nTrials x self._nmap) synthetic maps
        """
        maps = [generate_hpmaps(self.cosz_cdf(),
                                np.random.poisson(self.nevents_expected()),
                                self._nside) for _ in range(nTrials)]
        return np.vstack(maps).reshape((nTrials, self._nmap, self._npix))




class AstroGenerator(Cache):
    """Class to generate astrophysical events

    The events are generated follow a pdf, and then down-selected based on the effective area
    """
    def __init__(self, nmap, **kwargs):
        """C'tor

        Parameters
        ----------
        nmap : `int`
            Number of maps (i.e., energy bins) to consider
        """
        kwcopy = kwargs.copy()
        self._nmap = nmap
        self._nside = kwcopy.pop('nside', Defaults.NSIDE)
        self._npix = hp.pixelfunc.nside2npix(self._nside)

        self.nevents_expected = CachedArray(self, "_nevents", [self._nmap])
        self.pdf = CachedArray(self, "_pdf", [self._npix])
        self.aeff = CachedArray(self, "_aeff", [self._nmap, self._npix])
        self.prob_reject = CachedArray(self, self._prob_reject, [self._nmap, self._npix])
        self.mean_reject = CachedArray(self, self._mean_reject, [self._nmap])
        Cache.__init__(self, **kwcopy)


    def _prob_reject(self):
        aeff_data = self.aeff()
        max_aeff = aeff_data.max(1)
        return (aeff_data.T / max_aeff).T

    def _mean_reject(self):
        return self.prob_reject().mean(1)

    def generate_event_maps(self, n_trials, **kwargs):
        """Generate a set of `healpy` maps

        Parameters
        ----------
        nTrials : `int`
            Number of trials to generay

        Returns
        -------
        maps : `np.ndarray`
            An array of (nTrials x self._nmap) synthetic maps
        """
        pdf_expand = np.expand_dims(self.pdf(), 0)
        nevents_expand = np.expand_dims(self.nevents_expected()/self.mean_reject(), -1)

        full_pdf = pdf_expand * nevents_expand
        full_pdf *= self.prob_reject()
        return hp_utils.vectorize_gen_syn_data(np.random.poisson,
                                               full_pdf.clip(0, np.inf), n_trials, **kwargs)





class AstroGenerator_v2(Cache):
    """Class to generate astrophysical events

    The events are generated follow a pdf, and then down-selected based on the effective area
    """
    def __init__(self, nmap, f_gal, **kwargs):
        """C'tor

        Parameters
        ----------
        nmap : `int`
            Number of maps (i.e., energy bins) to consider
        """
        kwcopy = kwargs.copy()
        self._nmap = nmap
        self._nside = kwcopy.pop('nside', Defaults.NSIDE)
        self._npix = hp.pixelfunc.nside2npix(self._nside)
        self._ncl = kwcopy.pop('ncl', Defaults.NCL_galaxyInput)
        self.f_gal = f_gal
        self.cl = CachedArray(self, "_cl", [1, self._ncl])
        self.nevents_expected = CachedArray(self, "_nevents", [self._nmap])
        self.aeff = CachedArray(self, "_aeff", [self._nmap, self._npix])
        self.prob_reject = CachedArray(self, self._prob_reject, [self._nmap, self._npix])
        self.mean_reject = CachedArray(self, self._mean_reject, [self._nmap])
        Cache.__init__(self, **kwcopy)


    def _prob_reject(self):
        aeff_data = self.aeff()
        max_aeff = aeff_data.max(1)
        return (aeff_data.T / max_aeff).T

    def _mean_reject(self):
        return self.prob_reject().mean(1)

    def generate_event_maps(self, n_trials, **kwargs):
        """Generate a set of `healpy` maps

        Parameters
        ----------
        nTrials : `int`
            Number of trials to generay

        Returns
        -------
        maps : `np.ndarray`
            An array of (nTrials x self._nmap) synthetic maps
        """
        #pdf_expand = np.expand_dims(self.pdf(), 0)
        #nevents_expand = np.expand_dims(self.nevents_expected()/self.mean_reject(), -1)        
        #nevents_expand = np.expand_dims(self.nevents_expected(), -1)

        syn_overdensities = hp_utils.vector_generate_overdensity_from_cl(self.f_gal*self.cl(), self._nside, n_trials)
        
        event_map_list = []
        for i in range(n_trials):
            normalized_counts_map = np.exp(syn_overdensities[i])
            normalized_counts_map /= normalized_counts_map.sum()
            for j in range(self._nmap):
                expected_counts_map = self.prob_reject()[j] * normalized_counts_map * self.nevents_expected()[j]
                observed_counts_map = np.random.poisson(expected_counts_map)
                event_map_list.append(observed_counts_map)

        return np.vstack(event_map_list).reshape((n_trials, self._nmap, self._npix))
