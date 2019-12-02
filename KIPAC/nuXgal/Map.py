"""Defines the `Map` class which implements transformations between
various map types and reciprical space representations"""

import healpy as hp

from . import Defaults

from . import hp_utils

from . import file_utils

from .utilities import CachedArray, Cache


class Map(Cache):
    """Implements transformations between various map types and reciprical space representations"""
    def __init__(self, nmap, **kwargs):
        """C'tor

        Parameters
        ----------
        nmap : `int`
            Number of Maps


        Keywords
        --------
        nside : `int`
            `healpy` nside parameter defining the size of the maps
        ncl : `int`
            Number of cl to store
        remaining keywords passed to `Cache.__init__` and used to

        """
        kwcopy = kwargs.copy()
        self._nmap = nmap
        self._nside = kwcopy.pop('nside', Defaults.NSIDE)
        self._npix = hp.pixelfunc.nside2npix(self._nside)
        self._ncl = kwcopy.pop('ncl', 3 * self._nside)
        self._nalm = int(self._ncl * (self._ncl + 1) / 2)
        self.pdf = CachedArray(self, "_pdf", (self._nmap, self._npix))
        self.counts = CachedArray(self, "_counts", (self._nmap, self._npix))
        self.exposure = CachedArray(self, "_exposure", (self._nmap, self._npix))
        self.intensity = CachedArray(self, self._intensity, (self._nmap, self._npix))
        self.overdensity = CachedArray(self, self._overdensity, (self._nmap, self._npix))
        self.alm = CachedArray(self, self._alm, (self._nmap, self._nalm))
        self.cl = CachedArray(self, self._cl, (self._nmap, self._ncl))
        self.syn_counts = CachedArray(self, self._syn_counts, (None, self._nmap, self._npix))
        self.syn_intensity = CachedArray(self, "_syn_intensity", (None, self._nmap, self._npix))
        self.syn_overdensity = CachedArray(self, self._syn_overdensity, (None, self._nmap, self._npix))
        self.syn_alm = CachedArray(self, "_syn_alm", (None, self._nmap, self._nalm))
        self.syn_cl = CachedArray(self, "_syn_cl", (None, self._nmap, self._ncl))
        Cache.__init__(self, **kwcopy)

    def _intensity(self):
        return hp_utils.vector_intensity_from_counts_and_exposure(self.counts(), self.exposure())

    def _overdensity(self):
        return hp_utils.vector_overdensity_from_intensity(self.intensity())

    def _alm(self):
        return hp_utils.vector_alm_from_overdensity(self.overdensity(), self._nalm)

    def _cl(self):
        use_alm = None
        if self.alm.cached is not None:
            use_alm = self.alm.cached
        elif self.overdensity.cached is not None:
            return hp_utils.vector_cl_from_overdensity(self.overdensity.cached, self._ncl)
        if use_alm is None:
            use_alm = self.alm()
        return hp_utils.vector_cl_from_alm(use_alm, self._ncl)

    def _syn_counts(self, norm, n_trials):
        if self.pdf.cached is not None:
            return hp_utils.vector_generate_counts_from_pdf(self.pdf.cached, norm, n_trials)
        return None

    def _syn_overdensity(self, n_trials, **kwargs):
        return hp_utils.vector_generate_overdensity_from_cl(self.cl(), self._nside, n_trials, **kwargs)


    def cross_correlation(self, other, **kwargs):
        """Compute the cross correlation with another map

        Parameter
        ---------
        other : `Map`
            The other map

        Returns
        -------
        cross : `np.ndarray`
            The array of cross correlation values
        """
        return hp_utils.vector_cross_correlate_alms(self.alm(), other.alm(), self._ncl, **kwargs)


    @classmethod
    def create_from_overdensity_maps(cls, fileformat, nmap=1):
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
        the_map : `Map`
            `Map` object
        """
        maps = file_utils.read_maps_from_fits(fileformat, nmap)
        return cls(nmap, overdensity=maps)


    @classmethod
    def create_from_counts_and_exposure_maps(cls, fileformat_counts, fileformat_exposure, nmap=1):
        """Read a map or series of maps from FITS files

        Parameter
        ---------
        fileformat_counts : `str`
            For a single map it is just the file path,
            for multiple maps, it should include {i} to format the filepath
        fileformat_exposure : `str`
            For a single map it is just the file path,
            for multiple maps, it should include {i} to format the filepath
        nmap : `int`
            Number of maps to read

        Returns
        -------
        the_map : `Map`
            `Map` object
        """
        counts_map = file_utils.read_maps_from_fits(fileformat_counts, nmap)
        exposure_map = file_utils.read_maps_from_fits(fileformat_exposure, nmap)
        if counts_map.shape != exposure_map.shape:
            raise ValueError("Counts and Exposure map shapes do not match %s:%s %s:%s" %
                             (fileformat_counts, fileformat_exposure,
                              str(counts_map.shape), str(exposure_map.shape)))
        return cls(nmap, counts=counts_map, exposure=exposure_map)



    @classmethod
    def create_from_cl(cls, fileformat, nmap=1, ncl=500):
        """Read a cl data from text fils

        Parameter
        ---------
        fileformat : `str`
            For a single map it is just the file path,
            for multiple maps, it should include {i} to format the filepath
        nmap : `int`
            Number of maps to read
        ncl : `int`
            Number of cl to read

        Returns
        -------
        the_map : `Map`
            `Map` object
        """
        cl_data = file_utils.read_cls_from_txt(fileformat, nmap, ncl)
        nmap = cl_data.shape[0]
        return cls(nmap, cl=cl_data, ncl=ncl)
