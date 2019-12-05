"""Functions to help make plots and collect figures"""

import healpy as hp

import numpy as np

import matplotlib.pyplot as plt



FONT = {'family':'Arial', 'weight':'normal', 'size': 20}
LEGENDFONT = {'fontsize':18, 'frameon':False}


class FigureDict:
    """Object to make and collect figures.

    This is implemented as a dictionary of dictionaries,

    Each value is a dictionary with `matplotlib` objects for each figure
    """
    def __init__(self):
        """C'tor"""
        self._fig_dict = {}

    def add_figure(self, key, fig):
        """Added a figure

        Parameters
        ----------
        key : `str`
            Name of the figure
        fig : `matplotlib.figure.Figure`
            The figure we are adding
        """
        self._fig_dict[key] = dict(fig=fig)

    def get_figure(self, key):
        """Return a `Figure` by name"""
        return self._fig_dict[key]['fig']

    def keys(self):
        """Return the names of the figures"""
        return self._fig_dict.keys()

    def values(self):
        """Returns the sub-dictionary of `matplotlib` objects"""
        return self._fig_dict.values()

    def items(self):
        """Return the name : sub-dictionary pairs"""
        return self._fig_dict.items()

    def __getitem__(self, key):
        """Return a particular sub-dictionary by name"""
        return self._fig_dict[key]

    def get_obj(self, key, key2):
        """Return some other `matplotlib` object besides a `Figure`

        Parameters
        ----------
        key : `str`
            Key for the figure
        key2 : `str`
            Key for the object

        Returns
        -------
        retval : `object`
            Requested object
        """
        return self._fig_dict[key][key2]

    def setup_figure(self, key, **kwargs):
        """Set up a figure with requested labeling

        Parameters
        ----------
        key : str
            Key for the figure.

        Keywords
        --------
        title : `str`
            Figure title
        xlabel : `str`
            X-axis label
        ylabel : `str`
            Y-axis label
        figsize : `tuple`
            Figure width, height in inches

        Returns
        -------
        fig : `Figure`
            The newly created `Figure`
        axes : `AxesSubplot`
            The axes objects
        """
        if key in self._fig_dict:
            return self._fig_dict[key]

        title = kwargs.get('title', None)
        xlabel = kwargs.get('xlabel', None)
        ylabel = kwargs.get('ylabel', None)
        figsize = kwargs.get('figsize', (15, 10))

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        plt.rc('font', **FONT)
        plt.rc('legend', **LEGENDFONT)

        if title is not None:
            fig.suptitle(title)
        if xlabel is not None:
            axes.set_xlabel(xlabel)
        if ylabel is not None:
            axes.set_ylabel(ylabel)

        o_dict = dict(fig=fig, axes=axes)
        self._fig_dict[key] = o_dict
        return o_dict


    def savefig(self, key, filename):
        """Save a single figure

        Parameters
        ----------
        key : `str`
            Key for the figure.
        filename : `str`
            Name of the output file
        """
        fig = self._fig_dict[key]['fig']
        fig.savefig(filename)
        plt.close(fig)

    def save_all(self, basename, ftype='png'):
        """Save all the figures

        The files will be named {basename}_{key}.png

        If basename is None then the file will be shown on the display and not saved

        Parameters
        ----------
        basename : `str` or `None`
            Base of the output file names
        ftype : `str`
            File type to same, also filename extension
        """
        if basename is None:
            plt.ion()
            plt.show()
            return

        for key, val in self._fig_dict.items():
            fig = val['fig']
            fig.savefig("%s_%s.%s" % (basename, key, ftype))
            plt.close(fig)

    def mollview(self, key, data, **kwargs):
        """Make a Mollweide projection of a single `healpy` map

        Parameters
        ----------
        key : `str`
            Key for the figure.
        data : `np.ndarray`
            The map

        Keywords
        --------
        figsize : `tuple`
            The figure size (in inches)

        Returns
        -------
        fig : `Figure`
            The newly created `Figure`
        """
        figsize = kwargs.get('figsize', (8, 6))
        fig = plt.figure(figsize=figsize)
        hp.mollview(data, fig=fig.number)
        o_dict = dict(fig=fig)
        self._fig_dict[key] = o_dict
        return o_dict


    def mollview_maps(self, basekey, data, **kwargs):
        """Make a series of Mollweide projections of `healpy` maps

        Parameters
        ----------
        basekey : `str`
            Key for the figure
        data : `np.ndarray`
            The map

        Keywords
        --------
        figsize : `tuple`
            The figure size (in inches)
        """
        for i, _data in enumerate(data):
            self.mollview("%s%i" % (basekey, i), _data, **kwargs)


    def plot(self, key, xvals, yvals, **kwargs):
        """Make a simple plot

        Parameters
        ----------
        key : `str`
            Key for the figure
        xvals : `np.ndarray`
            The x values
        yvals_array : `np.ndarray`
            The y values
        """
        kwcopy = kwargs.copy()
        kwsetup = {}
        for kw in ['title', 'xlabel', 'ylabel', 'figsize']:
            if kw in kwcopy:
                kwsetup[kw] = kwcopy.pop(kw)
        o_dict = self.setup_figure(key, **kwsetup)
        axes = o_dict['axes']
        axes.plot(xvals, yvals, **kwcopy)
        return o_dict

    def plot_yvals(self, key, xvals, yvals_list, **kwargs):
        """Make a simple plot

        Parameters
        ----------
        key : `str`
            Key for the figure
        xvals : `np.ndarray`
            The x values
        yvals_list : `np.ndarray` or `list`
            The y values
        """
        kwcopy = kwargs.copy()
        kwsetup = {}
        for kw in ['title', 'xlabel', 'ylabel', 'figsize']:
            if kw in kwcopy:
                kwsetup[kw] = kwcopy.pop(kw)
        o_dict = self.setup_figure(key, **kwsetup)        
        fig = o_dict['fig']
        axes = o_dict['axes']
        for i, yvals in enumerate(yvals_list):
            axes.plot(xvals, yvals, label=str(i), **kwcopy)
        o_dict['leg'] = fig.legend()
        return o_dict

    def plot_xyvals(self, key, xvals_list, yvals_list, **kwargs):
        """Make a simple plot

        Parameters
        ----------
        key : `str`
            Key for the figure
        xvals_list : `np.ndarray` or `list`
            The x values
        yvals_list : `np.ndarray` or `list`
            The y values
        """
        kwcopy = kwargs.copy()
        kwsetup = {}
        for kw in ['title', 'xlabel', 'ylabel', 'figsize']:
            if kw in kwcopy:
                kwsetup[kw] = kwcopy.pop(kw)
        o_dict = self.setup_figure(key, **kwsetup)        
        labels = kwcopy.pop('labels', None)
        fig = o_dict['fig']
        axes = o_dict['axes']
        for i, (xvals, yvals) in enumerate(zip(xvals_list, yvals_list)):
            if labels is None:
                label = str(i)
            else:
                label = labels[i]
            axes.scatter(xvals, yvals, label=label, **kwcopy)
        o_dict['leg'] = fig.legend()
        return o_dict


    def plot_cl(self, key, xvals, cl_data, **kwargs):
        """Plot a series of power spectra

        Parameters
        ----------
        key : `str`
            Key for the figure
        xvals : `np.ndarray`
            The l values
        cl_data : `np.ndarray`
            The power spectrum values

        Keywords
        --------
        ymin : `float`
            Min for y-axis
        ymax : `float`
            Max for y-axis
        yerr : `list`
            Means and stds to plot as the y-errors
        colors : `list`
            Colors to use for the plots

        Returns
        -------
        fig : `Figure`
            The newly created `Figure`
        axes : `AxesSubplot`
            The axes objects
        """
        kwcopy = kwargs.copy()
        colors = kwcopy.pop('colors', None)
        labels = kwcopy.pop('labels', None)
        ymin = kwcopy.pop('ymin', 1e-8)
        ymax = kwcopy.pop('ymax', 10.)
        yerr = kwcopy.pop('yerr', None)
        band_1sig = kwcopy.pop('band_1sig', None)
        band_2sig = kwcopy.pop('band_2sig', None)
        lw = kwcopy.pop('lw', None)

        o_dict = self.setup_figure(key, **kwcopy)

        fig = o_dict['fig']
        axes = o_dict['axes']

        axes.set_xscale('log')
        if ymin >= 0:
            axes.set_yscale('log')
        axes.set_ylim(ymin, ymax)

        do_errs = bool(yerr is not None)

        c_dict = {}
        for i, _cl_data in enumerate(cl_data):
            if labels is None:
                c_dict['label'] = str(i)
            else:
                c_dict['label'] = labels[i]
            if colors is not None:
                c_dict['color'] = colors[i]
            if band_2sig is not None:
                axes.fill_between(xvals, band_2sig[i][0],  band_2sig[i][1])

            if band_1sig is not None:
                axes.fill_between(xvals, band_1sig[i][0],  band_1sig[i][1])
            axes.plot(xvals, _cl_data.clip(ymin, ymax), lw=lw, **c_dict)
            if do_errs:
                axes.errorbar(xvals, _cl_data, yerr=yerr[i], **c_dict)
            else:
                axes.plot(xvals, _cl_data, **c_dict)

        #o_dict['leg'] = fig.legend(ncol=2)
        fig.subplots_adjust(left=0.18, top=0.9, right=0.9)
        return o_dict


    def plot_w_cross_norm(self, key, xvals, data, **kwargs):
        """Plot a series of power spectra

        Parameters
        ----------
        key : `str`
            Key for the figure
        xvals : `np.ndarray`
            The l values
        data : `np.ndarray`
            The cross-correlations valeus

        Keywords
        --------
        ymin : `float`
            Min for y-axis
        ymax : `float`
            Max for y-axis
        yerr : `list`
            Means and stds to plot as the y-errors
        colors : `list`
            Colors to use for the plots

        Returns
        -------
        fig : `Figure`
            The newly created `Figure`
        axes : `AxesSubplot`
            The axes objects
        """
        kwcopy = kwargs.copy()
        colors = kwcopy.pop('colors', None)
        labels = kwcopy.pop('labels', None)
        ymin = kwcopy.pop('ymin', -5)
        ymax = kwcopy.pop('ymax', 5.)
        yerr = kwcopy.pop('yerr', None)

        o_dict = self.setup_figure(key, **kwcopy)

        fig = o_dict['fig']
        axes = o_dict['axes']

        axes.set_xscale('log')
        axes.set_ylim(ymin, ymax)

        do_errs = bool(yerr is not None)

        c_dict = {}
        for i, _data in enumerate(data):
            if labels is None:
                c_dict['label'] = str(i)
            else:
                c_dict['label'] = labels[i]
            if colors is not None:
                c_dict['color'] = colors[i]
            axes.plot(xvals, _data.clip(ymin, ymax), **c_dict)
            if do_errs:
                axes.errorbar(xvals, yerr[0][i], yerr=yerr[1][i], color='grey')

        o_dict['leg'] = fig.legend(ncol=2)
        fig.subplots_adjust(left=0.18, top=0.9, right=0.9)
        return o_dict


    def plot_intesity_E2(self, key, eVals, intensities, **kwargs):
        """Plot a series of intensities in E**2

        Parameters
        ----------
        key : `str`
            Key for the figure
        eVals : `np.ndarray`
            The energy values
        intensities : `np.ndarray`
            The Intensity values

        Keywords
        --------
        colors : `list`
            Colors to use for the plots
        markers : `list`
            Colors to use for the plots

        Returns
        -------
        fig : `Figure`
            The newly created `Figure`
        axes : `AxesSubplot`
            The axes objects
        """

        kwcopy = kwargs.copy()
        markers = kwcopy.pop('markers', None)
        colors = kwcopy.pop('colors', None)
        o_dict = self.setup_figure(key, **kwcopy)


        fig = o_dict['fig']
        axes = o_dict['axes']
        axes.set_xscale('log')
        axes.set_yscale('log')
        axes.set_xlabel('E [GeV]')
        axes.set_ylabel(r'$E^2 \Phi\,[GeV\,cm^{-2}\,s^{-1}\,sr^{-1}]$')
        axes.set_xlim(1e2, 1e8)
        axes.set_ylim(1e-9, 1e-2)

        eVals_sq = eVals*eVals

        c_dict = {}
        for i, intensity in enumerate(intensities):
            if markers is not None:
                c_dict['marker'] = markers[i]
            if colors is not None:
                c_dict['c'] = colors[i]

            axes.scatter(eVals, intensity * eVals_sq, **c_dict)

        fig.subplots_adjust(left=0.18, bottom=0.2, right=0.9)
        return o_dict


    def plot_hists(self, key, bins, vals_list, **kwargs):
        """Make a simple plot

        Parameters
        ----------
        key : `str`
            Key for the figure
        bins : `np.ndarray`
            The x edges
        vals_list : `np.ndarray` or `list`
            The values to histogram
        """
        kwcopy = kwargs.copy()
        kwsetup = {}
        for kw in ['title', 'xlabel', 'ylabel', 'figsize']:
            if kw in kwcopy:
                kwsetup[kw] = kwcopy.pop(kw)
        o_dict = self.setup_figure(key, **kwsetup)
        labels = kwcopy.pop('labels', None)
   
        fig = o_dict['fig']
        axes = o_dict['axes']
        for i, vals in enumerate(vals_list):
            if labels is None:
                label = str(i)
            else:
                label = labels[i]
            axes.hist(vals.flat, bins, label=label, **kwcopy)
        o_dict['leg'] = fig.legend()
        return o_dict



    def plot_hist_verus_l(self, key, bins, l_array, data, **kwargs):
        """Make a simple plot

        Parameters
        ----------
        key : `str`
            Key for the figure
        bins : `np.ndarray`
            The x and y bin edges
        l_values : `np.ndarray`
            An array of the l_values for the data
        data : `np.ndarray` or `list`
            The values to histogram as a function on l
        """
        kwcopy = kwargs.copy()
        kwsetup = {}
        for kw in ['title', 'xlabel', 'ylabel', 'figsize']:
            if kw in kwcopy:
                kwsetup[kw] = kwcopy.pop(kw)
        o_dict = self.setup_figure(key, **kwsetup)
        labels = kwcopy.pop('labels', None)
   
        fig = o_dict['fig']
        axes = o_dict['axes']

        hist_2d = np.histogram2d(l_array.flatten(), data.T.flatten(), bins=bins)
        extent = (bins[0][0], bins[0][-1], bins[1][0], bins[1][-1])
        
        img = axes.imshow(hist_2d[0].T, extent=extent, aspect='auto', origin='lower')
        cbar = plt.colorbar(img)
        
        o_dict['img'] = img
        o_dict['cbar'] = cbar
        return o_dict

