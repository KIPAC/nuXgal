"""Functions from fermipy/castro.py and fermipy/utils.py --- CHECK LICENSE ---
"""
import numpy as np
import scipy
import scipy.special as special


def onesided_cl_to_dlnl(cl):
    """Compute the delta-loglikehood values that corresponds to an
    upper limit of the given confidence level.

    Parameters
    ----------
    cl : float
        Confidence level.

    Returns
    -------
    dlnl : float
        Delta-loglikelihood value with respect to the maximum of the
        likelihood function.
    """
    alpha = 1.0 - cl
    return 0.5 * np.power(np.sqrt(2.) * special.erfinv(1 - 2 * alpha), 2.)

def twosided_cl_to_dlnl(cl):
    """Compute the delta-loglikehood value that corresponds to a
    two-sided interval of the given confidence level.

    Parameters
    ----------
    cl : float
        Confidence level.

    Returns
    -------
    dlnl : float
        Delta-loglikelihood value with respect to the maximum of the
        likelihood function.
    """
    return 0.5 * np.power(np.sqrt(2.) * special.erfinv(cl), 2)


class Interpolator(object):
    """ Helper class for interpolating a 1-D function from a
    set of tabulated values.

    Safely deals with overflows and underflows
    """

    def __init__(self, x, y):
        """ C'tor, take input array of x and y value
        """
        from scipy.interpolate import UnivariateSpline, splrep

        x = np.squeeze(np.array(x, ndmin=1))
        y = np.squeeze(np.array(y, ndmin=1))

        msk = np.isfinite(y)
        x = x[msk]
        y = y[msk]

        if len(x) == 0 or len(y) == 0:
            raise ValueError("Failed to build interpolate, empty axis.")

        self._x = x
        self._y = y
        self._xmin = x[0]
        self._xmax = x[-1]
        self._ymin = y[0]
        self._ymax = y[-1]
        self._dydx_lo = (y[1] - y[0]) / (x[1] - x[0])
        self._dydx_hi = (y[-1] - y[-2]) / (x[-1] - x[-2])

        self._fn = UnivariateSpline(x, y, s=0, k=1)
        self._sp = splrep(x, y, k=1, s=0)

    @property
    def xmin(self):
        """ return the minimum value over which the spline is defined
        """
        return self._xmin

    @property
    def xmax(self):
        """ return the maximum value over which the spline is defined
        """
        return self._xmax

    @property
    def x(self):
        """ return the x values used to construct the split
        """
        return self._x

    @property
    def y(self):
        """ return the y values used to construct the split
        """
        return self._y

    def derivative(self, x, der=1):
        """ return the derivative a an array of input values

        x   : the inputs
        der : the order of derivative
        """
        from scipy.interpolate import splev
        return splev(x, self._sp, der=der)

    def __call__(self, x):
        """ Return the interpolated values for an array of inputs

        x : the inputs

        Note that if any x value is outside the interpolation ranges
        this will return a linear extrapolation based on the slope
        at the endpoint
        """
        x = np.array(x, ndmin=1)

        below_bounds = x < self._xmin
        above_bounds = x > self._xmax

        dxhi = np.array(x - self._xmax)
        dxlo = np.array(x - self._xmin)

        # UnivariateSpline will only accept 1-D arrays so this
        # passes a flattened version of the array.
        y = self._fn(x.ravel())
        y.resize(x.shape)

        y[above_bounds] = (self._ymax + dxhi[above_bounds] * self._dydx_hi)
        y[below_bounds] = (self._ymin + dxlo[below_bounds] * self._dydx_lo)
        return y


class LnLFn(object):
    """Helper class for interpolating a 1-D log-likelihood function from a
    set of tabulated values.
    """

    def __init__(self, x, y, norm_type=0):
        """C'tor, takes input arrays of x and y values

        Parameters
        ----------
        x : array-like
           Set of values of the free parameter

        y : array-like
           Set of values for the _negative_ log-likelhood

        norm_type :  str
           String specifying the type of quantity used for the `x`
           parameter.

        Notes
        -----
        Note that class takes and returns the _negative log-likelihood
        as fitters typically minimize rather than maximize.

        """
        self._interp = Interpolator(x, y)
        self._mle = None
        self._norm_type = norm_type

    @property
    def interp(self):
        """ return the underlying Interpolator object
        """
        return self._interp

    @property
    def norm_type(self):
        """Return a string specifying the quantity used for the normalization.
        This isn't actually used in this class, but it is carried so
        that the class is self-describing.  The possible values are
        open-ended.
        """
        return self._norm_type

    def _compute_mle(self):
        """Compute the maximum likelihood estimate.

        Calls `scipy.optimize.brentq` to find the roots of the derivative.
        """
        min_y = np.min(self._interp.y)
        if self._interp.y[0] == min_y:
            self._mle = self._interp.x[0]
        elif self._interp.y[-1] == min_y:
            self._mle = self._interp.x[-1]
        else:
            argmin_y = np.argmin(self._interp.y)
            ix0 = max(argmin_y - 4, 0)
            ix1 = min(argmin_y + 4, len(self._interp.x) - 1)

            while np.sign(self._interp.derivative(self._interp.x[ix0])) == \
                    np.sign(self._interp.derivative(self._interp.x[ix1])):
                ix0 += 1

            self._mle = scipy.optimize.brentq(self._interp.derivative,
                                              self._interp.x[ix0],
                                              self._interp.x[ix1],
                                              xtol=1e-10 *
                                              np.median(self._interp.x))

    def mle(self):
        """ return the maximum likelihood estimate

        This will return the cached value, if it exists
        """
        if self._mle is None:
            self._compute_mle()
        return self._mle

    def fn_mle(self):
        """ return the function value at the maximum likelihood estimate """
        return self._interp(self.mle())

    def TS(self):
        """ return the Test Statistic """
        return 2. * (self._interp(0.) - self._interp(self.mle()))

    def getDeltaLogLike(self, dlnl, upper=True):
        """Find the point at which the log-likelihood changes by a
        given value with respect to its value at the MLE."""
        mle_val = self.mle()
        # A little bit of paranoia to avoid zeros
        if mle_val <= 0.:
            mle_val = self._interp.xmin
        if mle_val <= 0.:
            mle_val = self._interp.x[1]
        log_mle = np.log10(mle_val)
        lnl_max = self.fn_mle()

        if upper:
            x = np.logspace(log_mle, np.log10(self._interp.xmax), 100)
            retVal = np.interp(dlnl, self.interp(x) - lnl_max, x)
        else:
            x = np.linspace(self._interp.xmin, self._mle, 100)
            retVal = np.interp(dlnl, self.interp(x)[::-1] - lnl_max, x[::-1])

        return retVal

    def getLimit(self, alpha, upper=True):
        """ Evaluate the limits corresponding to a C.L. of (1-alpha)%.

        Parameters
        ----------
        alpha :  limit confidence level.
        upper :  upper or lower limits.
        """
        dlnl = onesided_cl_to_dlnl(1.0 - alpha)
        return self.getDeltaLogLike(dlnl, upper=upper)

    def getInterval(self, alpha):
        """ Evaluate the interval corresponding to a C.L. of (1-alpha)%.

        Parameters
        ----------
        alpha : limit confidence level.
        """
        dlnl = twosided_cl_to_dlnl(1.0 - alpha)
        lo_lim = self.getDeltaLogLike(dlnl, upper=False)
        hi_lim = self.getDeltaLogLike(dlnl, upper=True)
        return (lo_lim, hi_lim)
