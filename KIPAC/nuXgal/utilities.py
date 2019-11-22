"""Utility clases for nuXgal"""

import numpy as np

class CachedObject:
    """An object that can be used to cache an object

    The object is stored in `self._cached`
    The object is constructed on the fly using the `self._fget` function if the cache is empty
    The object can be set using the `self.set_value` function
    The cache can be cleared using the `self.clear` function

    """
    def __init__(self, parent, fget, obj_type):
        """Ctor

        Parameters
        ----------
        parent : `Cache`
            Object that is holding this `CachedArray`

        fget : `func`
            Function used to build the array for this `CachedArray`

        obj_type : `type` or `None`
            Expected type for the cached object
        """
        self._parent = parent
        self._obj_type = obj_type
        if isinstance(fget, str):
            self._fget = None
            self._func_name = fget
        else:
            self._fget = fget
            self._func_name = fget.__name__
        self._cached = None

    @property
    def cached(self):
        """Returns the cached value"""
        return self._cached

    def __call__(self, *args, **kwargs):
        """Builds `self._cached` array using `self._fget` if it does not exist
        Passes args and kwargs to `self._fget`

        Returns self._cached
        """
        if self._cached is None:
            if self._fget is None:
                raise ValueError("%s is not implmented and value was not provided" % self._func_name)
            val = self._fget(*args, **kwargs)
            if val is None:
                raise ValueError("%s is returned None" % self._func_name)
            self.set_value(val, clear_parent=False)
        return self._cached

    def set_value(self, val, clear_parent=True):
        """Sets `self._cached`

        Calls self._check_value to insure that the array shape matches expectation

        Parameters
        ----------
        val : `numpy.ndarray`
            Array we are putting in the cache

        clear_parent :
            If True, the Parent cache will be cleared
        """
        if clear_parent:
            self._parent.clear_cache()
        self._check_value(val)
        self._cached = val

    def _check_value(self, val):
        """Check the shape of an input array

        Parameters
        ----------
        val : `object`
            Object to check

        Raises
        ------
        ValueError : if the object doesn't match expectation
        """
        if val is None:
            return
        if self._obj_type is None:
            return
        if not isinstance(val, self._obj_type):
            raise ValueError("CachedObject %s type %s != %s" % (self._func_name,
                                                                type(val),
                                                                self._obj_type))

    def clear(self):
        """Clear the cache"""
        self._cached = None



class CachedArray(CachedObject):
    """An object that can be used to cache a numpy array

    """
    def __init__(self, parent, fget, shape):
        """Ctor

        Parameters
        ----------
        parent : `Cache`
            Object that is holding this `CachedArray`

        fget : `func`
            Function used to build the array for this `CachedArray`

        shape : `tuple`
            Expected shape for the cached array
        """
        self._shape = shape
        CachedObject.__init__(self, parent, fget, np.ndarray)


    def _check_value(self, val):
        """Check the shape of an input array

        Parameters
        ----------
        val : `numpy.ndarray`
            Array to check

        Raises
        ------
        ValueError : if the shape doesn't match expectation
        """
        CachedObject._check_value(self, val)
        if val is None:
            return
        if self._shape is None:
            return
        if len(val.shape) != len(self._shape):
            raise ValueError("CachedArray %s dimension %i != %i" % (self._func_name,
                                                                    len(val.shape),
                                                                    len(self._shape)))
        for i, j in zip(val.shape, self._shape):
            if j is None:
                continue
            if i != j:
                raise ValueError("CachedArray %s shape %s != %s" % (self._func_name,
                                                                    str(val.shape),
                                                                    str(self._shape)))




class Cache:
    """An object that stores `CachedArray` objects
    """
    def __init__(self, **kwargs):
        """C'tor

        Passes keywords to `self.set_cache` to initialize the CachedObject objects"""
        self.set_cache(**kwargs)

    def print_cache_status(self):
        """Iterates through the CachedObject objects in `self.__dict__`
        And prints if their cache is set or not"""
        for key, val in self.__dict__.items():
            if isinstance(val, CachedObject):
                made = bool(val.cached is not None)
                print("  %s : %s" % (key, str(made)))

    def clear_cache(self):
        """Iterates through the CachedObject objects in `self.__dict__`
        and calls CachedArray.clear on each"""
        for val in self.__dict__.values():
            if isinstance(val, CachedObject):
                val.clear()

    def set_cache(self, **kwargs):
        """Iterates throught the keywords and calls `CachedObject.set_value` on
        each key, value pair"""
        self.clear_cache()
        for key, val in kwargs.items():
            cobj = getattr(self, key)
            cobj.set_value(val, clear_parent=False)
