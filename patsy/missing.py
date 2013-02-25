# This file is part of Patsy
# Copyright (C) 2013 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

# Missing data detection/handling

# First, how do we represent missing data? (i.e., which values count as
# "missing"?) In the long run, we want to use numpy's NA support... but that
# doesn't exist yet. Until then, people use various sorts of ad-hoc
# things. Some things that might be considered NA:
#   NA (eventually)
#   NaN  (in float or object arrays)
#   None (in object arrays)
# Compatibility consideration: pandas 'isnull' treats NaN and None as
# missing.
#
# However, None (and object arrays in general) can only occur for categorical
# data, and our Categorical class has its own missing-data handling. So here
# we only have to worry about Categorical missing data, and NaNs in numeric
# arrays.

# Next, what should be done once we find missing data? R's options:
#   -- throw away those rows (from all aligned matrices)
#      -- with or without preserving information on which rows were discarded
#   -- error out
#   -- carry on
# The 'carry on' option requires that we have some way to represent NA in our
# output array. To avoid further solidifying the use of NaN for this purpose,
# we'll leave this option out for now, until real NA support is
# available. Also, we always preserve information on which rows were
# discarded, using the pandas index functionality (currently this is only
# returned to the original caller if they used return_type="dataframe",
# though).

# All of this behaviour is encapsulated by a function that just takes a
# set of evaluated factors and returns a new set of factors (with the new set
# being what will actually be used).

import numpy as np
from patsy import PatsyError
from patsy.util import safe_isnan
from patsy.categorical import Categorical

# These are made available in the patsy.* namespace
__all__ = ["NAAction"]

import threading
current_NA_action = threading.local()
current_NA_action.value = None

_valid_NA_types = ["None", "NaN", "numpy.ma"]
_valid_NA_responses = ["raise", "drop"]
def _desc_options(options):
    return ", ".join([repr(opt) for opt in options])

class NAAction(object):
    """An NAAction object defines a strategy for handling missing data.

    "NA" is short for "Not Available", and is used to refer to any value which
    is somehow unmeasured or unavailable. In the long run, it is devoutly
    hoped that numpy will gain first-class missing value support. Until then,
    we work around this lack as best we're able.

    There are two parts to this: First, we have to determine what counts as
    missing data. For categorical factors, Patsy's :class:`Categorical` does
    its own auto-detection of missing values: if you have a vector of
    categorical data, then :meth:`Categorical.from_sequence` will treat as
    missing any values which are the Python object `None`, a masked entry in a
    numpy masked array, or a floating point object which has the value 'not a
    number' (also known as NaN, and available as `numpy.nan`). For numerical
    factors, by default we treat NaN values a missing.

    Second, we have to decide what to do with any missing data when we
    encounter it. One option is to simply discard any rows which contain
    missing data from our design matrices (`drop`). Another option is to raise
    an error (`raise`). A third option would be to simply let the missing
    values pass through into the returned design matrices. However, this last
    option is not yet implemented, because of the lack of any standard way to
    represent missing values in arbitrary numpy matrices; we're hoping numpy
    will get this sorted out before we standardize on anything ourselves.

    You can control how patsy handles missing data through the `NA_action=`
    argument to functions like :func:`build_design_matrices` and
    :func:`dmatrix`. If all you want to do is to choose between `drop` and
    `raise` behaviour, you can pass one of those strings as the `NA_action=`
    argument directly. If you want more fine-grained control over how missing
    values are detected and handled, then you can create an instance of this
    class, or your own object that implements :meth:`handle_NA`, and pass that
    as the `NA_action=` argument instead.
    """
    def __init__(self, on_NA="drop", NA_types=["None", "NaN", "numpy.ma"]):
        """The `NAAction` constructor takes the following arguments:
        
        :arg on_NA: How to handle missing values. The default is "drop", which
          removes all rows from all matrices which contain any missing
          values. Also available is "raise", which raises an exception when
          any missing values are encountered.
        :arg NA_types: Which values count as missing, as a list of
          strings. 
        """
        self.on_NA = on_NA
        if self.on_NA not in _valid_NA_responses:
            raise ValueError("invalid on_NA action %r "
                             "(should be one of %s)"
                             % (on_NA, _desc_options(_valid_NA_responses)))
        if isinstance(NA_types, basestring):
            raise ValueError("NA_types should be a list of strings")
        self.NA_types = tuple(NA_types)
        for NA_type in self.NA_types:
            if NA_type not in _valid_NA_types:
                raise ValueError("invalid NA_type %r "
                                 "(should be one of %s)"
                                 % (NA_type, _desc_options(_valid_NA_types)))

    def is_NA(self, arr):
        if isinstance(arr, Categorical):
            return (arr.int_array == -1)
        else:
            mask = np.zeros(arr.shape, dtype=bool)
            if "NaN" in self.NA_types:
                if np.issubdtype(vector.dtype, np.inexact):
                    mask |= np.isnan(vector)
                if vector.dtype == np.dtype(object):
                    mask |= safe_isnan(vector)
            if mask.ndim > 1:
                mask = np.any(mask, axis=1)
        return mask

    def handle_NA(self, index, factor_values, origins):
        """Takes a set of factor values that may have NAs, and handles them
        appropriately.

        If you want to somehow create your own custom handling for missing
        values, you can do that by creating a class which defines a compatible
        `handle_NA` method, and then pass an instance of your class as your
        `NA_action=` argument.

        :arg index: An `ndarray` or :class:`pandas.Index` which represents the
          original index for the incoming data. Comparing the input index to
          the output index lets downstream code determine which rows in our
          return value match to which rows in the original input, and which
          rows were deleted.
        :arg factor_values: A list of `ndarray` and :class:`Categorical`
          objects representing the data. The `ndarray` objects are always
          2-dimensional floating point arrays. All have the same number of
          rows. (The `Categorical` objects, which are 1-dimensional, have the
          same number of entries.)
        :arg origins: A list with the same number of entries as
          `factor_values`, containing information on the origin of each
          value. If we encounter a problem with some particular value, we use
          the corresponding entry in `origins` as the origin argument when
          raising a :class:`PatsyError`.
        :returns: A tuple `(new_index, new_factor_values)`.
        """
        if not factor_values:
            return (index, factor_values)
        if self.on_NA == "raise":
            return self._handle_NA_raise(index, factor_values, origins)
        elif self.on_NA == "drop":
            return self._handle_NA_drop(index, factor_values, origins)
        else: # pragma: no cover
            assert False

    def _handle_NA_raise(self, index, factor_values, origins):
        for factor_value, origin in zip(factor_values, origins):
            this_mask = self.is_NA(factor_value)
            if np.any(this_mask):
                raise PatsyError("factor contains missing values", origin)
        return (index, factor_values)

    def _handle_NA_drop(self, index, factor_values, origins):
        # this works no matter whether factor_values[0] is 1- or 2-dimensional
        if isinstance(factor_values[0], Categorical):
            num_rows = factor_values[0].int_array.shape[0]
        else:
            num_rows = factor_values[0].shape[0]
        total_mask = np.zeros(num_rows, dtype=bool)
        for factor_value in factor_values:
            total_mask |= self._where_NA(factor_value)
        good_mask = ~total_mask
        def select_rows(v):
            if isinstance(v, Categorical):
                return Categorical(v.int_array[good_mask],
                                   levels=v.levels,
                                   contrast=v.contrast)
            else:
                return v[good_mask, ...]
        return index[good_mask], [select_rows(v) for v in factor_values]

def test_NAAction_basic():
    from nose.tools import assert_raises
    assert_raises(ValueError, NAAction, on_NA="pord")
    assert_raises(ValueError, NAAction, NA_types=("NaN", "asdf"))
    assert_raises(ValueError, NAAction, NA_types="NaN")

    index, factor_values = NAAction().handle_NA(np.asarray([0, 1, 2]),
                                                [np.asarray([np.nan, 0.0, 1.0]),
                                                 np.asarray(["a", "b", "c"])],
                                                [None])
    assert np.array_equal(index, [1, 2])
    assert np.array_equal(factor_values[0], [0.0, 1.0])
    assert np.all(factor_values[1] == ["b", "c"])

def test_NAAction_NA_types():
    for NA_types in [[], ["NaN"]]:
        action = NAAction(NA_types=NA_types)
        for val, dtype in [("hi", object), (1.0, float), (1, int)]:
            for ndim in [1, 2]:
                arr = np.array([val] * 6, dtype=dtype)
                nan_rows = [0, 2]
                exp_NA_mask = np.zeros(6, dtype=bool)
                if ndim == 2:
                    arr = np.column_stack((arr, arr))
                    nan_idxs = zip(nan_rows, [1, 0])
                else:
                    nan_idxs = nan_rows
                if dtype in (object, float):
                    for nan_idx in nan_idxs:
                        arr[nan_idx] = np.nan
                    if "NaN" in NA_types:
                        exp_NA_mask[nan_rows] = True
                mask = action._where_NA(arr)
                assert np.array_equal(mask, exp_NA_mask)
        # Also check that Categorical missing values are detected regardless
        # of NA_types
        c = Categorical.from_sequence(["a1", None, "a2"], contrast="asdf")
        assert np.array_equal(action._where_NA(c), [False, True, False])

def test_NAAction_drop():
    def t(in_index, in_arrs, exp_index, exp_arrs):
        action = NAAction(on_NA="drop")
        out_index, out_arrs = action.handle_NA(in_index, in_arrs,
                                               [None] * len(in_arrs))
        assert np.array_equal(out_index, exp_index)
        assert len(out_arrs) == len(in_arrs) == len(exp_arrs)
        for out_arr, exp_arr in zip(out_arrs, exp_arrs):
            assert np.array_equal(out_arr, exp_arr)

    t(np.arange(5)[::-1],
      [np.asarray([1, 2, 3, 4, 5]),
       np.asarray([[1.0, 2.0],
                   [3.0, 4.0],
                   [np.nan, 5.0],
                   [6.0, 7.0],
                   [8.0, np.nan]])],
      np.asarray([4, 3, 1]),
      [np.asarray([1, 2, 4]),
       np.asarray([[1.0, 2.0], [3.0, 4.0], [6.0, 7.0]])])
    
def test_NAAction_drop_categorical():
    numeric = np.asarray([1.0, np.nan, 3.0, 4.0])
    cat = Categorical.from_sequence(["a1", "a3", None, "a2"],
                                    contrast="asdf")
    index = np.arange(4)
    out_index, (out_numeric, out_cat) = (
        NAAction().handle_NA(index, [numeric, cat], [None, None]))
    assert np.array_equal(out_index, [0, 3])
    assert np.array_equal(out_numeric, [1.0, 4.0])
    assert np.array_equal(out_cat.int_array, [0, 1])
    assert out_cat.levels == ("a1", "a2", "a3")
    assert out_cat.contrast == "asdf"

def test_NAAction_raise():
    action = NAAction(on_NA="raise")

    # no-NA just passes through:
    in_idx = np.arange(2)
    in_arrs = [np.asarray(["a", "b"], dtype=object),
               np.asarray([1, 2])]
    got_idx, got_arrs = action.handle_NA(in_idx, in_arrs, [None, None])
    assert np.array_equal(got_idx, in_idx)
    assert np.array_equal(got_arrs[0], in_arrs[0])
    assert np.array_equal(got_arrs[1], in_arrs[1])

    from patsy.origin import Origin
    o1 = Origin("asdf", 0, 1)
    o2 = Origin("asdf", 2, 3)

    # NA raises an error with a correct origin
    in_idx = np.arange(2)
    in_arrs = [np.asarray(["a", "b"], dtype=object),
               np.asarray([1.0, np.nan], dtype=float)]
    try:
        action.handle_NA(in_idx, in_arrs, [o1, o2])
        assert False
    except PatsyError, e:
        assert e.origin is o2
