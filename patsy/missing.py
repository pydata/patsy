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
#   np.ma.masked (in numpy.ma masked arrays)
# Pandas compatibility considerations:
#   For numeric arrays, None is unconditionally converted to NaN.
#   For object arrays (including string arrays!), None and NaN are preserved,
#     but pandas.isnull() returns True for both.
# np.ma compatibility considerations:
#   Preserving array subtypes is a huge pain, because it means that we can't
#   just call 'asarray' and be done... we already jump through tons of hoops
#   to write code that can handle both ndarray's and pandas objects, and
#   just thinking about adding another item to this list makes me tired. So
#   for now we don't support np.ma missing values. Use pandas!

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

import numpy as np
from patsy import PatsyError
from patsy.util import safe_isnan, safe_scalar_isnan

# These are made available in the patsy.* namespace
__all__ = ["NAAction"]

_valid_NA_types = ["None", "NaN"]
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
    missing data. For numerical data, the default is to treat NaN values
    (e.g., `numpy.nan`) as missing. For categorical data, the default is to
    treat NaN values, and also the Python object None, as missing. (This is
    consistent with how pandas does things, so if you're already using
    None/NaN to mark missing data in your pandas DataFrames, you're good to
    go.)

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
    class, or your own object that implements the same interface, and pass
    that as the `NA_action=` argument instead.
    """
    def __init__(self, on_NA="drop", NA_types=["None", "NaN"]):
        """The `NAAction` constructor takes the following arguments:
        
        :arg on_NA: How to handle missing values. The default is "drop", which
          removes all rows from all matrices which contain any missing
          values. Also available is "raise", which raises an exception when
          any missing values are encountered.
        :arg NA_types: Which rules are used to identify missing values, as a
          list of strings. Allowed values are:
          * "None": treat the `None` object as missing in categorical data.
          * "NaN": treat floating point NaN values as missing in categorical
            and numerical data.
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

    def is_categorical_NA(self, obj):
        """Return True if `obj` is a categorical NA value.

        Note that here `obj` is a single scalar value."""
        if "NaN" in self.NA_types and safe_scalar_isnan(obj):
            return True
        if "None" in self.NA_types and obj is None:
            return True
        return False

    def is_numerical_NA(self, arr):
        """Returns a 1-d mask array indicating which rows in an array of
        numerical values contain at least one NA value.

        Note that here `arr` is a numpy array or pandas DataFrame."""
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
        num_rows = factor_values[0].shape[0]
        total_mask = np.zeros(num_rows, dtype=bool)
        for factor_value in factor_values:
            total_mask |= self._where_NA(factor_value)
        good_mask = ~total_mask
        return index[good_mask], [v[good_mask, ...] for v in factor_values]

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
