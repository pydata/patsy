# This file is part of Patsy
# Copyright (C) 2011-2013 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

# These are made available in the patsy.* namespace
__all__ = ["C"]

# How we handle categorical data: the big picture
# -----------------------------------------------
#
# There is no Python/NumPy standard for how to represent categorical data.
# There is no Python/NumPy standard for how to represent missing data.
#
# Together, these facts mean that when we receive some data object, we must be
# able to heuristically infer what levels it has -- and this process must be
# sensitive to the current missing data handling, because maybe 'None' is a
# level and maybe it is missing data.
#
# We don't know how missing data is represented until we get into the actual
# builder code, so anything which runs before this -- e.g., the 'C()' builtin
# -- cannot actually do *anything* meaningful with the data.
#
# Therefore, C() simply takes some data and arguments, and boxes them all up
# together into an object called (appropriately enough) _CategoricalBox. All
# the actual work of handling the various different sorts of categorical data
# (lists, string arrays, bool arrays, pandas.Categorical, etc.) happens inside
# the builder code, and we just extend this so that it also accepts
# _CategoricalBox objects as yet another categorical type.
#
# Originally this file contained a container type (called 'Categorical'), and
# the various sniffing, conversion, etc., functions were written as methods on
# that type. But we had to get rid of that type, so now this file just
# provides a set of plain old functions which are used by patsy.build to
# handle the different stages of categorical data munging.

import numpy as np
from patsy import PatsyError
from patsy.state import stateful_transform
from patsy.util import (SortAnythingKey,
                        have_pandas, have_pandas_categorical,
                        asarray_or_pandas,
                        pandas_friendly_reshape,
                        safe_scalar_isnan)

if have_pandas:
    import pandas

# Objects of this type will always be treated as categorical, with the
# specified levels and contrast (if given).
class _CategoricalBox(object):
    def __init__(self, data, contrast, levels):
        self.data = _NA_safe_asarray(data)
        self.contrast = contrast
        self.levels = levels

def C(data, contrast=None, levels=None):
    """
    Marks some `data` as being categorical, and specifies how to interpret
    it.

    This is used for three reasons:

    * To explicitly mark some data as categorical. For instance, integer data
      is by default treated as numerical. If you have data that is stored
      using an integer type, but where you want patsy to treat each different
      value as a different level of a categorical factor, you can wrap it in a
      call to `C` to accomplish this. E.g., compare::

        dmatrix("a", {"a": [1, 2, 3]})
        dmatrix("C(a)", {"a": [1, 2, 3]})

    * To explicitly set the levels or override the default level ordering for
      categorical data, e.g.::

        dmatrix("C(a, levels=["a2", "a1"])", balanced(a=2))
    * To override the default coding scheme for categorical data. The
      `contrast` argument can be any of:

      * A :class:`ContrastMatrix` object
      * A simple 2d ndarray (which is treated the same as a ContrastMatrix
        object except that you can't specify column names)
      * An object with methods called `code_with_intercept` and
        `code_without_intercept`, like the built-in contrasts
        (:class:`Treatment`, :class:`Diff`, :class:`Poly`, etc.). See
        :ref:`categorical-coding` for more details.
      * A callable that returns one of the above.
    """
    return _CategoricalBox(data, contrast, levels)

def guess_categorical(data):
    if have_pandas_categorical and isinstance(data, pandas.Categorical):
        return True
    if isinstance(data, _CategoricalBox):
        return True
    data = np.asarray(data)
    if np.issubdtype(value.dtype, np.number):
        return False
    return True

def test_guess_categorical():
    if have_pandas_categorical:
        assert guess_categorical(pandas.Categorical([1, 2, 3]))
    assert guess_categorical(C([1, 2, 3]))
    assert guess_categorical([True, False])
    assert guess_categorical(["a", "b"])
    assert not guess_categorical([1, 2, 3])
    assert not guess_categorical([1.0, 2.0, 3.0])

class CatLevelSniffer(object):
    def __init__(self, NA_action):
        self._NA_action = NA_action
        self._levels = None
        self._level_set = set()

    def levels(self):
        if self._levels is not None:
            return self._levels
        else:
            levels = list(self._level_set)
            levels.sort(key=SortAnythingKey)
            return levels

    def sniff_levels(self, data):
        # returns a bool: are we confident that we found all the levels?
        if have_pandas_categorical and isinstance(data, pandas.Categorical):
            # pandas.Categorical has its own NA detection, so don't try to
            # second-guess it.
            self._levels = tuple(data.levels)
            return True
        if isinstance(data, _CategoricalBox):
            if data.levels is not None:
                self._levels = tuple(data.levels)
                return True
            else:
                # unbox and fall through
                data = data.data
        for value in data:
            if self._NA_action.is_NA_scalar(value):
                continue
            if value is True or value is False:
                self._level_set.add([True, False])
            else:
                try:
                    self._level_set.add(value)
                except TypeError:
                    raise PatsyError("Error interpreting categorical data: "
                                     "all items must be hashable")
        # If everything we've seen is boolean, assume that everything else
        # would be too. Otherwise we need to keep looking.
        return self._level_set == set([True, False])

def test_CatLevelSniffer(object):
    from patsy.missing import NAAction
    def t(NA_types, datas, exp_finish_fast, exp_levels):
        sniffer = CatLevelSniffer(NAAction(NA_types=NA_types))
        for data in datas:
            done = sniffer.sniff_levels(data)
            if done:
                assert exp_finish_fast
                break
            else:
                assert not exp_finish_fast
        assert sniffer.levels() == exp_levels
    
    if have_pandas_categorical:
        t([], [pandas.Categorical.from_array([1, 2, None])],
          True, (1, 2))
        # check order preservation
        t([], [pandas.Categorical([1, 0], ["a", "b"])],
          True, ("a", "b"))
        t([], [pandas.Categorical([1, 0], ["b", "a"])],
          True, ("b", "a"))

    t([], [C([1, 2]), C([3, 2])], False, (1, 2, 3))
    # check order preservation
    t([], [C([1, 2], levels=[1, 2, 3]), C([4, 2])], True, (1, 2, 3))
    t([], [C([1, 2], levels=[3, 2, 1]), C([4, 2])], True, (3, 2, 1))

    # do some actual sniffing with NAs in
    t(["None", "NaN"], [C([1, np.nan]), C([10, None])],
      False, (1, 10))

    # bool special case
    t(["None", "NaN"], [C([True, np.nan, None])],
      True, (False, True))
    t([], [C([10, 20]), C([False]), C([30, 40])],
      False, (False, True, 10, 20, 30, 40))

    # check tuples too
    t(["None", "Nan"], [C([("b", 2), None, ("a", 1), np.nan, ("c", None)])],
      False, (("a", 1), ("b", 2), ("c", None)))

    # unhashable level error:
    from nose.tools import assert_raises
    sniffer = CatLevelSniffer(NAAction())
    assert_raises(PatsyError, sniffer.sniff_levels, [{}])

def categorical_to_int(data, levels, NA_action):
    # In this function, missing values are always mapped to -1
    if have_pandas_categorical and isinstance(data, pandas.Categorical):
        if not data.levels.equals(levels):
            raise PatsyError("mismatching levels: expected %r, got %r"
                             % (levels, data.levels))
        # pandas.Categorical also uses -1 to indicate NA, and we don't try to
        # second-guess its NA detection, so we can just pass it back.
        return data.labels
    if isinstance(data, _CategoricalBox):
        if data.levels is not None and data.levels != levels:
            raise PatsyError("mismatching levels: expected %r, got %r"
                             % (levels, data.levels))
        data = data.data
    try:
        level_to_int = dict(zip(levels, xrange(len(levels))))
    except TypeError:
        raise PatsyError("Error interpreting categorical data: "
                         "all items must be hashable")
    out = np.empty(len(data), dtype=int)
    for i, value in enumerate(data):
        if NA_action.is_scalar_NA(value):
            out[i] = -1
        else:
            try:
                out[i] = level_to_int[value]
            except KeyError:
                SHOW_LEVELS = 4
                level_strs = []
                if len(levels) <= SHOW_LEVELS:
                    level_strs += [repr(level) for level in levels]
                else:
                    level_strs += [repr(level)
                                   for level in levels[:SHOW_LEVELS//2]]
                    level_strs.append("...")
                    level_strs += [repr(level)
                                   for level in levels[-SHOW_LEVELS//2:]]
                level_str = "[%s]" % (", ".join(level_strs))
                raise PatsyError("Error converting data to categorical: "
                                 "observation with value %r does not match "
                                 "any of the expected levels (expected: %s)"
                                 % (entry, level_str))
    if have_pandas and isinstance(data, pandas.Series):
        out = pandas.Series(out, index=data.index)
    return out

def test_Categorical():
    c = Categorical([0, 1, 2], levels=["a", "b", "c"])
    assert isinstance(c.int_array, np.ndarray)
    assert np.all(c.int_array == [0, 1, 2])
    assert isinstance(c.levels, tuple)
    assert c.levels == ("a", "b", "c")
    assert c.contrast is None

    if have_pandas:
        s = pandas.Series([0, 1, 2], index=[10, 20, 30])
        c_pandas = Categorical(s, levels=["a", "b", "c"])
        assert np.all(c_pandas.int_array == [0, 1, 2])
        assert isinstance(c_pandas.levels, tuple)
        assert np.all(c_pandas.int_array.index == [10, 20, 30])
        c_pandas2 = Categorical(pandas.DataFrame({10: s}),
                                levels=["a", "b", "c"])
        assert np.all(c_pandas2.int_array == [0, 1, 2])
        assert isinstance(c_pandas2.levels, tuple)
        assert np.all(c_pandas2.int_array.index == [10, 20, 30])

    c2 = Categorical.from_sequence(["b", "a", "c"])
    assert c2.levels == ("a", "b", "c")
    assert np.all(c2.int_array == [1, 0, 2])

    c3 = Categorical.from_sequence(["b", "a", "c"],
                                   levels=["a", "c", "d", "b"])
    assert c3.levels == ("a", "c", "d", "b")
    print c3.int_array
    assert np.all(c3.int_array == [3, 0, 1])
    assert c3.contrast is None

    c4 = Categorical.from_sequence(["a"] * 100, levels=["b", "a"])
    assert c4.levels == ("b", "a")
    assert np.all(c4.int_array == 1)
    assert c4.contrast is None

    c5 = Categorical([[0.0], [1.0], [2.0]], levels=["a", "b", "c"])
    assert np.all(c5.int_array == [0, 1, 2])
    assert c5.int_array.dtype == np.dtype(int)

    if have_pandas:
        c6 = Categorical.from_sequence(pandas.Series(["a", "c", "b"],
                                                     index=[10, 20, 30]))
        assert isinstance(c6.int_array, pandas.Series)
        assert np.array_equal(c6.int_array.index, [10, 20, 30])
        assert np.array_equal(c6.int_array, [0, 2, 1])
        assert c6.levels == ("a", "b", "c")

    from nose.tools import assert_raises
    assert_raises(PatsyError,
                  Categorical, 0, levels=["a", "b", "c"])
    assert_raises(PatsyError,
                  Categorical, [[0, 1]], levels=["a", "b", "c"])
    if have_pandas:
            assert_raises(PatsyError,
                          Categorical, pandas.DataFrame([[0, 1]]),
                          levels=["a", "b", "c"])

    assert_raises(PatsyError,
                  Categorical.from_sequence, ["a", "b", "q"], levels=["a", "b"])
    # Smoke test for the branch that formats the ellipsized list of levels in
    # the error message:
    assert_raises(PatsyError,
                  Categorical.from_sequence, ["a", "b", "q"],
                  levels=["a", "b", "c", "d", "e", "f", "g", "h"])

    assert_raises(PatsyError,
                  Categorical.from_sequence, ["a", "b", {}])
    assert_raises(PatsyError,
                  Categorical.from_sequence, ["a", "b"], levels=["a", "b", {}])

def test_Categorical_missing():
    seqs = [["a", "c", None, np.nan, "b"],
            np.asarray(["a", "c", None, np.nan, "b"], dtype=object),
            [("hi", 1), ("zzz", 1), None, np.nan, ("hi", 2)],
            ]
    if have_pandas:
        seqs.append(pandas.Series(["a", "c", None, np.nan, "b"]))
    for seq in seqs:
        c = Categorical.from_sequence(seq)
        assert len(c.levels) == 3
        assert np.array_equal(c.int_array, [0, 2, -1, -1, 1])

    c = Categorical.from_sequence(["a", "c", None, np.nan, "b"],
                                  levels=["c", "a", "b"])
    assert c.levels == ("c", "a", "b")
    assert np.array_equal(c.int_array, [1, 0, -1, -1, 2])

    if have_pandas_categorical:
        # Make sure that from_pandas_categorical works too
        pc = pandas.Categorical.from_array(["a", "c", None, np.nan, "b"])
        from patsy.util import safe_isnan
        assert np.array_equal(safe_isnan(pc),
                              [False, False, True, True, False])
        c = Categorical.from_pandas_categorical(pc)
        assert np.array_equal(c.int_array, [0, 2, -1, -1, 1])

    marr = np.ma.masked_array(["a", "c", "a", "z", "b"],
                              mask=[False, False, True, True, False])
    c = Categorical.from_sequence(marr)
    assert c.levels == ("a", "b", "c")
    assert np.array_equal(c.int_array, [0, 2, -1, -1, 1])

# contrast= can be:
#   -- a ContrastMatrix
#   -- a simple np.ndarray
#   -- an object with code_with_intercept and code_without_intercept methods
#   -- a function returning one of the above
#   -- None
class CategoricalTransform(object):
    """C(data, contrast=None, levels=None)

    Converts some `data` into :class:`Categorical` form. (It is also called
    implicitly any time a formula contains a bare categorical factor.)

    This is used in two cases:

    * To explicitly set the levels or override the default level ordering for
      categorical data, e.g.::

        dmatrix("C(a, levels=["a2", "a1"])", balanced(a=2))
    * To override the default coding scheme for categorical data. The
      `contrast` argument can be any of:

      * A :class:`ContrastMatrix` object
      * A simple 2d ndarray (which is treated the same as a ContrastMatrix
        object except that you can't specify column names)
      * An object with methods called `code_with_intercept` and
        `code_without_intercept`, like the built-in contrasts
        (:class:`Treatment`, :class:`Diff`, :class:`Poly`, etc.). See
        :ref:`categorical-coding` for more details.
      * A callable that returns one of the above.

    In order to properly detect and remember the levels in your data, this is
    a :ref:`stateful transform <stateful-transforms>`.
    """
    def __init__(self, levels=None):
        self._levels = set()
        self._levels_tuple = None
        # 'levels' argument is for the use of the building code
        if levels is not None:
            self._levels_tuple = tuple(levels)

    def memorize_chunk(self, data, contrast=None, levels=None):
        if levels is None and not isinstance(data, Categorical):
            if isinstance(data, np.ndarray):
                data = data.ravel()
            self._levels.update(data)

    def memorize_finish(self):
        assert self._levels_tuple is None
        self._levels_tuple = tuple(sorted(self._levels, key=SortAnythingKey))

    def transform(self, data, contrast=None, levels=None):
        kwargs = {"contrast": contrast}
        if have_pandas_categorical and isinstance(data, pandas.Categorical):
            data = Categorical.from_pandas_categorical(data)
            # fall through to the next 'if':
        if isinstance(data, Categorical):
            if levels is not None and data.levels != levels:
                raise PatsyError("changing levels of categorical data "
                                    "not supported yet")
            return Categorical(data.int_array, data.levels, **kwargs)
        if levels is None:
            levels = self._levels_tuple
        return Categorical.from_sequence(data, levels, **kwargs)

    # This is for the use of the building code, which uses this transform to
    # convert string arrays (and similar) into Categoricals, and after
    # memorizing the data it needs to know what the levels were.
    def levels(self):
        assert self._levels_tuple is not None
        return self._levels_tuple

C = stateful_transform(CategoricalTransform)

def test_CategoricalTransform():
    t1 = CategoricalTransform()
    t1.memorize_chunk(["a", "b"])
    t1.memorize_chunk(["a", "c"])
    t1.memorize_finish()
    c1 = t1.transform(["a", "c"])
    assert c1.levels == ("a", "b", "c")
    assert np.all(c1.int_array == [0, 2])

    t2 = CategoricalTransform()
    t2.memorize_chunk(["a", "b"], contrast="foo", levels=["c", "b", "a"])
    t2.memorize_chunk(["a", "c"], contrast="foo", levels=["c", "b", "a"])
    t2.memorize_finish()
    c2 = t2.transform(["a", "c"], contrast="foo", levels=["c", "b", "a"])
    assert c2.levels == ("c", "b", "a")
    assert np.all(c2.int_array == [2, 0])
    assert c2.contrast == "foo"

    # Check that it passes through already-categorical data correctly,
    # changing the attributes on a copy only:
    c = Categorical.from_sequence(["a", "b"], levels=["b", "a"],
                                 contrast="foo")
    t3 = CategoricalTransform()
    t3.memorize_chunk(c, contrast="bar")
    t3.memorize_finish()
    c_t = t3.transform(c, contrast="bar")
    assert np.all(c_t.int_array == c.int_array)
    assert c_t.levels == c.levels
    assert c.contrast == "foo"
    assert c_t.contrast == "bar"
    
    # Check interpretation of non-keyword arguments
    t4 = CategoricalTransform()
    t4.memorize_chunk(["a", "b"], "foo", ["b", "c", "a"])
    t4.memorize_finish()
    c4 = t4.transform(["a", "b"], "foo", ["b", "c", "a"])
    assert np.all(c4.int_array == [2, 0])
    assert c4.levels == ("b", "c", "a")
    assert c4.contrast == "foo"

def test_C_pandas():
    if have_pandas:
        s_noidx = pandas.Series([3, 2, 1])
        s_idx = pandas.Series([3, 2, 1], index=[10, 20, 30])
        cat = C(s_noidx)
        assert isinstance(cat.int_array, pandas.Series)
        assert np.array_equal(cat.int_array, [2, 1, 0])
        assert np.array_equal(cat.int_array.index, [0, 1, 2])
        cat2 = C(s_idx, levels=[4, 3, 2, 1])
        assert isinstance(cat2.int_array, pandas.Series)
        assert np.array_equal(cat2.int_array, [1, 2, 3])
        assert np.array_equal(cat2.int_array.index, [10, 20, 30])
        assert cat2.contrast is None
        cat3 = C(cat2, "asdf")
        assert isinstance(cat3.int_array, pandas.Series)
        assert np.array_equal(cat3.int_array, [1, 2, 3])
        assert np.array_equal(cat3.int_array.index, [10, 20, 30])
        assert cat3.contrast == "asdf"

def test_categorical_from_pandas_categorical():
    if have_pandas_categorical:
        pandas_categorical = pandas.Categorical.from_array(["a", "b", "a"])
        c = Categorical.from_pandas_categorical(pandas_categorical)
        assert np.array_equal(c.int_array, [0, 1, 0])
        assert c.levels == ("a", "b")

def test_categorical_non_strings():
    cat = C([1, "foo", ("a", "b")])
    assert set(cat.levels) == set([1, "foo", ("a", "b")])
    # have to use list() here because tuple.index does not exist before Python
    # 2.6.
    assert cat.int_array[0] == list(cat.levels).index(1)
    assert cat.int_array[1] == list(cat.levels).index("foo")
    assert cat.int_array[2] == list(cat.levels).index(("a", "b"))
