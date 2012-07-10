# This file is part of Patsy
# Copyright (C) 2011 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

# These are made available in the patsy.* namespace
__all__ = ["Categorical", "C"]

import numpy as np
from patsy import PatsyError
from patsy.state import stateful_transform
from patsy.util import (SortAnythingKey,
                           have_pandas, asarray_or_pandas,
                           pandas_friendly_reshape)

if have_pandas:
    import pandas

# A simple wrapper around some categorical data. Provides basically no
# services, but it holds data fine... eventually it'd be nice to make a custom
# dtype for this, but doing that right will require fixes to numpy itself.
class Categorical(object):
    """This is a simple class for holding categorical data, along with
    (possibly) a preferred contrast coding.

    You should not normally need to use this class directly; it's mostly used
    as a way for :func:`C` to pass information back to the formula evaluation
    machinery.
    """
    def __init__(self, int_array, levels, contrast=None):
        self.int_array = asarray_or_pandas(int_array, dtype=int)
        if self.int_array.ndim != 1:
            if self.int_array.ndim == 2 and self.int_array.shape[1] == 1:
                new_shape = (self.int_array.shape[0],)
                self.int_array = pandas_friendly_reshape(self.int_array,
                                                         new_shape)
            else:
                raise PatsyError("Categorical data must be 1 dimensional "
                                    "or column vector")
        self.levels = tuple(levels)
        self.contrast = contrast

    @classmethod
    def from_sequence(cls, sequence, levels=None, **kwargs):
        """from_sequence(sequence, levels=None, contrast=None)

        Create a Categorical object given a sequence of data. Levels will be
        auto-detected if not given.
        """
        if levels is None:
            try:
                levels = list(set(sequence))
            except TypeError:
                raise PatsyError("Error converting data to categorical: "
                                    "all items must be hashable")
            levels.sort(key=SortAnythingKey)
        level_to_int = {}
        for i, level in enumerate(levels):
            try:
                level_to_int[level] = i
            except TypeError:
                raise PatsyError("Error converting data to categorical: "
                                    "all levels must be hashable (and %r isn't)"
                                    % (level,))
        int_array = np.empty(len(sequence), dtype=int)
        for i, entry in enumerate(sequence):
            try:
                int_array[i] = level_to_int[entry]
            except KeyError:
                raise PatsyError("Error converting data to categorical: "
                                    "object %r does not match any of the "
                                    "expected levels" % (entry,))
        if have_pandas and isinstance(sequence, pandas.Series):
            int_array = pandas.Series(int_array, index=sequence.index)
        return cls(int_array, levels, **kwargs)

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

    assert_raises(PatsyError,
                  Categorical.from_sequence, ["a", "b", {}])
    assert_raises(PatsyError,
                  Categorical.from_sequence, ["a", "b"], levels=["a", "b", {}])

# contrast= can be:
#   -- a ContrastMatrix
#   -- a simple np.ndarray
#   -- an object with code_with_intercept and code_without_intercept methods
#   -- a function returning one of the above
#   -- None
class CategoricalTransform(object):
    """C(data, contrast=None, levels=None)

    Converts some `data` into :class:`Categorical` form. (It is also used
    called implicitly any time a formula contains a bare categorical factor.)

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

def test_categorical_non_strings():
    cat = C([1, "foo", ("a", "b")])
    assert set(cat.levels) == set([1, "foo", ("a", "b")])
    # have to use list() here because tuple.index does not exist before Python
    # 2.6.
    assert cat.int_array[0] == list(cat.levels).index(1)
    assert cat.int_array[1] == list(cat.levels).index("foo")
    assert cat.int_array[2] == list(cat.levels).index(("a", "b"))
