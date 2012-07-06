# This file is part of Charlton
# Copyright (C) 2011 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

# These are made available in the charlton.* namespace
__all__ = ["Categorical", "C"]

import numpy as np
from charlton import CharltonError
from charlton.state import stateful_transform
from charlton.util import SortAnythingKey, have_pandas, asarray_or_pandas

# A simple wrapper around some categorical data. Provides basically no
# services, but it holds data fine... eventually it'd be nice to make a custom
# dtype for this, but doing that right will require fixes to numpy itself.
class Categorical(object):
    def __init__(self, int_array, levels, contrast=None):
        self.int_array = asarray_or_pandas(int_array, dtype=int)
        if self.int_array.ndim != 1:
            raise CharltonError("Categorical data must be 1 dimensional")
        self.levels = tuple(levels)
        self.contrast = contrast

    @classmethod
    def from_sequence(cls, sequence, levels=None, **kwargs):
        if levels is None:
            try:
                levels = list(set(sequence))
            except TypeError:
                raise CharltonError("Error converting data to categorical: "
                                    "all items must be hashable")
            levels.sort(key=SortAnythingKey)
        level_to_int = {}
        for i, level in enumerate(levels):
            try:
                level_to_int[level] = i
            except TypeError:
                raise CharltonError("Error converting data to categorical: "
                                    "all levels must be hashable (and %r isn't)"
                                    % (level,))
        int_array = np.empty(len(sequence), dtype=int)
        for i, entry in enumerate(sequence):
            try:
                int_array[i] = level_to_int[entry]
            except KeyError:
                raise CharltonError("Error converting data to categorical: "
                                    "object %r does not match any of the "
                                    "expected levels" % (entry,))
        return cls(int_array, levels, **kwargs)

def test_Categorical():
    c = Categorical([0, 1, 2], levels=["a", "b", "c"])
    assert isinstance(c.int_array, np.ndarray)
    assert np.all(c.int_array == [0, 1, 2])
    assert isinstance(c.levels, tuple)
    assert c.levels == ("a", "b", "c")
    assert c.contrast is None

    if have_pandas:
        import pandas
        s = pandas.Series([0, 1, 2], index=[10, 20, 30])
        c_pandas = Categorical(s, levels=["a", "b", "c"])
        assert np.all(c_pandas.int_array == [0, 1, 2])
        assert isinstance(c_pandas.levels, tuple)
        assert np.all(c_pandas.int_array.index == [10, 20, 30])

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

    c5 = Categorical([0.0, 1.0, 2.0], levels=["a", "b", "c"])
    assert np.all(c5.int_array == [0, 1, 2])
    assert c5.int_array.dtype == np.dtype(int)

    from nose.tools import assert_raises
    assert_raises(CharltonError,
                  Categorical, 0, levels=["a", "b", "c"])
    assert_raises(CharltonError,
                  Categorical, [[0]], levels=["a", "b", "c"])
    if have_pandas:
            assert_raises(CharltonError,
                          Categorical, pandas.DataFrame([[0]]),
                          levels=["a", "b", "c"])

    assert_raises(CharltonError,
                  Categorical.from_sequence, ["a", "b", "q"], levels=["a", "b"])

    assert_raises(CharltonError,
                  Categorical.from_sequence, ["a", "b", {}])
    assert_raises(CharltonError,
                  Categorical.from_sequence, ["a", "b"], levels=["a", "b", {}])

class CategoricalTransform(object):
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
                raise CharltonError("changing levels of categorical data "
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

def test_categorical_non_strings():
    cat = C([1, "foo", ("a", "b")])
    assert set(cat.levels) == set([1, "foo", ("a", "b")])
    # have to use list() here because tuple.index does not exist before Python
    # 2.6.
    assert cat.int_array[0] == list(cat.levels).index(1)
    assert cat.int_array[1] == list(cat.levels).index("foo")
    assert cat.int_array[2] == list(cat.levels).index(("a", "b"))
