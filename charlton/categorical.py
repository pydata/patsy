# This file is part of Charlton
# Copyright (C) 2011 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

__all__ = ["Categorical", "code"]

import numpy as np
from charlton import CharltonError

# A simple wrapper around some categorical data. Provides basically no
# services, but it holds data fine... eventually it'd be nice to make a custom
# dtype for this, but doing that right will require fixes to numpy itself.
class Categorical(object):
    def __init__(self, int_array, levels, ordered=False, contrast=None):
        self.int_array = np.asarray(int_array, dtype=int).ravel()
        self.levels = tuple(levels)
        self.ordered = ordered
        self.contrast = contrast

    @classmethod
    def from_strings(cls, sequence, levels=None, **kwargs):
        if levels is None:
            try:
                levels = list(set(sequence))
            except TypeError:
                raise CharltonError("Error converting data to categorical: "
                                    "all items must be hashable")
            levels.sort()
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
            except ValueError:
                raise CharltonError("Error converting data to categorical: "
                                    "object '%r' does not match any of the "
                                    "expected levels" % (entry,))
        return cls(int_array, levels, **kwargs)

def test_categorical():
    c = Categorical([0, 1, 2], levels=["a", "b", "c"])
    assert isinstance(c.int_array, np.ndarray)
    assert np.all(c.int_array == [0, 1, 2])
    assert isinstance(c.levels, tuple)
    assert c.levels == ("a", "b", "c")
    assert not c.ordered
    assert c.contrast is None

    c2 = Categorical.from_strings(["b", "a", "c"])
    assert c2.levels == ("a", "b", "c")
    assert np.all(c2.int_array == [1, 0, 2])
    assert not c2.ordered

    c3 = Categorical.from_strings(["b", "a", "c"],
                                  levels=["a", "c", "d", "b"],
                                  ordered=True)
    assert c3.levels == ("a", "c", "d", "b")
    print c3.int_array
    assert np.all(c3.int_array == [3, 0, 1])
    assert c3.ordered
    assert c3.contrast is None

    c4 = Categorical.from_strings(["a"] * 100, levels=["b", "a"])
    assert c4.levels == ("b", "a")
    assert np.all(c4.int_array == 1)
    assert not c4.ordered
    assert c4.contrast is None

    c5 = Categorical([[0.0], [1.0], [2.0]], levels=["a", "b", "c"])
    assert np.all(c5.int_array == [0, 1, 2])
    assert c5.int_array.dtype == np.dtype(int)

class CategoricalTransform(object):
    def __init__(self):
        self._levels = set()
        self._levels_tuple = None

    def memorize_chunk(self, data, levels=None, **kwargs):
        if levels is None and not isinstance(data, Categorical):
            for item in data:
                self._levels.add(item)

    def memorize_finish(self):
        self._levels_tuple = tuple(sorted(self._levels))

    def transform(self, data, levels=None, **kwargs):
        if isinstance(data, Categorical):
            if levels is not None and data.levels != levels:
                raise CharltonError("changing levels of categorical data "
                                    "not supported yet")
            return Categorical(data.int_array, data.levels, **kwargs)
        if levels is None:
            levels = self._levels_tuple
        return Categorical.from_strings(data, levels=levels, **kwargs)

    # This is for the use of the ModelSpec building code, which has a special
    # case where it uses this transform to convert string arrays (and similar)
    # into Categoricals, and after memorizing the data it needs to know what
    # the levels were.
    def levels():
        assert self._levels_tuple is not None
        return self._levels_tuple

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
    c = Categorical.from_strings(["a", "b"], levels=["b", "a"],
                                 ordered=False, contrast="foo")
    t3 = CategoricalTransform()
    t3.memorize_chunk(c, ordered=True, contrast="bar")
    t3.memorize_finish()
    c_t = t3.transform(c, ordered=True, contrast="bar")
    assert np.all(c_t.int_array == c.int_array)
    assert c_t.levels == c.levels
    assert not c.ordered
    assert c_t.ordered
    assert c.contrast == "foo"
    assert c_t.contrast == "bar"
    

# This is just an alias for the above, but with rearranged arguments
class ContrastTransform(CategoricalTransform):
    def memorize_chunk(self, data, new_contrast):
        CategoricalTransform.memorize_chunk(self, data, contrast=new_contrast)

    def transform(self, data, new_contrast):
        return CategoricalTransform.transform(self, data, contrast=new_contrast)

def test_ContrastTransform():
    t1 = ContrastTransform()
    t1.memorize_chunk(["a", "b"], "foo")
    t1.memorize_chunk(["a", "c"], "foo")
    t1.memorize_finish()
    c1 = t1.transform(["a", "c"], "foo")
    assert c1.levels == ("a", "b", "c")
    assert np.all(c1.int_array == [0, 2])
    assert c1.contrast == "foo"
