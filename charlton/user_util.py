# This file is part of Charlton
# Copyright (C) 2012 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

# Miscellaneous utilities that are useful to users (as compared to
# charlton.util, which is misc. utilities useful for implementing charlton).

# These are made available in the charlton.* namespace
__all__ = ["balanced", "demo_data"]

import numpy as np
from charlton import CharltonError
from charlton.compat import itertools_product

def balanced(**kwargs):
    repeat = kwargs.pop("repeat", 1)
    levels = []
    names = sorted(kwargs)
    for name in names:
        level_count = kwargs[name]
        levels.append(["%s%s" % (name, i) for i in xrange(1, level_count + 1)])
    # zip(*...) does an "unzip"
    values = zip(*itertools_product(*levels))
    data = {}
    for name, value in zip(names, values):
        data[name] = list(value) * repeat
    return data

def test_balanced():
    data = balanced(a=2, b=3)
    assert data["a"] == ["a1", "a1", "a1", "a2", "a2", "a2"]
    assert data["b"] == ["b1", "b2", "b3", "b1", "b2", "b3"]
    data = balanced(a=2, b=3, repeat=2)
    assert data["a"] == ["a1", "a1", "a1", "a2", "a2", "a2",
                         "a1", "a1", "a1", "a2", "a2", "a2"]
    assert data["b"] == ["b1", "b2", "b3", "b1", "b2", "b3",
                         "b1", "b2", "b3", "b1", "b2", "b3"]

def demo_data(*names, **kwargs):
    nlevels = kwargs.pop("nlevels", 2)
    min_rows = kwargs.pop("min_rows", 5)
    if kwargs:
        raise TypeError, "unexpected keyword arguments %r" % (kwargs)
    numerical = set()
    categorical = {}
    for name in names:
        if name[0] in "abcdefghijklmn":
            categorical[name] = nlevels
        elif name[0] in "pqrstuvwxyz":
            numerical.add(name)
        else:
            raise CharltonError, "bad name %r" % (name,)
    balanced_design_size = np.prod(categorical.values())
    repeat = int(np.ceil(min_rows * 1.0 / balanced_design_size))
    num_rows = repeat * balanced_design_size
    data = balanced(repeat=repeat, **categorical)
    r = np.random.RandomState(0)
    for name in sorted(numerical):
        data[name] = r.normal(size=num_rows)
    return data
    
def test_demo_data():
    d1 = demo_data("a", "b", "x")
    assert sorted(d1.keys()) == ["a", "b", "x"]
    assert d1["a"] == ["a1", "a1", "a2", "a2", "a1", "a1", "a2", "a2"]
    assert d1["b"] == ["b1", "b2", "b1", "b2", "b1", "b2", "b1", "b2"]
    assert d1["x"].dtype == np.dtype(float)
    assert d1["x"].shape == (8,)

    d2 = demo_data("x", "y")
    assert sorted(d2.keys()) == ["x", "y"]
    assert len(d2["x"]) == len(d2["y"]) == 5

    assert len(demo_data("x", min_rows=10)["x"]) == 10
    assert len(demo_data("a", "b", "x", min_rows=10)["x"]) == 12
    assert len(demo_data("a", "b", "x", min_rows=10, nlevels=3)["x"]) == 18

    from nose.tools import assert_raises
    assert_raises(CharltonError, demo_data, "a", "b", "__123")
    assert_raises(TypeError, demo_data, "a", "b", asdfasdf=123)
