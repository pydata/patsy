# This file is part of Charlton
# Copyright (C) 2011 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

# Some generic utilities.

__all__ = ["atleast_2d_column_default", "to_unique_tuple",
           "widest_float", "widest_complex", "wide_dtype_for", "widen",
           "repr_pretty_delegate", "repr_pretty_impl",
           "SortAnythingKey",
           ]

import numpy as np
from cStringIO import StringIO
from compat import optional_dep_ok

# Like np.atleast_2d, but this converts lower-dimensional arrays into columns,
# instead of rows. It also converts ndarray subclasses into basic ndarrays --
# this is useful for objects that have odd behavior. E.g., pandas.Series
# cannot be made 2 dimensional.
def atleast_2d_column_default(a):
    a = np.asarray(a)
    a = np.atleast_1d(a)
    if a.ndim <= 1:
        a = a.reshape((-1, 1))
    assert a.ndim >= 2
    return a

def test_atleast_2d_column_default():
    assert np.all(atleast_2d_column_default([1, 2, 3]) == [[1], [2], [3]])

    assert atleast_2d_column_default(1).shape == (1, 1)
    assert atleast_2d_column_default([1]).shape == (1, 1)
    assert atleast_2d_column_default([[1]]).shape == (1, 1)
    assert atleast_2d_column_default([[[1]]]).shape == (1, 1, 1)

    assert atleast_2d_column_default([1, 2, 3]).shape == (3, 1)
    assert atleast_2d_column_default([[1], [2], [3]]).shape == (3, 1)

    assert type(atleast_2d_column_default(np.matrix(1))) == np.ndarray

def to_unique_tuple(seq):
    seq_new = []
    for obj in seq:
        if obj not in seq_new:
            seq_new.append(obj)
    return tuple(seq_new)

def test_to_unique_tuple():
    assert to_unique_tuple([1, 2, 3]) == (1, 2, 3)
    assert to_unique_tuple([1, 3, 3, 2, 3, 1]) == (1, 3, 2)
    assert to_unique_tuple([3, 2, 1, 4, 1, 2, 3]) == (3, 2, 1, 4)


for float_type in ("float128", "float96", "float64"):
    if hasattr(np, float_type):
        widest_float = getattr(np, float_type)
        break
else: # pragma: no cover
    assert False
for complex_type in ("complex256", "complex196", "complex128"):
    if hasattr(np, complex_type):
        widest_complex = getattr(np, complex_type)
        break
else: # pragma: no cover
    assert False

def wide_dtype_for(arr):
    arr = np.asarray(arr)
    if (np.issubdtype(arr.dtype, np.integer)
        or np.issubdtype(arr.dtype, np.floating)):
        return widest_float
    elif np.issubdtype(arr.dtype, np.complexfloating):
        return widest_complex
    raise ValueError, "cannot widen a non-numeric type %r" % (arr.dtype,)

def widen(arr):
    return np.asarray(arr, dtype=wide_dtype_for(arr))

def test_wide_dtype_for_and_widen():
    assert np.allclose(widen([1, 2, 3]), [1, 2, 3])
    assert widen([1, 2, 3]).dtype == widest_float
    assert np.allclose(widen([1.0, 2.0, 3.0]), [1, 2, 3])
    assert widen([1.0, 2.0, 3.0]).dtype == widest_float
    assert np.allclose(widen([1+0j, 2, 3]), [1, 2, 3])
    assert widen([1+0j, 2, 3]).dtype == widest_complex
    from nose.tools import assert_raises
    assert_raises(ValueError, widen, ["hi"])

class PushbackAdapter(object):
    def __init__(self, it):
        self._it = it
        self._pushed = []

    def __iter__(self):
        return self

    def push_back(self, obj):
        self._pushed.append(obj)

    def next(self):
        if self._pushed:
            return self._pushed.pop()
        else:
            # May raise StopIteration
            return self._it.next()

    def peek(self):
        try:
            obj = self.next()
        except StopIteration:
            raise ValueError, "no more data"
        self.push_back(obj)
        return obj

    def has_more(self):
        try:
            self.peek()
        except ValueError:
            return False
        else:
            return True

def test_PushbackAdapter():
    it = PushbackAdapter(iter([1, 2, 3, 4]))
    assert it.has_more()
    assert it.next() == 1
    it.push_back(0)
    assert it.next() == 0
    assert it.next() == 2
    assert it.peek() == 3
    it.push_back(10)
    assert it.peek() == 10
    it.push_back(20)
    assert it.peek() == 20
    assert it.has_more()
    assert list(it) == [20, 10, 3, 4]
    assert not it.has_more()

# The IPython pretty-printer gives very nice output that is difficult to get
# otherwise, e.g., look how much more readable this is than if it were all
# smooshed onto one line:
# 
#    ModelDesc(input_code='y ~ x*asdf',
#              lhs_terms=[Term([EvalFactor('y')])],
#              rhs_terms=[Term([]),
#                         Term([EvalFactor('x')]),
#                         Term([EvalFactor('asdf')]),
#                         Term([EvalFactor('x'), EvalFactor('asdf')])],
#              )
#              
# But, we don't want to assume it always exists; nor do we want to be
# re-writing every repr function twice, once for regular repr and once for
# the pretty printer. So, here's an ugly fallback implementation that can be
# used unconditionally to implement __repr__ in terms of _pretty_repr_.
#
# Pretty printer docs:
#   http://ipython.org/ipython-doc/dev/api/generated/IPython.lib.pretty.html

from cStringIO import StringIO
class _MiniPPrinter(object):
    def __init__(self):
        self._out = StringIO()
        self.indentation = 0

    def text(self, text):
        self._out.write(text)

    def breakable(self, sep=" "):
        self._out.write(sep)

    def begin_group(self, _, text):
        self.text(text)

    def end_group(self, _, text):
        self.text(text)

    def pretty(self, obj):
        if hasattr(obj, "_repr_pretty_"):
            obj._repr_pretty_(self, False)
        else:
            self.text(repr(obj))

    def getvalue(self):
        return self._out.getvalue()

def _mini_pretty(obj):
   printer = _MiniPPrinter()
   printer.pretty(obj)
   return printer.getvalue()

if optional_dep_ok:
    try:
        from IPython.lib.pretty import pretty as repr_pretty_delegate
    except ImportError:
        repr_pretty_delegate = _mini_pretty
else:
    repr_pretty_delegate = _mini_pretty

def repr_pretty_impl(p, obj, args, kwargs=[]):
    name = obj.__class__.__name__
    p.begin_group(len(name) + 1, "%s(" % (name,))
    started = [False]
    def new_item():
        if started[0]:
            p.text(",")
            p.breakable()
        started[0] = True
    for arg in args:
        new_item()
        p.pretty(arg)
    for label, value in kwargs:
        new_item()
        p.begin_group(len(label) + 1, "%s=" % (label,))
        p.pretty(value)
        p.end_group(len(label) + 1, "")
    p.end_group(len(name) + 1, ")")

def test_repr_pretty():
    assert repr_pretty_delegate("asdf") == "'asdf'"
    printer = _MiniPPrinter()
    class MyClass(object):
        pass
    repr_pretty_impl(printer, MyClass(),
                     ["a", 1], [("foo", "bar"), ("asdf", "asdf")])
    assert printer.getvalue() == "MyClass('a', 1, foo='bar', asdf='asdf')"

# In Python 3, objects of different types are not generally comparable, so a
# list of heterogenous types cannot be sorted. This implements a Python 2
# style comparison for arbitrary types. (It works on Python 2 too, but just
# gives you the built-in ordering.) To understand why this is tricky, consider
# this example:
#   a = 1    # type 'int'
#   b = 1.5  # type 'float'
#   class gggg:
#       pass
#   c = gggg()
#   sorted([a, b, c])
# The fallback ordering sorts by class name, so according to the fallback
# ordering, we have b < c < a. But, of course, a and b are comparable (even
# though they're of different types), so we also have a < b. This is
# inconsistent. There is no general solution to this problem (which I guess is
# why Python 3 stopped trying), but the worst offender is all the different
# "numeric" classes (int, float, complex, decimal, rational...), so as a
# special-case, we sort all numeric objects to the start of the list.
# (In Python 2, there is also a similar special case for str and unicode, but
# we don't have to worry about that for Python 3.)
class SortAnythingKey(object):
    def __init__(self, obj):
        self.obj = obj

    def _python_lt(self, other_obj):
        # On Py2, < never raises an error, so this is just <. (Actually it
        # does raise a TypeError for comparing complex to numeric, but not for
        # comparisons of complex to other types. Sigh. Whatever.)
        # On Py3, this returns a bool if available, and otherwise returns
        # NotImplemented
        try:
            return self.obj < other_obj
        except TypeError:
            return NotImplemented

    def __lt__(self, other):
        assert isinstance(other, SortAnythingKey)
        result = self._python_lt(other.obj)
        if result is not NotImplemented:
            return result
        # Okay, that didn't work, time to fall back.
        # If one of these is a number, then it is smaller.
        if self._python_lt(0) is not NotImplemented:
            return True
        if other._python_lt(0) is not NotImplemented:
            return False
        # Also check ==, since it may well be defined for otherwise
        # unorderable objects, and if so then we should be consistent with
        # it:
        if self.obj == other.obj:
            return False
        # Otherwise, we break ties based on class name and memory position
        return ((self.obj.__class__.__name__, id(self.obj))
                < (other.obj.__class__.__name__, id(other.obj)))

def test_SortAnythingKey():
    assert sorted([20, 10, 0, 15], key=SortAnythingKey) == [0, 10, 15, 20]
    assert sorted([10, -1.5], key=SortAnythingKey) == [-1.5, 10]
    assert sorted([10, "a", 20.5, "b"], key=SortAnythingKey) == [10, 20.5, "a", "b"]
    class a(object):
        pass
    class b(object):
        pass
    class z(object):
        pass
    a_obj = a()
    b_obj = b()
    z_obj = z()
    o_obj = object()
    assert (sorted([z_obj, a_obj, 1, b_obj, o_obj], key=SortAnythingKey)
            == [1, a_obj, b_obj, o_obj, z_obj])
