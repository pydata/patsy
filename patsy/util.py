# This file is part of Patsy
# Copyright (C) 2011-2013 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

# Some generic utilities.

__all__ = ["atleast_2d_column_default", "is_valid_python_varname",
           "uniqueify_list", "widest_float", "widest_complex", "wide_dtype_for",
           "widen", "repr_pretty_delegate", "repr_pretty_impl",
           "SortAnythingKey", "safe_scalar_isnan", "safe_isnan", "iterable",
           ]

import re
import sys
import numpy as np
from cStringIO import StringIO
from compat import optional_dep_ok

try:
    import pandas
except ImportError:
    have_pandas = False
else:
    have_pandas = True

# Pandas versions < 0.9.0 don't have Categorical
# Can drop this guard whenever we drop support for such older versions of
# pandas.
have_pandas_categorical = (have_pandas and hasattr(pandas, "Categorical"))

# Passes through Series and DataFrames, call np.asarray() on everything else
def asarray_or_pandas(a, copy=False, dtype=None, subok=False):
    if have_pandas:
        if isinstance(a, (pandas.Series, pandas.DataFrame)):
            # The .name attribute on Series is discarded when passing through
            # the constructor:
            #   https://github.com/pydata/pandas/issues/1578
            extra_args = {}
            if hasattr(a, "name"):
                extra_args["name"] = a.name
            return a.__class__(a, copy=copy, dtype=dtype, **extra_args)
    return np.array(a, copy=copy, dtype=dtype, subok=subok)

def test_asarray_or_pandas():
    assert type(asarray_or_pandas([1, 2, 3])) is np.ndarray
    assert type(asarray_or_pandas(np.matrix([[1, 2, 3]]))) is np.ndarray
    assert type(asarray_or_pandas(np.matrix([[1, 2, 3]]), subok=True)) is np.matrix
    a = np.array([1, 2, 3])
    assert asarray_or_pandas(a) is a
    a_copy = asarray_or_pandas(a, copy=True)
    assert np.array_equal(a, a_copy)
    a_copy[0] = 100
    assert not np.array_equal(a, a_copy)
    assert np.allclose(asarray_or_pandas([1, 2, 3], dtype=float),
                       [1.0, 2.0, 3.0])
    assert asarray_or_pandas([1, 2, 3], dtype=float).dtype == np.dtype(float)
    a_view = asarray_or_pandas(a, dtype=a.dtype)
    a_view[0] = 99
    assert a[0] == 99
    global have_pandas
    if have_pandas:
        s = pandas.Series([1, 2, 3], name="A", index=[10, 20, 30])
        s_view1 = asarray_or_pandas(s)
        assert s_view1.name == "A"
        assert np.array_equal(s_view1.index, [10, 20, 30])
        s_view1[10] = 101
        assert s[10] == 101
        s_copy = asarray_or_pandas(s, copy=True)
        assert s_copy.name == "A"
        assert np.array_equal(s_copy.index, [10, 20, 30])
        assert np.array_equal(s_copy, s)
        s_copy[10] = 100
        assert not np.array_equal(s_copy, s)
        assert asarray_or_pandas(s, dtype=float).dtype == np.dtype(float)
        s_view2 = asarray_or_pandas(s, dtype=s.dtype)
        assert s_view2.name == "A"
        assert np.array_equal(s_view2.index, [10, 20, 30])
        s_view2[10] = 99
        assert s[10] == 99

        df = pandas.DataFrame([[1, 2, 3]],
                              columns=["A", "B", "C"],
                              index=[10])
        df_view1 = asarray_or_pandas(df)
        df_view1.ix[10, "A"] = 101
        assert np.array_equal(df_view1.columns, ["A", "B", "C"])
        assert np.array_equal(df_view1.index, [10])
        assert df.ix[10, "A"] == 101
        df_copy = asarray_or_pandas(df, copy=True)
        assert np.array_equal(df_copy, df)
        assert np.array_equal(df_copy.columns, ["A", "B", "C"])
        assert np.array_equal(df_copy.index, [10])
        df_copy.ix[10, "A"] = 100
        assert not np.array_equal(df_copy, df)
        df_converted = asarray_or_pandas(df, dtype=float)
        assert df_converted["A"].dtype == np.dtype(float)
        assert np.allclose(df_converted, df)
        assert np.array_equal(df_converted.columns, ["A", "B", "C"])
        assert np.array_equal(df_converted.index, [10])
        df_view2 = asarray_or_pandas(df, dtype=df["A"].dtype)
        assert np.array_equal(df_view2.columns, ["A", "B", "C"])
        assert np.array_equal(df_view2.index, [10])
        # This actually makes a copy, not a view, because of a pandas bug:
        #   https://github.com/pydata/pandas/issues/1572
        assert np.array_equal(df, df_view2)
        # df_view2[0][0] = 99
        # assert df[0][0] == 99

        had_pandas = have_pandas
        try:
            have_pandas = False
            assert (type(asarray_or_pandas(pandas.Series([1, 2, 3])))
                    is np.ndarray)
            assert (type(asarray_or_pandas(pandas.DataFrame([[1, 2, 3]])))
                    is np.ndarray)
        finally:
            have_pandas = had_pandas

# Like np.atleast_2d, but this converts lower-dimensional arrays into columns,
# instead of rows. It also converts ndarray subclasses into basic ndarrays,
# which makes it easier to guarantee correctness. However, there are many
# places in the code where we want to preserve pandas indexing information if
# present, so there is also an option 
def atleast_2d_column_default(a, preserve_pandas=False):
    if preserve_pandas and have_pandas:
        if isinstance(a, pandas.Series):
            return pandas.DataFrame(a)
        elif isinstance(a, pandas.DataFrame):
            return a
        # fall through
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

    global have_pandas
    if have_pandas:
        assert (type(atleast_2d_column_default(pandas.Series([1, 2])))
                == np.ndarray)
        assert (type(atleast_2d_column_default(pandas.DataFrame([[1], [2]])))
                == np.ndarray)
        assert (type(atleast_2d_column_default(pandas.Series([1, 2]),
                                               preserve_pandas=True))
                == pandas.DataFrame)
        assert (type(atleast_2d_column_default(pandas.DataFrame([[1], [2]]),
                                               preserve_pandas=True))
                == pandas.DataFrame)
        s = pandas.Series([10, 11,12], name="hi", index=["a", "b", "c"])
        df = atleast_2d_column_default(s, preserve_pandas=True)
        assert isinstance(df, pandas.DataFrame)
        assert np.all(df.columns == ["hi"])
        assert np.all(df.index == ["a", "b", "c"])
    assert (type(atleast_2d_column_default(np.matrix(1),
                                           preserve_pandas=True))
            == np.ndarray)
    assert (type(atleast_2d_column_default([1, 2, 3],
                                           preserve_pandas=True))
            == np.ndarray)
        
    if have_pandas:
        had_pandas = have_pandas
        try:
            have_pandas = False
            assert (type(atleast_2d_column_default(pandas.Series([1, 2]),
                                                   preserve_pandas=True))
                    == np.ndarray)
            assert (type(atleast_2d_column_default(pandas.DataFrame([[1], [2]]),
                                                   preserve_pandas=True))
                    == np.ndarray)
        finally:
            have_pandas = had_pandas

def is_valid_python_varname(name):
    # see: http://stackoverflow.com/a/10134719/809705
    return (isinstance(name, str) and
            re.match(r"^[^\d\W]\w*\Z", name) is not None)

def test_is_valid_python_varname():
    tests = {"a": True,
             "a1": True,
             "_a1": True,
             "1a": False,
             "a.b": False,
             1: False}
    for k, v in tests.iteritems():
        assert is_valid_python_varname(k) == v

# A version of .reshape() that knows how to down-convert a 1-column
# pandas.DataFrame into a pandas.Series. Useful for code that wants to be
# agnostic between 1d and 2d data, with the pattern:
#   new_a = atleast_2d_column_default(a, preserve_pandas=True)
#   # do stuff to new_a, which can assume it's always 2 dimensional
#   return pandas_friendly_reshape(new_a, a.shape)
def pandas_friendly_reshape(a, new_shape):
    if not have_pandas:
        return a.reshape(new_shape)
    if not isinstance(a, pandas.DataFrame):
        return a.reshape(new_shape)
    # we have a DataFrame. Only supported reshapes are no-op, and
    # single-column DataFrame -> Series.
    if new_shape == a.shape:
        return a
    if len(new_shape) == 1 and a.shape[1] == 1:
        if new_shape[0] != a.shape[0]:
            raise ValueError, "arrays have incompatible sizes"
        return a[a.columns[0]]
    raise ValueError("cannot reshape a DataFrame with shape %s to shape %s"
                     % (a.shape, new_shape))

def test_pandas_friendly_reshape():
    from nose.tools import assert_raises
    global have_pandas
    assert np.allclose(pandas_friendly_reshape(np.arange(10).reshape(5, 2),
                                               (2, 5)),
                       np.arange(10).reshape(2, 5))
    if have_pandas:
        df = pandas.DataFrame({"x": [1, 2, 3]}, index=["a", "b", "c"])
        noop = pandas_friendly_reshape(df, (3, 1))
        assert isinstance(noop, pandas.DataFrame)
        assert np.array_equal(noop.index, ["a", "b", "c"])
        assert np.array_equal(noop.columns, ["x"])
        squozen = pandas_friendly_reshape(df, (3,))
        assert isinstance(squozen, pandas.Series)
        assert np.array_equal(squozen.index, ["a", "b", "c"])
        assert squozen.name == "x"

        assert_raises(ValueError, pandas_friendly_reshape, df, (4,))
        assert_raises(ValueError, pandas_friendly_reshape, df, (1, 3))
        assert_raises(ValueError, pandas_friendly_reshape, df, (3, 3))

        had_pandas = have_pandas
        try:
            have_pandas = False
            # this will try to do a reshape directly, and DataFrames *have* no
            # reshape method
            assert_raises(AttributeError, pandas_friendly_reshape, df, (3,))
        finally:
            have_pandas = had_pandas

def uniqueify_list(seq):
    seq_new = []
    seen = set()
    for obj in seq:
        if obj not in seen:
            seq_new.append(obj)
            seen.add(obj)
    return seq_new

def test_to_uniqueify_list():
    assert uniqueify_list([1, 2, 3]) == [1, 2, 3]
    assert uniqueify_list([1, 3, 3, 2, 3, 1]) == [1, 3, 2]
    assert uniqueify_list([3, 2, 1, 4, 1, 2, 3]) == [3, 2, 1, 4]

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

def repr_pretty_delegate(obj):
    # If IPython is already loaded, then might as well use it. (Most commonly
    # this will occur if we are in an IPython session, but somehow someone has
    # called repr() directly. This can happen for example if printing an
    # container like a namedtuple that IPython lacks special code for
    # pretty-printing.)  But, if IPython is not already imported, we do not
    # attempt to import it. This makes patsy itself faster to import (as of
    # Nov. 2012 I measured the extra overhead from loading IPython as ~4
    # seconds on a cold cache), it prevents IPython from automatically
    # spawning a bunch of child processes (!) which may not be what you want
    # if you are not otherwise using IPython, and it avoids annoying the
    # pandas people who have some hack to tell whether you are using IPython
    # in their test suite (see patsy bug #12).
    if optional_dep_ok and "IPython" in sys.modules:
        from IPython.lib.pretty import pretty
        return pretty(obj)
    else:
        return _mini_pretty(obj)

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

# NaN checking functions that work on arbitrary objects, on old Python
# versions (math.isnan is only in 2.6+), etc.
def safe_scalar_isnan(x):
    try:
        return np.isnan(float(x))
    except (TypeError, ValueError, NotImplementedError):
        return False
safe_isnan = np.vectorize(safe_scalar_isnan, otypes=[bool])

def test_safe_scalar_isnan():
    assert not safe_scalar_isnan(True)
    assert not safe_scalar_isnan(None)
    assert not safe_scalar_isnan("sadf")
    assert not safe_scalar_isnan((1, 2, 3))
    assert not safe_scalar_isnan(np.asarray([1, 2, 3]))
    assert not safe_scalar_isnan([np.nan])
    assert safe_scalar_isnan(np.nan)
    assert safe_scalar_isnan(np.float32(np.nan))
    assert safe_scalar_isnan(float(np.nan))

def test_safe_isnan():
    assert np.array_equal(safe_isnan([1, True, None, np.nan, "asdf"]),
                          [False, False, False, True, False])
    assert safe_isnan(np.nan).ndim == 0
    assert safe_isnan(np.nan)
    assert not safe_isnan(None)
    # raw isnan raises a *different* error for strings than for objects:
    assert not safe_isnan("asdf")
    
def iterable(obj):
    try:
        iter(obj)
    except Exception:
        return False
    return True

def test_iterable():
    assert iterable("asdf")
    assert iterable([])
    assert iterable({"a": 1})
    assert not iterable(1)
    assert not iterable(iterable)
