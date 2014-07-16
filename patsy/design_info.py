# This file is part of Patsy
# Copyright (C) 2011-2012 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

# This file defines the main class for storing metadata about a model
# design. It also defines a 'value-added' design matrix type -- a subclass of
# ndarray that represents a design matrix and holds metadata about its
# columns.  The intent is that these are useful and usable data structures
# even if you're not using *any* of the rest of patsy to actually build
# your matrices.

from __future__ import print_function

# These are made available in the patsy.* namespace
__all__ = ["DesignInfo", "DesignMatrix"]

import six
import numpy as np
from patsy import PatsyError
from patsy.util import atleast_2d_column_default
from patsy.compat import OrderedDict
from patsy.util import repr_pretty_delegate, repr_pretty_impl
from patsy.constraint import linear_constraint

class DesignInfo(object):
    """A DesignInfo object holds metadata about a design matrix.

    This is the main object that Patsy uses to pass information to
    statistical libraries. Usually encountered as the `.design_info` attribute
    on design matrices.
    """
    def __init__(self, column_names,
                 term_slices=None, term_name_slices=None,
                 builder=None):
        self.column_name_indexes = OrderedDict(zip(column_names,
                                                   range(len(column_names))))
        if term_slices is not None:
            #: An OrderedDict mapping :class:`Term` objects to Python
            #: func:`slice` objects. May be None, for design matrices which
            #: were constructed directly rather than by using the patsy
            #: machinery. If it is not None, then it
            #: is guaranteed to list the terms in order, and the slices are
            #: guaranteed to exactly cover all columns with no overlap or
            #: gaps.
            self.term_slices = OrderedDict(term_slices)
            if term_name_slices is not None:
                raise ValueError("specify only one of term_slices and "
                                 "term_name_slices")
            term_names = [term.name() for term in self.term_slices]
            #: And OrderedDict mapping term names (as strings) to Python
            #: :func:`slice` objects. Guaranteed never to be None. Guaranteed
            #: to list the terms in order, and the slices are
            #: guaranteed to exactly cover all columns with no overlap or
            #: gaps. Name overlap is allowed between term names and column
            #: names, but it is guaranteed that if it occurs, then they refer
            #: to exactly the same column.
            self.term_name_slices = OrderedDict(zip(term_names,
                                                    self.term_slices.values()))
        else: # term_slices is None
            self.term_slices = None
            if term_name_slices is None:
                # Make up one term per column
                term_names = column_names
                slices = [slice(i, i + 1) for i in range(len(column_names))]
                term_name_slices = zip(term_names, slices)
            self.term_name_slices = OrderedDict(term_name_slices)

        self.builder = builder

        # Guarantees:
        #   term_name_slices is never None
        #   The slices in term_name_slices are in order and exactly cover the
        #     whole range of columns.
        #   term_slices may be None
        #   If term_slices is not None, then its slices match the ones in
        #     term_name_slices.
        #   If there is any name overlap between terms and columns, they refer
        #     to the same columns.
        assert self.term_name_slices is not None
        if self.term_slices is not None:
            assert (list(self.term_slices.values())
                    == list(self.term_name_slices.values()))
        covered = 0
        for slice_ in six.itervalues(self.term_name_slices):
            start, stop, step = slice_.indices(len(column_names))
            if start != covered:
                raise ValueError("bad term slices")
            if step != 1:
                raise ValueError("bad term slices")
            covered = stop
        if covered != len(column_names):
            raise ValueError("bad term indices")
        for column_name, index in six.iteritems(self.column_name_indexes):
            if column_name in self.term_name_slices:
                slice_ = self.term_name_slices[column_name]
                if slice_ != slice(index, index + 1):
                    raise ValueError("term/column name collision")

    __repr__ = repr_pretty_delegate
    def _repr_pretty_(self, p, cycle):
        assert not cycle
        if self.term_slices is None:
            kwargs = [("term_name_slices", self.term_name_slices)]
        else:
            kwargs = [("term_slices", self.term_slices)]
        if self.builder is not None:
            kwargs.append(("builder", self.builder))
        repr_pretty_impl(p, self, [self.column_names], kwargs)

    @property
    def column_names(self):
        "A list of the column names, in order."
        return list(self.column_name_indexes)

    @property
    def terms(self):
        "A list of :class:`Terms`, in order, or else None."
        if self.term_slices is None:
            return None
        return list(self.term_slices)

    @property
    def term_names(self):
        "A list of terms, in order."
        return list(self.term_name_slices)

    def slice(self, columns_specifier):
        """Locate a subset of design matrix columns, specified symbolically.

        A patsy design matrix has two levels of structure: the individual
        columns (which are named), and the :ref:`terms <formulas>` in
        the formula that generated those columns. This is a one-to-many
        relationship: a single term may span several columns. This method
        provides a user-friendly API for locating those columns.

        (While we talk about columns here, this is probably most useful for
        indexing into other arrays that are derived from the design matrix,
        such as regression coefficients or covariance matrices.)

        The `columns_specifier` argument can take a number of forms:

        * A term name
        * A column name
        * A :class:`Term` object
        * An integer giving a raw index
        * A raw slice object

        In all cases, a Python :func:`slice` object is returned, which can be
        used directly for indexing.

        Example::

          y, X = dmatrices("y ~ a", demo_data("y", "a", nlevels=3))
          betas = np.linalg.lstsq(X, y)[0]
          a_betas = betas[X.design_info.slice("a")]

        (If you want to look up a single individual column by name, use
        ``design_info.column_name_indexes[name]``.)
        """
        if isinstance(columns_specifier, slice):
            return columns_specifier
        if np.issubsctype(type(columns_specifier), np.integer):
            return slice(columns_specifier, columns_specifier + 1)
        if (self.term_slices is not None
            and columns_specifier in self.term_slices):
            return self.term_slices[columns_specifier]
        if columns_specifier in self.term_name_slices:
            return self.term_name_slices[columns_specifier]
        if columns_specifier in self.column_name_indexes:
            idx = self.column_name_indexes[columns_specifier]
            return slice(idx, idx + 1)
        raise PatsyError("unknown column specified '%s'"
                            % (columns_specifier,))

    def linear_constraint(self, constraint_likes):
        """Construct a linear constraint in matrix form from a (possibly
        symbolic) description.

        Possible inputs:

        * A dictionary which is taken as a set of equality constraint. Keys
          can be either string column names, or integer column indexes.
        * A string giving a arithmetic expression referring to the matrix
          columns by name.
        * A list of such strings which are ANDed together.
        * A tuple (A, b) where A and b are array_likes, and the constraint is
          Ax = b. If necessary, these will be coerced to the proper
          dimensionality by appending dimensions with size 1.

        The string-based language has the standard arithmetic operators, / * +
        - and parentheses, plus "=" is used for equality and "," is used to
        AND together multiple constraint equations within a string. You can
        If no = appears in some expression, then that expression is assumed to
        be equal to zero. Division is always float-based, even if
        ``__future__.true_division`` isn't in effect.

        Returns a :class:`LinearConstraint` object.

        Examples::

          di = DesignInfo(["x1", "x2", "x3"])

          # Equivalent ways to write x1 == 0:
          di.linear_constraint({"x1": 0})  # by name
          di.linear_constraint({0: 0})  # by index
          di.linear_constraint("x1 = 0")  # string based
          di.linear_constraint("x1")  # can leave out "= 0"
          di.linear_constraint("2 * x1 = (x1 + 2 * x1) / 3")
          di.linear_constraint(([1, 0, 0], 0))  # constraint matrices

          # Equivalent ways to write x1 == 0 and x3 == 10
          di.linear_constraint({"x1": 0, "x3": 10})
          di.linear_constraint({0: 0, 2: 10})
          di.linear_constraint({0: 0, "x3": 10})
          di.linear_constraint("x1 = 0, x3 = 10")
          di.linear_constraint("x1, x3 = 10")
          di.linear_constraint(["x1", "x3 = 0"])  # list of strings
          di.linear_constraint("x1 = 0, x3 - 10 = x1")
          di.linear_constraint([[1, 0, 0], [0, 0, 1]], [0, 10])

          # You can also chain together equalities, just like Python:
          di.linear_constraint("x1 = x2 = 3")
        """
        return linear_constraint(constraint_likes, self.column_names)

    def describe(self):
        """Returns a human-readable string describing this design info.

        Example:

        .. ipython::

          In [1]: y, X = dmatrices("y ~ x1 + x2", demo_data("y", "x1", "x2"))

          In [2]: y.design_info.describe()
          Out[2]: 'y'

          In [3]: X.design_info.describe()
          Out[3]: '1 + x1 + x2'

        .. warning::

           There is no guarantee that the strings returned by this
           function can be parsed as formulas. They are best-effort descriptions
           intended for human users.
        """

        names = []
        for name in self.term_names:
            if name == "Intercept":
                names.append("1")
            else:
                names.append(name)
        return " + ".join(names)

    @classmethod
    def from_array(cls, array_like, default_column_prefix="column"):
        """Find or construct a DesignInfo appropriate for a given array_like.

        If the input `array_like` already has a ``.design_info``
        attribute, then it will be returned. Otherwise, a new DesignInfo
        object will be constructed, using names either taken from the
        `array_like` (e.g., for a pandas DataFrame with named columns), or
        constructed using `default_column_prefix`.

        This is how :func:`dmatrix` (for example) creates a DesignInfo object
        if an arbitrary matrix is passed in.

        :arg array_like: An ndarray or pandas container.
        :arg default_column_prefix: If it's necessary to invent column names,
          then this will be used to construct them.
        :returns: a DesignInfo object
        """
        if hasattr(array_like, "design_info") and isinstance(array_like.design_info, cls):
            return array_like.design_info
        arr = atleast_2d_column_default(array_like, preserve_pandas=True)
        if arr.ndim > 2:
            raise ValueError("design matrix can't have >2 dimensions")
        columns = getattr(arr, "columns", range(arr.shape[1]))
        if (isinstance(columns, np.ndarray)
            and not np.issubdtype(columns.dtype, np.integer)):
            column_names = [str(obj) for obj in columns]
        else:
            column_names = ["%s%s" % (default_column_prefix, i)
                            for i in columns]
        return DesignInfo(column_names)

def test_DesignInfo():
    from nose.tools import assert_raises
    class _MockTerm(object):
        def __init__(self, name):
            self._name = name

        def name(self):
            return self._name
    t_a = _MockTerm("a")
    t_b = _MockTerm("b")
    di = DesignInfo(["a1", "a2", "a3", "b"],
                    [(t_a, slice(0, 3)), (t_b, slice(3, 4))],
                    builder="asdf")
    assert di.column_names == ["a1", "a2", "a3", "b"]
    assert di.term_names == ["a", "b"]
    assert di.terms == [t_a, t_b]
    assert di.column_name_indexes == {"a1": 0, "a2": 1, "a3": 2, "b": 3}
    assert di.term_name_slices == {"a": slice(0, 3), "b": slice(3, 4)}
    assert di.term_slices == {t_a: slice(0, 3), t_b: slice(3, 4)}
    assert di.describe() == "a + b"
    assert di.builder == "asdf"

    assert di.slice(1) == slice(1, 2)
    assert di.slice("a1") == slice(0, 1)
    assert di.slice("a2") == slice(1, 2)
    assert di.slice("a3") == slice(2, 3)
    assert di.slice("a") == slice(0, 3)
    assert di.slice(t_a) == slice(0, 3)
    assert di.slice("b") == slice(3, 4)
    assert di.slice(t_b) == slice(3, 4)
    assert di.slice(slice(2, 4)) == slice(2, 4)
    assert_raises(PatsyError, di.slice, "asdf")

    # smoke test
    repr(di)

    # One without term objects
    di = DesignInfo(["a1", "a2", "a3", "b"],
                    term_name_slices=[("a", slice(0, 3)),
                                      ("b", slice(3, 4))])
    assert di.column_names == ["a1", "a2", "a3", "b"]
    assert di.term_names == ["a", "b"]
    assert di.terms is None
    assert di.column_name_indexes == {"a1": 0, "a2": 1, "a3": 2, "b": 3}
    assert di.term_name_slices == {"a": slice(0, 3), "b": slice(3, 4)}
    assert di.term_slices is None
    assert di.describe() == "a + b"

    assert di.slice(1) == slice(1, 2)
    assert di.slice("a") == slice(0, 3)
    assert di.slice("a1") == slice(0, 1)
    assert di.slice("a2") == slice(1, 2)
    assert di.slice("a3") == slice(2, 3)
    assert di.slice("b") == slice(3, 4)

    # smoke test
    repr(di)

    # One without term objects *or* names
    di = DesignInfo(["a1", "a2", "a3", "b"])
    assert di.column_names == ["a1", "a2", "a3", "b"]
    assert di.term_names == ["a1", "a2", "a3", "b"]
    assert di.terms is None
    assert di.column_name_indexes == {"a1": 0, "a2": 1, "a3": 2, "b": 3}
    assert di.term_name_slices == {"a1": slice(0, 1),
                                   "a2": slice(1, 2),
                                   "a3": slice(2, 3),
                                   "b": slice(3, 4)}
    assert di.term_slices is None
    assert di.describe() == "a1 + a2 + a3 + b"

    assert di.slice(1) == slice(1, 2)
    assert di.slice("a1") == slice(0, 1)
    assert di.slice("a2") == slice(1, 2)
    assert di.slice("a3") == slice(2, 3)
    assert di.slice("b") == slice(3, 4)

    # Check intercept handling in describe()
    assert DesignInfo(["Intercept", "a", "b"]).describe() == "1 + a + b"

    # Can't specify both term_slices and term_name_slices
    assert_raises(ValueError,
                  DesignInfo,
                  ["a1", "a2"],
                  term_slices=[(t_a, slice(0, 2))],
                  term_name_slices=[("a", slice(0, 2))])
    # out-of-order slices are bad
    assert_raises(ValueError, DesignInfo, ["a1", "a2", "a3", "a4"],
                  term_slices=[(t_a, slice(3, 4)), (t_b, slice(0, 3))])
    # gaps in slices are bad
    assert_raises(ValueError, DesignInfo, ["a1", "a2", "a3", "a4"],
                  term_slices=[(t_a, slice(0, 2)), (t_b, slice(3, 4))])
    assert_raises(ValueError, DesignInfo, ["a1", "a2", "a3", "a4"],
                  term_slices=[(t_a, slice(1, 3)), (t_b, slice(3, 4))])
    assert_raises(ValueError, DesignInfo, ["a1", "a2", "a3", "a4"],
                  term_slices=[(t_a, slice(0, 2)), (t_b, slice(2, 3))])
    # overlapping slices ditto
    assert_raises(ValueError, DesignInfo, ["a1", "a2", "a3", "a4"],
                  term_slices=[(t_a, slice(0, 3)), (t_b, slice(2, 4))])
    # no step arguments
    assert_raises(ValueError, DesignInfo, ["a1", "a2", "a3", "a4"],
                  term_slices=[(t_a, slice(0, 4, 2))])
    # no term names that mismatch column names
    assert_raises(ValueError, DesignInfo, ["a1", "a2", "a3", "a4"],
                  term_name_slices=[("a1", slice(0, 3)), ("b", slice(3, 4))])

def test_DesignInfo_from_array():
    di = DesignInfo.from_array([1, 2, 3])
    assert di.column_names == ["column0"]
    di2 = DesignInfo.from_array([[1, 2], [2, 3], [3, 4]])
    assert di2.column_names == ["column0", "column1"]
    di3 = DesignInfo.from_array([1, 2, 3], default_column_prefix="x")
    assert di3.column_names == ["x0"]
    di4 = DesignInfo.from_array([[1, 2], [2, 3], [3, 4]],
                                default_column_prefix="x")
    assert di4.column_names == ["x0", "x1"]
    m = DesignMatrix([1, 2, 3], di3)
    assert DesignInfo.from_array(m) is di3
    # But weird objects are ignored
    m.design_info = "asdf"
    di_weird = DesignInfo.from_array(m)
    assert di_weird.column_names == ["column0"]

    from patsy.util import have_pandas
    if have_pandas:
        import pandas
        # with named columns
        di5 = DesignInfo.from_array(pandas.DataFrame([[1, 2]],
                                                     columns=["a", "b"]))
        assert di5.column_names == ["a", "b"]
        # with irregularly numbered columns
        di6 = DesignInfo.from_array(pandas.DataFrame([[1, 2]],
                                                     columns=[0, 10]))
        assert di6.column_names == ["column0", "column10"]
        # with .design_info attr
        df = pandas.DataFrame([[1, 2]])
        df.design_info = di6
        assert DesignInfo.from_array(df) is di6

def test_lincon():
    di = DesignInfo(["a1", "a2", "a3", "b"],
                    term_name_slices=[("a", slice(0, 3)),
                                      ("b", slice(3, 4))])
    con = di.linear_constraint(["2 * a1 = b + 1", "a3"])
    assert con.variable_names == ["a1", "a2", "a3", "b"]
    assert np.all(con.coefs == [[2, 0, 0, -1], [0, 0, 1, 0]])
    assert np.all(con.constants == [[1], [0]])

# Idea: format with a reasonable amount of precision, then if that turns out
# to be higher than necessary, remove as many zeros as we can. But only do
# this while we can do it to *all* the ordinarily-formatted numbers, to keep
# decimal points aligned.
def _format_float_column(precision, col):
    format_str = "%." + str(precision) + "f"
    assert col.ndim == 1
    # We don't want to look at numbers like "1e-5" or "nan" when stripping.
    simple_float_chars = set("+-0123456789.")
    col_strs = np.array([format_str % (x,) for x in col], dtype=object)
    # Really every item should have a decimal, but just in case, we don't want
    # to strip zeros off the end of "10" or something like that.
    mask = np.array([simple_float_chars.issuperset(col_str) and "." in col_str
                     for col_str in col_strs])
    mask_idxes = np.nonzero(mask)[0]
    strip_char = "0"
    if np.any(mask):
        while True:
            if np.all([s.endswith(strip_char) for s in col_strs[mask]]):
                for idx in mask_idxes:
                    col_strs[idx] = col_strs[idx][:-1]
            else:
                if strip_char == "0":
                    strip_char = "."
                else:
                    break
    return col_strs
            
def test__format_float_column():
    def t(precision, numbers, expected):
        got = _format_float_column(precision, np.asarray(numbers))
        print(got, expected)
        assert np.array_equal(got, expected)
    # This acts weird on old python versions (e.g. it can be "-nan"), so don't
    # hardcode it:
    nan_string = "%.3f" % (np.nan,)
    t(3, [1, 2.1234, 2.1239, np.nan], ["1.000", "2.123", "2.124", nan_string])
    t(3, [1, 2, 3, np.nan], ["1", "2", "3", nan_string])
    t(3, [1.0001, 2, 3, np.nan], ["1", "2", "3", nan_string])
    t(4, [1.0001, 2, 3, np.nan], ["1.0001", "2.0000", "3.0000", nan_string])

# http://docs.scipy.org/doc/numpy/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array
class DesignMatrix(np.ndarray):
    """A simple numpy array subclass that carries design matrix metadata.

    .. attribute:: design_info

       A :class:`DesignInfo` object containing metadata about this design
       matrix.

    This class also defines a fancy __repr__ method with labeled
    columns. Otherwise it is identical to a regular numpy ndarray.

    .. warning::

       You should never check for this class using
       :func:`isinstance`. Limitations of the numpy API mean that it is
       impossible to prevent the creation of numpy arrays that have type
       DesignMatrix, but that are not actually design matrices (and such
       objects will behave like regular ndarrays in every way). Instead, check
       for the presence of a ``.design_info`` attribute -- this will be
       present only on "real" DesignMatrix objects.
    """

    def __new__(cls, input_array, design_info=None,
                default_column_prefix="column"):
        """Create a DesignMatrix, or cast an existing matrix to a DesignMatrix.

        A call like::

          DesignMatrix(my_array)

        will convert an arbitrary array_like object into a DesignMatrix.

        The return from this function is guaranteed to be a two-dimensional
        ndarray with a real-valued floating point dtype, and a
        ``.design_info`` attribute which matches its shape. If the
        `design_info` argument is not given, then one is created via
        :meth:`DesignInfo.from_array` using the given
        `default_column_prefix`.

        Depending on the input array, it is possible this will pass through
        its input unchanged, or create a view.
        """
        # Pass through existing DesignMatrixes. The design_info check is
        # necessary because numpy is sort of annoying and cannot be stopped
        # from turning non-design-matrix arrays into DesignMatrix
        # instances. (E.g., my_dm.diagonal() will return a DesignMatrix
        # object, but one without a design_info attribute.)
        if (isinstance(input_array, DesignMatrix)
            and hasattr(input_array, "design_info")):
            return input_array
        self = atleast_2d_column_default(input_array).view(cls)
        # Upcast integer to floating point
        if np.issubdtype(self.dtype, np.integer):
            self = np.asarray(self, dtype=float).view(cls)
        if self.ndim > 2:
            raise ValueError("DesignMatrix must be 2d")
        assert self.ndim == 2
        if design_info is None:
            design_info = DesignInfo.from_array(self, default_column_prefix)
        if len(design_info.column_names) != self.shape[1]:
            raise ValueError("wrong number of column names for design matrix "
                             "(got %s, wanted %s)"
                             % (len(design_info.column_names), self.shape[1]))
        self.design_info = design_info
        if not np.issubdtype(self.dtype, np.floating):
            raise ValueError("design matrix must be real-valued floating point")
        return self

    __repr__ = repr_pretty_delegate
    def _repr_pretty_(self, p, cycle):
        if not hasattr(self, "design_info"):
            # Not a real DesignMatrix
            p.pretty(np.asarray(self))
            return
        assert not cycle

        # XX: could try calculating width of the current terminal window:
        #   http://stackoverflow.com/questions/566746/how-to-get-console-window-width-in-python
        # sadly it looks like ipython does not actually pass this information
        # in, even if we use _repr_pretty_ -- the pretty-printer object has a
        # fixed width it always uses. (As of IPython 0.12.)
        MAX_TOTAL_WIDTH = 78
        SEP = 2
        INDENT = 2
        MAX_ROWS = 30
        PRECISION = 5

        names = self.design_info.column_names
        column_name_widths = [len(name) for name in names]
        min_total_width = (INDENT + SEP * (self.shape[1] - 1)
                           + np.sum(column_name_widths))
        if min_total_width <= MAX_TOTAL_WIDTH:
            printable_part = np.asarray(self)[:MAX_ROWS, :]
            formatted_cols = [_format_float_column(PRECISION,
                                                   printable_part[:, i])
                              for i in range(self.shape[1])]
            column_num_widths = [max([len(s) for s in col])
                                 for col in formatted_cols]
            column_widths = [max(name_width, num_width)
                             for (name_width, num_width)
                             in zip(column_name_widths, column_num_widths)]
            total_width = (INDENT + SEP * (self.shape[1] - 1)
                           + np.sum(column_widths))
            print_numbers = (total_width < MAX_TOTAL_WIDTH)
        else:
            print_numbers = False   

        p.begin_group(INDENT, "DesignMatrix with shape %s" % (self.shape,))
        p.breakable("\n" + " " * p.indentation)
        if print_numbers:
            # We can fit the numbers on the screen
            sep = " " * SEP
            # list() is for Py3 compatibility
            for row in [names] + list(zip(*formatted_cols)):
                cells = [cell.rjust(width)
                         for (width, cell) in zip(column_widths, row)]
                p.text(sep.join(cells))
                p.text("\n" + " " * p.indentation)
            if MAX_ROWS < self.shape[0]:
                p.text("[%s rows omitted]" % (self.shape[0] - MAX_ROWS,))
                p.text("\n" + " " * p.indentation)
        else:
            p.begin_group(2, "Columns:")
            p.breakable("\n" + " " * p.indentation)
            p.pretty(names)
            p.end_group(2, "")
            p.breakable("\n" + " " * p.indentation)

        p.begin_group(2, "Terms:")
        p.breakable("\n" + " " * p.indentation)
        for term_name, span in six.iteritems(self.design_info.term_name_slices):
            if span.start != 0:
                p.breakable(", ")
            p.pretty(term_name)
            if span.stop - span.start == 1:
                coltext = "column %s" % (span.start,)
            else:
                coltext = "columns %s:%s" % (span.start, span.stop)
            p.text(" (%s)" % (coltext,))
        p.end_group(2, "")

        if not print_numbers or self.shape[0] > MAX_ROWS:
            # some data was not shown
            p.breakable("\n" + " " * p.indentation)
            p.text("(to view full data, use np.asarray(this_obj))")

        p.end_group(INDENT, "")

    # No __array_finalize__ method, because we don't want slices of this
    # object to keep the design_info (they may have different columns!), or
    # anything fancy like that.

def test_design_matrix():
    from nose.tools import assert_raises

    di = DesignInfo(["a1", "a2", "a3", "b"],
                    term_name_slices=[("a", slice(0, 3)),
                                      ("b", slice(3, 4))])
    mm = DesignMatrix([[12, 14, 16, 18]], di)
    assert mm.design_info.column_names == ["a1", "a2", "a3", "b"]

    bad_di = DesignInfo(["a1"])
    assert_raises(ValueError, DesignMatrix, [[12, 14, 16, 18]], bad_di)

    mm2 = DesignMatrix([[12, 14, 16, 18]])
    assert mm2.design_info.column_names == ["column0", "column1", "column2",
                                            "column3"]

    mm3 = DesignMatrix([12, 14, 16, 18])
    assert mm3.shape == (4, 1)

    # DesignMatrix always has exactly 2 dimensions
    assert_raises(ValueError, DesignMatrix, [[[1]]])

    # DesignMatrix constructor passes through existing DesignMatrixes
    mm4 = DesignMatrix(mm)
    assert mm4 is mm
    # But not if they are really slices:
    mm5 = DesignMatrix(mm.diagonal())
    assert mm5 is not mm

    mm6 = DesignMatrix([[12, 14, 16, 18]], default_column_prefix="x")
    assert mm6.design_info.column_names == ["x0", "x1", "x2", "x3"]

    # Only real-valued matrices can be DesignMatrixs
    assert_raises(ValueError, DesignMatrix, [1, 2, 3j])
    assert_raises(ValueError, DesignMatrix, ["a", "b", "c"])
    assert_raises(ValueError, DesignMatrix, [1, 2, object()])

    # Just smoke tests
    repr(mm)
    repr(DesignMatrix(np.arange(100)))
    repr(DesignMatrix(np.arange(100) * 2.0))
    repr(mm[1:, :])
    repr(DesignMatrix(np.arange(100).reshape((1, 100))))
    repr(DesignMatrix([np.nan, np.inf]))
    repr(DesignMatrix([np.nan, 0, 1e20, 20.5]))
