# This file is part of Charlton
# Copyright (C) 2011-2012 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

# This file defines a 'value-added' design matrix type -- a subclass of
# ndarray that represents a design matrix and holds metadata about its
# columns.  The intent is that this is a useful and usable data structure even
# if you're not using *any* of the rest of charlton to actually build the
# matrix.

# These are made available in the charlton.* namespace
__all__ = ["DesignInfo", "DesignMatrix"]

import numpy as np
from charlton import CharltonError
from charlton.util import atleast_2d_column_default
from charlton.compat import OrderedDict
from charlton.util import repr_pretty_delegate

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
    pass

class DesignInfo(object):
    # term_name_to_columns and term_to_columns are separate in case someone
    # wants to make a DesignMatrix that isn't derived from a ModelDesc, and
    # thus has names, but not Term objects.
    def __init__(self, column_names,
                 term_slices=None, term_name_slices=None,
                 builder=None):
        self.column_name_indexes = OrderedDict(zip(column_names,
                                                   range(len(column_names))))
        if term_slices is not None:
            self.term_slices = OrderedDict(term_slices)
            if term_name_slices is not None:
                raise ValueError("specify only one of term_slices and "
                                 "term_name_slices")
            term_names = [term.name() for term in self.term_slices]
            self.term_name_slices = OrderedDict(zip(term_names,
                                                    self.term_slices.values()))
        else: # term_slices is None
            self.term_slices = None
            if term_name_slices is None:
                # Make up one term per column
                term_names = column_names
                slices = [slice(i, i + 1) for i in xrange(len(column_names))]
                term_name_slices = zip(term_names, slices)
            self.term_name_slices = OrderedDict(term_name_slices)

        from charlton.build import DesignMatrixBuilder
        if builder is not None and not isinstance(builder, DesignMatrixBuilder):
            raise ValueError, "invalid builder"
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
            assert self.term_slices.values() == self.term_name_slices.values()
        covered = 0
        for slice_ in self.term_name_slices.itervalues():
            start, stop, step = slice_.indices(len(column_names))
            if start != covered:
                raise ValueError, "bad term slices"
            if step != 1:
                raise ValueError, "bad term slices"
            covered = stop
        if covered != len(column_names):
            raise ValueError, "bad term indices"
        for column_name, index in self.column_name_indexes.iteritems():
            if column_name in self.term_name_slices:
                slice_ = self.term_name_slices[column_name]
                if slice_ != slice(index, index + 1):
                    raise ValueError, "term/column name collision"

    @property
    def column_names(self):
        return self.column_name_indexes.keys()

    @property
    def terms(self):
        if self.term_slices is None:
            return None
        return self.term_slices.keys()

    @property
    def term_names(self):
        return self.term_name_slices.keys()

    def slice(self, column_specifier):
        """Take anything (raw indices, terms, term names, column names...) and
        return a slice object that can be used as an index into the model
        matrix ndarray."""
        if np.issubsctype(type(column_specifier), np.integer):
            return slice(column_specifier, column_specifier + 1)
        if (self.term_slices is not None
            and column_specifier in self.term_slices):
            return self.term_slices[column_specifier]
        if column_specifier in self.term_name_slices:
            return self.term_name_slices[column_specifier]
        if column_specifier in self.column_name_indexes:
            idx = self.column_name_indexes[column_specifier]
            return slice(idx, idx + 1)
        raise CharltonError("unknown column specified '%s'"
                            % (column_specifier,))

    def linear_constraint(self, constraint_likes):
        from charlton.constraint import linear_constraint
        return linear_constraint(constraint_likes, self.column_names)

def test_DesignInfo():
    from nose.tools import assert_raises
    class _MockTerm(object):
        def __init__(self, name):
            self._name = name

        def name(self):
            return self._name
    t_a = _MockTerm("a")
    t_b = _MockTerm("b")
    ci = DesignInfo(["a1", "a2", "a3", "b"],
                    [(t_a, slice(0, 3)), (t_b, slice(3, 4))])
    assert ci.column_names == ["a1", "a2", "a3", "b"]
    assert ci.term_names == ["a", "b"]
    assert ci.terms == [t_a, t_b]
    assert ci.column_name_indexes == {"a1": 0, "a2": 1, "a3": 2, "b": 3}
    assert ci.term_name_slices == {"a": slice(0, 3), "b": slice(3, 4)}
    assert ci.term_slices == {t_a: slice(0, 3), t_b: slice(3, 4)}

    assert ci.slice(1) == slice(1, 2)
    assert ci.slice("a1") == slice(0, 1)
    assert ci.slice("a2") == slice(1, 2)
    assert ci.slice("a3") == slice(2, 3)
    assert ci.slice("a") == slice(0, 3)
    assert ci.slice(t_a) == slice(0, 3)
    assert ci.slice("b") == slice(3, 4)
    assert ci.slice(t_b) == slice(3, 4)
    assert_raises(CharltonError, ci.slice, "asdf")

    # One without term objects
    ci = DesignInfo(["a1", "a2", "a3", "b"],
                    term_name_slices=[("a", slice(0, 3)),
                                      ("b", slice(3, 4))])
    assert ci.column_names == ["a1", "a2", "a3", "b"]
    assert ci.term_names == ["a", "b"]
    assert ci.terms is None
    assert ci.column_name_indexes == {"a1": 0, "a2": 1, "a3": 2, "b": 3}
    assert ci.term_name_slices == {"a": slice(0, 3), "b": slice(3, 4)}
    assert ci.term_slices is None

    assert ci.slice(1) == slice(1, 2)
    assert ci.slice("a") == slice(0, 3)
    assert ci.slice("a1") == slice(0, 1)
    assert ci.slice("a2") == slice(1, 2)
    assert ci.slice("a3") == slice(2, 3)
    assert ci.slice("b") == slice(3, 4)

    # One without term objects *or* names
    ci = DesignInfo(["a1", "a2", "a3", "b"])
    assert ci.column_names == ["a1", "a2", "a3", "b"]
    assert ci.term_names == ["a1", "a2", "a3", "b"]
    assert ci.terms is None
    assert ci.column_name_indexes == {"a1": 0, "a2": 1, "a3": 2, "b": 3}
    assert ci.term_name_slices == {"a1": slice(0, 1),
                                   "a2": slice(1, 2),
                                   "a3": slice(2, 3),
                                   "b": slice(3, 4)}
    assert ci.term_slices is None

    assert ci.slice(1) == slice(1, 2)
    assert ci.slice("a1") == slice(0, 1)
    assert ci.slice("a2") == slice(1, 2)
    assert ci.slice("a3") == slice(2, 3)
    assert ci.slice("b") == slice(3, 4)

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

def test_lincon():
    ci = DesignInfo(["a1", "a2", "a3", "b"],
                    term_name_slices=[("a", slice(0, 3)),
                                      ("b", slice(3, 4))])
    con = ci.linear_constraint(["2 * a1 = b + 1", "a3"])
    assert con.variable_names == ["a1", "a2", "a3", "b"]
    assert np.all(con.coefs == [[2, 0, 0, -1], [0, 0, 1, 0]])
    assert np.all(con.constants == [[1], [0]])

# http://docs.scipy.org/doc/numpy/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array
class DesignMatrix(np.ndarray):
    def __new__(cls, input_array, design_info=None,
                default_column_prefix="column"):
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
            raise ValueError, "DesignMatrix must be 2d"
        assert self.ndim == 2
        if design_info is None:
            column_names = ["%s%s" % (default_column_prefix, i)
                            for i in xrange(self.shape[1])]
            design_info = DesignInfo(column_names)
        if len(design_info.column_names) != self.shape[1]:
            raise ValueError("wrong number of column names for design matrix "
                             "(got %s, wanted %s)"
                             % (len(design_info.column_names), self.shape[1]))
        self.design_info = design_info
        if not np.issubdtype(self.dtype, np.floating):
            raise ValueError, "design matrix must be real-valued floating point"
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
                              for i in xrange(self.shape[1])]
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
        for term_name, span in self.design_info.term_name_slices.iteritems():
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

    ci = DesignInfo(["a1", "a2", "a3", "b"],
                    term_name_slices=[("a", slice(0, 3)),
                                      ("b", slice(3, 4))])
    mm = DesignMatrix([[12, 14, 16, 18]], ci)
    assert mm.design_info.column_names == ["a1", "a2", "a3", "b"]

    bad_ci = DesignInfo(["a1"])
    assert_raises(ValueError, DesignMatrix, [[12, 14, 16, 18]], bad_ci)

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
