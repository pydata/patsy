# This file is part of Charlton
# Copyright (C) 2011 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

# This file defines a 'value-added' design matrix type -- a subclass of
# ndarray that represents a design matrix and holds metadata about its
# columns.  The intent is that this is a useful and usable data structure even
# if you're not using *any* of the rest of charlton to actually build the
# matrix.

# These are made available in the charlton.* namespace
__all__ = ["DesignMatrixColumnInfo", "DesignMatrix"]

import numpy as np
from charlton import CharltonError
from charlton.util import atleast_2d_column_default
from charlton.compat import OrderedDict

class DesignMatrixColumnInfo(object):
    # term_name_to_columns and term_to_columns are separate in case someone
    # wants to make a DesignMatrix that isn't derived from a ModelDesc, and
    # thus has names, but not Term objects.
    def __init__(self, column_names,
                 term_slices=None, term_name_slices=None):
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

def test_DesignMatrixColumnInfo():
    class _MockTerm(object):
        def __init__(self, name):
            self._name = name

        def name(self):
            return self._name
    t_a = _MockTerm("a")
    t_b = _MockTerm("b")
    ci = DesignMatrixColumnInfo(["a1", "a2", "a3", "b"],
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

    # One without term objects
    ci = DesignMatrixColumnInfo(["a1", "a2", "a3", "b"],
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
    ci = DesignMatrixColumnInfo(["a1", "a2", "a3", "b"])
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

    from nose.tools import assert_raises
    # Can't specify both term_slices and term_name_slices
    assert_raises(ValueError,
                  DesignMatrixColumnInfo,
                  ["a1", "a2"],
                  term_slices=[(t_a, slice(0, 2))],
                  term_name_slices=[("a", slice(0, 2))])
    # out-of-order slices are bad
    assert_raises(ValueError, DesignMatrixColumnInfo, ["a1", "a2", "a3", "a4"],
                  term_slices=[(t_a, slice(3, 4)), (t_b, slice(0, 3))])
    # gaps in slices are bad
    assert_raises(ValueError, DesignMatrixColumnInfo, ["a1", "a2", "a3", "a4"],
                  term_slices=[(t_a, slice(0, 2)), (t_b, slice(3, 4))])
    assert_raises(ValueError, DesignMatrixColumnInfo, ["a1", "a2", "a3", "a4"],
                  term_slices=[(t_a, slice(1, 3)), (t_b, slice(3, 4))])
    assert_raises(ValueError, DesignMatrixColumnInfo, ["a1", "a2", "a3", "a4"],
                  term_slices=[(t_a, slice(1, 2)), (t_b, slice(2, 3))])
    # overlapping slices ditto
    assert_raises(ValueError, DesignMatrixColumnInfo, ["a1", "a2", "a3", "a4"],
                  term_slices=[(t_a, slice(1, 3)), (t_b, slice(2, 4))])
    # no step arguments
    assert_raises(ValueError, DesignMatrixColumnInfo, ["a1", "a2", "a3", "a4"],
                  term_slices=[(t_a, slice(1, 3, 2)), (t_b, slice(3, 4))])
    # no term names that don't match column names
    assert_raises(ValueError, DesignMatrixColumnInfo, ["a1", "a2", "a3", "a4"],
                  term_name_slices=[("a1", slice(1, 3, 2)), ("b", slice(3, 4))])

def test_lincon():
    ci = DesignMatrixColumnInfo(["a1", "a2", "a3", "b"],
                               term_name_slices=[("a", slice(0, 3)),
                                                 ("b", slice(3, 4))])
    con = ci.linear_constraint(["2 * a1 = b + 1", "a3"])
    assert con.variable_names == ["a1", "a2", "a3", "b"]
    assert np.all(con.coefs == [[2, 0, 0, -1], [0, 0, 1, 0]])
    assert np.all(con.constants == [[1], [0]])

# http://docs.scipy.org/doc/numpy/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array
try:
    import pandas
except ImportError:
    have_pandas = False
else:
    have_pandas = True
class DesignMatrix(np.ndarray):
    def __new__(cls, input_array, column_info=None, builder=None):
        # Pass through existing DesignMatrixes. The column_info check is
        # necessary because numpy is sort of annoying and cannot be stopped
        # from turning non-design-matrix arrays into DesignMatrix
        # instances. (E.g., my_dm.diagonal() will return a DesignMatrix
        # object, but one without a column_info attribute.)
        if (isinstance(input_array, DesignMatrix)
            and hasattr(input_array, "column_info")):
            return input_array
        self = atleast_2d_column_default(input_array).view(cls)
        if self.ndim > 2:
            raise ValueError, "DesignMatrix must be 2d"
        assert self.ndim == 2
        if column_info is None:
            column_names = ["column%s" % (i,) for i in xrange(self.shape[1])]
            column_info = DesignMatrixColumnInfo(column_names)
        if len(column_info.column_names) != self.shape[1]:
            raise ValueError("wrong number of column names for design matrix "
                             "(got %s, wanted %s)"
                             % (len(column_info.column_names), self.shape[1]))
        self.column_info = column_info
        self.builder = builder
        return self

    # 'use_pandas' argument makes testing easier
    def __repr__(self, use_pandas=True):
        if have_pandas and use_pandas:
            df = pandas.DataFrame(self, columns=self.column_info.column_names)
            matrix_repr = "DesignMatrix:\n" + repr(df)
        else:
            numbers = np.array2string(self, precision=2, separator=", ",
                                      prefix=self.__class__.__name__)
            matrix_repr = ("DesignMatrix(%s)\n"
                           "columns are: %r"
                           % (numbers, self.column_info.column_names))
        term_reprs = []
        for term_name, span in self.column_info.term_name_slices.iteritems():
            term_reprs.append("Term %s: " % (term_name,))
            if span.stop - span.start == 1:
                term_reprs.append("column %s\n" % (span.start,))
            else:
                term_reprs.append("columns %s:%s\n" % (span.start, span.stop))
        return matrix_repr + "\n" + "".join(term_reprs)

    # No __array_finalize__ method, because we don't want slices of this
    # object to keep the column_info (they may have different columns!), or
    # anything fancy like that.

def test_design_matrix():
    from nose.tools import assert_raises

    ci = DesignMatrixColumnInfo(["a1", "a2", "a3", "b"],
                               term_name_slices=[("a", slice(0, 3)),
                                                 ("b", slice(3, 4))])
    mm = DesignMatrix([[12, 14, 16, 18]], ci)
    assert mm.column_info.column_names == ["a1", "a2", "a3", "b"]

    bad_ci = DesignMatrixColumnInfo(["a1"])
    assert_raises(ValueError, DesignMatrix, [[12, 14, 16, 18]], bad_ci)

    mm2 = DesignMatrix([[12, 14, 16, 18]])
    assert mm2.column_info.column_names == ["column0", "column1", "column2",
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

    # Just a smoke test
    repr(mm)
    mm.__repr__(use_pandas=False)
