# This file is part of Charlton
# Copyright (C) 2011 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

# This file defines a 'value-added' model matrix type -- a subclass of ndarray
# that represents a model matrix and holds metadata about its columns.  The
# intent is that this is a useful and usable data structure even if you're not
# using *any* of the rest of charlton to actually build the matrix.

__all__ = ["ModelMatrixColumnInfo", "ModelMatrix", "model_matrix"]

import numpy as np
from charlton import CharltonError
from charlton.util import atleast_2d_column_default
from charlton.compat import OrderedDict

class ModelMatrixColumnInfo(object):
    # term_name_to_columns and term_to_columns are separate in case someone
    # wants to make a ModelMatrix that isn't derived from a ModelDesc, and
    # thus has names, but not Term objects.
    def __init__(self, column_names,
                 term_slices=None, term_name_slices=None):
        self.column_name_indexes = OrderedDict(zip(column_names,
                                                   range(len(column_names))))
        if term_slices is not None:
            assert term_name_slices is None
            term_names = [term.name() for term in term_slices]
            term_name_slices = OrderedDict(zip(term_names,
                                               term_slices.values()))
        else: # term_slices is None
            if term_name_slices is None:
                # Make up one term per column
                term_names = column_names
                slices = [slice(i, i + 1) for i in xrange(len(column_names))]
                term_name_to_slice = OrderedDict(zip(term_names, slices))
        self.term_slices = term_slices
        self.term_name_slices = term_name_slices

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

    def index(self, column_specifier):
        """Take anything (raw indices, terms, term names, column names...) and
        return something that can be used as an index into the model matrix
        ndarray."""
        if np.issubsctype(column_specifier, (np.integer, np.bool_)):
            return column_specifier
        if column_specifier in self.term_slices:
            return self.term_slices[column_specifier]
        if column_specifier in self.term_name_slices:
            return self.term_name_slices[column_specifier]
        if column_specifier in self.column_name_indexes:
            return self.column_name_indexes[column_specifier]
        raise CharltonError("unknown column specified '%s'"
                            % (column_specifier,))

    def linear_constraint(self, constraint_likes):
        from charlton.constraint import linear_constraint
        return linear_constraint(constraint_likes, self.column_names)

def test_ModelMatrixColumnInfo():
    t_a = object()
    t_b = object()
    ci = ModelMatrixColumnInfo(["a1", "a2", "a3", "b"],
                               {"a": (0, 3), "b": (3, 4)},
                               {t_a: (0, 3), t_b: (3, 4)})
    assert ci.column_name_to_column == {"a1": 0, "a2": 1, "a3": 2, "b": 3}
    assert np.all(ci.index(1) == [1])
    assert np.all(ci.index([1]) == [1])
    assert np.all(ci.index([True, False, True, True])
                  == [True, False, True, True])
    assert np.all(ci.index("a") == [0, 1, 2])
    assert np.all(ci.index(["a"]) == [0, 1, 2])
    assert np.all(ci.index(t_a) == [0, 1, 2])
    assert np.all(ci.index([t_a]) == [0, 1, 2])
    assert np.all(ci.index("b") == [3])
    assert np.all(ci.index(["b"]) == [3])
    assert np.all(ci.index(t_b) == [3])
    assert np.all(ci.index([t_b]) == [3])
    assert np.all(ci.index(["a2"]) == [1])
    assert np.all(ci.index(["b", "a2"]) == [3, 1])
    assert np.all(ci.index([t_b, "a1"]) == [3, 0])

# http://docs.scipy.org/doc/numpy/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array
try:
    import pandas
except ImportError:
    have_pandas = False
else:
    have_pandas = True
class ModelMatrix(np.ndarray):
    def __new__(cls, input_array, column_info=None):
        self = atleast_2d_column_default(input_array).view(cls)
        if column_info is None:
            names = ["column%s" % (i,) for i in xrange(self.shape[1])]
            column_info = ModelMatrixColumnInfo(column_names)
        self.column_info = column_info
        return self

    # 'use_pandas' argument makes testing easier
    def __repr__(self, use_pandas=True):
        if have_pandas and use_pandas:
            df = pandas.DataFrame(self, columns=self.column_info.column_names)
            matrix_repr = "ModelMatrix:\n" + repr(df)
        else:
            numbers = np.array2string(self, precision=2, separator=", ",
                                      prefix=self.__class__.__name__)
            matrix_repr = ("ModelMatrix(%s)\n"
                           "columns are: %r"
                           % (numbers, self.column_info.column_names))
        term_reprs = []
        term_name_columns = self.column_info.term_name_to_columns.items()
        term_name_columns.sort(key=lambda item: item[1])
        for term_name, (low, high) in term_name_columns:
            term_reprs.append("Term %s: " % (term_name,))
            if high - low == 1:
                term_reprs.append("column %s\n" % (low,))
            else:
                term_reprs.append("columns %s-%s\n" % (low, high - 1))
        return matrix_repr + "\n" + "".join(term_reprs)

    # No __array_finalize__ method, because we don't want slices of this
    # object to keep the column_info (they may have different columns!), or
    # anything fancy like that.

def model_matrix(input_array, *column_info_args, **column_info_kwargs):
    input_array = np.asarray(input_array)
    return ModelMatrix(input_array, ci)

def test_model_matrix():
    mm = model_matrix([[12, 14, 16, 18]],
                      ["a1", "a2", "a3", "b"],
                      term_name_to_columns={"a": (0, 3), "b": (3, 4)})
    assert np.all(mm.column_info.index(["a1", "a3"]) == [0, 2])
    mm2 = model_matrix([[12, 14, 16, 18]])
    assert np.all(mm2.column_info.index([1, 3]) == [1, 3])
    assert np.all(mm2.column_info.index(["column1", "column3"]) == [1, 3])

    con = mm.linear_constraint(["2 * a1 = b + 1", "a3"])
    assert con.variable_names == ["a1", "a2", "a3", "b"]
    assert np.all(con.coefs == [[2, 0, 0, -1], [0, 0, 1, 0]])
    assert np.all(con.constants == [[1], [0]])
