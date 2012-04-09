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

class ModelMatrixColumnInfo(object):
    def __init__(self, column_names=[], term_to_columns={}):
        self.column_names = column_names
        self.term_to_columns = term_to_columns
        term_names = [term.name() for term in term_to_columns.iterkeys()]
        self.term_name_to_columns = dict(zip(term_names, term_to_columns.values()))
        self.column_name_to_column = {}
        for i, name in enumerate(self.column_names):
            self.column_name_to_column[name] = i

    def index(self, column_specifier):
        """Take anything (raw indices, term names, column names...) and return
        something that can be used as an index into the model matrix
        ndarray."""
        column_specifier = np.atleast_1d(column_specifier)
        if np.issubdtype(column_specifier.dtype, int):
            return column_specifier
        if column_specifier.dtype.kind == "b":
            return column_specifier
        columns = []
        for name in column_specifier:
            if name in self.term_to_columns:
                columns += range(*self.term_to_columns[name])
            elif name in self.term_name_to_columns:
                columns += range(*self.term_name_to_columns[name])
            elif name in self.column_name_to_column:
                columns.append(self.column_name_to_column[name])
            else:
                raise CharltonError("unknown column specifier '%s'" % (name,))
        return columns

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
        self = np.asarray(input_array).view(cls)
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
    ci = ModelMatrixColumnInfo(*column_info_args, **column_info_kwargs)
    if not ci.column_names:
        names = ["column%s" % (i,) for i in xrange(input_array.shape[1])]
        ci = ModelMatrixColumnInfo(names)
    return ModelMatrix(input_array, ci)

def test_model_matrix():
    mm = model_matrix([[12, 14, 16, 18]],
                      ["a1", "a2", "a3", "b"],
                      {"a": (0, 3), "b": (3, 4)})
    assert np.all(mm.column_info.index(["a1", "a3"]) == [0, 2])
    mm2 = model_matrix([[12, 14, 16, 18]])
    assert np.all(mm2.column_info.index([1, 3]) == [1, 3])
    assert np.all(mm2.column_info.index(["column1", "column3"]) == [1, 3])
