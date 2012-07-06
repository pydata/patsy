# This file is part of Charlton
# Copyright (C) 2011 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

# car::Contrasts constructs contrasts with *easier to read names* than stats
#   treatment: type[T.prof], type[T.wc]
#   sum: S.
#   helmert: H.
# and I guess type[prof] for dummy coding? or type==prof? -- probably too hard
#   to read, the []'s set the level off more
#
# http://www.ats.ucla.edu/stat/r/library/contrast_coding.htm
# http://www.ats.ucla.edu/stat/sas/webbooks/reg/chapter5/sasreg5.htm

# These are made available in the charlton.* namespace
__all__ = ["ContrastMatrix", "Treatment", "Poly", "code_contrast_matrix",
           "Sum", "Helmert", "Diff"]

import sys
import numpy as np
from charlton import CharltonError
from charlton.compat import triu_indices, tril_indices, diag_indices

class ContrastMatrix(object):
    def __init__(self, matrix, column_suffixes):
        self.matrix = np.asarray(matrix)
        self.column_suffixes = column_suffixes
        assert self.matrix.shape[1] == len(column_suffixes)

# This always produces an object of the type that Python calls 'str' (whether
# that be a Python 2 string-of-bytes or a Python 3 string-of-unicode). It does
# *not* make any particular guarantees about being reversible or having other
# such useful programmatic properties -- it just produces something that will
# be nice for users to look at.
def _obj_to_readable_str(obj):
    if isinstance(obj, str):
        return obj
    elif sys.version_info >= (3,) and isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            return repr(obj)
    elif sys.version_info < (3,) and isinstance(obj, unicode):
        try:
            return obj.encode("ascii")
        except UnicodeEncodeError:
            return repr(obj)
    else:
        return repr(obj)

def test__obj_to_readable_str():
    def t(obj, expected):
        got = _obj_to_readable_str(obj)
        assert type(got) is str
        assert got == expected
    t(1, "1")
    t(1.0, "1.0")
    t("asdf", "asdf")
    t(u"asdf", "asdf")
    if sys.version_info >= (3,):
        # a utf-8 encoded euro-sign comes out as a real euro sign
        t(u"\u20ac".encode("utf-8"), "\u20ac")
        # but a iso-8859-15 euro sign can't be decoded, and we fall back on
        # repr()
        t(u"\u20ac".encode("iso-8859-15"), "b'\\xa4'")
    else:
        t(u"\u20ac", "u'\\u20ac'")

def _name_levels(prefix, levels):
    return ["[%s%s]" % (prefix, _obj_to_readable_str(level)) for level in levels]

def test__name_levels():
    assert _name_levels("a", ["b", "c"]) == ["[ab]", "[ac]"]

def _dummy_code(levels):
    return ContrastMatrix(np.eye(len(levels)), _name_levels("", levels))

class Treatment(object):
    def __init__(self, base=0):
        self.base = base

    def code_with_intercept(self, levels):
        return _dummy_code(levels)

    def code_without_intercept(self, levels):
        base = self.base
        if base < 0:
            base += len(levels)
        eye = np.eye(len(levels) - 1)
        contrasts = np.vstack((eye[:base, :],
                                np.zeros((1, len(levels) - 1)),
                                eye[base:, :]))
        names = _name_levels("T.", levels[:base] + levels[base + 1:])
        return ContrastMatrix(contrasts, names)

def test_Treatment():
    t1 = Treatment()
    matrix = t1.code_with_intercept(["a", "b", "c"])
    assert matrix.column_suffixes == ["[a]", "[b]", "[c]"]
    assert np.allclose(matrix.matrix, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    matrix = t1.code_without_intercept(["a", "b", "c"])
    assert matrix.column_suffixes == ["[T.b]", "[T.c]"]
    assert np.allclose(matrix.matrix, [[0, 0], [1, 0], [0, 1]])
    matrix = Treatment(base=1).code_without_intercept(["a", "b", "c"])
    assert matrix.column_suffixes == ["[T.a]", "[T.c]"]
    assert np.allclose(matrix.matrix, [[1, 0], [0, 0], [0, 1]])
    matrix = Treatment(base=-2).code_without_intercept(["a", "b", "c"])
    assert matrix.column_suffixes == ["[T.a]", "[T.c]"]
    assert np.allclose(matrix.matrix, [[1, 0], [0, 0], [0, 1]])


class Poly(object):
    def __init__(self, scores=None):
        self.scores = scores

    def _code_either(self, intercept, levels):
        n = len(levels)
        scores = self.scores
        if scores is None:
            scores = np.arange(n)
        scores = np.asarray(scores, dtype=float)
        if len(scores) != n:
            raise CharltonError("number of levels (%s) does not match"
                                " number of scores (%s)"
                                % (n, len(scores)))
        # Strategy: just make a matrix whose columns are naive linear,
        # quadratic, etc., functions of the raw scores, and then use 'qr' to
        # orthogonalize each column against those to its left.
        scores -= scores.mean()
        raw_poly = scores.reshape((-1, 1)) ** np.arange(n).reshape((1, -1))
        q, r = np.linalg.qr(raw_poly)
        q *= np.sign(np.diag(r))
        q /= np.sqrt(np.sum(q ** 2, axis=1))
        names = [".Constant", ".Linear", ".Quadratic", ".Cubic"]
        names += ["^%s" % (i,) for i in xrange(4, n)]
        names = names[:n]
        if intercept:
            return ContrastMatrix(q, names)
        else:
            # We always include the constant/intercept column as something to
            # orthogonalize against, but we don't always return it:
            return ContrastMatrix(q[:, 1:], names[1:])

    def code_with_intercept(self, levels):
        return self._code_either(True, levels)

    def code_without_intercept(self, levels):
        return self._code_either(False, levels)

def test_Poly():
    t1 = Poly()
    matrix = t1.code_with_intercept(["a", "b", "c"])
    assert matrix.column_suffixes == [".Constant", ".Linear", ".Quadratic"]
    # Values from R 'options(digits=15); contr.poly(3)'
    expected = [[1./3 ** (0.5), -7.07106781186548e-01, 0.408248290463863],
                [1./3 ** (0.5), 0, -0.816496580927726],
                [1./3 ** (0.5), 7.07106781186547e-01, 0.408248290463863]]
    print matrix.matrix
    assert np.allclose(matrix.matrix, expected)
    matrix = t1.code_without_intercept(["a", "b", "c"])
    assert matrix.column_suffixes == [".Linear", ".Quadratic"]
    # Values from R 'options(digits=15); contr.poly(3)'
    print matrix.matrix
    assert np.allclose(matrix.matrix,
                       [[-7.07106781186548e-01, 0.408248290463863],
                        [0, -0.816496580927726],
                        [7.07106781186547e-01, 0.408248290463863]])

    matrix = Poly(scores=[0, 10, 11]).code_with_intercept(["a", "b", "c"])
    assert matrix.column_suffixes == [".Constant", ".Linear", ".Quadratic"]
    # Values from R 'options(digits=15); contr.poly(3, scores=c(0, 10, 11))'
    print matrix.matrix
    assert np.allclose(matrix.matrix,
                       [[1./3 ** (0.5), -0.813733471206735, 0.0671156055214024],
                        [1./3 ** (0.5), 0.348742916231458, -0.7382716607354268],
                        [1./3 ** (0.5), 0.464990554975277, 0.6711560552140243]])

    # we had an integer/float handling bug for score vectors whose mean was
    # non-integer, so check one of those:
    matrix = Poly(scores=[0, 10, 12]).code_with_intercept(["a", "b", "c"])
    assert matrix.column_suffixes == [".Constant", ".Linear", ".Quadratic"]
    # Values from R 'options(digits=15); contr.poly(3, scores=c(0, 10, 12))'
    print matrix.matrix
    assert np.allclose(matrix.matrix,
                       [[1./3 ** (0.5), -0.806559132617443, 0.127000127000191],
                        [1./3 ** (0.5), 0.293294230042706, -0.762000762001143],
                        [1./3 ** (0.5), 0.513264902574736, 0.635000635000952]])

    matrix = t1.code_with_intercept(range(6))
    assert matrix.column_suffixes == [".Constant", ".Linear", ".Quadratic",
                                      ".Cubic", "^4", "^5"]


class Sum(object):
    """
    Deviation coding. Compares the mean of each level to
    the mean-of-means. (In a balanced design, compares the mean of each level
    to the overall mean.) Equivalent to R's `contr.sum`.
    
    ..warning:: There are multiple definitions of 'deviation coding' in
    use. Make sure this is the one you expect before trying to interpret your
    results!
    """
    def __init__(self, omit=-1):
        self.omit = omit

    # we need to be able to index one-past-the-omitted-level by writing
    # omit+1. Which doesn't work for negative indices. So we convert negative
    # indices into equivalent positive indices.
    def _omit_i(self, levels):
        omit_i = self.omit
        if omit_i < 0:
            omit_i += len(levels)
        return omit_i

    def _sum_contrast(self, levels):
        n = len(levels)
        omit_i = self._omit_i(levels)
        eye = np.eye(n - 1)
        out = np.empty((n, n - 1))
        out[:omit_i, :] = eye[:omit_i, :]
        out[omit_i, :] = -1
        out[omit_i + 1:, :] = eye[omit_i:, :]
        return out

    def code_with_intercept(self, levels):
        contrast = self.code_without_intercept(levels)
        matrix = np.column_stack((np.ones(len(levels)),
                                  contrast.matrix))
        column_suffixes = ["[mean]"] + contrast.column_suffixes
        return ContrastMatrix(matrix, column_suffixes)

    def code_without_intercept(self, levels):
        matrix = self._sum_contrast(levels)
        omit_i = self._omit_i(levels)
        included_levels = levels[:omit_i] + levels[omit_i + 1:]
        return ContrastMatrix(matrix, _name_levels("S.", included_levels))

def test_Sum():
    t1 = Sum()
    matrix = t1.code_with_intercept(["a", "b", "c"])
    assert matrix.column_suffixes == ["[mean]", "[S.a]", "[S.b]"]
    assert np.allclose(matrix.matrix, [[1, 1, 0], [1, 0, 1], [1, -1, -1]])
    matrix = t1.code_without_intercept(["a", "b", "c"])
    assert matrix.column_suffixes == ["[S.a]", "[S.b]"]
    assert np.allclose(matrix.matrix, [[1, 0], [0, 1], [-1, -1]])
    t2 = Sum(omit=1)
    matrix = t2.code_with_intercept(["a", "b", "c"])
    assert matrix.column_suffixes == ["[mean]", "[S.a]", "[S.c]"]
    assert np.allclose(matrix.matrix, [[1, 1, 0], [1, -1, -1], [1, 0, 1]])
    matrix = t2.code_without_intercept(["a", "b", "c"])
    assert matrix.column_suffixes == ["[S.a]", "[S.c]"]
    assert np.allclose(matrix.matrix, [[1, 0], [-1, -1], [0, 1]])
    t3 = Sum(omit=-3)
    matrix = t3.code_with_intercept(["a", "b", "c"])
    assert matrix.column_suffixes == ["[mean]", "[S.b]", "[S.c]"]
    assert np.allclose(matrix.matrix, [[1, -1, -1], [1, 1, 0], [1, 0, 1]])
    matrix = t3.code_without_intercept(["a", "b", "c"])
    assert matrix.column_suffixes == ["[S.b]", "[S.c]"]
    assert np.allclose(matrix.matrix, [[-1, -1], [1, 0], [0, 1]])

class Helmert(object):
    def _helmert_contrast(self, levels):
        n = len(levels)
        #http://www.ats.ucla.edu/stat/sas/webbooks/reg/chapter5/sasreg5.htm#HELMERT
        #contr = np.eye(n - 1)
        #int_range = np.arange(n - 1., 1, -1)
        #denom = np.repeat(int_range, np.arange(n - 2, 0, -1))
        #contr[np.tril_indices(n - 1, -1)] = -1. / denom

        #http://www.ats.ucla.edu/stat/r/library/contrast_coding.htm#HELMERT
        #contr = np.zeros((n - 1., n - 1))
        #int_range = np.arange(n, 1, -1)
        #denom = np.repeat(int_range[:-1], np.arange(n - 2, 0, -1))
        #contr[np.diag_indices(n - 1)] = (int_range - 1.) / int_range
        #contr[np.tril_indices(n - 1, -1)] = -1. / denom
        #contr = np.vstack((contr, -1./int_range))

        #r-like
        contr = np.zeros((n, n - 1))
        contr[1:][diag_indices(n - 1)] = np.arange(1, n)
        contr[triu_indices(n - 1)] = -1
        return contr

    def code_with_intercept(self, levels):
        contrast = np.column_stack((np.ones(len(levels)),
                                    self._helmert_contrast(levels)))
        column_suffixes = _name_levels("H.", ["intercept"] + list(levels[1:]))
        return ContrastMatrix(contrast, column_suffixes)

    def code_without_intercept(self, levels):
        contrast = self._helmert_contrast(levels)
        return ContrastMatrix(contrast,
                              _name_levels("H.", levels[1:]))

def test_Helmert():
    t1 = Helmert()
    for levels in (["a", "b", "c", "d"], ("a", "b", "c", "d")):
        matrix = t1.code_with_intercept(levels)
        assert matrix.column_suffixes == ["[H.intercept]",
                                          "[H.b]",
                                          "[H.c]",
                                          "[H.d]"]
        assert np.allclose(matrix.matrix, [[1, -1, -1, -1],
                                           [1, 1, -1, -1],
                                           [1, 0, 2, -1],
                                           [1, 0, 0, 3]])
        matrix = t1.code_without_intercept(levels)
        assert matrix.column_suffixes == ["[H.b]", "[H.c]", "[H.d]"]
        assert np.allclose(matrix.matrix, [[-1, -1, -1],
                                           [1, -1, -1],
                                           [0, 2, -1],
                                           [0, 0, 3]])

class Diff(object):
    def _diff_contrast(self, levels):
        nlevels = len(levels)
        contr = np.zeros((nlevels, nlevels-1))
        int_range = np.arange(1, nlevels)
        upper_int = np.repeat(int_range, int_range)
        row_i, col_i = triu_indices(nlevels-1)
        # we want to iterate down the columns not across the rows
        # it would be nice if the index functions had a row/col order arg
        col_order = np.argsort(col_i)
        contr[row_i[col_order],
              col_i[col_order]] = (upper_int-nlevels)/float(nlevels)
        lower_int = np.repeat(int_range, int_range[::-1])
        row_i, col_i = tril_indices(nlevels-1)
        # we want to iterate down the columns not across the rows
        col_order = np.argsort(col_i)
        contr[row_i[col_order]+1, col_i[col_order]] = lower_int/float(nlevels)
        return contr

    def code_with_intercept(self, levels):
        contrast = np.column_stack((np.ones(len(levels)),
                                    self._diff_contrast(levels)))
        return ContrastMatrix(contrast, _name_levels("D.", levels))

    def code_without_intercept(self, levels):
        contrast = self._diff_contrast(levels)
        return ContrastMatrix(contrast, _name_levels("D.", levels[:-1]))

def test_diff():
    t1 = Diff()
    matrix = t1.code_with_intercept(["a", "b", "c", "d"])
    assert matrix.column_suffixes == ["[D.a]", "[D.b]", "[D.c]",
                                      "[D.d]"]
    assert np.allclose(matrix.matrix, [[1, -3/4., -1/2., -1/4.],
                                        [1, 1/4., -1/2., -1/4.],
                                        [1, 1/4., 1./2, -1/4.],
                                        [1, 1/4., 1/2., 3/4.]])
    matrix = t1.code_without_intercept(["a", "b", "c", "d"])
    assert matrix.column_suffixes == ["[D.a]", "[D.b]", "[D.c]"]
    assert np.allclose(matrix.matrix, [[-3/4., -1/2., -1/4.],
                                        [1/4., -1/2., -1/4.],
                                        [1/4., 2./4, -1/4.],
                                        [1/4., 1/2., 3/4.]])

# contrast can be:
#   -- a ContrastMatrix
#   -- a simple np.ndarray
#   -- an object with code_with_intercept and code_without_intercept methods
#   -- a function returning one of the above
#   -- None, in which case the above rules are applied to 'default'
# This function always returns a ContrastMatrix.
def code_contrast_matrix(intercept, levels, contrast, default=None):
    if contrast is None:
        contrast = default
    if callable(contrast):
        contrast = contrast()
    if isinstance(contrast, ContrastMatrix):
        return contrast
    as_array = np.asarray(contrast)
    if np.issubdtype(as_array.dtype, np.number):
        return ContrastMatrix(as_array,
                              _name_levels("custom", range(as_array.shape[1])))
    if intercept:
        return contrast.code_with_intercept(levels)
    else:
        return contrast.code_without_intercept(levels)

