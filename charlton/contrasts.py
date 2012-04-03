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

__all__ = ["ContrastMatrix", "Treatment", "Poly", "code_contrast_matrix"]

import numpy as np
from charlton import CharltonError

class ContrastMatrix(object):
    def __init__(self, matrix, column_suffixes):
        self.matrix = np.asarray(matrix)
        self.column_suffixes = column_suffixes
        assert self.matrix.shape[1] == len(column_suffixes)

def _name_levels(prefix, levels):
    return ["[%s%s]" % (prefix, level) for level in levels]

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
        eye = np.eye(len(levels) - 1)
        contrasts = np.vstack((eye[:self.base, :],
                                np.zeros((1, len(levels) - 1)),
                                eye[self.base:, :]))
        names = _name_levels("T.", levels[:self.base] + levels[self.base + 1:])
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
                              _name_levels("custom", range(contrast.shape[1])))
    if intercept:
        return contrast.code_with_intercept(levels)
    else:
        return contrast.code_without_intercept(levels)

