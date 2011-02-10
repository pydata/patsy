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

import numpy as np

from charlton import CharltonError

def _name_levels(prefix, levels):
    return ["[%s%s]" % (prefix, level) for level in levels]

def test__name_levels():
    assert _name_levels("a", ["b", "c"]) == ["[ab]", "[ac]"]

def _dummy_code(levels):
    return _name_levels("", levels), np.eye(len(levels))

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
        return (names, contrasts)

def test_Treatment():
    t1 = Treatment()
    names, matrix = t1.code_with_intercept(["a", "b", "c"])
    assert names == ["[a]", "[b]", "[c]"]
    assert np.allclose(matrix, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    names, matrix = t1.code_without_intercept(["a", "b", "c"])
    assert names == ["[T.b]", "[T.c]"]
    assert np.allclose(matrix, [[0, 0], [1, 0], [0, 1]])
    names, matrix = Treatment(base=1).code_without_intercept(["a", "b", "c"])
    assert names == ["[T.a]", "[T.c]"]
    assert np.allclose(matrix, [[1, 0], [0, 0], [0, 1]])

class Poly(object):
    def __init__(self, scores=None):
        self.scores = scores

    def _code_either(self, intercept, levels):
        n = len(levels)
        scores = self.scores
        if scores is None:
            scores = np.arange(n)
        scores = np.asarray(scores)
        if len(scores) != n:
            raise CharltonError("number of levels (%s) does not match"
                                " number of scores (%s)"
                                % (n, len(scores)))
        # Strategy: just make a matrix whose columns are naive linear,
        # quadratic, etc., functions of the raw scores, and then use 'qr' to
        # orthogonalize each column against those to its left.
        scores -= scores.mean()
        raw_poly = scores.reshape((-1, 1)) ** np.arange(n).reshape((1, -1))
        q, _ = np.linalg.qr(raw_poly)
        q /= np.sqrt(np.sum(q ** 2, axis=1))
        names = ["^%s" % (i,) for i in xrange(n)]
        names[0] = ".Constant"
        if n > 1:
            names[1] = ".Linear"
        if n > 2:
            names[2] = ".Quadratic"
        if n > 3:
            names[3] = ".Cubic"
        if intercept:
            return names, q
        else:
            # We always include the constant/intercept column as something to
            # orthogonalize against, but we don't always return it:
            return names[1:], q[:, 1:]

    def code_with_intercept(self, levels):
        return self._code_either(True, levels)

    def code_without_intercept(self, levels):
        return self._code_either(False, levels)

def test_Poly():
    t1 = Poly()
    names, matrix = t1.code_with_intercept(["a", "b", "c"])
    assert names == [".Constant", ".Linear", ".Quadratic"]
    # Values from R 'options(digits=15); contr.poly(3)'
    print matrix
    assert np.allclose(matrix,
                       [[1./3 ** (0.5), -7.07106781186548e-01, 0.408248290463863],
                        [1./3 ** (0.5), 0, -0.816496580927726],
                        [1./3 ** (0.5), 7.07106781186547e-01, 0.408248290463863]])
    names, matrix = t1.code_without_intercept(["a", "b", "c"])
    assert names == [".Linear", ".Quadratic"]
    # Values from R 'options(digits=15); contr.poly(3)'
    print matrix
    assert np.allclose(matrix,
                       [[-7.07106781186548e-01, 0.408248290463863],
                        [0, -0.816496580927726],
                        [7.07106781186547e-01, 0.408248290463863]])

    names, matrix = Poly(scores=[0, 10, 11]).code_with_intercept(["a", "b", "c"])
    assert names == [".Constant", ".Linear", ".Quadratic"]
    # Values from R 'options(digits=15); contr.poly(3, scores=c(0, 10, 11))'
    print matrix
    assert np.allclose(matrix,
                       [[1./3 ** (0.5), -0.813733471206735, 0.0671156055214024],
                        [1./3 ** (0.5), 0.348742916231458, -0.7382716607354268],
                        [1./3 ** (0.5), 0.464990554975277, 0.6711560552140243]])
    
    names, matrix = t1.code_with_intercept(range(6))
    assert names == [".Constant", ".Linear", ".Quadratic", ".Cubic",
                     "^4", "^5"]

def get_contrast(intercept, categorical, default=Treatment):
    contrast = categorical.contrast
    if contrast is None:
        contrast = default
    if np.issubdtype(np.asarray(contrast).dtype, np.number):
        return _name_levels("custom", range(contrast.shape[1])), contrast
    if issubclass(contrast, Contrast):
        contrast = contrast()
    if intercept:
        return contrast.code_with_intercept(categorical.levels)
    else:
        return contrast.code_without_intercept(categorical.levels)
