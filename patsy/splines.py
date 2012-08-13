# This file is part of Patsy
# Copyright (C) 2012 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

# R-compatible spline basis functions

import numpy as np

# make sure we correctly handle degree 0 (piecewise constant) 'spline'
# functions.

# TODO: add sub-matrix construction -- some way to ask a DesignMatrixBuilder
# for another DesignMatrixBuilder that builds only a single term. Then this
# can be used for single-term predictions, which seems like the most useful
# way to implement fitted spline function visualization.

def _eval_bspline_basis(x, knots, degree):
    # 'knots' are assumed to be already pre-processed. E.g. usually you
    # want to include duplicate copies of boundary knots; you should do
    # that *before* calling this constructor.
    knots = np.atleast_1d(np.asarray(knots, dtype=float))
    if knots.ndim > 1:
        raise ValueError("knots must be 1 dimensional")
    knots.sort()
    degree = int(degree)
    x = np.atleast_1d(x)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim > 1:
        raise ValueError("can't evaluate spline on multidim array")
    # Thanks to Charles Harris for explaining splev. It's not well
    # documented, but basically it computes an arbitrary b-spline basis
    # given knots and degree on some specificed points (or derivatives
    # thereof, but we don't use that functionality), and then returns some
    # linear combination of these basis functions. To get out the basis
    # functions themselves, we use linear combinations like [1, 0, 0], [0,
    # 1, 0], [0, 0, 1].
    # NB: This probably makes it rather inefficient (though I haven't checked
    # to be sure -- maybe the fortran code actually skips computing the basis
    # function for coefficients that are zero).
    # Note: the order of a spline is the same as its degree + 1.
    # Note: there are (len(knots) - order) basis functions.
    from scipy.interpolate import splev
    n_bases = len(knots) - (degree + 1)
    basis = np.empty((x.shape[0], n_bases), dtype=float)
    for i in xrange(n_bases):
        coefs = np.zeros((n_bases,))
        coefs[i] = 1
        basis[:, i] = splev(x, (knots, coefs, degree))
    return basis

class BS(object):
    def __init__(self):
        self._xs = []
        self._lower_bound = np.inf
        self._upper_bound = -np.inf

    def memorize_chunk(self, x, df=None, knots=None, degree=3,
                       span_intercept=False,
                       lower_bound=None, upper_bound=None):
        x = np.atleast_1d(x)
        if x.ndim > 1:
            raise ValueError, "input to 'bs' must be 1-d vector"
        self._need_knots = (knots is None)
        if self._need_knots:
            if df is None:
                raise ValueError, "must specify either df or knots"
            self._knots_wanted = df - (degree + 1)
            if span_intercept:
                self._knots_wanted -= 1
            # There's no better way to compute exact quantiles than 
            self._xs.append(x)
        if lower_bound is None:
            self._lower_bound = min(self._lower_bound, np.min(x))
        if upper_bound is None:
            self._upper_bound = max(self._upper_bound, np.max(x))

    def memorize_finish(self):
        if not self._need_memorize:
            return
        # These are guaranteed to all be 1d vectors by the code above
        x = np.concatenate(self._xs)
        # Just go ahead and calculate these unconditionally, 
        self._lower_bound = np.min(x)
        self._upper_bound = np.max(x)
