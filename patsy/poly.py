# This file is part of Patsy
# Copyright (C) 2012-2013 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

# R-compatible poly function

# These are made available in the patsy.* namespace
__all__ = ["poly"]

import numpy as np

from patsy.util import have_pandas, no_pickling, assert_no_pickling
from patsy.state import stateful_transform

if have_pandas:
    import pandas

class Poly(object):
    """poly(x, degree=1, raw=False)

    Generates an orthogonal polynomial transformation of x of degree.
    Generic usage is something along the lines of::

      y ~ 1 + poly(x, 4)

    to fit ``y`` as a function of ``x``, with a 4th degree polynomial.

    :arg degree: The number of degrees for the polynomial expansion.
    :arg raw: When raw is False (the default), will return orthogonal
      polynomials.

    .. versionadded:: 0.4.1
    """
    def __init__(self):
        self._tmp = {}
        self._degree = None
        self._raw = None

    def memorize_chunk(self, x, degree=3, raw=False):
        args = {"degree": degree,
                "raw": raw
                }
        self._tmp["args"] = args
        # XX: check whether we need x values before saving them
        x = np.atleast_1d(x)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim > 1:
            raise ValueError("input to 'poly' must be 1-d, "
                             "or a 2-d column vector")
        # There's no better way to compute exact quantiles than memorizing
        # all data.
        x = np.array(x, dtype=float)
        self._tmp.setdefault("xs", []).append(x)

    def memorize_finish(self):
        tmp = self._tmp
        args = tmp["args"]
        del self._tmp

        if args["degree"] < 1:
            raise ValueError("degree must be greater than 0 (not %r)"
                             % (args["degree"],))
        if int(args["degree"]) != args["degree"]:
            raise ValueError("degree must be an integer (not %r)"
                             % (self._degree,))

        # These are guaranteed to all be 1d vectors by the code above
        scores = np.concatenate(tmp["xs"])
        scores_mean = scores.mean()
        # scores -= scores_mean
        self.scores_mean = scores_mean
        n = args['degree']
        self.degree = n
        raw_poly = scores.reshape((-1, 1)) ** np.arange(n + 1).reshape((1, -1))
        raw = args['raw']
        self.raw = raw
        if not raw:
            q, r = np.linalg.qr(raw_poly)
            # Q is now orthognoal of degree n. To match what R is doing, we
            # need to use the three-term recurrence technique to calculate
            # new alpha, beta, and norm.

            self.alpha = (np.sum(scores.reshape((-1, 1)) * q[:, :n] ** 2,
                                 axis=0) /
                          np.sum(q[:, :n] ** 2, axis=0))

            # For reasons I don't understand, the norms R uses are based off
            # of the diagonal of the r upper triangular matrix.

            self.norm = np.linalg.norm(q * np.diag(r), axis=0)
            self.beta = (self.norm[1:] / self.norm[:n]) ** 2

    def transform(self, x, degree=3, raw=False):
        if have_pandas:
            if isinstance(x, (pandas.Series, pandas.DataFrame)):
                to_pandas = True
                idx = x.index
            else:
                to_pandas = False
        else:
            to_pandas = False
        x = np.array(x, ndmin=1).flatten()

        if self.raw:
            n = self.degree
            p = x.reshape((-1, 1)) ** np.arange(n + 1).reshape((1, -1))
        else:
            # This is where the three-term recurrance technique is unwound.

            p = np.empty((x.shape[0], self.degree + 1))
            p[:, 0] = 1

            for i in np.arange(self.degree):
                p[:, i + 1] = (x - self.alpha[i]) * p[:, i]
                if i > 0:
                    p[:, i + 1] = (p[:, i + 1] -
                                   (self.beta[i - 1] * p[:, i - 1]))
            p /= self.norm

        p = p[:, 1:]
        if to_pandas:
            p = pandas.DataFrame(p)
            p.index = idx
        return p

    __getstate__ = no_pickling

poly = stateful_transform(Poly)


def test_poly_compat():
    from patsy.test_state import check_stateful
    from patsy.test_poly_data import (R_poly_test_x,
                                      R_poly_test_data,
                                      R_poly_num_tests)
    lines = R_poly_test_data.split("\n")
    tests_ran = 0
    start_idx = lines.index("--BEGIN TEST CASE--")
    while True:
        if not lines[start_idx] == "--BEGIN TEST CASE--":
            break
        start_idx += 1
        stop_idx = lines.index("--END TEST CASE--", start_idx)
        block = lines[start_idx:stop_idx]
        test_data = {}
        for line in block:
            key, value = line.split("=", 1)
            test_data[key] = value
        # Translate the R output into Python calling conventions
        kwargs = {
            # integer
            "degree": int(test_data["degree"]),
            # boolen
            "raw": (test_data["raw"] == 'TRUE')
            }
        # Special case: in R, setting intercept=TRUE increases the effective
        # dof by 1. Adjust our arguments to match.
        # if kwargs["df"] is not None and kwargs["include_intercept"]:
        #     kwargs["df"] += 1
        output = np.asarray(eval(test_data["output"]))
        # Do the actual test
        check_stateful(Poly, False, R_poly_test_x, output, **kwargs)
        tests_ran += 1
        # Set up for the next one
        start_idx = stop_idx + 1
    assert tests_ran == R_poly_num_tests


def test_poly_errors():
    from nose.tools import assert_raises
    x = np.arange(27)
    # Invalid input shape
    assert_raises(ValueError, poly, x.reshape((3, 3, 3)))
    assert_raises(ValueError, poly, x.reshape((3, 3, 3)), raw=True)
    # Invalid degree
    assert_raises(ValueError, poly, x, degree=-1)
    assert_raises(ValueError, poly, x, degree=0)
    assert_raises(ValueError, poly, x, degree=3.5)
