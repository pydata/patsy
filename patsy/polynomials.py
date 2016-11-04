# This file is part of Patsy
# Copyright (C) 2012-2013 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

# R-compatible poly function

# These are made available in the patsy.* namespace
import numpy as np

from patsy.util import have_pandas, no_pickling, assert_no_pickling
from patsy.state import stateful_transform

__all__ = ["poly"]

if have_pandas:
    import pandas


class Poly(object):
    """poly(x, degree=3, raw=False)

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
                             % (args['degree'],))

        # These are guaranteed to all be 1d vectors by the code above
        scores = np.concatenate(tmp["xs"])

        n = args['degree']
        self.degree = n
        self.raw = args['raw']

        if not self.raw:
            raw_poly = self.vander(scores, n)
            self.alpha, self.norm, self.beta = self.gen_qr(raw_poly, n)

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

        n = self.degree
        p = self.vander(x, n)

        if not self.raw:
            p = self.apply_qr(p, n, self.alpha, self.norm, self.beta)

        p = p[:, 1:]
        if to_pandas:
            p = pandas.DataFrame(p)
            p.index = idx
        return p

    @staticmethod
    def vander(x, n):
        raw_poly = np.polynomial.polynomial.polyvander(x, n)
        return raw_poly

    @staticmethod
    def gen_qr(raw_poly, n):
        x = raw_poly[:, 1]
        q, r = np.linalg.qr(raw_poly)
        # Q is now orthognoal of degree n. To match what R is doing, we
        # need to use the three-term recurrence technique to calculate
        # new alpha, beta, and norm.
        alpha = (np.sum(x.reshape((-1, 1)) * q[:, :n] ** 2, axis=0) /
                 np.sum(q[:, :n] ** 2, axis=0))

        # For reasons I don't understand, the norms R uses are based off
        # of the diagonal of the r upper triangular matrix.

        norm = np.linalg.norm(q * np.diag(r), axis=0)
        beta = (norm[1:] / norm[:n]) ** 2
        return alpha, norm, beta

    @staticmethod
    def apply_qr(x, n, alpha, norm, beta):
        # This is where the three-term recurrence is unwound for the QR
        # decomposition.
        if np.ndim(x) == 2:
            x = x[:, 1]
        p = np.empty((x.shape[0], n + 1))
        p[:, 0] = 1

        for i in np.arange(n):
            p[:, i + 1] = (x - alpha[i]) * p[:, i]
            if i > 0:
                p[:, i + 1] = (p[:, i + 1] - (beta[i - 1] * p[:, i - 1]))
        p /= norm
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

    p = poly(np.arange(1, 10), degree=3)
    assert_no_pickling(p)
