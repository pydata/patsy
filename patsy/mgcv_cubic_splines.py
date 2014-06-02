# This file is part of Patsy
# Copyright (C) 2014 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

# R package 'mgcv' compatible cubic spline basis functions

# These are made available in the patsy.* namespace
__all__ = ["cr", "cs", "cc", "ms", "te"]

import warnings
import numpy as np

try:
    from scipy import linalg
except ImportError:
    raise ImportError("Cubic spline functionality requires scipy.")

from patsy.util import have_pandas
from patsy.state import stateful_transform

if have_pandas:
    import pandas


def _get_natural_f(knots):
    """Returns mapping of natural cubic spline values to 2nd derivatives.

    .. note:: See 'Generalized Additive Models', Simon N. Wood, 2006, pp 145-146

    :param knots: The 1-d array knots used for cubic spline parametrization,
     must be sorted in ascending order.
    :return: A 2-d array mapping natural cubic spline values at
     knots to second derivatives.
    """
    h = knots[1:] - knots[:-1]
    diag = (h[:-1] + h[1:]) / 3.
    ul_diag = h[1:-1] / 6.
    banded_b = np.array([np.r_[0., ul_diag], diag, np.r_[ul_diag, 0.]])
    d = np.zeros((knots.size - 2, knots.size))
    for i in range(knots.size - 2):
        d[i, i] = 1. / h[i]
        d[i, i + 2] = 1. / h[i + 1]
        d[i, i + 1] = - d[i, i] - d[i, i + 2]

    fm = linalg.solve_banded((1, 1), banded_b, d)

    return np.vstack([np.zeros(knots.size), fm, np.zeros(knots.size)])


# Cyclic Cubic Regression Splines

def _map_cyclic(x, lbound, ubound):
    """Maps values into the interval [lbound, ubound] in a cyclic fashion.

    :param x: The 1-d array values to be mapped.
    :param lbound: The lower bound of the interval.
    :param ubound: The upper bound of the interval.
    :return: A new 1-d array containing mapped x values.

    :raise ValueError: if lbound >= ubound.
    """
    if lbound >= ubound:
        raise ValueError("Invalid argument: lbound (%r) should be "
                         "less than ubound (%r)."
                         % (lbound, ubound))

    x = np.copy(x)
    x[x > ubound] = lbound + (x[x > ubound] - ubound) % (ubound - lbound)
    x[x < lbound] = ubound - (lbound - x[x < lbound]) % (ubound - lbound)

    return x


def test__map_cyclic():
    x = np.array([1.5, 2.6, 0.1, 4.4, 10.7])
    x_orig = np.copy(x)
    expected_mapped_x = np.array([3.0, 2.6, 3.1, 2.9, 3.2])
    mapped_x = _map_cyclic(x, 2.1, 3.6)
    assert np.allclose(x, x_orig)
    assert np.allclose(mapped_x, expected_mapped_x)


def test__map_cyclic_errors():
    from nose.tools import assert_raises
    x = np.linspace(0.2, 5.7, 10)
    assert_raises(ValueError, _map_cyclic, x, 4.5, 3.6)
    assert_raises(ValueError, _map_cyclic, x, 4.5, 4.5)


def _get_cyclic_f(knots):
    """Returns mapping of cyclic cubic spline values to 2nd derivatives.

    .. note:: See 'Generalized Additive Models', Simon N. Wood, 2006, pp 146-147

    :param knots: The 1-d array knots used for cubic spline parametrization,
     must be sorted in ascending order.
    :return: A 2-d array mapping cyclic cubic spline values at
     knots to second derivatives.
    """
    h = knots[1:] - knots[:-1]
    n = knots.size - 1
    b = np.zeros((n, n))
    d = np.zeros((n, n))

    b[0, 0] = (h[n - 1] + h[0]) / 3.
    b[0, n - 1] = h[n - 1] / 6.
    b[n - 1, 0] = h[n - 1] / 6.

    d[0, 0] = -1. / h[0] - 1. / h[n - 1]
    d[0, n - 1] = 1. / h[n - 1]
    d[n - 1, 0] = 1. / h[n - 1]

    for i in range(1, n):
        b[i, i] = (h[i - 1] + h[i]) / 3.
        b[i, i - 1] = h[i - 1] / 6.
        b[i - 1, i] = h[i - 1] / 6.

        d[i, i] = -1. / h[i - 1] - 1. / h[i]
        d[i, i - 1] = 1. / h[i - 1]
        d[i - 1, i] = 1. / h[i - 1]

    return linalg.solve(b, d)


# Tensor Product


def _row_tensor_product(dms):
    """Computes row-wise tensor product of given arguments.

    .. note:: Custom algorithm to precisely match what is done in 'mgcv',
    in particular look out for order of result columns!
    For reference implementation see 'mgcv' source code,
    file 'mat.c', mgcv_tensor_mm(), l.62

    :param dms: A sequence of 2-d arrays (marginal design matrices).
    :return: The 2-d array row-wise tensor product of given arguments.

    :raise ValueError: if argument sequence is empty, does not contain only
     2-d arrays or if the arrays number of rows does not match.
    """
    if len(dms) == 0:
        raise ValueError("Tensor product arrays sequence should not be empty.")
    for dm in dms:
        if dm.ndim != 2:
            raise ValueError("Tensor product arguments should be 2-d arrays.")

    tp_nrows = dms[0].shape[0]
    tp_ncols = 1
    for dm in dms:
        if dm.shape[0] != tp_nrows:
            raise ValueError("Tensor product arguments should have "
                             "same number of rows.")
        tp_ncols *= dm.shape[1]
    tp = np.zeros((tp_nrows, tp_ncols))
    tp[:, -dms[-1].shape[1]:] = dms[-1]
    filled_tp_ncols = dms[-1].shape[1]
    for dm in dms[-2::-1]:
        p = - filled_tp_ncols * dm.shape[1]
        for j in range(dm.shape[1]):
            xj = dm[:, j]
            for t in range(-filled_tp_ncols, 0):
                tp[:, p] = tp[:, t] * xj
                p += 1
        filled_tp_ncols *= dm.shape[1]

    return tp


def test__row_tensor_product_errors():
    from nose.tools import assert_raises
    assert_raises(ValueError, _row_tensor_product, [])
    assert_raises(ValueError, _row_tensor_product, [np.arange(1, 5)])
    assert_raises(ValueError, _row_tensor_product,
                  [np.arange(1, 5), np.arange(1, 5)])
    assert_raises(ValueError, _row_tensor_product,
                  [np.arange(1, 13).reshape((3, 4)),
                   np.arange(1, 13).reshape((4, 3))])


def test__row_tensor_product():
    # Testing cases where main input array should not be modified
    dm1 = np.arange(1, 17).reshape((4, 4))
    assert np.array_equal(_row_tensor_product([dm1]), dm1)
    ones = np.ones(4).reshape((4, 1))
    tp1 = _row_tensor_product([ones, dm1])
    assert np.array_equal(tp1, dm1)
    tp2 = _row_tensor_product([dm1, ones])
    assert np.array_equal(tp2, dm1)

    # Testing cases where main input array should be scaled
    twos = 2 * ones
    tp3 = _row_tensor_product([twos, dm1])
    assert np.array_equal(tp3, 2 * dm1)
    tp4 = _row_tensor_product([dm1, twos])
    assert np.array_equal(tp4, 2 * dm1)

    # Testing main cases
    dm2 = np.array([[1, 2], [1, 2]])
    dm3 = np.arange(1, 7).reshape((2, 3))
    expected_tp5 = np.array([[1,  2,  3,  2,  4,  6],
                             [4,  5,  6,  8, 10, 12]])
    tp5 = _row_tensor_product([dm2, dm3])
    assert np.array_equal(tp5, expected_tp5)
    expected_tp6 = np.array([[1,  2,  2,  4,  3,  6],
                             [4,  8,  5, 10,  6, 12]])
    tp6 = _row_tensor_product([dm3, dm2])
    assert np.array_equal(tp6, expected_tp6)


# Common code


def _find_knots_lower_bounds(x, knots):
    """Finds knots lower bounds for given values.

    Returns an array of indices ``I`` such that
    ``0 <= I[i] <= knots.size - 2`` for all ``i``
    and
    ``knots[I[i]] < x[i] <= knots[I[i] + 1]`` if
    ``np.min(knots) < x[i] <= np.max(knots)``,
    ``I[i] = 0`` if ``x[i] <= np.min(knots)``
    ``I[i] = knots.size - 2`` if ``np.max(knots) < x[i]``
    
    :param x: The 1-d array values whose knots lower bounds are to be found.
    :param knots: The 1-d array knots used for cubic spline parametrization,
     must be sorted in ascending order.
    :return: An array of knots lower bounds indices.
    """
    lb = np.searchsorted(knots, x) - 1

    # I[i] = 0 for x[i] <= np.min(knots)
    lb[lb == -1] = 0

    # I[i] = knots.size - 2 for x[i] > np.max(knots)
    lb[lb == knots.size - 1] = knots.size - 2

    return lb


def _compute_base_functions(x, knots, j=None):
    """Computes base functions used for building cubic splines basis.

    .. note:: See 'Generalized Additive Models', Simon N. Wood, 2006, p. 146
      and for the special treatment of ``x`` values outside ``knots`` range
      see 'mgcv' source code, file 'mgcv.c', function 'crspl()', l.249

    :param x: The 1-d array values for which base functions should be computed.
    :param knots: The 1-d array knots used for cubic spline parametrization,
     must be sorted in ascending order.
    :param j: The 1-d array of knots lower bounds indices corresponding to
     the given ``x`` values. If none provided, it will be computed using
     :ref:`_find_knots_lower_bounds`.
    :return: 4 arrays corresponding to the 4 base functions ajm, ajp, cjm, cjp.
    """
    if j is None:
        j = _find_knots_lower_bounds(x, knots)

    h = knots[1:] - knots[:-1]
    hj = h[j]
    xj1_x = knots[j + 1] - x
    x_xj = x - knots[j]

    ajm = xj1_x / hj
    ajp = x_xj / hj

    cjm_3 = xj1_x * xj1_x * xj1_x / (6. * hj)
    cjm_3[x > np.max(knots)] = 0.
    cjm_1 = hj * xj1_x / 6.
    cjm = cjm_3 - cjm_1

    cjp_3 = x_xj * x_xj * x_xj / (6. * hj)
    cjp_3[x < np.min(knots)] = 0.
    cjp_1 = hj * x_xj / 6.
    cjp = cjp_3 - cjp_1

    return ajm, ajp, cjm, cjp


def _absorb_constraints(dm, pc):
    """Absorb the parameters constraints ``pc`` into the design matrix ``dm``.

    :param dm: The (2-d array) initial design matrix.
    :param pc: The 2-d array defining parameters (``betas``) constraints
     (``np.dot(pc, betas) = 0``).
    :return: The new design matrix with absorbed parameters constraints.
    """
    m = pc.shape[0]
    q, r = linalg.qr(np.transpose(pc))

    return np.dot(dm, q[:, m:])


def _get_free_crs_dmatrix(x, knots, cyclic=False):
    """Builds an unconstrained cubic regression spline design matrix.

    Returns design matrix with dimensions ``len(x) x n``
    for a cubic regression spline smoother
    where 
     - ``n = len(knots)`` for natural CRS
     - ``n = len(knots) - 1`` for cyclic CRS

    .. note:: See 'Generalized Additive Models', Simon N. Wood, 2006, p. 145

    :param x: The 1-d array values.
    :param knots: The 1-d array knots used for cubic spline parametrization,
     must be sorted in ascending order.
    :param cyclic: Indicates whether used cubic regression splines should
     be cyclic or not. Default is ``False``.
    :return: The (2-d array) design matrix.

    :raise ValueError: if for natural CRS some data points fall outside the
     outermost knots.
    """
    n = knots.size
    if cyclic:
        x = _map_cyclic(x, min(knots), max(knots))
        n -= 1

    j = _find_knots_lower_bounds(x, knots)
    j1 = j + 1
    if cyclic:
        j1[j1 == n] = 0

    i = np.identity(n)

    if cyclic:
        f = _get_cyclic_f(knots)
    else:
        f = _get_natural_f(knots)

    ajm, ajp, cjm, cjp = _compute_base_functions(x, knots, j)
    dmt = ajm * i[j, :].T + ajp * i[j1, :].T + \
        cjm * f[j, :].T + cjp * f[j1, :].T

    return dmt.T


def _get_crs_dmatrix(x, knots, pc=None, cyclic=False):
    """Builds a cubic regression spline design matrix.

    Returns design matrix with dimensions len(x) x n
    where:
     - ``n = len(knots) - nrows(pc)`` for natural CRS
     - ``n = len(knots) - nrows(pc) - 1`` for cyclic CRS
    for a cubic regression spline smoother

    :param x: The 1-d array values.
    :param knots: The 1-d array knots used for cubic spline parametrization,
     must be sorted in ascending order.
    :param pc: The 2-d array defining parameters (``betas``) constraints
     (``np.dot(pc, betas) = 0``).
    :param cyclic: Indicates whether used cubic regression splines should
     be cyclic or not. Default is ``False``.
    :return: The (2-d array) design matrix.
    """
    dm = _get_free_crs_dmatrix(x, knots, cyclic)
    if pc is not None:
        dm = _absorb_constraints(dm, pc)

    return dm


def _get_te_dmatrix(dms, pc=None):
    """Builds tensor product design matrix, given the marginal design matrices.

    :param dms: A sequence of 2-d arrays (marginal design matrices).
    :param pc: The 2-d array defining parameters (``betas``)
     constraints (``np.dot(pc, betas) = 0``).
    :return: The (2-d array) design matrix.
    """
    dm = _row_tensor_product(dms)
    if pc is not None:
        dm = _absorb_constraints(dm, pc)

    return dm


# Stateful Transforms


def _compute_knots(x, k):
    """Places knots evenly wrt to the given data values CDF.

    :param x: The 1-d array data values.
    :param k: The number of knots to place.
    :return: The array of ``k`` knots.
    """
    xu = np.unique(x)
    q = np.linspace(0, 100, k).tolist()

    return np.asarray(np.percentile(xu, q))


def _get_actual_sorted_knots(k, get_knots, default_k, min_k):
    """Determines actual (sorted) knots to use given available arguments.

    :param k: Either a 1-d array/list of knots or an int equal to the number
     of desired knots.
    :param get_knots: A function to retrieve knots from the desired number.
    :param default_k: The default number of knots.
    :param min_k: The minimum value of number of knots.
    :return: The actual (sorted) knots to use.

    :raise ValueError: if actual number of knots is less than ``min_k``. This
     could happen through invalid user input or invalid result from 'get_knots'.
    """
    if isinstance(k, int):
        if k < 0:
            warnings.warn("Invalid number of knots (k=%r) "
                          "replaced by default value (default_k=%r)"
                          % (k, default_k))
            k = default_k
        if k < min_k:
            warnings.warn("Provided number of knots (k=%r) increased to "
                          "minimum value (min_k=%r)"
                          % (k, min_k))
            k = min_k
        k = get_knots(k)
    elif k is None:
        k = get_knots(default_k)

    k = np.asarray(k)
    if k.size < min_k:
        raise ValueError("At least %r knots should be specified, but %r given."
                         % (min_k, k.size))

    return np.sort(k)


def test__get_actual_knots():
    get_knots = lambda k: np.arange(k)[::-1]
    default_k = 10
    min_k = 3
    knots = np.arange(8)[::-1]
    knots_list = range(8)[::-1]
    assert np.array_equal(
        _get_actual_sorted_knots(None, get_knots, default_k, min_k),
        np.arange(default_k))
    assert np.array_equal(
        _get_actual_sorted_knots(-5, get_knots, default_k, min_k),
        np.arange(default_k))
    assert np.array_equal(
        _get_actual_sorted_knots(2, get_knots, default_k, min_k),
        np.arange(min_k))
    assert np.array_equal(
        _get_actual_sorted_knots(7, get_knots, default_k, min_k),
        np.arange(7))
    assert np.array_equal(
        _get_actual_sorted_knots(knots, get_knots, default_k, min_k),
        np.arange(knots.size))
    assert np.array_equal(
        _get_actual_sorted_knots(knots_list, get_knots, default_k, min_k),
        np.arange(knots.size))
    from nose.tools import assert_raises
    assert_raises(ValueError, _get_actual_sorted_knots, range(2),
                  get_knots, default_k, min_k)
    assert_raises(ValueError, _get_actual_sorted_knots, 7,
                  lambda k: range(2), default_k, min_k)


def _get_actual_constraints(cons, get_constraints):
    """Determines actual parameters constraints to use.

    :param cons: Either a 2-d array defining the constraints or a boolean
     indicating whether we should retrieve the constraints using the
     function parameter.
    :param get_constraints: A function used to retrieve parameters
     constraints if needed.
    :return: The actual parameters constraints to use.

    :raise ValueError: if centering constraint is specified but
     ``absorb_centering_constraint`` is set to ``False``.
    """
    if isinstance(cons, bool):
        if cons:
            cons = get_constraints()
        else:
            cons = None

    return cons


def test__get_actual_constraints():
    get_constraints = lambda: np.arange(20).reshape((4, 5))
    constraints = np.ones(20).reshape((4, 5))
    assert _get_actual_constraints(None, get_constraints) is None
    assert _get_actual_constraints(False, get_constraints) is None
    assert np.array_equal(_get_actual_constraints(True, get_constraints),
                          np.arange(20).reshape((4, 5)))
    assert np.array_equal(_get_actual_constraints(constraints, get_constraints),
                          np.ones(20).reshape((4, 5)))


def _get_centering_constraint_from_dmatrix(dm):
    """ Computes the centering constraint from the given design matrix.

    :param dm: The 2-d array design matrix.
    :return: A 2-d array (1 x ncols(dm)) defining the centering constraint.
    """
    return dm.mean(axis=0).reshape((1, dm.shape[1]))


class CubicRegressionSpline(object):
    """Base class for cubic regression spline stateful transforms

    This class contains all the functionality for the following stateful
    transforms:
     - ``cr(x, k=None, cons=None)`` for natural cubic regression spline
     - ``cs(x, k=None, cons=None)`` for natural cubic regression spline with
        shrinkage; in the context of patsy there is no difference between
        ``cs`` and ``cr``. These two symbols exist only for compatibility with
        the R package 'mgcv'.
     - ``cc(x, k=None, cons=None)`` for cyclic cubic regression spline

    Each of these stateful transforms generate a cubic spline basis for ``x``
    (with the option of absorbing centering or more general parameters
    constraints), allowing non-linear fits. The usual usage is something like::

      y ~ 1 + cr(x, k=5, cons=True)

    to fit ``y`` as a smooth function of ``x``, with a 5-dimensional basis
    used to represent the smooth term, and centering constraint absorbed in
    the resulting design matrix.

    :arg k: Either the dimension of the spline basis or the list of knots used
     to represent the smooth term. If only the dimension is provided, equally
     spaced quantiles of the input data are used as knots and will be remembered
     and re-used for prediction from the fitted model.
    :arg cons: Either a 2-d array defining the constraints or a boolean which,
     if set to ``True``, indicates that we should apply centering constraint
     (this constraint will be computed from the input data, remembered and
     re-used for prediction from the fitted model).
     The constraints are absorbed in the resulting design matrix. Note that
     ``cons=False`` is equivalent to no ``cons`` parameter.

    Using these functions requires scipy be installed.

    .. note:: These functions reproduce the cubic regression splines as
    implemented in the R package 'mgcv'.

    .. versionadded:: 0.2.1
    """
    def __init__(self, name, default_k, min_k, cyclic):
        self._name = name
        self._default_k = default_k
        self._min_k = min_k
        self._cyclic = cyclic
        self._xs = []
        self._k = None
        self._cons = None

    def memorize_chunk(self, x, k=None, cons=None):
        x = np.atleast_1d(x)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim > 1:
            raise ValueError("Input to '%r' must be 1-d, "
                             "or a 2-d column vector."
                             % (self._name,))
        self._xs.append(x)
        self._k = k
        self._cons = cons

    def memorize_finish(self):
        x = np.concatenate(self._xs)
        self._k = _get_actual_sorted_knots(
            self._k, lambda k: _compute_knots(x, k),
            default_k=self._default_k, min_k=self._min_k)
        self._cons = _get_actual_constraints(
            self._cons,
            lambda: _get_centering_constraint_from_dmatrix(
                _get_free_crs_dmatrix(x, self._k, cyclic=self._cyclic)))

    def transform(self, x, k=None, cons=None):
        x_orig = x
        x = np.atleast_1d(x)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim > 1:
            raise ValueError("Input to '%r' must be 1-d, "
                             "or a 2-d column vector."
                             % (self._name,))
        dm = _get_crs_dmatrix(x, self._k, self._cons, cyclic=self._cyclic)
        if have_pandas:
            if isinstance(x_orig, (pandas.Series, pandas.DataFrame)):
                dm = pandas.DataFrame(dm)
                dm.index = x_orig.index
        return dm


class CR(CubicRegressionSpline):
    """cr(x, k=None, cons=None)

    For more details see :ref:`CubicRegressionSpline`
    """
    def __init__(self):
        CubicRegressionSpline.__init__(
            self, name='cr', default_k=10, min_k=3, cyclic=False)

cr = stateful_transform(CR)


class CS(CubicRegressionSpline):
    """cs(x, k=None, cons=None)

    For more details see :ref:`CubicRegressionSpline`
    """
    def __init__(self):
        CubicRegressionSpline.__init__(
            self, name='cs', default_k=10, min_k=3, cyclic=False)

cs = stateful_transform(CS)


class CC(CubicRegressionSpline):
    """cc(x, k=None, cons=None)

    For more details see :ref:`CubicRegressionSpline`
    """
    def __init__(self):
        CubicRegressionSpline.__init__(
            self, name='cc', default_k=10, min_k=4, cyclic=True)

cc = stateful_transform(CC)


def test_crs_compat():
    from patsy.test_state import check_stateful
    from patsy.test_splines_crs_data import (R_crs_test_x,
                                             R_crs_test_data,
                                             R_crs_num_tests)
    lines = R_crs_test_data.split("\n")
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
        spline_type = eval(test_data["spline_type"].upper())
        kwargs = {"cons": (test_data["absorb_cons"] == "TRUE")}
        if test_data["knots"] != "None":
            kwargs["k"] = np.asarray(eval(test_data["knots"]))
        else:
            kwargs["k"] = eval(test_data["nb_knots"])
        output = np.asarray(eval(test_data["output"]))
        # Do the actual test
        check_stateful(spline_type, False, R_crs_test_x, output, **kwargs)
        tests_ran += 1
        # Set up for the next one
        start_idx = stop_idx + 1
    assert tests_ran == R_crs_num_tests


def test_crs_with_specific_constraint():
    from patsy.highlevel import incr_dbuilder, build_design_matrices, dmatrix
    x = (-1.5)**np.arange(20)
    # Hard coded R values for smooth: s(x, bs="cr", k=5)
    # R> knots <- smooth$xp
    knots = np.array([-2216.837820053100585937,
                      -50.456909179687500000,
                      -0.250000000000000000,
                      33.637939453125000000,
                      1477.891880035400390625])
    # R> centering.constraint <- t(qr.X(attr(smooth, "qrc")))
    centering_constraint = np.array([[0.064910676323168478574,
                                     1.4519875239407085132,
                                     -2.1947446912471946234,
                                     1.6129783104357671153,
                                     0.064868180547550072235]])
    # values for which we want a prediction
    new_x = np.array([-3000., -200., 300., 2000.])
    result1 = dmatrix("cr(new_x, k=knots, cons=centering_constraint)")

    data_chunked = [{"x": x[:10]}, {"x": x[10:]}]
    new_data = {"x": new_x}
    builder = incr_dbuilder("cr(x, k=5, cons=True)", lambda: iter(data_chunked))
    result2 = build_design_matrices([builder], new_data)[0]

    assert np.allclose(result1, result2)


def ms(smooth_st, **kwargs):
    """Defines a marginal smooth with its specific parameters.

    .. note:: Arguments specifying constraints, if any, are removed.

    :param smooth_st: The stateful_transform associated with the marginal
     smooth.
    :param kwargs: Smooth specific keyworded arguments.
    :return: Dictionary containing instantiated smooth object (key: 'smooth')
     and associated keyworded arguments (wo constraints, key: 'kwargs').

    :raise ValueError: if the requested smooth is not supported.
    """
    # Constraints arguments filter for CRS splines
    def remove_crs_cons(smooth_name, smooth_args):
        if "cons" in smooth_args:
            cons = smooth_args.pop("cons")
            warnings.warn("Removed requested constraint for smooth %r."
                          % (smooth_name,))
        return smooth_args

    # Supported smooths associated with their specific constraints filter:
    supported_smooths = {
        "BS": lambda smooth_args: smooth_args,
        "CR": lambda smooth_args: remove_crs_cons("cr", smooth_args),
        "CS": lambda smooth_args: remove_crs_cons("cs", smooth_args),
        "CC": lambda smooth_args: remove_crs_cons("cc", smooth_args)}

    if smooth_st.__name__ not in supported_smooths:
        raise ValueError("Unsupported marginal smooth '%r'."
                         % (smooth_st.__name__,))

    # Remove requested constraints, if any, for the marginal smooth:
    supported_smooths[smooth_st.__name__](kwargs)

    return {"smooth": smooth_st.__patsy_stateful_transform__(),
            "kwargs": kwargs}


class TE(object):
    """te(x1, .., xn, s=None, cons=None)

    Generates smooth of several covariates as a tensor product of the bases
    of marginal univariate smooths. The resulting basis dimension is the
    product of the basis dimensions of the marginal smooths. The usual usage
    is something like::

      y ~ 1 + te(x1, x2, s=(ms(cr, k=5), ms(cc, k=7)), cons=True)

    to fit ``y`` as a smooth function of both ``x1`` and ``x2``, with a natural
    cubic spline for ``x1`` marginal smooth and a cyclic cubic spline for
    ``x2`` (and centering constraint absorbed in the resulting design matrix).

    :arg s: A tuple describing each marginal smooth and their specific
     arguments using the function ``ms()``. Supported marginal smooths are
     ``cr``, ``cs``, ``cc`` and ``bs``.
    :arg cons: Either a 2-d array defining the constraints or a boolean which,
     if set to ``True``, indicates that we should apply centering constraint
     (this constraint will be computed from the input data, remembered and
     re-used for prediction from the fitted model).
     The constraints are absorbed in the resulting design matrix. Note that
     ``cons=False`` is equivalent to no ``cons`` parameter.
     Marginal smooths ``cons`` parameter is ignored, only a global constraint
     on the whole tensor product smooth is applied if requested.

    Using this function requires scipy be installed.

    .. note:: This function reproduce the tensor product smooth 'te' as
      implemented in the R package 'mgcv'.
      See also 'Generalized Additive Models', Simon N. Wood, 2006, pp 158-163

    .. versionadded:: 0.2.1
    """
    def __init__(self):
        self._concat_args = None
        self._cons = None
        self._marginal_smooths = None

    def memorize_chunk(self, *args, **kwargs):
        self._cons = kwargs.get('cons', None)
        if self._marginal_smooths is None:
            self._marginal_smooths = kwargs.get('s', ())
        if len(args) != len(self._marginal_smooths):
            raise ValueError("Tensor product: %r arguments but %r marginal "
                             "smooth definitions given (keyword argument 's=')."
                             % (len(args), len(self._marginal_smooths)))
        if self._concat_args is None:
            self._concat_args = [[] for _ in range(len(args))]

        for i in range(len(args)):
            x = args[i]
            x = np.atleast_1d(x)
            if x.ndim == 2 and x.shape[1] == 1:
                x = x[:, 0]
            if x.ndim > 1:
                raise ValueError("Input to 'te' must be 1-d, "
                                 "or 2-d column vectors.")
            st = self._marginal_smooths[i]['smooth']
            st_kwargs = self._marginal_smooths[i]['kwargs']
            st.memorize_chunk(x, **st_kwargs)
            self._concat_args[i].append(x)

    def memorize_finish(self):
        dms = []
        for i in range(len(self._marginal_smooths)):
            st = self._marginal_smooths[i]['smooth']
            st_kwargs = self._marginal_smooths[i]['kwargs']
            st.memorize_finish()
            dms.append(st.transform(np.concatenate(self._concat_args[i]),
                                    **st_kwargs))
        tp = _row_tensor_product(dms)
        self._cons = _get_actual_constraints(
            self._cons,
            lambda: _get_centering_constraint_from_dmatrix(tp))

    def transform(self, *args, **kwargs):
        assert len(args) == len(self._marginal_smooths)
        dms = []
        for i in range(len(self._marginal_smooths)):
            st = self._marginal_smooths[i]['smooth']
            st_args = self._marginal_smooths[i]['kwargs']
            dms.append(st.transform(args[i], **st_args))

        return _get_te_dmatrix(dms, self._cons)

te = stateful_transform(TE)


def test_te():
    from patsy.highlevel import incr_dbuilder, build_design_matrices, dmatrix
    height = np.array([70., 65., 63., 72., 81., 83., 66., 75., 80., 75., 79., 76., 76., 69., 75., 74., 85., 86., 71., 64., 78., 80., 74., 72., 77., 81., 82., 80., 80., 80., 87.])
    girth = np.array([8.3, 8.6, 8.8, 10.5, 10.7, 10.8, 11., 11., 11.1, 11.2, 11.3, 11.4, 11.4, 11.7, 12., 12.9, 12.9, 13.3, 13.7, 13.8, 14., 14.2, 14.5, 16., 16.3, 17.3, 17.5, 17.9, 18., 18., 20.6])
    height_knots = np.array([63., 70., 76., 81., 87.])
    height_x = np.array([64., 82.]) # values for which we want a prediction
    girth_knots = np.array([8.30000, 10.73333, 11.26667, 12.90000, 14.06667, 16.96667, 20.60000])
    girth_x = np.array([10.8, 15.7]) # values for which we want a prediction
    result1 = dmatrix("te(height_x, girth_x, s=(ms(cs, k=height_knots), ms(cc, k=girth_knots)))")
    data_chunked = [{"height_x": height[:17], "girth_x": girth[:17]}, {"height_x": height[17:], "girth_x": girth[17:]}]
    new_data = {"height_x": height_x, "girth_x": girth_x} # values for which we want a prediction
    builder = incr_dbuilder("te(height_x, girth_x, s=[ms(cs, k=5), ms(cc, k=7)])", lambda: iter(data_chunked))
    result2 = build_design_matrices([builder], new_data)[0]
    assert np.allclose(result1, result2, atol=1e-5) # TODO: use full precision girth_knots
