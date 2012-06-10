# This file is part of Charlton
# Copyright (C) 2012 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

# Exhaustive end-to-end tests of the top-level API.

import numpy as np
from nose.tools import assert_raises
from charlton import CharltonError
from charlton.design_matrix import DesignMatrix
from charlton.eval import EvalEnvironment
from charlton.desc import ModelDesc, Term, LookupFactor, INTERCEPT
from charlton.categorical import C
from charlton.build import make_design_matrix_builders
from charlton.test_build import assert_full_rank, make_test_factors

from charlton.highlevel import *

# End-to-end tests need to include:
# - user-specified coding
# - transformations from the environment
# - term order
# - with and without response variable
# - incremental building with nested stateful transforms
# - use of builtins
# - test I(a / b) varies depending on __future__ state of caller

# incremental building:
#   ModelDesign.from_desc_and_data_iter_maker(desc,
#                                             data_iter_maker, *args, **kwargs)
#   incr_design(formula_like, eval_env, data_iter_maker, *args, **kwargs)
#   _get_design(formula_like, eval_env, data_iter_maker) # may return None

# XX what term ordering *do* we want?
# I guess:
#   1) all 0-order no-numeric
#   2) all 1st-order no-numeric
#   3) all 2nd-order no-numeric
#   4) ...
#   5) all 0-order with the first numeric interaction encountered
#   6) all 1st-order with the first numeric interaction encountered
#   7) ...
#   8) all 0-order with the second numeric interaction encountered
#   9) ...

def check_result(design, lhs, rhs, data,
                 expected_rhs_values, expected_rhs_names,
                 expected_lhs_values, expected_lhs_names): # pragma: no cover
    assert np.allclose(rhs, expected_rhs_values)
    assert rhs.column_info.column_names == expected_rhs_names
    if lhs is not None:
        assert np.allclose(lhs, expected_lhs_values)
        assert lhs.column_info.column_names == expected_lhs_names
    else:
        assert expected_lhs_values is None
        assert expected_lhs_names is None
    if design is not None:
        new_lhs, new_rhs = design.make_matrices(data)
        if lhs is None:
            assert new_lhs.shape == (new_rhs.shape[0], 0)
        else:
            assert np.allclose(new_lhs, lhs)
            assert new_lhs.column_info.column_names == expected_lhs_names
        assert np.allclose(new_rhs, rhs)
        assert new_rhs.column_info.column_names == expected_rhs_names

def t(formula_like, data, depth,
      expect_model_design,
      expected_rhs_values, expected_rhs_names,
      expected_lhs_values=None, expected_lhs_names=None,
      expected_design=None): # pragma: no cover
    if isinstance(depth, int):
        depth += 1
    if isinstance(formula_like, (basestring, ModelDesc, ModelDesign)):
        design = incr_design(formula_like, depth, iter, [data])
        lhs, rhs = design.make_matrices(data)
        if lhs.shape[1] == 0:
            lhs = None
        check_result(design, lhs, rhs, data,
                     expected_rhs_values, expected_rhs_names,
                     expected_lhs_values, expected_lhs_names)
    else:
        assert_raises(CharltonError, incr_design, formula_like, None,
                      iter, [data])
    if expected_lhs_values is None:
        (design, rhs) = design_and_matrix(formula_like, data, depth)
        assert (design is not None) == expect_model_design
        if expected_design is not None:
            assert design is expected_design
        check_result(design, None, rhs, data,
                     expected_rhs_values, expected_rhs_names,
                     expected_lhs_values, expected_lhs_names)

        rhs = dmatrix(formula_like, data, depth)
        check_result(None, None, rhs, data,
                     expected_rhs_values, expected_rhs_names,
                     expected_lhs_values, expected_lhs_names)

        # We inline assert_raises here to avoid complications with the
        # depth argument.
        for f in (design_and_matrices, dmatrices):
            try:
                f(formula_like, data, depth)
            except CharltonError:
                pass
            else:
                raise AssertionError
    else:
        for f in (design_and_matrix, dmatrix):
            try:
                f(formula_like, data, depth)
            except CharltonError:
                pass
            else:
                raise AssertionError

        (design, lhs, rhs) = design_and_matrices(formula_like, data,
                                                 depth)
        assert (design is not None) == expect_model_design
        if expected_design is not None:
            assert design is expected_design
        check_result(design, lhs, rhs, data,
                     expected_rhs_values, expected_rhs_names,
                     expected_lhs_values, expected_lhs_names)

        (lhs, rhs) = dmatrices(formula_like, data, depth)
        check_result(None, lhs, rhs, data,
                     expected_rhs_values, expected_rhs_names,
                     expected_lhs_values, expected_lhs_names)

def t_invalid(formula_like, data, depth, exc=CharltonError): # pragma: no cover
    if isinstance(depth, int):
        depth += 1
    for f in (design_and_matrix, dmatrix, design_and_matrices, dmatrices):
        try:
            f(formula_like, data, depth)
        except exc:
            pass
        else:
            raise AssertionError

# Exercise all the different calling conventions for the high-level API
def test_formula_likes():
    # Plain array-like, rhs only
    t([[1, 2, 3], [4, 5, 6]], {}, 0,
      False,
      [[1, 2, 3], [4, 5, 6]], ["x0", "x1", "x2"])
    t((None, [[1, 2, 3], [4, 5, 6]]), {}, 0,
      False,
      [[1, 2, 3], [4, 5, 6]], ["x0", "x1", "x2"])
    t(np.asarray([[1, 2, 3], [4, 5, 6]]), {}, 0,
      False,
      [[1, 2, 3], [4, 5, 6]], ["x0", "x1", "x2"])
    t((None, np.asarray([[1, 2, 3], [4, 5, 6]])), {}, 0,
      False,
      [[1, 2, 3], [4, 5, 6]], ["x0", "x1", "x2"])
    dm = DesignMatrix([[1, 2, 3], [4, 5, 6]], default_column_prefix="foo")
    t(dm, {}, 0,
      False,
      [[1, 2, 3], [4, 5, 6]], ["foo0", "foo1", "foo2"])
    t((None, dm), {}, 0,
      False,
      [[1, 2, 3], [4, 5, 6]], ["foo0", "foo1", "foo2"])
      
    # Plain array-likes, lhs and rhs
    t(([1, 2], [[1, 2, 3], [4, 5, 6]]), {}, 0,
      False,
      [[1, 2, 3], [4, 5, 6]], ["x0", "x1", "x2"],
      [[1], [2]], ["y0"])
    t(([[1], [2]], [[1, 2, 3], [4, 5, 6]]), {}, 0,
      False,
      [[1, 2, 3], [4, 5, 6]], ["x0", "x1", "x2"],
      [[1], [2]], ["y0"])
    t((np.asarray([1, 2]), np.asarray([[1, 2, 3], [4, 5, 6]])), {}, 0,
      False,
      [[1, 2, 3], [4, 5, 6]], ["x0", "x1", "x2"],
      [[1], [2]], ["y0"])
    t((np.asarray([[1], [2]]), np.asarray([[1, 2, 3], [4, 5, 6]])), {}, 0,
      False,
      [[1, 2, 3], [4, 5, 6]], ["x0", "x1", "x2"],
      [[1], [2]], ["y0"])
    x_dm = DesignMatrix([[1, 2, 3], [4, 5, 6]], default_column_prefix="foo")
    y_dm = DesignMatrix([1, 2], default_column_prefix="bar")
    t((y_dm, x_dm), {}, 0,
      False,
      [[1, 2, 3], [4, 5, 6]], ["foo0", "foo1", "foo2"],
      [[1], [2]], ["bar0"])
    # number of rows must match
    t_invalid(([1, 2, 3], [[1, 2, 3], [4, 5, 6]]), {}, 0)

    # tuples must have the right size
    t_invalid(([[1, 2, 3]],), {}, 0)
    t_invalid(([[1, 2, 3]], [[1, 2, 3]], [[1, 2, 3]]), {}, 0)

    # Foreign ModelDesign-like objects
    class MockModelDesign(object):
        def make_matrices(self, data):
            return (data["Y"], data["X"])
        def __charlton_get_model_design__(self, data):
            return self
    mock_design = MockModelDesign()
    # make_matrices must return DesignMatrixes
    t_invalid(mock_design, {"Y": [1, 2], "X": DesignMatrix([[1, 2], [3, 4]])}, 0)
    t_invalid(mock_design, {"Y": DesignMatrix([1, 2]), "X": [[1, 2], [3, 4]]}, 0)
    # And they must have valid metadata (which is not preserved over slicing)
    t_invalid(mock_design, {"Y": DesignMatrix([1, 2])[:, 0],
                            "X": DesignMatrix([[1, 2], [3, 4]])}, 0)
    t_invalid(mock_design, {"Y": DesignMatrix([1, 2]),
                            "X": DesignMatrix([[1, 2], [3, 4]])[:, 0]}, 0)
    # Number of rows must match
    t_invalid(mock_design, {"Y": DesignMatrix([1, 2, 3]),
                            "X": DesignMatrix([[1, 2], [3, 4]])}, 0)

    t(mock_design,
      {"Y": DesignMatrix([1, 2], default_column_prefix="Y"),
       "X": DesignMatrix([[1, 2], [3, 4]], default_column_prefix="X")},
      0,
      True,
      [[1, 2], [3, 4]], ["X0", "X1"],
      [[1], [2]], ["Y0"],
      expected_design=mock_design)

    # string formulas
    t("y ~ x", {"y": [1, 2], "x": [3, 4]}, 0,
      True,
      [[1, 3], [1, 4]], ["Intercept", "x"],
      [[1], [2]], ["y"])
    t("~ x", {"y": [1, 2], "x": [3, 4]}, 0,
      True,
      [[1, 3], [1, 4]], ["Intercept", "x"])
    t("x + y", {"y": [1, 2], "x": [3, 4]}, 0,
      True,
      [[1, 3, 1], [1, 4, 2]], ["Intercept", "x", "y"])
    
    # ModelDesc
    desc = ModelDesc(None, [], [Term([LookupFactor("x")])])
    t(desc, {"x": [1.5, 2.5, 3.5]}, 0,
      True,
      [[1.5], [2.5], [3.5]], ["x"])
    desc = ModelDesc(None, [], [Term([]), Term([LookupFactor("x")])])
    t(desc, {"x": [1.5, 2.5, 3.5]}, 0,
      True,
      [[1, 1.5], [1, 2.5], [1, 3.5]], ["Intercept", "x"])
    desc = ModelDesc(None,
                     [Term([LookupFactor("y")])],
                     [Term([]), Term([LookupFactor("x")])])
    t(desc, {"x": [1.5, 2.5, 3.5], "y": [10, 20, 30]}, 0,
      True,
      [[1, 1.5], [1, 2.5], [1, 3.5]], ["Intercept", "x"],
      [[10], [20], [30]], ["y"])

    # ModelDesign
    termlists = ([], [Term([]), Term([LookupFactor("x")])])
    builders = make_design_matrix_builders(termlists,
                                           iter, [{"x": [1, 2, 3]}])
    design = ModelDesign(None, *builders)
    t(design, {"x": [10, 20, 30]}, 0,
      True,
      [[1, 10], [1, 20], [1, 30]], ["Intercept", "x"],
      expected_design=design)
    
    # check depth arguments
    x_in_env = [1, 2, 3]
    t("~ x_in_env", {}, 0,
      True,
      [[1, 1], [1, 2], [1, 3]], ["Intercept", "x_in_env"])
    t("~ x_in_env", {"x_in_env": [10, 20, 30]}, 0,
      True,
      [[1, 10], [1, 20], [1, 30]], ["Intercept", "x_in_env"])
    # Trying to pull x_in_env out of our *caller* shouldn't work.
    t_invalid("~ x_in_env", {}, 1, exc=NameError)
    # But then again it should, if called from one down on the stack:
    def check_nested_call():
        x_in_env = "asdf"
        t("~ x_in_env", {}, 1,
          True,
          [[1, 1], [1, 2], [1, 3]], ["Intercept", "x_in_env"])
    check_nested_call()
    # passing in an explicit EvalEnvironment also works:
    e = EvalEnvironment.capture(1)
    t_invalid("~ x_in_env", {}, e, exc=NameError)
    e = EvalEnvironment.capture(0)
    def check_nested_call_2():
        x_in_env = "asdf"
        t("~ x_in_env", {}, e,
          True,
          [[1, 1], [1, 2], [1, 3]], ["Intercept", "x_in_env"])
    check_nested_call_2()

def test_term_info():
    data = {}
    data["a"], data["b"] = make_test_factors(2, 2)
    (design, rhs) = design_and_matrix("a:b", data)
    assert rhs.column_info.column_names == ["Intercept", "b[T.b2]",
                                            "a[T.a2]:b[b1]", "a[T.a2]:b[b2]"]
    assert rhs.column_info.term_names == ["Intercept", "a:b"]
    assert len(rhs.column_info.terms) == 2
    assert rhs.column_info.terms[0] == INTERCEPT
    
def test_data_types():
    data = {"a": [1, 2, 3],
            "b": [1.0, 2.0, 3.0],
            "c": np.asarray([1, 2, 3], dtype=np.float32),
            "d": [True, False, True],
            "e": ["foo", "bar", "baz"],
            "f": C([1, 2, 3]),
            "g": C(["foo", "bar", "baz"]),
            "h": np.array(["foo", 1, (1, "hi")], dtype=object),
            }
    t("~ 0 + a", data, 0, True,
      [[1], [2], [3]], ["a"])
    t("~ 0 + b", data, 0, True,
      [[1], [2], [3]], ["b"])
    t("~ 0 + c", data, 0, True,
      [[1], [2], [3]], ["c"])
    t("~ 0 + d", data, 0, True,
      [[0, 1], [1, 0], [0, 1]], ["d[False]", "d[True]"])
    t("~ 0 + e", data, 0, True,
      [[0, 0, 1], [1, 0, 0], [0, 1, 0]], ["e[bar]", "e[baz]", "e[foo]"])
    t("~ 0 + f", data, 0, True,
      [[1, 0, 0], [0, 1, 0], [0, 0, 1]], ["f[1]", "f[2]", "f[3]"])
    t("~ 0 + g", data, 0, True,
      [[0, 0, 1], [1, 0, 0], [0, 1, 0]], ["g[bar]", "g[baz]", "g[foo]"])
    # This depends on Python's sorting behavior:
    t("~ 0 + h", data, 0, True,
      [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
      ["h[1]", "h[foo]", "h[(1, 'hi')]"])
    
def test_categorical():
    data = {}
    data["a"], data["b"] = make_test_factors(2, 2)
    

# End-to-end tests on all the wacky things you can do with formulas.

