# This file is part of Patsy
# See file LICENSE.txt for license information.

import numpy as np
import pytest

from patsy import Poly, dmatrix, build_design_matrices


@pytest.mark.parametrize("intercept", [True, False])
@pytest.mark.parametrize("kind", ["float", "readonly", "view", "integer", "list"])
def test_poly_scores_input_unchanged(intercept, kind):
    owner = np.array([0.0, 91.0, 10.0, 92.0, 12.0, 93.0])
    if kind == "view":
        scores = owner[::2]
    elif kind == "integer":
        scores = np.array([0, 10, 12])
    elif kind == "list":
        scores = [0.0, 10.0, 12.0]
    else:
        scores = np.array([0.0, 10.0, 12.0])
    if kind == "readonly":
        scores.setflags(write=False)
    before = np.asarray(scores).copy()
    owner_before = owner.copy()
    method = "code_with_intercept" if intercept else "code_without_intercept"
    expected = getattr(Poly(before.tolist()), method)(["a", "b", "c"])
    actual = getattr(Poly(scores), method)(["a", "b", "c"])
    np.testing.assert_allclose(actual.matrix, expected.matrix)
    assert actual.column_suffixes == expected.column_suffixes
    np.testing.assert_array_equal(scores, before)
    np.testing.assert_array_equal(owner, owner_before)


@pytest.mark.parametrize("intercept", [True, False])
@pytest.mark.parametrize("readonly", [True, False])
def test_poly_scores_formula_input_unchanged(intercept, readonly):
    scores = np.array([0.0, 10.0, 12.0])
    before = scores.copy()
    if readonly:
        scores.setflags(write=False)
    formula = "C(a, Poly(scores))" if intercept else "0 + C(a, Poly(scores))"
    a = ["a", "b", "c", "a", "b", "c"]
    actual = dmatrix(formula, {"a": a, "scores": scores})
    expected = dmatrix(formula, {"a": a, "scores": before.tolist()})
    np.testing.assert_allclose(actual, expected)
    assert actual.design_info.column_names == expected.design_info.column_names
    prediction = build_design_matrices([actual.design_info], {"a": ["c", "a", "b"]})[0]
    np.testing.assert_allclose(prediction, actual[[2, 0, 1]])
    np.testing.assert_array_equal(scores, before)
