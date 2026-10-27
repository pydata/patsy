# This file is part of Patsy
# Copyright (C) 2013 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

# Regression tests for fixed bugs (when not otherwise better covered somewhere
# else)

from patsy import EvalEnvironment, dmatrix, build_design_matrices, PatsyError, Origin
import numpy as np
import pytest
from patsy.util import have_pandas


def test_issue_11():
    # Give a sensible error message for level mismatches
    # (At some points we've failed to put an origin= on these errors)
    env = EvalEnvironment.capture()
    data = {"X": [0, 1, 2, 3], "Y": [1, 2, 3, 4]}
    formula = "C(X) + Y"
    new_data = {"X": [0, 0, 1, 2, 3, 3, 4], "Y": [1, 2, 3, 4, 5, 6, 7]}
    info = dmatrix(formula, data)
    try:
        build_design_matrices([info.design_info], new_data)
    except PatsyError as e:
        assert e.origin == Origin(formula, 0, 4)
    else:
        assert False


@pytest.mark.parametrize("level", [False, True])
def test_single_boolean_level_array(level):
    matrix = dmatrix(
        "0 + C(x, levels=levels)",
        {"x": np.array([level, level]), "levels": [level]},
    )
    np.testing.assert_array_equal(matrix, [[1], [1]])
    info = matrix.design_info
    assert next(iter(info.factor_infos.values())).categories == (level,)
    prediction = build_design_matrices(
        [info], {"x": np.array([level]), "levels": [level]}
    )[0]
    np.testing.assert_array_equal(prediction, [[1]])


@pytest.mark.parametrize("level", [False, True])
def test_single_boolean_level_prediction(level):
    matrix = dmatrix("0 + C(x, levels=levels)", {"x": [level], "levels": [level]})
    prediction = build_design_matrices(
        [matrix.design_info], {"x": np.array([level]), "levels": [level]}
    )[0]
    np.testing.assert_array_equal(prediction, [[1]])
    with pytest.raises(PatsyError, match="does not match any of the expected levels"):
        build_design_matrices(
            [matrix.design_info], {"x": np.array([not level]), "levels": [level]}
        )


@pytest.mark.skipif(not have_pandas, reason="pandas is not available")
@pytest.mark.parametrize("level", [False, True])
def test_single_boolean_level_series(level):
    import pandas

    matrix = dmatrix(
        "0 + C(x, levels=levels)",
        {"x": pandas.Series([level, level], index=[10, 20]), "levels": [level]},
        return_type="dataframe",
    )
    np.testing.assert_array_equal(matrix, [[1], [1]])
    assert list(matrix.index) == [10, 20]
