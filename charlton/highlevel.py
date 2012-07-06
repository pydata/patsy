# This file is part of Charlton
# Copyright (C) 2011 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

# These are made available in the charlton.* namespace:
__all__ = ["dmatrix", "dmatrices", "incr_dbuilder", "incr_dbuilders"]

# problems:
#   statsmodels reluctant to pass around separate eval environment, suggesting
#     that design_and_matrices-equivalent should return a formula_like
#   is ModelDesc really the high-level thing?
#   ModelDesign doesn't work -- need to work with the builder set
#   want to be able to return either a matrix or a pandas dataframe

import numpy as np
from charlton import CharltonError
from charlton.design_matrix import DesignMatrix, DesignInfo
from charlton.eval import EvalEnvironment
from charlton.desc import ModelDesc
from charlton.build import (design_matrix_builders,
                            build_design_matrices,
                            DesignMatrixBuilder)

def _get_env(eval_env):
    if isinstance(eval_env, int):
        # Here eval_env=0 refers to our caller's caller.
        return EvalEnvironment.capture(eval_env + 2)
    return eval_env

# Tries to build a (lhs, rhs) design given a formula_like and an incremental
# data source. If formula_like is not capable of doing this, then returns
# None.
def _try_incr_builders(formula_like, eval_env, data_iter_maker):
    if isinstance(formula_like, DesignMatrixBuilder):
        return (design_matrix_builders([[]], data_iter_maker)[0],
                formula_like)
    if (isinstance(formula_like, tuple)
        and len(formula_like) == 2
        and isinstance(formula_like[0], DesignMatrixBuilder)
        and isinstance(formula_like[1], DesignMatrixBuilder)):
        return formula_like
    if hasattr(formula_like, "__charlton_get_model_desc__"):
        formula_like = formula_like.__charlton_get_model_desc__(eval_env)
        if not isinstance(formula_like, ModelDesc):
            raise CharltonError("bad value from %r.__charlton_get_model_desc__"
                                % (formula_like,))
        # fallthrough
    if isinstance(formula_like, basestring):
        eval_env = _get_env(eval_env)
        formula_like = ModelDesc.from_formula(formula_like, eval_env)
        # fallthrough
    if isinstance(formula_like, ModelDesc):
        return design_matrix_builders([formula_like.lhs_termlist,
                                       formula_like.rhs_termlist],
                                      data_iter_maker)
    else:
        return None

def incr_dbuilder(formula_like, eval_env, data_iter_maker):
    builders = _try_incr_builders(formula_like, _get_env(eval_env),
                                  data_iter_maker)
    if builders is None:
        raise CharltonError("bad formula-like object")
    if len(builders[0].design_info.column_names) > 0:
        raise CharltonError("encountered outcome variables for a model "
                            "that does not expect them")
    return builders[1]

def incr_dbuilders(formula_like, eval_env, data_iter_maker):
    builders = _try_incr_builders(formula_like, _get_env(eval_env),
                                  data_iter_maker)
    if builders is None:
        raise CharltonError("bad formula-like object")
    if len(builders[0].design_info.column_names) == 0:
        raise CharltonError("model is missing required outcome variables")
    return builders

# This always returns a length-two tuple,
#   response, predictors
# where
#   response is a DesignMatrix (possibly with 0 columns)
#   predictors is a DesignMatrix
# The input 'formula_like' could be like:
#   (np.ndarray, np.ndarray)
#   (DesignMatrix, DesignMatrix)
#   (None, DesignMatrix)
#   np.ndarray  # for predictor-only models
#   DesignMatrix
#   (None, np.ndarray)
#   "y ~ x"
#   ModelDesc(...)
#   DesignMatrixBuilder
#   (DesignMatrixBuilder, DesignMatrixBuilder)
#   any object with a special method __charlton_get_model_desc__
def _do_highlevel_design(formula_like, data, eval_env):
    def data_iter_maker():
        return iter([data])
    builders = _try_incr_builders(formula_like, eval_env, data_iter_maker)
    if builders is not None:
        return build_design_matrices(builders, data)
    else:
        # No builders, but maybe we can still get matrices
        if isinstance(formula_like, tuple):
            if len(formula_like) != 2:
                raise CharltonError("don't know what to do with a length %s "
                                    "matrices tuple"
                                    % (len(formula_like),))
            (lhs, rhs) = formula_like
        else:
            # asanyarray is necessary here to allow DesignMatrixes to pass
            # through
            (lhs, rhs) = (None, np.asanyarray(formula_like))
        # some sort of explicit matrix or matrices were given, normalize their
        # format
        rhs = DesignMatrix(rhs, default_column_prefix="x")
        if lhs is None:
            lhs = np.zeros((rhs.shape[0], 0), dtype=float)
        lhs = DesignMatrix(lhs, default_column_prefix="y")

        assert isinstance(lhs, DesignMatrix)
        assert isinstance(getattr(lhs, "design_info", None), DesignInfo)
        assert isinstance(rhs, DesignMatrix)
        assert isinstance(getattr(rhs, "design_info", None), DesignInfo)
        if lhs.shape[0] != rhs.shape[0]:
            raise CharltonError("shape mismatch: outcome matrix has %s rows, "
                                "predictor matrix has %s rows"
                                % (lhs.shape[0], rhs.shape[0]))
        return (lhs, rhs)

def dmatrix(formula_like, data={}, eval_env=0):
    (lhs, rhs) = _do_highlevel_design(formula_like, data, _get_env(eval_env))
    if lhs.shape[1] != 0:
        raise CharltonError("encountered outcome variables for a model "
                            "that does not expect them")
    return rhs

def dmatrices(formula_like, data={}, eval_env=0):
    (lhs, rhs) = _do_highlevel_design(formula_like, data, _get_env(eval_env))
    if lhs.shape[1] == 0:
        raise CharltonError("model is missing required outcome variables")
    return (lhs, rhs)
