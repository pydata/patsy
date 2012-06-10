# This file is part of Charlton
# Copyright (C) 2011 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

# These are made available in the charlton.* namespace:
__all__ = ["ModelDesign", "dmatrix", "dmatrices",
           "design_and_matrix", "design_and_matrices",
           "incr_design"]

import numpy as np
from charlton import CharltonError
from charlton.design_matrix import DesignMatrix, DesignMatrixColumnInfo
from charlton.eval import EvalEnvironment
from charlton.desc import ModelDesc
from charlton.build import make_design_matrix_builders, make_design_matrices

class ModelDesign(object):
    def __init__(self, desc, lhs_builder, rhs_builder):
        self.desc = desc
        self.lhs_builder = lhs_builder
        self.rhs_builder = rhs_builder

    def make_matrices(self, data):
        return make_design_matrices([self.lhs_builder, self.rhs_builder], data)

def _get_env(eval_env):
    if isinstance(eval_env, int):
        # Here eval_env=0 refers to our caller's caller.
        return EvalEnvironment.capture(eval_env + 2)
    return eval_env

# This always returns a length-three tuple,
#   design, response, predictors
# where
#   design is an object with a make_matrices method, or else None
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
#   ModelDesign(...)
#   any object with a special method __charlton_get_model_design__

# Tries to build a design given a formula_like and an incremental data
# source. If formula_like is not capable of doing this, then returns None. (At
# the moment this requires that formula_like be a charlton formula or
# similar.)
def _try_incr_design(formula_like, eval_env, data_iter_maker, *args, **kwargs):
    if isinstance(formula_like, basestring):
        eval_env = _get_env(eval_env)
        formula_like = ModelDesc.from_formula(formula_like, eval_env)
        # fallthrough
    if isinstance(formula_like, ModelDesc):
        termlists = [formula_like.lhs_terms, formula_like.rhs_terms]
        builders = make_design_matrix_builders(termlists,
                                               data_iter_maker, *args, **kwargs)
        formula_like = ModelDesign(formula_like, *builders)
        # fallthrough
    if isinstance(formula_like, ModelDesign):
        return formula_like
    return None

def incr_design(formula_like, eval_env, data_iter_maker, *args, **kwargs):
    design_like = _try_incr_design(formula_like, _get_env(eval_env),
                                   data_iter_maker, *args, **kwargs)
    if design_like is None:
        raise CharltonError("bad formula-like object")
    return design_like

def _design_and_matrices(formula_like, data, eval_env):
    # Invariant: only one of these will be non-None at once
    if hasattr(formula_like, "__charlton_get_model_design__"):
        design_like = formula_like.__charlton_get_model_design__(data)
    else:
        design_like = _try_incr_design(formula_like, eval_env, iter, [data])
    # Both branch of this 'if' statement set up the variables (lhs, rhs),
    # which are the matrices we will validate and return.
    if design_like is not None:
        # We explicitly do *not* normalize the format of matrices that come
        # out of a design_like object -- but we do validate them for
        # correctness. This is because any downstream code that wants to do
        # predictions will call design_like.make_matrices directly, so we want
        # to make sure that make_matrices returns things in a good format to
        # start with.
        (lhs, rhs) = design_like.make_matrices(data)
    else:
        # No design, but maybe we can still get matrices
        assert design_like is None
        if isinstance(formula_like, tuple):
            assert design_like is None
            if len(formula_like) != 2:
                raise CharltonError("don't know what to do with a length %s "
                                    "matrices tuple"
                                    % (len(formula_like),))
            matrices = formula_like
        else:
            # asanyarray is necessary here to allow DesignMatrixes to pass
            # through
            matrices = (None, np.asanyarray(formula_like))
        # some sort of explicit matrix or matrices were given, normalize their
        # format
        assert isinstance(matrices, tuple)
        assert len(matrices) == 2
        (lhs, rhs) = matrices
        rhs = DesignMatrix(rhs, default_column_prefix="x")
        if lhs is None:
            lhs = np.zeros((rhs.shape[0], 0), dtype=float)
        lhs = DesignMatrix(lhs, default_column_prefix="y")

    if not isinstance(lhs, DesignMatrix):
        raise CharltonError("lhs matrix must be DesignMatrix")
    if not isinstance(getattr(lhs, "column_info", None),
                      DesignMatrixColumnInfo):
        raise CharltonError("lhs DesignMatrix has invalid format")
    if not isinstance(rhs, DesignMatrix):
        raise CharltonError("rhs matrix must be DesignMatrix")
    if not isinstance(getattr(rhs, "column_info", None),
                      DesignMatrixColumnInfo):
        raise CharltonError("rhs DesignMatrix has invalid format")
    if lhs.shape[0] != rhs.shape[0]:
        raise CharltonError("shape mismatch: outcome matrix has %s rows, "
                            "predictor matrix has %s rows"
                            % (lhs.shape[0], rhs.shape[0]))

    return (design_like, lhs, rhs)

def design_and_matrices(formula_like, data, eval_env=0):
    (design, lhs, rhs) = _design_and_matrices(formula_like, data,
                                              _get_env(eval_env))
    if lhs.shape[1] == 0:
        raise CharltonError("model is missing required outcome variables")
    return (design, lhs, rhs)

def design_and_matrix(formula_like, data, eval_env=0):
    (design, lhs, rhs) = _design_and_matrices(formula_like, data,
                                              _get_env(eval_env))
    if lhs.shape[1] != 0:
        raise CharltonError("encountered outcome variables for a model "
                            "that does not expect them")
    return (design, rhs)

def dmatrix(formula_like, data, eval_env=0):
    return design_and_matrix(formula_like, data, _get_env(eval_env))[1]

def dmatrices(formula_like, data, eval_env=0):
    return design_and_matrices(formula_like, data, _get_env(eval_env))[1:]
