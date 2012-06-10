# This file is part of Charlton
# Copyright (C) 2011 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

# These are made available in the charlton.* namespace:
__all__ = ["ModelDesign", "dmatrix", "dmatrices",
           "design_and_matrix", "design_and_matrices"]

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

    @classmethod
    def from_desc_and_data(cls, desc, data):
        def data_gen():
            yield data
        builders = make_design_matrix_builders([desc.lhs_terms, desc.rhs_terms],
                                               data_gen)
        return cls(desc, builders[0], builders[1])

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
def _design_and_matrices(formula_like, data, eval_env):
    # Invariant: only one of these will be non-None at once
    design_like = None
    matrices = None
    
    if hasattr(formula_like, "__charlton_get_model_design__"):
        assert design_like is None
        design_like = formula_like.__charlton_get_model_design__(data)
    if isinstance(formula_like, basestring):
        eval_env = _get_env(eval_env)
        formula_like = ModelDesc.from_formula(formula_like, eval_env)
        # fallthrough
    if isinstance(formula_like, ModelDesc):
        formula_like = ModelDesign.from_desc_and_data(formula_like, data)
        # fallthrough
    if isinstance(formula_like, ModelDesign):
        assert design_like is None
        design_like = formula_like

    if isinstance(formula_like, tuple):
        assert matrices is None
        if len(formula_like) != 2:
            raise CharltonError("don't know what to do with a length %s "
                                "matrices tuple"
                                % (len(formula_like),))
        matrices = formula_like

    if design_like is None and matrices is None:
        # asanyarray is necessary here to allow DesignMatrixs to pass through
        formula_like = np.asanyarray(formula_like)
        matrices = (None, formula_like)

    # Both branches of this 'if' statement set up two variables lhs, rhs,
    # which are the matrices we will validate and return.
    if matrices is not None:
        assert design_like is None
        # some sort of explicit matrix or matrices were given, normalize their
        # format
        assert isinstance(matrices, tuple)
        assert len(matrices) == 2
        (lhs, rhs) = matrices
        rhs = DesignMatrix(rhs, default_column_prefix="x")
        if lhs is None:
            lhs = np.zeros((rhs.shape[0], 0), dtype=float)
        lhs = DesignMatrix(lhs, default_column_prefix="y")
    else:
        assert design_like is not None
        # We explicitly do *not* normalize the format of matrices that come
        # out of a design_like object -- but we do validate them for
        # correctness. This is because any downstream code that wants to do
        # predictions will call design_like.make_matrices directly, so we want
        # to make sure that make_matrices returns things in a good format to
        # start with.
        (lhs, rhs) = design_like.make_matrices(data)

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
