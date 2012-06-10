# This file is part of Charlton
# Copyright (C) 2011 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

# These are made available in the charlton.* namespace:
__all__ = ["ModelDesign", "dmatrix", "dmatrices",
           "design_and_matrix", "design_and_matrices"]

import numpy as np
from charlton import CharltonError
from charlton.design_matrix import DesignMatrix
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
    
# This always returns a length-three tuple,
#   design, response, predictors
# where
#   design is a ModelDesign or None
#   response is a DesignMatrix or None
#   predictors is a DesignMatrix
# The input 'formula_like' could be like:
#   (np.ndarray, np.ndarray)
#   (None, np.ndarray) # for predictor-only models
#   (DesignMatrix, DesignMatrix)
#   (None, DesignMatrix)
#   "y ~ x"
#   ModelDesc(...)
#   ModelDesign(...)
#   any object with a __charlton_make_modeldesign_alike__ function
def _design_and_matrices(formula_like, data, depth=0):
    if isinstance(formula_like, np.ndarray):
        formula_like = (None, formula_like)
    if isinstance(formula_like, tuple):
        if len(formula_like) != 2:
            raise CharltonError("don't know what to do with a length-%s tuple"
                                % (len(formula_like),))
        lhs = formula_like[0]
        if lhs is not None:
            lhs = DesignMatrix(lhs)
        return (None, lhs, DesignMatrix(formula_like[1]))
    eval_env = EvalEnvironment.capture(depth + 1)
    if hasattr(formula_like, "__charlton_make_modeldesign_alike__"):
        design_like = formula_like.__charlton_make_modeldesign_alike__(data, eval_env)
        return (design_like,) + tuple(design_like.make_matrices(data))
    if isinstance(formula_like, basestring):
        formula_like = ModelDesc.from_formula(formula_like, eval_env)
    if isinstance(formula_like, ModelDesc):
        formula_like = ModelDesign.from_desc_and_data(formula_like, data)
    if isinstance(formula_like, ModelDesign):
        return (formula_like,) + tuple(formula_like.make_matrices(data))
    raise CharltonError, "don't know what to do with %r" % (formula_like,)

def design_and_matrices(formula_like, data, depth=0):
    (design, lhs, rhs) = _design_and_matrices(formula_like, data,
                                              depth=depth + 1)
    if lhs is None:
        raise CharltonError("model is missing required outcome variables")
    return (design, lhs, rhs)

def design_and_matrix(formula_like, data, depth=0):
    (design, lhs, rhs) = _design_and_matrices(formula_like, data,
                                              depth=depth + 1)
    if lhs is not None:
        raise CharltonError("encountered outcome variables for a model "
                            "that does not expect them")
    return (design, rhs)

def dmatrix(formula_like, data, depth=0):
    return design_and_matrix(formula_like, data, depth=depth + 1)[1]

def dmatrices(formula_like, data, depth=0):
    return design_and_matrices(formula_like, data, depth=depth + 1)[1:]
