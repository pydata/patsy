# This file is part of Charlton
# Copyright (C) 2011 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

# The user-level convenience API:
__all__ = ["model_spec_and_matrices", "model_matrix", "model_matrices"]

import numpy as np
from charlton import CharltonError
from charlton.model_matrix import ModelMatrix
from charlton.eval import EvalEnvironment
from charlton.desc import ModelDesc
from charlton.build import make_model_matrix_builders, make_model_matrices

class ModelSpec(object):
    def __init__(self, desc, lhs_builder, rhs_builder):
        self.desc = desc
        self.lhs_builder = lhs_builder
        self.rhs_builder = rhs_builder

    @classmethod
    def from_desc_and_data(cls, desc, data):
        def data_gen():
            yield data
        builders = make_model_matrix_builders([desc.lhs_terms, desc.rhs_terms],
                                              data_gen)
        return cls(desc, builders[0], builders[1])

    def make_matrices(self, data):
        return make_model_matrices([self.lhs_builder, self.rhs_builder], data)
    
# This always returns a length-three tuple,
#   spec, response, predictors
# where spec is a ModelSpec or None
#   response is a ModelMatrix or None
#   predictors is a ModelMatrix
# 'formula_like' could be like:
#   (np.ndarray, np.ndarray)
#   (None, np.ndarray) # for predictor-only models
#   "y ~ x"
#   ModelDesc(...)
#   ModelSpec(...)
#   any object with a __charlton_make_modelspec_alike__ function
def model_spec_and_matrices(formula_like, data, depth=0):
    if isinstance(formula_like, np.ndarray):
        formula_like = (None, formula_like)
    if isinstance(formula_like, tuple):
        if len(formula_like) != 2:
            raise CharltonError("don't know what to do with a length-%s tuple"
                                % (len(formula_like),))
        lhs = formula_like[0]
        if lhs is not None:
            lhs = ModelMatrix(lhs)
        return (None, lhs, ModelMatrix(formula_like[1]))
    eval_env = EvalEnvironment.capture(depth + 1)
    if hasattr(formula_like, "__charlton_make_modelspec_alike__"):
        spec_like = formula_like.__charlton_make_modelspec_alike__(data, eval_env)
        return (spec_like,) + tuple(spec_like.make_matrices(data))
    if isinstance(formula_like, basestring):
        formula_like = ModelDesc.from_formula(formula_like, eval_env)
    if isinstance(formula_like, ModelDesc):
        formula_like = ModelSpec.from_desc_and_data(formula_like, data)
    if isinstance(formula_like, ModelSpec):
        return (formula_like,) + tuple(formula_like.make_matrices(data))
    raise CharltonError, "don't know what to do with %r" % (formula_like,)

def model_matrices(formula_like, data, depth=0):
    spec_and_matrices = model_spec_and_matrices(formula_like,
                                                data,
                                                depth=depth + 1)
    return spec_and_matrices[1:]

def model_matrix(formula_like, data, depth=0):
    spec_and_matrices = model_spec_and_matrices(formula_like,
                                                data,
                                                depth=depth + 1)
    return spec_and_matrices[2]
