# This file is part of Charlton
# Copyright (C) 2011 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

# These are made available in the charlton.* namespace:
__all__ = ["dmatrix", "dmatrices",
           "incr_dbuilder", "incr_dbuilders"]

# problems:
#   statsmodels reluctant to pass around separate eval environment, suggesting
#     that design_and_matrices-equivalent should return a formula_like
#   is ModelDesc really the high-level thing?
#   ModelDesign doesn't work -- need to work with the builder set
#   want to be able to return either a matrix or a pandas dataframe

import numpy as np
from charlton import CharltonError
from charlton.design_info import DesignMatrix, DesignInfo
from charlton.eval import EvalEnvironment
from charlton.desc import ModelDesc
from charlton.build import (design_matrix_builders,
                            build_design_matrices,
                            DesignMatrixBuilder)
from charlton.util import (have_pandas, asarray_or_pandas,
                           atleast_2d_column_default)

if have_pandas:
    import pandas

def _get_env(eval_env):
    if isinstance(eval_env, int):
        # Here eval_env=0 refers to our caller's caller.
        return EvalEnvironment.capture(eval_env + 2)
    return eval_env

# Tries to build a (lhs, rhs) design given a formula_like and an incremental
# data source. If formula_like is not capable of doing this, then returns
# None.
def _try_incr_builders(formula_like, data_iter_maker, eval_env):
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

def incr_dbuilder(formula_like, data_iter_maker, eval_env=0):
    builders = _try_incr_builders(formula_like, data_iter_maker,
                                  _get_env(eval_env))
    if builders is None:
        raise CharltonError("bad formula-like object")
    if len(builders[0].design_info.column_names) > 0:
        raise CharltonError("encountered outcome variables for a model "
                            "that does not expect them")
    return builders[1]

def incr_dbuilders(formula_like, data_iter_maker, eval_env=0):
    builders = _try_incr_builders(formula_like, data_iter_maker,
                                  _get_env(eval_env))
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
def _do_highlevel_design(formula_like, data, eval_env, output_type):
    if output_type == "dataframe" and not have_pandas:
        raise CharltonError("pandas.DataFrame was requested, but pandas "
                            "is not installed")
    if output_type not in ("matrix", "dataframe"):
        raise CharltonError("unrecognized output type %r, should be "
                            "'matrix' or 'dataframe'" % (output_type,))
    def data_iter_maker():
        return iter([data])
    builders = _try_incr_builders(formula_like, data_iter_maker, eval_env)
    if builders is not None:
        return build_design_matrices(builders, data,
                                     output_type=output_type)
    else:
        # No builders, but maybe we can still get matrices
        if isinstance(formula_like, tuple):
            if len(formula_like) != 2:
                raise CharltonError("don't know what to do with a length %s "
                                    "matrices tuple"
                                    % (len(formula_like),))
            (lhs, rhs) = formula_like
        else:
            # subok=True is necessary here to allow DesignMatrixes to pass
            # through
            (lhs, rhs) = (None, asarray_or_pandas(formula_like, subok=True))
        # some sort of explicit matrix or matrices were given. Currently we
        # have them in one of these forms:
        #   -- an ndarray or subclass
        #   -- a DesignMatrix
        #   -- a pandas.Series
        #   -- a pandas.DataFrame
        # and we have to produce a standard output format.
        def _regularize_matrix(m, default_column_prefix):
            di = DesignInfo.from_array(m, default_column_prefix)
            if have_pandas and isinstance(m, (pandas.Series, pandas.DataFrame)):
                orig_index = m.index
            else:
                orig_index = None
            if output_type == "dataframe":
                m = atleast_2d_column_default(m, preserve_pandas=True)
                m = pandas.DataFrame(m)
                m.columns = di.column_names
                m.design_info = di
                return (m, orig_index)
            else:
                return (DesignMatrix(m, di), orig_index)
        rhs, rhs_orig_index = _regularize_matrix(rhs, "x")
        if lhs is None:
            lhs = np.zeros((rhs.shape[0], 0), dtype=float)
        lhs, lhs_orig_index = _regularize_matrix(lhs, "y")

        assert isinstance(getattr(lhs, "design_info", None), DesignInfo)
        assert isinstance(getattr(rhs, "design_info", None), DesignInfo)
        if lhs.shape[0] != rhs.shape[0]:
            raise CharltonError("shape mismatch: outcome matrix has %s rows, "
                                "predictor matrix has %s rows"
                                % (lhs.shape[0], rhs.shape[0]))
        if rhs_orig_index is not None and lhs_orig_index is not None:
            if not np.array_equal(rhs_orig_index, lhs_orig_index):
                raise CharltonError("index mismatch: outcome and "
                                    "predictor have incompatible indexes")
        if output_type == "dataframe":
            if rhs_orig_index is not None and lhs_orig_index is None:
                lhs.index = rhs.index
            if rhs_orig_index is None and lhs_orig_index is not None:
                rhs.index = lhs.index
        return (lhs, rhs)

def dmatrix(formula_like, data={}, eval_env=0, output_type="matrix"):
    """Construct a single design matrix given a formula_like and data.

    """
    (lhs, rhs) = _do_highlevel_design(formula_like, data, _get_env(eval_env),
                                      output_type)
    if lhs.shape[1] != 0:
        raise CharltonError("encountered outcome variables for a model "
                            "that does not expect them")
    return rhs

def dmatrices(formula_like, data={}, eval_env=0, output_type="matrix"):
    """Construct two design matrices given a formula_like and data.

    This function is identical to :func:`dmatrix`, except that it requires the
    formula to specify both a left-hand side outcome matrix and a right-hand
    side predictors matrix, which are return as a tuple. See :func:`dmatrix`
    for details.
    """
    (lhs, rhs) = _do_highlevel_design(formula_like, data, _get_env(eval_env),
                                      output_type)
    if lhs.shape[1] == 0:
        raise CharltonError("model is missing required outcome variables")
    return (lhs, rhs)
