# This file is part of Charlton
# Copyright (C) 2011 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

# This file defines the model specification class, ModelSpec. Unlike a
# ModelDesc (which describes a model in general terms), a ModelSpec specifies
# all the details about it --- it knows how many columns the model matrix will
# have (and what they're called), it knows which predictors are categorical
# (and how they're coded), and it holds state on behalf of any stateful
# factors.

import numpy as np
from charlton.origin import CharltonErrorWithOrigin
from charlton.categorical import CategoricalTransform, Categorical
from charlton.util import atleast_2d_column_default, odometer_iter
from charlton.eval import DictStack

class _MockFactor(object):
    def eval(self, state, env):
        return env["mock"]

    def name(self):
        return "MOCKMOCK"

def _max_allowed_dim(dim, arr, factor):
    if arr.ndim > dim:
        msg = ("factor '%s' evaluates to an %s-dimensional array; I only "
               "handle arrays with dimension <= %s"
               % (factor.name(), arr.ndim, dim))
        raise CharltonErrorWithOrigin(msg, factor)

def test__max_allowed_dim():
    from nose.tools import assert_raises
    f = _MockFactor()
    _max_allowed_dim(1, np.array(1), f)
    _max_allowed_dim(1, np.array([1]), f)
    assert_raises(CharltonErrorWithOrigin, _max_allowed_dim, 1, np.array([[1]]), f)
    assert_raises(CharltonErrorWithOrigin, _max_allowed_dim, 1, np.array([[[1]]]), f)
    _max_allowed_dim(2, np.array(1), f)
    _max_allowed_dim(2, np.array([1]), f)
    _max_allowed_dim(2, np.array([[1]]), f)
    assert_raises(CharltonErrorWithOrigin, _max_allowed_dim, 2, np.array([[[1]]]), f)

class _BoolToCategorical(object):
    def __init__(self, factor):
        self.factor = factor

    def transform(self, data):
        data = np.asarray(data)
        _max_allowed_dim(1, data, self.factor)
        # issubdtype(int, bool) is true! So we can't use it:
        if not data.dtype.kind == "b":
            raise CharltonErrorWithOrigin("factor %s, which I thought was "
                                          "boolean, gave non-boolean data "
                                          "of dtype %s"
                                          % (self.factor.name(), data.dtype),
                                          self.factor)
        return Categorical(data, levels=[False, True])

def test__BoolToCategorical():
    from nose.tools import assert_raises
    f = _MockFactor()
    btc = _BoolToCategorical(f)
    cat = btc.transform([True, False, True, True])
    assert cat.levels == (False, True)
    assert np.issubdtype(cat.int_array.dtype, int)
    assert np.all(cat.int_array == [1, 0, 1, 1])
    assert_raises(CharltonErrorWithOrigin, btc.transform, [1, 0, 1])
    assert_raises(CharltonErrorWithOrigin, btc.transform, ["a", "b"])
    assert_raises(CharltonErrorWithOrigin, btc.transform, [[True]])

class NumericFactorEvaluator(object):
    def __init__(self, factor, state, expected_columns):
        self.factor = factor
        self.state = state
        self.expected_columns = expected_columns

    def eval(self, env):
        result = self.factor.eval(self.state, env)
        result = atleast_2d_column_default(result)
        _max_allowed_dim(2, result, self.factor)
        if result.shape[1] != self.expected_columns:
            raise CharltonErrorWithOrigin("when evaluating factor %s, I got "
                                          "%s columns instead of the %s "
                                          "I was expecting"
                                          % (self.factor.name(),
                                             self.expected_columns,
                                             result.shape[1]),
                                          self.factor)
        if not np.issubdtype(result.dtype, np.number):
            raise CharltonErrorWithOrigin("when evaluating numeric factor %s, "
                                          "I got non-numeric data of type '%s'"
                                          % (self.factor.name(),
                                             result.dtype),
                                          self.factor)
        return result

def test_NumericFactorEvaluator():
    from nose.tools import assert_raises
    nf1 = NumericFactorEvaluator(_MockFactor(), {}, 1)
    eval123 = nf1.eval({"mock": [1, 2, 3]})
    assert eval123.shape == (3, 1)
    assert np.all(eval123 == [[1], [2], [3]])
    assert_raises(CharltonErrorWithOrigin, nf1.eval, {"mock": [[[1]]]})
    assert_raises(CharltonErrorWithOrigin, nf1.eval, {"mock": [[1, 2]]})
    assert_raises(CharltonErrorWithOrigin, nf1.eval, {"mock": ["a", "b"]})
    assert_raises(CharltonErrorWithOrigin, nf1.eval, {"mock": [True, False]})
    nf2 = NumericFactorEvaluator(_MockFactor(), {}, 2)
    eval123321 = nf2.eval({"mock": [[1, 3], [2, 2], [3, 1]]})
    assert eval123321.shape == (3, 2)
    assert np.all(eval123321 == [[1, 3], [2, 2], [3, 1]])
    assert_raises(CharltonErrorWithOrigin, nf2.eval, {"mock": [1, 2, 3]})
    assert_raises(CharltonErrorWithOrigin, nf2.eval, {"mock": [[1, 2, 3]]})

class CategoricFactorEvaluator(object):
    def __init__(self, factor, state, postprocessor, expected_levels):
        self.factor = factor
        self.state = state
        self.postprocessor = postprocessor
        self.expected_levels = tuple(expected_levels)

    def eval(self, env):
        result = self.factor.eval(self.state, env)
        if self.postprocessor is not None:
            result = self.postprocessor.transform(result)
        if not isinstance(result, Categorical):
            msg = ("when evaluating categoric factor %s, I got a "
                   "result that is not of type Categorical (but rather %s)"
                   # result.__class__.__name__ would be better, but not
                   # defined for old-style classes:
                   % (self.factor.name(), result.__class__))
            raise CharltonErrorWithOrigin(msg, self.factor)
        if result.levels != self.expected_levels:
            msg = ("when evaluating categoric factor %s, I got Categorical "
                   " data with unexpected levels (wanted %s, got %s)"
                   % (self.factor.name(), self.expected_levels, result.levels))
            raise CharltonErrorWithOrigin(msg, self.factor)
        _max_allowed_dim(1, result.int_array, self.factor)
        # For consistency, evaluators *always* return 2d arrays (though in
        # this case it will always have only 1 column):
        return atleast_2d_column_default(result.int_array)

def test_CategoricFactorEvaluator():
    from nose.tools import assert_raises
    from charlton.categorical import Categorical
    cf1 = CategoricFactorEvaluator(_MockFactor(), {}, None, ["a", "b"])
    cat1 = cf1.eval({"mock": Categorical.from_strings(["b", "a", "b"])})
    assert cat1.shape == (3, 1)
    assert np.all(cat1 == [[1], [0], [1]])
    assert_raises(CharltonErrorWithOrigin, cf1.eval, {"mock": ["c"]})
    assert_raises(CharltonErrorWithOrigin, cf1.eval,
                  {"mock": Categorical.from_strings(["a", "c"])})
    assert_raises(CharltonErrorWithOrigin, cf1.eval,
                  {"mock": Categorical.from_strings(["a", "b"],
                                                    levels=["b", "a"])})
    assert_raises(CharltonErrorWithOrigin, cf1.eval, {"mock": [1, 0, 1]})
    bad_cat = Categorical.from_strings(["b", "a", "a", "b"])
    bad_cat.int_array.resize((2, 2))
    assert_raises(CharltonErrorWithOrigin, cf1.eval, {"mock": bad_cat})

    btc = _BoolToCategorical(_MockFactor())
    cf2 = CategoricFactorEvaluator(_MockFactor(), {}, btc, [False, True])
    cat2 = cf2.eval({"mock": [True, False, False, True]})
    assert cat2.shape == (4, 1)
    assert np.all(cat2 == [[1], [0], [0], [1]])

# This class is responsible for producing some columns in a final model matrix
# output:
class ColumnBuilder(object):
    def __init__(self, factors, contrasts):
        self.factors = factors
        self.contrasts = contrasts

    def build(self, factor_values, out):
        # use odometer_iter to go through all the columns of all the factors
        columns_per_factor = []
        for factor in self.factors:
            if factor in self.contrasts:
                columns_per_factor.append(self.contrasts[factor].shape[1])
            else:
                columns_per_factor.append(factor_values[factor].shape[1])
        assert np.prod(columns_per_factor, dtype=int) == out.shape[1]
        out[:] = 1
        for i, column_idxs in enumerate(odometer_iter(columns_per_factor)):
            for factor, column_idx in zip(self.factors, column_idxs):
                if factor in self.contrasts:
                    contrast = self.contrasts[factor]
                    out[:, i] *= contrast[factor_values[factor].ravel(),
                                          column_idx]
                else:
                    out[:, i] *= factor_values[factor][:, column_idx]

def test_ColumnBuilder():
    f1 = _MockFactor()
    f2 = _MockFactor()
    f3 = _MockFactor()
    contrast = np.array([[0, 0.5],
                         [3, 0]])
    cb = ColumnBuilder([f1, f2, f3], {f2: contrast})
    mat = np.empty((3, 2))
    cb.build({f1: atleast_2d_column_default([1, 2, 3]),
              f2: atleast_2d_column_default([0, 0, 1]),
              f3: atleast_2d_column_default([7.5, 2, -12])},
             mat)
    assert np.allclose(mat, [[0, 0.5 * 1 * 7.5],
                             [0, 0.5 * 2 * 2],
                             [3 * 3 * -12, 0]])
    mat2 = np.empty((3, 4))
    cb.build({f1: atleast_2d_column_default([[1, 2], [3, 4], [5, 6]]),
              f2: atleast_2d_column_default([0, 0, 1]),
              f3: atleast_2d_column_default([7.5, 2, -12])},
             mat2)
    # Columns should be, in order:
    #   f1.1:f2.1, f1.1:f2.2, f1.2:f2.1, f1.2:f2.2
    assert np.allclose(mat2, [[0, 0.5 * 1 * 7.5, 0, 0.5 * 2 * 7.5],
                              [0, 0.5 * 3 * 2, 0, 0.5 * 4 * 2],
                              [3 * 5 * -12, 0, 3 * 6 * -12, 0]])
    # Check intercept building:
    cb_intercept = ColumnBuilder([], {})
    mat3 = np.empty((3, 1))
    cb_intercept.build({f1: [1, 2, 3], f2: [1, 2, 3], f3: [1, 2, 3]}, mat3)
    assert np.allclose(mat3, 1)

def _factors_memorize(stateful_transforms, default_env, factors,
                      data_iter_maker):
    # First, start off the memorization process by setting up each factor's
    # state and finding out how many passes it will need:
    factor_states = {}
    memorize_passes = {}
    for factor in all_factors:
        state = {}
        which_pass = factor.memorize_passes_needed(state, stateful_transforms)
        factor_states[factor] = state
        memorize_passes[factor] = which_pass
    # Now, cycle through the data until all the factors have finished
    # memorizing everything:
    memorize_needed = set()
    for factor, passes in factor_passes.iteritems():
        if passes > 0:
            memorize_needed.add(factor)
    which_pass = 0
    while memorize_needed:
        for data in data_iter_maker():
            for factor in memorize_needed:
                state = factor_states[factor]
                factor.memorize_chunk(state, which_pass,
                                      DictStack(data, default_env))
        for factor in memorize_needed:
            factor.memorize_finish(factor_states[factor], which_pass)
            if which_pass == factor_passes[factor] - 1:
                memorize_needed.remove(factor)
        which_pass += 1
    return factor_states

def _examine_factor_types(factors, data_iter_maker, args, kwargs):
    numeric_column_counts = {}
    categorical_postprocessors = {}
    categorical_levels_contrasts = {}
    examine_needed = set(factors)
    for data in data_iter_maker(*args, **kwargs):
        # We might have gathered all the information we need after the first
        # chunk of data. If so, then we shouldn't spend time loading all the
        # rest of the chunks.
        if not examine_needed:
            break
        for factor in list(examine_needed):
            value = factor.eval(factor_states[factor], data)
            if isinstance(value, Categorical):
                categorical_levels_contrasts[factor] = (value.levels,
                                                        value.contrast)
                examine_needed.remove(factor)
            value = atleast_2d_column_default(value)
            _max_allowed_dim(2, value, factor)
            if np.issubdtype(value.dtype, np.number):
                column_count = _get_numeric_column_count(value, factor)
                numeric_column_counts[factor] = column_count
                examine_needed.remove(factor)
            # issubdtype(X, bool) isn't reliable -- it returns true for
            # X == int! So check the kind code instead:
            elif value.dtype.kind == "b":
                # Special case: give it a transformer, but don't bother
                # processing the rest of the data
                _max_allowed_dim(1, value, factor)
                categorical_postprocessors[factor] = _BoolToCategorical(factor)
                examine_needed.remove(factor)
            else:
                _max_allowed_dim(1, value, factor)
                if factor not in categorical_postprocessors:
                    categorical_postprocessors[factor] = CategoricalTransform()
                processor = categorical_postprocessors[factor]
                processor.memorize_chunk(value)
    for processor in categorical_postprocessors.itervalues():
        processor.memorize_finish()
    return (numeric_column_counts,
            categorical_postprocessors,
            categorical_levels_contrasts)

def _make_model_builder(terms,
                        numeric_column_counts,
                        categorical_postprocessors,
                        categorical_levels_contrasts):
    # Sort each term into bins based on the set of numeric factors it
    # contains:
    term_buckets = {}
    for term in terms:
        numeric_factors = []
        for factor in term.factors:
            if factor in numeric_column_counts:
                numeric_factors.append(factor)
        bucket = frozenset(numeric_factors)
        term_buckets.setdefault(bucket, []).append(term)
    # The intercept acts as a 0-degree interaction
    term_to_term_spec = {}
    for numeric_factors, bucket in term_buckets.iteritems():
        # Sort by degree of interaction
        bucket.sort(cmp=lambda t1, t2: cmp(len(t1.factors), len(t2.factors)))
        extant_expanded_factors = set()
        for term in bucket:
            expanded = list(_expand_categorical_part(term, numeric_fators))
            expanded.sort(cmp=lambda s1, s2: cmp(len(s1), len(s2)))

def make_model_specs(stateful_transforms, default_env,
                     model_descs,
                     data_iter_maker, *args, **kwargs):
    all_factors = set()
    for model_desc in model_descs:
        for term in model_desc.terms:
            all_factors.update(term.lhs_factors)
            all_factors.update(term.rhs_factors)
    def data_iter_maker_thunk():
        return data_iter_maker(*args, **kwargs)
    factor_states = _factors_memorize(stateful_transforms, default_env,
                                      all_factors, data_iter_maker_thunk)
    # Now all the factors have working eval methods, so we can evaluate them
    # on some data to find out what type of data they return.
    (numeric_column_counts,
     categorical_postprocessors,
     categorical_levels_contrasts) = _examine_factor_types(all_factors,
                                                           data_iter_maker_thunk)
    # Now we know everything there is to know about each factor; we can
    # finally build the ModelSpecs. To do this, we need to convert our
    # knowledge about factors into knowledge about terms -- in particular, for
    # each term, we need to know:
    #   -- how many columns it produces
    #   -- what those columns are named
    #   -- the contrast coding for each categorical factor
    #   -- whether each categorical factor should include the intercept or not
    #      within this particular term
    model_specs = []
    for model_desc in model_descs:
        model_specs.append((_make_model_builder(model_desc.lhs_intercept,
                                                model_desc.lhs_terms,
                                                *factor_type_data),
                            _make_model_builder(model_desc.rhs_intercept,
                                                model_desc.rhs_terms,
                                                *factor_type_data)))
    return model_specs

def model_spec_from_model_desc_and_data(model_desc, data):
    def data_gen():
        yield data
    return make_model_specs([model_desc], data_gen)[0]

# No reason for this to take a data_iter_maker... you can just call it once
# for each data chunk
def make_model_matrices(model_specs, data_iter_maker, *args, **kwargs):
    pass

# Example of Factor protocol:
class LookupFactor(object):
    def __init__(self, name):
        self.name = name

    def name(self):
        return self.name

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.name)
        
    def __eq__(self, other):
        return isinstance(other, LookupFactor) and self.name == other.name

    def __hash__(self):
        return hash((LookupFactor, self.name))

    def memorize_passes_needed(self, stateful_transforms, state):
        return 0

    def memorize_chunk(self, state, which_pass, env):
        assert False

    def memorize_finish(self, state, which_pass):
        assert False

    def eval(self, memorize_state, env):
        return env[self.name]


# Issue: you have to evaluate the model to get the types of the
# predictors. For all the memorization functions that I can think of, the key
# facts about the final computed data is known before memorization finishes
# (i.e., # of columns, dtype, levels for categorical data). centering doesn't
# change any of these things, splines change # of columns but in known way,
# cut() knows what the levels will be even if it doesn't know where the cut
# points will be.
# So the naive approach is:
#   -- cycle through all data calculating memorization
#   -- cycle through all data (basically calculating all entries in the model
#      matrix) once to figure out columns and categorical coding
#      -- for this, we only need to pass in one row for most terms
#         exception is categorical data without explicit levels attached --
#         for those we need to cycle through everything. But we can detect
#         those from a single row (i.e., result has string dtype).
#   -- cycle through all data again to *actually* calculate the model matrix
# As above, in principle we might be able to avoid the second step here by
# merging it into the first step in a somewhat nasty way. But actually the
# second step doesn't look so bad, so never mind.
# ---- IF we have some way to cut down the size of the data being passed in! a
# rule that data can't be pulled out of the environment, only functions?
# blehh...
# the other approach is to merge (2) and (3), of course.
# For a large (incremental-necessary) model matrix, we can't hold the whole
# set of factor columns in memory anyway -- we *have* to cycle through them
# once to identify levels, and then a second time to build the (pieces of the)
# matrix.
# For a small (ordinary) model matrix, we might as well calculate it twice
# anyway...
# Maybe the way to think of it is, fully process the first chunk. Hold onto
# it, and process the rest of the chunks where we need to see everything, in
# order to set up the levels. then if we're making the first model matrix at
# the same time, go back and do that, with the first chunk already
# calculated.
#
# possible future optimization: let a memorize_chunk() function raise
# Stateless to indicate that actually, ha-ha, it doesn't need to memorize
# anything after all (b/c the relevant data turns out to be in *arg, **kwargs)

# stateful transform categorical()?
