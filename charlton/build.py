# This file is part of Charlton
# Copyright (C) 2011-2012 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

# This file defines the core design matrix building functions. There are two
# basic operations:
#   - make_design_matrix_builders: Takes a set of termlists (each one
#     representing a design matrix -- e.g., you might have one termlist for
#     your response variables and one for your predictor variables) together
#     with some exemplary data, and produces a set of
#     DesignMatrixBuilders. Unlike a termlist, which is a high-level
#     description of a model, a DesignMatrixBuilder knows every
#     detail about the resulting matrix: how many columns it will have, which
#     predictors are categorical (and how they are coded), how to perform any
#     stateful transforms, etc.
#   - make_design_matrices: Takes a set of DesignMatrixBuilders and some data,
#     and produces the corresponding set of design matrices.

# These are made available in the charlton.* namespace
__all__ = ["ModelDesign"]

import numpy as np
from charlton import CharltonError
from charlton.categorical import CategoricalTransform, Categorical
from charlton.util import atleast_2d_column_default
from charlton.design_matrix import DesignMatrix, DesignMatrixColumnInfo
from charlton.redundancy import pick_contrasts_for_term
from charlton.desc import ModelDesc
from charlton.contrasts import code_contrast_matrix, Treatment
from charlton.compat import itertools_product, OrderedDict

class _MockFactor(object):
    def __init__(self, name="MOCKMOCK"):
        self._name = name

    def eval(self, state, env):
        return env["mock"]

    def name(self):
        return self._name

def _max_allowed_dim(dim, arr, factor):
    if arr.ndim > dim:
        msg = ("factor '%s' evaluates to an %s-dimensional array; I only "
               "handle arrays with dimension <= %s"
               % (factor.name(), arr.ndim, dim))
        raise CharltonError(msg, factor)

def test__max_allowed_dim():
    from nose.tools import assert_raises
    f = _MockFactor()
    _max_allowed_dim(1, np.array(1), f)
    _max_allowed_dim(1, np.array([1]), f)
    assert_raises(CharltonError, _max_allowed_dim, 1, np.array([[1]]), f)
    assert_raises(CharltonError, _max_allowed_dim, 1, np.array([[[1]]]), f)
    _max_allowed_dim(2, np.array(1), f)
    _max_allowed_dim(2, np.array([1]), f)
    _max_allowed_dim(2, np.array([[1]]), f)
    assert_raises(CharltonError, _max_allowed_dim, 2, np.array([[[1]]]), f)

class _BoolToCat(object):
    def __init__(self, factor):
        self.factor = factor

    def memorize_finish(self):
        pass

    def levels(self):
        return (False, True)

    def transform(self, data):
        data = np.asarray(data)
        _max_allowed_dim(1, data, self.factor)
        # issubdtype(int, bool) is true! So we can't use it:
        if not data.dtype.kind == "b":
            raise CharltonError("factor %s, which I thought was boolean, "
                                "gave non-boolean data of dtype %s"
                                % (self.factor.name(), data.dtype),
                                self.factor)
        return Categorical(data, levels=[False, True])

def test__BoolToCat():
    from nose.tools import assert_raises
    f = _MockFactor()
    btc = _BoolToCat(f)
    cat = btc.transform([True, False, True, True])
    assert cat.levels == (False, True)
    assert np.issubdtype(cat.int_array.dtype, int)
    assert np.all(cat.int_array == [1, 0, 1, 1])
    assert_raises(CharltonError, btc.transform, [1, 0, 1])
    assert_raises(CharltonError, btc.transform, ["a", "b"])
    assert_raises(CharltonError, btc.transform, [[True]])

class _NumFactorEvaluator(object):
    def __init__(self, factor, state, expected_columns):
        # This one instance variable is part of our public API:
        self.factor = factor
        self._state = state
        self._expected_columns = expected_columns

    def eval(self, data):
        result = self.factor.eval(self._state, data)
        result = atleast_2d_column_default(result)
        _max_allowed_dim(2, result, self.factor)
        if result.shape[1] != self._expected_columns:
            raise CharltonError("when evaluating factor %s, I got %s columns "
                                "instead of the %s I was expecting"
                                % (self.factor.name(), self._expected_columns,
                                   result.shape[1]),
                                self.factor)
        if not np.issubdtype(result.dtype, np.number):
            raise CharltonError("when evaluating numeric factor %s, "
                                "I got non-numeric data of type '%s'"
                                % (self.factor.name(), result.dtype),
                                self.factor)
        return result

def test__NumFactorEvaluator():
    from nose.tools import assert_raises
    f = _MockFactor()
    nf1 = _NumFactorEvaluator(f, {}, 1)
    assert nf1.factor is f
    eval123 = nf1.eval({"mock": [1, 2, 3]})
    assert eval123.shape == (3, 1)
    assert np.all(eval123 == [[1], [2], [3]])
    assert_raises(CharltonError, nf1.eval, {"mock": [[[1]]]})
    assert_raises(CharltonError, nf1.eval, {"mock": [[1, 2]]})
    assert_raises(CharltonError, nf1.eval, {"mock": ["a", "b"]})
    assert_raises(CharltonError, nf1.eval, {"mock": [True, False]})
    nf2 = _NumFactorEvaluator(_MockFactor(), {}, 2)
    eval123321 = nf2.eval({"mock": [[1, 3], [2, 2], [3, 1]]})
    assert eval123321.shape == (3, 2)
    assert np.all(eval123321 == [[1, 3], [2, 2], [3, 1]])
    assert_raises(CharltonError, nf2.eval, {"mock": [1, 2, 3]})
    assert_raises(CharltonError, nf2.eval, {"mock": [[1, 2, 3]]})

class _CatFactorEvaluator(object):
    def __init__(self, factor, state, postprocessor, expected_levels):
        # This one instance variable is part of our public API:
        self.factor = factor
        self._state = state
        self._postprocessor = postprocessor
        self._expected_levels = tuple(expected_levels)

    def eval(self, data):
        result = self.factor.eval(self._state, data)
        if self._postprocessor is not None:
            result = self._postprocessor.transform(result)
        if not isinstance(result, Categorical):
            msg = ("when evaluating categoric factor %r, I got a "
                   "result that is not of type Categorical (but rather %s)"
                   # result.__class__.__name__ would be better, but not
                   # defined for old-style classes:
                   % (self.factor.name(), result.__class__))
            raise CharltonError(msg, self.factor)
        if result.levels != self._expected_levels:
            msg = ("when evaluating categoric factor %r, I got Categorical "
                   "data with unexpected levels (wanted %s, got %s)"
                   % (self.factor.name(), self._expected_levels, result.levels))
            raise CharltonError(msg, self.factor)
        _max_allowed_dim(1, result.int_array, self.factor)
        # For consistency, evaluators *always* return 2d arrays (though in
        # this case it will always have only 1 column):
        return atleast_2d_column_default(result.int_array)

def test__CatFactorEvaluator():
    from nose.tools import assert_raises
    from charlton.categorical import Categorical
    f = _MockFactor()
    cf1 = _CatFactorEvaluator(f, {}, None, ["a", "b"])
    assert cf1.factor is f
    cat1 = cf1.eval({"mock": Categorical.from_strings(["b", "a", "b"])})
    assert cat1.shape == (3, 1)
    assert np.all(cat1 == [[1], [0], [1]])
    assert_raises(CharltonError, cf1.eval, {"mock": ["c"]})
    assert_raises(CharltonError, cf1.eval,
                  {"mock": Categorical.from_strings(["a", "c"])})
    assert_raises(CharltonError, cf1.eval,
                  {"mock": Categorical.from_strings(["a", "b"],
                                                    levels=["b", "a"])})
    assert_raises(CharltonError, cf1.eval, {"mock": [1, 0, 1]})
    bad_cat = Categorical.from_strings(["b", "a", "a", "b"])
    bad_cat.int_array.resize((2, 2))
    assert_raises(CharltonError, cf1.eval, {"mock": bad_cat})

    btc = _BoolToCat(_MockFactor())
    cf2 = _CatFactorEvaluator(_MockFactor(), {}, btc, [False, True])
    cat2 = cf2.eval({"mock": [True, False, False, True]})
    assert cat2.shape == (4, 1)
    assert np.all(cat2 == [[1], [0], [0], [1]])

def _column_combinations(columns_per_factor):
    # For consistency with R, the left-most item iterates fastest:
    iterators = [xrange(n) for n in reversed(columns_per_factor)]
    for reversed_combo in itertools_product(*iterators):
        yield reversed_combo[::-1]

def test__column_combinations():
    assert list(_column_combinations([2, 3])) == [(0, 0),
                                                  (1, 0),
                                                  (0, 1),
                                                  (1, 1),
                                                  (0, 2),
                                                  (1, 2)]
    assert list(_column_combinations([3])) == [(0,), (1,), (2,)]

# This class is responsible for producing some columns in a final design
# matrix output:
class _ColumnBuilder(object):
    def __init__(self, factors, num_columns, cat_contrasts):
        self._factors = factors
        self._num_columns = num_columns
        self._cat_contrasts = cat_contrasts
        self._columns_per_factor = []
        for factor in self._factors:
            if factor in self._cat_contrasts:
                columns = self._cat_contrasts[factor].matrix.shape[1]
            else:
                columns = num_columns[factor]
            self._columns_per_factor.append(columns)
        self.total_columns = np.prod(self._columns_per_factor, dtype=int)

    def column_names(self):
        if not self._factors:
            return ["Intercept"]
        column_names = []
        for i, column_idxs in enumerate(_column_combinations(self._columns_per_factor)):
            name_pieces = []
            for factor, column_idx in zip(self._factors, column_idxs):
                if factor in self._num_columns:
                    if self._num_columns[factor] > 1:
                        name_pieces.append("%s[%s]"
                                           % (factor.name(), column_idx))
                    else:
                        assert column_idx == 0
                        name_pieces.append(factor.name())
                else:
                    contrast = self._cat_contrasts[factor]
                    suffix = contrast.column_suffixes[column_idx]
                    name_pieces.append("%s%s" % (factor.name(), suffix))
            column_names.append(":".join(name_pieces))
        assert len(column_names) == self.total_columns
        return column_names

    def build(self, factor_values, out):
        assert self.total_columns == out.shape[1]
        out[:] = 1
        for i, column_idxs in enumerate(_column_combinations(self._columns_per_factor)):
            for factor, column_idx in zip(self._factors, column_idxs):
                if factor in self._cat_contrasts:
                    contrast = self._cat_contrasts[factor]
                    out[:, i] *= contrast.matrix[factor_values[factor].ravel(),
                                                 column_idx]
                else:
                    assert (factor_values[factor].shape[1]
                            == self._num_columns[factor])
                    out[:, i] *= factor_values[factor][:, column_idx]

def test__ColumnBuilder():
    from charlton.contrasts import ContrastMatrix
    f1 = _MockFactor("f1")
    f2 = _MockFactor("f2")
    f3 = _MockFactor("f3")
    contrast = ContrastMatrix(np.array([[0, 0.5],
                                        [3, 0]]),
                              ["[c1]", "[c2]"])
                             
    cb = _ColumnBuilder([f1, f2, f3], {f1: 1, f3: 1}, {f2: contrast})
    mat = np.empty((3, 2))
    assert cb.column_names() == ["f1:f2[c1]:f3", "f1:f2[c2]:f3"]
    cb.build({f1: atleast_2d_column_default([1, 2, 3]),
              f2: atleast_2d_column_default([0, 0, 1]),
              f3: atleast_2d_column_default([7.5, 2, -12])},
             mat)
    assert np.allclose(mat, [[0, 0.5 * 1 * 7.5],
                             [0, 0.5 * 2 * 2],
                             [3 * 3 * -12, 0]])
    cb2 = _ColumnBuilder([f1, f2, f3], {f1: 2, f3: 1}, {f2: contrast})
    mat2 = np.empty((3, 4))
    cb2.build({f1: atleast_2d_column_default([[1, 2], [3, 4], [5, 6]]),
               f2: atleast_2d_column_default([0, 0, 1]),
               f3: atleast_2d_column_default([7.5, 2, -12])},
              mat2)
    assert cb2.column_names() == ["f1[0]:f2[c1]:f3",
                                  "f1[1]:f2[c1]:f3",
                                  "f1[0]:f2[c2]:f3",
                                  "f1[1]:f2[c2]:f3"]
    assert np.allclose(mat2, [[0, 0, 0.5 * 1 * 7.5, 0.5 * 2 * 7.5],
                              [0, 0, 0.5 * 3 * 2, 0.5 * 4 * 2],
                              [3 * 5 * -12, 3 * 6 * -12, 0, 0]])
    # Check intercept building:
    cb_intercept = _ColumnBuilder([], {}, {})
    assert cb_intercept.column_names() == ["Intercept"]
    mat3 = np.empty((3, 1))
    cb_intercept.build({f1: [1, 2, 3], f2: [1, 2, 3], f3: [1, 2, 3]}, mat3)
    assert np.allclose(mat3, 1)

def _factors_memorize(factors, data_iter_maker):
    # First, start off the memorization process by setting up each factor's
    # state and finding out how many passes it will need:
    factor_states = {}
    passes_needed = {}
    for factor in factors:
        state = {}
        which_pass = factor.memorize_passes_needed(state)
        factor_states[factor] = state
        passes_needed[factor] = which_pass
    # Now, cycle through the data until all the factors have finished
    # memorizing everything:
    memorize_needed = set()
    for factor, passes in passes_needed.iteritems():
        if passes > 0:
            memorize_needed.add(factor)
    which_pass = 0
    while memorize_needed:
        for data in data_iter_maker():
            for factor in memorize_needed:
                state = factor_states[factor]
                factor.memorize_chunk(state, which_pass, data)
        for factor in list(memorize_needed):
            factor.memorize_finish(factor_states[factor], which_pass)
            if which_pass == passes_needed[factor] - 1:
                memorize_needed.remove(factor)
        which_pass += 1
    return factor_states

def test__factors_memorize():
    class MockFactor(object):
        def __init__(self, requested_passes, token):
            self._requested_passes = requested_passes
            self._token = token
            self._chunk_in_pass = 0
            self._seen_passes = 0

        def memorize_passes_needed(self, state):
            state["calls"] = []
            state["token"] = self._token
            return self._requested_passes

        def memorize_chunk(self, state, which_pass, data):
            state["calls"].append(("memorize_chunk", which_pass))
            assert data["chunk"] == self._chunk_in_pass
            self._chunk_in_pass += 1

        def memorize_finish(self, state, which_pass):
            state["calls"].append(("memorize_finish", which_pass))
            self._chunk_in_pass = 0

    class Data(object):
        CHUNKS = 3
        def __init__(self):
            self.calls = 0
            self.data = [{"chunk": i} for i in xrange(self.CHUNKS)]
        def __call__(self):
            self.calls += 1
            return iter(self.data)
    data = Data()
    f0 = MockFactor(0, "f0")
    f1 = MockFactor(1, "f1")
    f2a = MockFactor(2, "f2a")
    f2b = MockFactor(2, "f2b")
    factor_states = _factors_memorize(set([f0, f1, f2a, f2b]), data)
    assert data.calls == 2
    mem_chunks0 = [("memorize_chunk", 0)] * data.CHUNKS
    mem_chunks1 = [("memorize_chunk", 1)] * data.CHUNKS
    expected = {
        f0: {
            "calls": [],
            "token": "f0",
            },
        f1: {
            "calls": mem_chunks0 + [("memorize_finish", 0)],
            "token": "f1",
            },
        f2a: {
            "calls": mem_chunks0 + [("memorize_finish", 0)]
                     + mem_chunks1 + [("memorize_finish", 1)],
            "token": "f2a",
            },
        f2b: {
            "calls": mem_chunks0 + [("memorize_finish", 0)]
                     + mem_chunks1 + [("memorize_finish", 1)],
            "token": "f2b",
            },
        }
    assert factor_states == expected

def _examine_factor_types(factors, factor_states, data_iter_maker):
    num_column_counts = {}
    cat_levels_contrasts = {}
    cat_postprocessors = {}
    prefinished_postprocessors = {}
    examine_needed = set(factors)
    for data in data_iter_maker():
        # We might have gathered all the information we need after the first
        # chunk of data. If so, then we shouldn't spend time loading all the
        # rest of the chunks.
        if not examine_needed:
            break
        for factor in list(examine_needed):
            value = factor.eval(factor_states[factor], data)
            if isinstance(value, Categorical):
                postprocessor = CategoricalTransform(levels=value.levels)
                prefinished_postprocessors[factor] = postprocessor
                cat_levels_contrasts[factor] = (value.levels,
                                                value.contrast)
                examine_needed.remove(factor)
                continue
            value = atleast_2d_column_default(value)
            _max_allowed_dim(2, value, factor)
            if np.issubdtype(value.dtype, np.number):
                column_count = value.shape[1]
                num_column_counts[factor] = column_count
                examine_needed.remove(factor)
            # issubdtype(X, bool) isn't reliable -- it returns true for
            # X == int! So check the kind code instead:
            elif value.dtype.kind == "b":
                # Special case: give it a transformer, but don't bother
                # processing the rest of the data
                if value.shape[1] > 1:
                    msg = ("factor '%s' evaluates to a boolean array with "
                           "%s columns; I can only handle single-column "
                           "boolean arrays" % (factor.name(), value.shape[1]))
                    raise CharltonError(msg, factor)
                cat_postprocessors[factor] = _BoolToCat(factor)
                examine_needed.remove(factor)
            else:
                if value.shape[1] > 1:
                    msg = ("factor '%s' appears to be categorical but has "
                           "%s columns; I can only handle single-column "
                           "categorical factors"
                           % (factor.name(), value.shape[1]))
                    raise CharltonError(msg, factor)
                if factor not in cat_postprocessors:
                    cat_postprocessors[factor] = CategoricalTransform()
                processor = cat_postprocessors[factor]
                processor.memorize_chunk(value)
    for factor, processor in cat_postprocessors.iteritems():
        processor.memorize_finish()
        cat_levels_contrasts[factor] = (processor.levels(), None)
    cat_postprocessors.update(prefinished_postprocessors)
    assert set(cat_postprocessors) == set(cat_levels_contrasts)
    return (num_column_counts,
            cat_levels_contrasts,
            cat_postprocessors)

def test__examine_factor_types():
    class MockFactor(object):
        def __init__(self):
            # You should check this using 'is', not '=='
            from charlton.origin import Origin
            self.origin = Origin("MOCK", 1, 2)

        def eval(self, state, data):
            return state[data]

        def name(self):
            return "MOCK MOCK"

    # This hacky class can only be iterated over once, but it keeps track of
    # how far it got.
    class DataIterMaker(object):
        def __init__(self):
            self.i = -1

        def __call__(self):
            return self

        def __iter__(self):
            return self

        def next(self):
            self.i += 1
            if self.i > 1:
                raise StopIteration
            return self.i

    num_1dim = MockFactor()
    num_1col = MockFactor()
    num_4col = MockFactor()
    categ_1col = MockFactor()
    bool_1col = MockFactor()
    string_1col = MockFactor()
    object_1col = MockFactor()
    object_levels = (object(), object(), object())
    factor_states = {
        num_1dim: ([1, 2, 3], [4, 5, 6]),
        num_1col: ([[1], [2], [3]], [[4], [5], [6]]),
        num_4col: (np.zeros((3, 4)), np.ones((3, 4))),
        categ_1col: (Categorical([0, 1, 2], levels=("a", "b", "c"),
                                 contrast="MOCK CONTRAST"),
                     Categorical([2, 1, 0], levels=("a", "b", "c"),
                                 contrast="MOCK CONTRAST")),
        bool_1col: ([True, True, False], [False, True, True]),
        # It has to read through all the data to see all the possible levels:
        string_1col: (["a", "a", "a"], ["c", "b", "a"]),
        object_1col: ([object_levels[0]] * 3, object_levels),
        }

    it = DataIterMaker()
    (num_column_counts, cat_levels_contrasts, cat_postprocessors
     ) = _examine_factor_types(factor_states.keys(), factor_states, it)
    assert it.i == 2
    iterations = 0
    assert num_column_counts == {num_1dim: 1, num_1col: 1, num_4col: 4}
    assert cat_levels_contrasts == {
        categ_1col: (("a", "b", "c"), "MOCK CONTRAST"),
        bool_1col: ((False, True), None),
        string_1col: (("a", "b", "c"), None),
        object_1col: (tuple(sorted(object_levels)), None),
        }
    assert (set(cat_postprocessors.keys())
            == set([categ_1col, bool_1col, string_1col, object_1col]))

    # Check that it doesn't read through all the data if that's not necessary:
    it = DataIterMaker()
    no_read_necessary = [num_1dim, num_1col, num_4col, categ_1col, bool_1col]
    (num_column_counts, cat_levels_contrasts, cat_postprocessors
     ) = _examine_factor_types(no_read_necessary, factor_states, it)
    assert it.i == 1
    assert num_column_counts == {num_1dim: 1, num_1col: 1, num_4col: 4}
    assert cat_levels_contrasts == {
        categ_1col: (("a", "b", "c"), "MOCK CONTRAST"),
        bool_1col: ((False, True), None),
        }
    assert set(cat_postprocessors) == set([categ_1col, bool_1col])

    # Illegal inputs:
    bool_3col = MockFactor()
    num_3dim = MockFactor()
    # no such thing as a multi-dimensional Categorical
    # categ_3dim = MockFactor()
    string_3col = MockFactor()
    object_3col = MockFactor()
    illegal_factor_states = {
        bool_3col: (np.zeros((3, 3), dtype=bool), np.ones((3, 3), dtype=bool)),
        num_3dim: (np.zeros((3, 3, 3)), np.ones((3, 3, 3))),
        string_3col: ([["a", "b", "c"]], [["b", "c", "a"]]),
        object_3col: ([[[object()]]], [[[object()]]]),
        }
    from nose.tools import assert_raises
    for illegal_factor in illegal_factor_states:
        it = DataIterMaker()
        try:
            _examine_factor_types([illegal_factor], illegal_factor_states, it)
        except CharltonError, e:
            assert e.origin is illegal_factor.origin
        else:
            assert False

def _make_term_column_builders(terms,
                               num_column_counts,
                               cat_levels_contrasts):
    # Sort each term into a bucket based on the set of numeric factors it
    # contains:
    term_buckets = OrderedDict()
    bucket_ordering = []
    for term in terms:
        num_factors = []
        for factor in term.factors:
            if factor in num_column_counts:
                num_factors.append(factor)
        bucket = frozenset(num_factors)
        if bucket not in term_buckets:
            bucket_ordering.append(bucket)
        term_buckets.setdefault(bucket, []).append(term)
    # Special rule: if there is a no-numerics bucket, then it always comes
    # first:
    if frozenset() in term_buckets:
        bucket_ordering.remove(frozenset())
        bucket_ordering.insert(0, frozenset())
    term_to_column_builders = {}
    new_term_order = []
    # Then within each bucket, work out which sort of contrasts we want to use
    # for each term to avoid redundancy
    for bucket in bucket_ordering:
        bucket_terms = term_buckets[bucket]
        # Sort by degree of interaction
        bucket_terms.sort(key=lambda t: len(t.factors))
        new_term_order += bucket_terms
        used_subterms = set()
        for term in bucket_terms:
            column_builders = []
            factor_codings = pick_contrasts_for_term(term,
                                                     num_column_counts,
                                                     used_subterms)
            # Construct one _ColumnBuilder for each subterm
            for factor_coding in factor_codings:
                builder_factors = []
                num_columns = {}
                cat_contrasts = {}
                # In order to preserve factor ordering information, the
                # coding_for_term just returns dicts, and we refer to
                # the original factors to figure out which are included in
                # each subterm, and in what order
                for factor in term.factors:
                    # Numeric factors are included in every subterm
                    if factor in num_column_counts:
                        builder_factors.append(factor)
                        num_columns[factor] = num_column_counts[factor]
                    elif factor in factor_coding:
                        builder_factors.append(factor)
                        levels, contrast = cat_levels_contrasts[factor]
                        # This is where the default coding is set to
                        # Treatment:
                        coded = code_contrast_matrix(factor_coding[factor],
                                                     levels, contrast,
                                                     default=Treatment)
                        cat_contrasts[factor] = coded
                column_builder = _ColumnBuilder(builder_factors,
                                                num_columns,
                                                cat_contrasts)
                column_builders.append(column_builder)
            term_to_column_builders[term] = column_builders
    return new_term_order, term_to_column_builders
                        
def _make_design_matrix_builders(termlists, data_iter_maker, *args, **kwargs):
    all_factors = set()
    for termlist in termlists:
        for term in termlist:
            all_factors.update(term.factors)
    def data_iter_maker_thunk():
        return data_iter_maker(*args, **kwargs)
    factor_states = _factors_memorize(all_factors, data_iter_maker_thunk)
    # Now all the factors have working eval methods, so we can evaluate them
    # on some data to find out what type of data they return.
    (num_column_counts,
     cat_levels_contrasts,
     cat_postprocessors) = _examine_factor_types(all_factors,
                                                 factor_states,
                                                 data_iter_maker_thunk)
    # Now we need the factor evaluators, which encapsulate the knowledge of
    # how to turn any given factor into a chunk of data:
    factor_evaluators = {}
    for factor in all_factors:
        if factor in num_column_counts:
            evaluator = _NumFactorEvaluator(factor,
                                            factor_states[factor],
                                            num_column_counts[factor])
        else:
            assert factor in cat_levels_contrasts
            postprocessor = cat_postprocessors.get(factor)
            levels = cat_levels_contrasts[factor][0]
            evaluator = _CatFactorEvaluator(factor, factor_states[factor],
                                            postprocessor, levels)
        factor_evaluators[factor] = evaluator
    # And now we can construct the DesignMatrixBuilder for each termlist:
    builders = []
    for termlist in termlists:
        result = _make_term_column_builders(termlist,
                                            num_column_counts,
                                            cat_levels_contrasts)
        new_term_order, term_to_column_builders = result
        assert frozenset(new_term_order) == frozenset(termlist)
        term_evaluators = set()
        for term in termlist:
            for factor in term.factors:
                term_evaluators.add(factor_evaluators[factor])
        builders.append(DesignMatrixBuilder(new_term_order,
                                            term_evaluators,
                                            term_to_column_builders))
    return builders

class DesignMatrixBuilder(object):
    def __init__(self, terms, evaluators, term_to_column_builders):
        self._terms = terms
        self._evaluators = evaluators
        self._term_to_column_builders = term_to_column_builders
        term_column_count = []
        column_names = []
        for term in self._terms:
            column_builders = self._term_to_column_builders[term]
            this_count = 0
            for column_builder in column_builders:
                this_names = column_builder.column_names()
                this_count += len(this_names)
                column_names += this_names
            term_column_count.append(this_count)
        term_column_starts = np.concatenate(([0], np.cumsum(term_column_count)))
        term_slices = []
        for i, term in enumerate(self._terms):
            span = slice(term_column_starts[i], term_column_starts[i + 1])
            term_slices.append((term, span))
        self.total_columns = np.sum(term_column_count, dtype=int)
        self.column_info = DesignMatrixColumnInfo(column_names, term_slices)

    def _build(self, evaluator_to_values, dtype):
        factor_to_values = {}
        need_reshape = False
        num_rows = None
        for evaluator, value in evaluator_to_values.iteritems():
            if evaluator in self._evaluators:
                factor_to_values[evaluator.factor] = value
                if num_rows is not None:
                    assert num_rows == value.shape[0]
                else:
                    num_rows = value.shape[0]
        if num_rows is None:
            # We have no dependence on the data -- e.g. an empty termlist, or
            # only an intercept term.
            num_rows = 1
            need_reshape = True
        m = DesignMatrix(np.empty((num_rows, self.total_columns), dtype=dtype),
                         self.column_info, builder=self)
        start_column = 0
        for term in self._terms:
            for column_builder in self._term_to_column_builders[term]:
                end_column = start_column + column_builder.total_columns
                m_slice = m[:, start_column:end_column]
                column_builder.build(factor_to_values, m_slice)
                start_column = end_column
        assert start_column == self.total_columns
        return need_reshape, m

def _make_design_matrices(builders, data, dtype=float):
    evaluator_to_values = {}
    num_rows = None
    for builder in builders:
        # We look at evaluators rather than factors here, because it might
        # happen that we have the same factor twice, but with different
        # memorized state.
        for evaluator in builder._evaluators:
            if evaluator not in evaluator_to_values:
                value = evaluator.eval(data)
                assert value.ndim == 2
                if num_rows is None:
                    num_rows = value.shape[0]
                else:
                    if num_rows != value.shape[0]:
                        msg = ("Row mismatch: factor %s had %s rows, when "
                               "previous factors had %s rows"
                               % (evaluator.factor.name(), value.shape[0],
                                  num_rows))
                        raise CharltonError(msg, evaluator.factor)
                evaluator_to_values[evaluator] = value
    results = []
    for builder in builders:
        results.append(builder._build(evaluator_to_values, dtype))
    matrices = []
    for need_reshape, matrix in results:
        if need_reshape and num_rows is not None:
            assert matrix.shape[0] == 1
            matrices.append(DesignMatrix(np.repeat(matrix, num_rows, axis=0),
                                         matrix.column_info,
                                         builder=matrix.builder))
        else:
            # There is no data-dependence, at all -- a formula like "1 ~ 1". I
            # guess we'll just return some single-row matrices. Perhaps it
            # would be better to figure out how many rows are in the input
            # data and broadcast to that size, but eh. Input data is optional
            # in the first place, so even that would be no guarantee... let's
            # wait until someone actually has a relevant use case before we
            # worry about it.
            matrices.append(matrix)
    return matrices

class ModelDesign(object):
    def __init__(self, desc, termlists, builders):
        self.desc = desc
        self.termlists = termlists
        self.builders = builders

    @classmethod
    def from_desc(cls, desc, data_iter_maker, *args, **kwargs):
        self = cls.from_termlists([desc.lhs_terms, desc.rhs_terms],
                                  data_iter_maker, *args, **kwargs)
        self.desc = desc
        return self

    @classmethod
    def from_termlists(cls, termlists, data_iter_maker, *args, **kwargs):
        builders = _make_design_matrix_builders(termlists, data_iter_maker,
                                                *args, **kwargs)
        return cls(None, termlists, builders)

    def make_matrices(self, data):
        return _make_design_matrices(self.builders, data)

# It should be possible to do just the factors -> factor evaluators stuff
# alone, since that, well, makes logical sense to do. though categorical
# coding has to happen afterwards, hmm.
