# This file is part of Patsy
# Copyright (C) 2011-2013 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

# This file defines the core design matrix building functions.

# These are made available in the patsy.* namespace
__all__ = ["design_matrix_builders", "DesignMatrixBuilder",
           "build_design_matrices"]

import itertools
import six

import numpy as np
from patsy import PatsyError
from patsy.categorical import (guess_categorical,
                               CategoricalSniffer,
                               categorical_to_int)
from patsy.util import (atleast_2d_column_default,
                        have_pandas, have_pandas_categorical,
                        asarray_or_pandas)
from patsy.design_info import DesignMatrix, DesignInfo
from patsy.redundancy import pick_contrasts_for_term
from patsy.desc import ModelDesc
from patsy.eval import EvalEnvironment
from patsy.contrasts import code_contrast_matrix, Treatment
from patsy.compat import OrderedDict
from patsy.missing import NAAction

if have_pandas:
    import pandas

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
        raise PatsyError(msg, factor)

def test__max_allowed_dim():
    from nose.tools import assert_raises
    f = _MockFactor()
    _max_allowed_dim(1, np.array(1), f)
    _max_allowed_dim(1, np.array([1]), f)
    assert_raises(PatsyError, _max_allowed_dim, 1, np.array([[1]]), f)
    assert_raises(PatsyError, _max_allowed_dim, 1, np.array([[[1]]]), f)
    _max_allowed_dim(2, np.array(1), f)
    _max_allowed_dim(2, np.array([1]), f)
    _max_allowed_dim(2, np.array([[1]]), f)
    assert_raises(PatsyError, _max_allowed_dim, 2, np.array([[[1]]]), f)

class _NumFactorEvaluator(object):
    def __init__(self, factor, state, expected_columns):
        # This one instance variable is part of our public API:
        self.factor = factor
        self._state = state
        self._expected_columns = expected_columns

    # Returns either a 2d ndarray, or a DataFrame, plus is_NA mask
    def eval(self, data, NA_action):
        result = self.factor.eval(self._state, data)
        result = atleast_2d_column_default(result, preserve_pandas=True)
        _max_allowed_dim(2, result, self.factor)
        if result.shape[1] != self._expected_columns:
            raise PatsyError("when evaluating factor %s, I got %s columns "
                                "instead of the %s I was expecting"
                                % (self.factor.name(), self._expected_columns,
                                   result.shape[1]),
                                self.factor)
        if not np.issubdtype(np.asarray(result).dtype, np.number):
            raise PatsyError("when evaluating numeric factor %s, "
                                "I got non-numeric data of type '%s'"
                                % (self.factor.name(), result.dtype),
                                self.factor)
        return result, NA_action.is_numerical_NA(result)

def test__NumFactorEvaluator():
    from nose.tools import assert_raises
    naa = NAAction()
    f = _MockFactor()
    nf1 = _NumFactorEvaluator(f, {}, 1)
    assert nf1.factor is f
    eval123, is_NA = nf1.eval({"mock": [1, 2, 3]}, naa)
    assert eval123.shape == (3, 1)
    assert np.all(eval123 == [[1], [2], [3]])
    assert is_NA.shape == (3,)
    assert np.all(~is_NA)
    assert_raises(PatsyError, nf1.eval, {"mock": [[[1]]]}, naa)
    assert_raises(PatsyError, nf1.eval, {"mock": [[1, 2]]}, naa)
    assert_raises(PatsyError, nf1.eval, {"mock": ["a", "b"]}, naa)
    assert_raises(PatsyError, nf1.eval, {"mock": [True, False]}, naa)
    nf2 = _NumFactorEvaluator(_MockFactor(), {}, 2)
    eval123321, is_NA = nf2.eval({"mock": [[1, 3], [2, 2], [3, 1]]}, naa)
    assert eval123321.shape == (3, 2)
    assert np.all(eval123321 == [[1, 3], [2, 2], [3, 1]])
    assert is_NA.shape == (3,)
    assert np.all(~is_NA)
    assert_raises(PatsyError, nf2.eval, {"mock": [1, 2, 3]}, naa)
    assert_raises(PatsyError, nf2.eval, {"mock": [[1, 2, 3]]}, naa)

    ev_nan, is_NA = nf1.eval({"mock": [1, 2, np.nan]},
                             NAAction(NA_types=["NaN"]))
    assert np.array_equal(is_NA, [False, False, True])
    ev_nan, is_NA = nf1.eval({"mock": [1, 2, np.nan]},
                             NAAction(NA_types=[]))
    assert np.array_equal(is_NA, [False, False, False])

    if have_pandas:
        eval_ser, _ = nf1.eval({"mock":
                                pandas.Series([1, 2, 3], index=[10, 20, 30])},
                               naa)
        assert isinstance(eval_ser, pandas.DataFrame)
        assert np.array_equal(eval_ser, [[1], [2], [3]])
        assert np.array_equal(eval_ser.index, [10, 20, 30])
        eval_df1, _ = nf1.eval({"mock":
                                    pandas.DataFrame([[2], [1], [3]],
                                                     index=[20, 10, 30])},
                               naa)
        assert isinstance(eval_df1, pandas.DataFrame)
        assert np.array_equal(eval_df1, [[2], [1], [3]])
        assert np.array_equal(eval_df1.index, [20, 10, 30])
        eval_df2, _ = nf2.eval({"mock":
                                    pandas.DataFrame([[2, 3], [1, 4], [3, -1]],
                                                     index=[20, 30, 10])},
                               naa)
        assert isinstance(eval_df2, pandas.DataFrame)
        assert np.array_equal(eval_df2, [[2, 3], [1, 4], [3, -1]])
        assert np.array_equal(eval_df2.index, [20, 30, 10])
        
        assert_raises(PatsyError,
                      nf2.eval,
                      {"mock": pandas.Series([1, 2, 3], index=[10, 20, 30])},
                      naa)
        assert_raises(PatsyError,
                      nf1.eval,
                      {"mock":
                       pandas.DataFrame([[2, 3], [1, 4], [3, -1]],
                                        index=[20, 30, 10])},
                      naa)


class _CatFactorEvaluator(object):
    def __init__(self, factor, state, levels):
        # This one instance variable is part of our public API:
        self.factor = factor
        self._state = state
        self._levels = tuple(levels)

    # returns either a 1d ndarray or a pandas.Series, plus is_NA mask
    def eval(self, data, NA_action):
        result = self.factor.eval(self._state, data)
        result = categorical_to_int(result, self._levels, NA_action,
                                    origin=self.factor)
        assert result.ndim == 1
        return result, np.asarray(result == -1)

def test__CatFactorEvaluator():
    from nose.tools import assert_raises
    from patsy.categorical import C
    naa = NAAction()
    f = _MockFactor()
    cf1 = _CatFactorEvaluator(f, {}, ["a", "b"])
    assert cf1.factor is f
    cat1, _ = cf1.eval({"mock": ["b", "a", "b"]}, naa)
    assert cat1.shape == (3,)
    assert np.all(cat1 == [1, 0, 1])
    assert_raises(PatsyError, cf1.eval, {"mock": ["c"]}, naa)
    assert_raises(PatsyError, cf1.eval, {"mock": C(["a", "c"])}, naa)
    assert_raises(PatsyError, cf1.eval,
                  {"mock": C(["a", "b"], levels=["b", "a"])}, naa)
    assert_raises(PatsyError, cf1.eval, {"mock": [1, 0, 1]}, naa)
    bad_cat = np.asarray(["b", "a", "a", "b"])
    bad_cat.resize((2, 2))
    assert_raises(PatsyError, cf1.eval, {"mock": bad_cat}, naa)

    cat1_NA, is_NA = cf1.eval({"mock": ["a", None, "b"]},
                              NAAction(NA_types=["None"]))
    assert np.array_equal(is_NA, [False, True, False])
    assert np.array_equal(cat1_NA, [0, -1, 1])
    assert_raises(PatsyError, cf1.eval,
                  {"mock": ["a", None, "b"]}, NAAction(NA_types=[]))

    cf2 = _CatFactorEvaluator(_MockFactor(), {}, [False, True])
    cat2, _ = cf2.eval({"mock": [True, False, False, True]}, naa)
    assert cat2.shape == (4,)
    assert np.all(cat2 == [1, 0, 0, 1])

    if have_pandas:
        s = pandas.Series(["b", "a"], index=[10, 20])
        cat_s, _ = cf1.eval({"mock": s}, naa)
        assert isinstance(cat_s, pandas.Series)
        assert np.array_equal(cat_s, [1, 0])
        assert np.array_equal(cat_s.index, [10, 20])
        sbool = pandas.Series([True, False], index=[11, 21])
        cat_sbool, _ = cf2.eval({"mock": sbool}, naa)
        assert isinstance(cat_sbool, pandas.Series)
        assert np.array_equal(cat_sbool, [1, 0])
        assert np.array_equal(cat_sbool.index, [11, 21])

def _column_combinations(columns_per_factor):
    # For consistency with R, the left-most item iterates fastest:
    iterators = [range(n) for n in reversed(columns_per_factor)]
    for reversed_combo in itertools.product(*iterators):
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
                    if np.any(factor_values[factor] < 0):
                        raise PatsyError("can't build a design matrix "
                                         "containing missing values", factor)
                    out[:, i] *= contrast.matrix[factor_values[factor],
                                                 column_idx]
                else:
                    assert (factor_values[factor].shape[1]
                            == self._num_columns[factor])
                    out[:, i] *= factor_values[factor][:, column_idx]

def test__ColumnBuilder():
    from nose.tools import assert_raises
    from patsy.contrasts import ContrastMatrix
    from patsy.categorical import C
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
              f2: np.asarray([0, 0, 1]),
              f3: atleast_2d_column_default([7.5, 2, -12])},
             mat)
    assert np.allclose(mat, [[0, 0.5 * 1 * 7.5],
                             [0, 0.5 * 2 * 2],
                             [3 * 3 * -12, 0]])
    # Check that missing categorical values blow up
    assert_raises(PatsyError, cb.build,
                  {f1: atleast_2d_column_default([1, 2, 3]),
                   f2: np.asarray([0, -1, 1]),
                   f3: atleast_2d_column_default([7.5, 2, -12])},
                  mat)

    cb2 = _ColumnBuilder([f1, f2, f3], {f1: 2, f3: 1}, {f2: contrast})
    mat2 = np.empty((3, 4))
    cb2.build({f1: atleast_2d_column_default([[1, 2], [3, 4], [5, 6]]),
               f2: np.asarray([0, 0, 1]),
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
    for factor, passes in six.iteritems(passes_needed):
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
            self.data = [{"chunk": i} for i in range(self.CHUNKS)]
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

def _examine_factor_types(factors, factor_states, data_iter_maker, NA_action):
    num_column_counts = {}
    cat_sniffers = {}
    examine_needed = set(factors)
    for data in data_iter_maker():
        for factor in list(examine_needed):
            value = factor.eval(factor_states[factor], data)
            if factor in cat_sniffers or guess_categorical(value):
                if factor not in cat_sniffers:
                    cat_sniffers[factor] = CategoricalSniffer(NA_action,
                                                              factor.origin)
                done = cat_sniffers[factor].sniff(value)
                if done:
                    examine_needed.remove(factor)
            else:
                # Numeric
                value = atleast_2d_column_default(value)
                _max_allowed_dim(2, value, factor)
                column_count = value.shape[1]
                num_column_counts[factor] = column_count
                examine_needed.remove(factor)
        if not examine_needed:
            break
    # Pull out the levels
    cat_levels_contrasts = {}
    for factor, sniffer in six.iteritems(cat_sniffers):
        cat_levels_contrasts[factor] = sniffer.levels_contrast()
    return (num_column_counts, cat_levels_contrasts)

def test__examine_factor_types():
    from patsy.categorical import C
    class MockFactor(object):
        def __init__(self):
            # You should check this using 'is', not '=='
            from patsy.origin import Origin
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
        __next__ = next

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
        categ_1col: (C(["a", "b", "c"], levels=("a", "b", "c"),
                       contrast="MOCK CONTRAST"),
                     C(["c", "b", "a"], levels=("a", "b", "c"),
                       contrast="MOCK CONTRAST")),
        bool_1col: ([True, True, False], [False, True, True]),
        # It has to read through all the data to see all the possible levels:
        string_1col: (["a", "a", "a"], ["c", "b", "a"]),
        object_1col: ([object_levels[0]] * 3, object_levels),
        }

    it = DataIterMaker()
    (num_column_counts, cat_levels_contrasts,
     ) = _examine_factor_types(factor_states.keys(), factor_states, it,
                               NAAction())
    assert it.i == 2
    iterations = 0
    assert num_column_counts == {num_1dim: 1, num_1col: 1, num_4col: 4}
    assert cat_levels_contrasts == {
        categ_1col: (("a", "b", "c"), "MOCK CONTRAST"),
        bool_1col: ((False, True), None),
        string_1col: (("a", "b", "c"), None),
        object_1col: (tuple(sorted(object_levels, key=id)), None),
        }

    # Check that it doesn't read through all the data if that's not necessary:
    it = DataIterMaker()
    no_read_necessary = [num_1dim, num_1col, num_4col, categ_1col, bool_1col]
    (num_column_counts, cat_levels_contrasts,
     ) = _examine_factor_types(no_read_necessary, factor_states, it,
                               NAAction())
    assert it.i == 0
    assert num_column_counts == {num_1dim: 1, num_1col: 1, num_4col: 4}
    assert cat_levels_contrasts == {
        categ_1col: (("a", "b", "c"), "MOCK CONTRAST"),
        bool_1col: ((False, True), None),
        }

    # Illegal inputs:
    bool_3col = MockFactor()
    num_3dim = MockFactor()
    # no such thing as a multi-dimensional Categorical
    # categ_3dim = MockFactor()
    string_3col = MockFactor()
    object_3col = MockFactor()
    illegal_factor_states = {
        num_3dim: (np.zeros((3, 3, 3)), np.ones((3, 3, 3))),
        string_3col: ([["a", "b", "c"]], [["b", "c", "a"]]),
        object_3col: ([[[object()]]], [[[object()]]]),
        }
    from nose.tools import assert_raises
    for illegal_factor in illegal_factor_states:
        it = DataIterMaker()
        try:
            _examine_factor_types([illegal_factor], illegal_factor_states, it,
                                  NAAction())
        except PatsyError as e:
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

def design_matrix_builders(termlists, data_iter_maker, NA_action="drop"):
    """Construct several :class:`DesignMatrixBuilders` from termlists.

    This is one of Patsy's fundamental functions. This function and
    :func:`build_design_matrices` together form the API to the core formula
    interpretation machinery.

    :arg termlists: A list of termlists, where each termlist is a list of
      :class:`Term` objects which together specify a design matrix.
    :arg data_iter_maker: A zero-argument callable which returns an iterator
      over dict-like data objects. This must be a callable rather than a
      simple iterator because sufficiently complex formulas may require
      multiple passes over the data (e.g. if there are nested stateful
      transforms).
    :arg NA_action: An :class:`NAAction` object or string, used to determine
      what values count as 'missing' for purposes of determining the levels of
      categorical factors.
    :returns: A list of :class:`DesignMatrixBuilder` objects, one for each
      termlist passed in.

    This function performs zero or more iterations over the data in order to
    sniff out any necessary information about factor types, set up stateful
    transforms, pick column names, etc.

    See :ref:`formulas` for details.

    .. versionadded:: 0.2.0
       The ``NA_action`` argument.
    """
    if isinstance(NA_action, str):
        NA_action = NAAction(NA_action)
    all_factors = set()
    for termlist in termlists:
        for term in termlist:
            all_factors.update(term.factors)
    factor_states = _factors_memorize(all_factors, data_iter_maker)
    # Now all the factors have working eval methods, so we can evaluate them
    # on some data to find out what type of data they return.
    (num_column_counts,
     cat_levels_contrasts) = _examine_factor_types(all_factors,
                                                   factor_states,
                                                   data_iter_maker,
                                                   NA_action)
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
            levels = cat_levels_contrasts[factor][0]
            evaluator = _CatFactorEvaluator(factor, factor_states[factor],
                                            levels)
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
    """An opaque class representing Patsy's knowledge about
    how to build a specific design matrix.

    You get these objects from :func:`design_matrix_builders`, and pass them
    to :func:`build_design_matrices`.
    """
    def __init__(self, terms, evaluators, term_to_column_builders):
        self._termlist = terms
        self._evaluators = evaluators
        self._term_to_column_builders = term_to_column_builders
        term_column_count = []
        self._column_names = []
        for term in self._termlist:
            column_builders = self._term_to_column_builders[term]
            this_count = 0
            for column_builder in column_builders:
                this_names = column_builder.column_names()
                this_count += len(this_names)
                self._column_names += this_names
            term_column_count.append(this_count)
        term_column_starts = np.concatenate(([0], np.cumsum(term_column_count)))
        self._term_slices = []
        for i, term in enumerate(self._termlist):
            span = slice(term_column_starts[i], term_column_starts[i + 1])
            self._term_slices.append((term, span))
        self.total_columns = np.sum(term_column_count, dtype=int)

    # Generate this on demand, to avoid a reference loop:
    @property
    def design_info(self):
        """A :class:`DesignInfo` object giving information about the design
        matrices that this DesignMatrixBuilder can be used to create."""
        return DesignInfo(self._column_names, self._term_slices,
                          builder=self)

    def subset(self, which_terms):
        """Create a new :class:`DesignMatrixBuilder` that includes only a
        subset of the terms that this object does.

        For example, if `builder` has terms `x`, `y`, and `z`, then::

          builder2 = builder.subset(["x", "z"])

        will return a new builder that will return design matrices with only
        the columns corresponding to the terms `x` and `z`. After we do this,
        then in general these two expressions will return the same thing (here
        we assume that `x`, `y`, and `z` each generate a single column of the
        output)::

          build_design_matrix([builder], data)[0][:, [0, 2]]
          build_design_matrix([builder2], data)[0]

        However, a critical difference is that in the second case, `data` need
        not contain any values for `y`. This is very useful when doing
        prediction using a subset of a model, in which situation R usually
        forces you to specify dummy values for `y`.

        If using a formula to specify the terms to include, remember that like
        any formula, the intercept term will be included by default, so use
        `0` or `-1` in your formula if you want to avoid this.

        :arg which_terms: The terms which should be kept in the new
          :class:`DesignMatrixBuilder`. If this is a string, then it is parsed
          as a formula, and then the names of the resulting terms are taken as
          the terms to keep. If it is a list, then it can contain a mixture of
          term names (as strings) and :class:`Term` objects.

        .. versionadded: 0.2.0
        """
        factor_to_evaluators = {}
        for evaluator in self._evaluators:
            factor_to_evaluators[evaluator.factor] = evaluator
        design_info = self.design_info
        term_name_to_term = dict(zip(design_info.term_names,
                                     design_info.terms))
        if isinstance(which_terms, str):
            # We don't use this EvalEnvironment -- all we want to do is to
            # find matching terms, and we can't do that use == on Term
            # objects, because that calls == on factor objects, which in turn
            # compares EvalEnvironments. So all we do with the parsed formula
            # is pull out the term *names*, which the EvalEnvironment doesn't
            # effect. This is just a placeholder then to allow the ModelDesc
            # to be created:
            env = EvalEnvironment({})
            desc = ModelDesc.from_formula(which_terms, env)
            if desc.lhs_termlist:
                raise PatsyError("right-hand-side-only formula required")
            which_terms = [term.name() for term in desc.rhs_termlist]
        terms = []
        evaluators = set()
        term_to_column_builders = {}
        for term_or_name in which_terms:
            if isinstance(term_or_name, six.string_types):
                if term_or_name not in term_name_to_term:
                    raise PatsyError("requested term %r not found in "
                                     "this DesignMatrixBuilder"
                                     % (term_or_name,))
                term = term_name_to_term[term_or_name]
            else:
                term = term_or_name
            if term not in self._termlist:
                raise PatsyError("requested term '%s' not found in this "
                                 "DesignMatrixBuilder" % (term,))
            for factor in term.factors:
                evaluators.add(factor_to_evaluators[factor])
            terms.append(term)
            column_builder = self._term_to_column_builders[term]
            term_to_column_builders[term] = column_builder
        return DesignMatrixBuilder(terms,
                                   evaluators,
                                   term_to_column_builders)

    def _build(self, evaluator_to_values, dtype):
        factor_to_values = {}
        need_reshape = False
        num_rows = None
        for evaluator, value in six.iteritems(evaluator_to_values):
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
                         self.design_info)
        start_column = 0
        for term in self._termlist:
            for column_builder in self._term_to_column_builders[term]:
                end_column = start_column + column_builder.total_columns
                m_slice = m[:, start_column:end_column]
                column_builder.build(factor_to_values, m_slice)
                start_column = end_column
        assert start_column == self.total_columns
        return need_reshape, m

class _CheckMatch(object):
    def __init__(self, name, eq_fn):
        self._name = name
        self._eq_fn = eq_fn
        self.value = None
        self._value_desc = None
        self._value_origin = None

    def check(self, seen_value, desc, origin):
        if self.value is None:
            self.value = seen_value
            self._value_desc = desc
            self._value_origin = origin
        else:
            if not self._eq_fn(self.value, seen_value):
                msg = ("%s mismatch between %s and %s"
                       % (self._name, self._value_desc, desc))
                if isinstance(self.value, int):
                    msg += " (%r versus %r)" % (self.value, seen_value)
                # XX FIXME: this is a case where having discontiguous Origins
                # would be useful...
                raise PatsyError(msg, origin)

def build_design_matrices(builders, data,
                          NA_action="drop",
                          return_type="matrix",
                          dtype=np.dtype(float)):
    """Construct several design matrices from :class:`DesignMatrixBuilder`
    objects.

    This is one of Patsy's fundamental functions. This function and
    :func:`design_matrix_builders` together form the API to the core formula
    interpretation machinery.

    :arg builders: A list of :class:`DesignMatrixBuilders` specifying the
      design matrices to be built.
    :arg data: A dict-like object which will be used to look up data.
    :arg NA_action: What to do with rows that contain missing values. You can
      ``"drop"`` them, ``"raise"`` an error, or for customization, pass an
      :class:`NAAction` object. See :class:`NAAction` for details on what
      values count as 'missing' (and how to alter this).
    :arg return_type: Either ``"matrix"`` or ``"dataframe"``. See below.
    :arg dtype: The dtype of the returned matrix. Useful if you want to use
      single-precision or extended-precision.

    This function returns either a list of :class:`DesignMatrix` objects (for
    ``return_type="matrix"``) or a list of :class:`pandas.DataFrame` objects
    (for ``return_type="dataframe"``). In both cases, all returned design
    matrices will have ``.design_info`` attributes containing the appropriate
    :class:`DesignInfo` objects.

    Note that unlike :func:`design_matrix_builders`, this function takes only
    a simple data argument, not any kind of iterator. That's because this
    function doesn't need a global view of the data -- everything that depends
    on the whole data set is already encapsulated in the `builders`. If you
    are incrementally processing a large data set, simply call this function
    for each chunk.

    Index handling: This function always checks for indexes in the following
    places:

    * If ``data`` is a :class:`pandas.DataFrame`, its ``.index`` attribute.
    * If any factors evaluate to a :class:`pandas.Series` or
      :class:`pandas.DataFrame`, then their ``.index`` attributes.

    If multiple indexes are found, they must be identical (same values in the
    same order). If no indexes are found, then a default index is generated
    using ``np.arange(num_rows)``. One way or another, we end up with a single
    index for all the data. If ``return_type="dataframe"``, then this index is
    used as the index of the returned DataFrame objects. Examining this index
    makes it possible to determine which rows were removed due to NAs.

    Determining the number of rows in design matrices: This is not as obvious
    as it might seem, because it's possible to have a formula like "~ 1" that
    doesn't depend on the data (it has no factors). For this formula, it's
    obvious what every row in the design matrix should look like (just the
    value ``1``); but, how many rows like this should there be? To determine
    the number of rows in a design matrix, this function always checks in the
    following places:

    * If ``data`` is a :class:`pandas.DataFrame`, then its number of rows.
    * The number of entries in any factors present in any of the design
    * matrices being built.

    All these values much match. In particular, if this function is called to
    generate multiple design matrices at once, then they must all have the
    same number of rows.

    .. versionadded:: 0.2.0
       The ``NA_action`` argument.
    """
    if isinstance(NA_action, str):
        NA_action = NAAction(NA_action)
    if return_type == "dataframe" and not have_pandas:
        raise PatsyError("pandas.DataFrame was requested, but pandas "
                            "is not installed")
    if return_type not in ("matrix", "dataframe"):
        raise PatsyError("unrecognized output type %r, should be "
                            "'matrix' or 'dataframe'" % (return_type,))
    # Evaluate factors
    evaluator_to_values = {}
    evaluator_to_isNAs = {}
    import operator
    rows_checker = _CheckMatch("Number of rows", lambda a, b: a == b)
    index_checker = _CheckMatch("Index", lambda a, b: a.equals(b))
    if have_pandas and isinstance(data, pandas.DataFrame):
        index_checker.check(data.index, "data.index", None)
        rows_checker.check(data.shape[0], "data argument", None)
    for builder in builders:
        # We look at evaluators rather than factors here, because it might
        # happen that we have the same factor twice, but with different
        # memorized state.
        for evaluator in builder._evaluators:
            if evaluator not in evaluator_to_values:
                value, is_NA = evaluator.eval(data, NA_action)
                evaluator_to_isNAs[evaluator] = is_NA
                # value may now be a Series, DataFrame, or ndarray
                name = evaluator.factor.name()
                origin = evaluator.factor.origin
                rows_checker.check(value.shape[0], name, origin)
                if (have_pandas
                    and isinstance(value, (pandas.Series, pandas.DataFrame))):
                    index_checker.check(value.index, name, origin)
                # Strategy: we work with raw ndarrays for doing the actual
                # combining; DesignMatrixBuilder objects never sees pandas
                # objects. Then at the end, if a DataFrame was requested, we
                # convert. So every entry in this dict is either a 2-d array
                # of floats, or a 1-d array of integers (representing
                # categories).
                value = np.asarray(value)
                evaluator_to_values[evaluator] = value
    # Handle NAs
    values = list(evaluator_to_values.values())
    is_NAs = list(evaluator_to_isNAs.values())
    origins = [evaluator.factor.origin for evaluator in evaluator_to_values]
    pandas_index = index_checker.value
    num_rows = rows_checker.value
    # num_rows is None iff evaluator_to_values (and associated sets like
    # 'values') are empty, i.e., we have no actual evaluators involved
    # (formulas like "~ 1").
    if return_type == "dataframe" and num_rows is not None:
        if pandas_index is None:
            pandas_index = np.arange(num_rows)
        values.append(pandas_index)
        is_NAs.append(np.zeros(len(pandas_index), dtype=bool))
        origins.append(None)
    new_values = NA_action.handle_NA(values, is_NAs, origins)
    # NA_action may have changed the number of rows.
    if new_values:
        num_rows = new_values[0].shape[0]
    if return_type == "dataframe" and num_rows is not None:
        pandas_index = new_values.pop()
    evaluator_to_values = dict(zip(evaluator_to_values, new_values))
    # Build factor values into matrices
    results = []
    for builder in builders:
        results.append(builder._build(evaluator_to_values, dtype))
    matrices = []
    for need_reshape, matrix in results:
        if need_reshape:
            # There is no data-dependence, at all -- a formula like "1 ~ 1".
            # In this case the builder just returns a single-row matrix, and
            # we have to broadcast it vertically to the appropriate size. If
            # we can figure out what that is...
            assert matrix.shape[0] == 1
            if num_rows is not None:
                matrix = DesignMatrix(np.repeat(matrix, num_rows, axis=0),
                                      matrix.design_info)
            else:
                raise PatsyError(
                    "No design matrix has any non-trivial factors, "
                    "the data object is not a DataFrame. "
                    "I can't tell how many rows the design matrix should "
                    "have!"
                    )
        matrices.append(matrix)
    if return_type == "dataframe":
        assert have_pandas
        for i, matrix in enumerate(matrices):
            di = matrix.design_info
            matrices[i] = pandas.DataFrame(matrix,
                                           columns=di.column_names,
                                           index=pandas_index)
            matrices[i].design_info = di
    return matrices

# It should be possible to do just the factors -> factor evaluators stuff
# alone, since that, well, makes logical sense to do. though categorical
# coding has to happen afterwards, hmm.
