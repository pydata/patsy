# This file is part of Charlton
# Copyright (C) 2012 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

# There are a number of unit tests in build.py, but this file contains more
# thorough tests of the overall design matrix building system. (These are
# still not exhaustive end-to-end tests, though -- for that see test_api.py.)

import numpy as np
from nose.tools import assert_raises
from charlton import CharltonError
from charlton.util import atleast_2d_column_default
from charlton.compat import itertools_product
from charlton.desc import Term, INTERCEPT, LookupFactor
from charlton.build import make_design_matrix_builders, make_design_matrices
from charlton.categorical import C

def assert_full_rank(m):
    m = atleast_2d_column_default(m)
    if m.shape[1] == 0:
        return True
    u, s, v = np.linalg.svd(m)
    rank = np.sum(s > 1e-10)
    assert rank == m.shape[1]
    
def test_assert_full_rank():
    assert_full_rank(np.eye(10))
    assert_full_rank([[1, 0], [1, 0], [1, 0], [1, 1]])
    assert_raises(AssertionError,
                  assert_full_rank, [[1, 0], [2, 0]])
    assert_raises(AssertionError,
                  assert_full_rank, [[1, 2], [2, 4]])
    assert_raises(AssertionError,
                  assert_full_rank, [[1, 2, 3], [1, 10, 100]])
    # col1 + col2 = col3
    assert_raises(AssertionError,
                  assert_full_rank, [[1, 2, 3], [1, 5, 6], [1, 6, 7]])
    
def make_test_factors(*level_counts, **kwargs):
    def levels(name, level_count):
        return ["%s%s" % (name, i) for i in xrange(1, level_count + 1)]
    # zip(*...) means "unzip":
    name_counts = zip("abcdefghijklmnopqrstuvwxyz", level_counts)
    all_levels = [levels(*name_count) for name_count in name_counts]
    values = [list(v) for v in zip(*itertools_product(*all_levels))]
    for i in xrange(len(values)):
        values[i] *= kwargs.get("repeat", 1)
    return values

def test_make_test_factors():
    a, b = make_test_factors(2, 3)
    assert a == ["a1", "a1", "a1", "a2", "a2", "a2"]
    assert b == ["b1", "b2", "b3", "b1", "b2", "b3"]
    a, b = make_test_factors(2, 3, repeat=2)
    assert a == ["a1", "a1", "a1", "a2", "a2", "a2",
                 "a1", "a1", "a1", "a2", "a2", "a2"]
    assert b == ["b1", "b2", "b3", "b1", "b2", "b3",
                 "b1", "b2", "b3", "b1", "b2", "b3"]

def make_termlist(*entries):
    terms = []
    for entry in entries:
        terms.append(Term([LookupFactor(name) for name in entry]))
    return terms

def check_design_matrix(mm, expected_rank, termlist, column_names=None):
    assert_full_rank(mm)
    assert sorted(mm.column_info.terms) == sorted(termlist)
    if column_names is not None:
        assert mm.column_info.column_names == column_names
    assert mm.ndim == 2
    assert mm.shape[1] == expected_rank

def make_matrix(data, expected_rank, entries, column_names=None):
    termlist = make_termlist(*entries)
    def iter_maker():
        yield data
    builders = make_design_matrix_builders([termlist], iter_maker)
    matrices = make_design_matrices(builders, data)
    matrix = matrices[0]
    assert matrix.builder is builders[0]
    check_design_matrix(matrix, expected_rank, termlist,
                        column_names=column_names)
    return matrix

def test_simple():
    data = {}
    data["a"], data["b"] = make_test_factors(2, 2)
    x1 = data["x1"] = np.linspace(0, 1, len(data["a"]))
    x2 = data["x2"] = data["x1"] ** 2

    m = make_matrix(data, 2, [["a"]], column_names=["a[a1]", "a[a2]"])
    assert np.allclose(m, [[1, 0], [1, 0], [0, 1], [0, 1]])

    m = make_matrix(data, 2, [[], ["a"]], column_names=["Intercept", "a[T.a2]"])
    assert np.allclose(m, [[1, 0], [1, 0], [1, 1], [1, 1]])

    m = make_matrix(data, 4, [["a", "b"]],
                    column_names=["a[a1]:b[b1]", "a[a2]:b[b1]",
                                  "a[a1]:b[b2]", "a[a2]:b[b2]"])
    assert np.allclose(m, [[1, 0, 0, 0],
                           [0, 0, 1, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1]])

    m = make_matrix(data, 4, [[], ["a"], ["b"], ["a", "b"]],
                    column_names=["Intercept", "a[T.a2]",
                                  "b[T.b2]", "a[T.a2]:b[T.b2]"])
    assert np.allclose(m, [[1, 0, 0, 0],
                           [1, 0, 1, 0],
                           [1, 1, 0, 0],
                           [1, 1, 1, 1]])

    m = make_matrix(data, 4, [[], ["b"], ["a"], ["b", "a"]],
                    column_names=["Intercept", "b[T.b2]",
                                  "a[T.a2]", "b[T.b2]:a[T.a2]"])
    assert np.allclose(m, [[1, 0, 0, 0],
                           [1, 1, 0, 0],
                           [1, 0, 1, 0],
                           [1, 1, 1, 1]])

    m = make_matrix(data, 4, [["a"], ["x1"], ["a", "x1"]],
                    column_names=["a[a1]", "a[a2]", "x1", "a[T.a2]:x1"])
    assert np.allclose(m, [[1, 0, x1[0], 0],
                           [1, 0, x1[1], 0],
                           [0, 1, x1[2], x1[2]],
                           [0, 1, x1[3], x1[3]]])
    
    m = make_matrix(data, 3, [["x1"], ["x2"], ["x2", "x1"]],
                    column_names=["x1", "x2", "x2:x1"])
    assert np.allclose(m, np.column_stack((x1, x2, x1 * x2)))
    
def test_R_bugs():
    data = {}
    data["a"], data["b"], data["c"] = make_test_factors(2, 2, 2)
    data["x"] = np.linspace(0, 1, len(data["a"]))
    # For "1 + a:b", R produces a design matrix with too many columns (5
    # instead of 4), because it can't tell that there is a redundancy between
    # the two terms.
    make_matrix(data, 4, [[], ["a", "b"]])
    # For "0 + a:x + a:b", R produces a design matrix with too few columns (4
    # instead of 6), because it thinks that there is a redundancy which
    # doesn't exist.
    make_matrix(data, 6, [["a", "x"], ["a", "b"]])
    # This can be compared with "0 + a:c + a:b", where the redundancy does
    # exist. Confusingly, adding another categorical factor increases the
    # baseline dimensionality to 8, and then the redundancy reduces it to 6
    # again, so the result is the same as before but for different reasons. (R
    # does get this one right, but we might as well test it.)
    make_matrix(data, 6, [["a", "c"], ["a", "b"]])

def test_redundancy_thoroughly():
    # To make sure there aren't any lurking bugs analogous to the ones that R
    # has (see above), we check that we get the correct matrix rank for every
    # possible combination of 2 categorical and 2 numerical factors.
    data = {}
    data["a"], data["b"] = make_test_factors(2, 2, repeat=5)
    data["x1"] = np.linspace(0, 1, len(data["a"]))
    data["x2"] = data["x1"] ** 2
    
    def all_subsets(l):
        if not l:
            yield tuple()
        else:
            obj = l[0]
            for subset in all_subsets(l[1:]):
                yield tuple(sorted(subset))
                yield tuple(sorted((obj,) + subset))

    all_terms = list(all_subsets(("a", "b", "x1", "x2")))
    all_termlist_templates = list(all_subsets(all_terms))
    print len(all_termlist_templates)
    # eliminate some of the symmetric versions to speed things up
    redundant = [[("b",), ("a",)],
                 [("x2",), ("x1")],
                 [("b", "x2"), ("a", "x1")],
                 [("a", "b", "x2"), ("a", "b", "x1")],
                 [("b", "x1", "x2"), ("a", "x1", "x2")]]
    for termlist_template in all_termlist_templates:
        termlist_set = set(termlist_template)
        for dispreferred, preferred in redundant:
            if dispreferred in termlist_set and preferred not in termlist_set:
                break
        else:
            expanded_terms = set()
            for term_template in termlist_template:
                numeric = tuple([t for t in term_template if t.startswith("x")])
                rest = [t for t in term_template if not t.startswith("x")]
                for subset_rest in all_subsets(rest):
                    expanded_terms.add(frozenset(subset_rest + numeric))
            # Because our categorical variables have 2 levels, each expanded
            # term corresponds to 1 unique dimension of variation
            expected_rank = len(expanded_terms)
            make_matrix(data, expected_rank, termlist_template)

test_redundancy_thoroughly.slow = 1
    
def test_data_types():
    basic_dict = {"a": ["a1", "a2", "a1", "a2"],
                  "x": [1, 2, 3, 4]}
    structured_array = np.array(zip(basic_dict["a"], basic_dict["x"]),
                                dtype=[("a", "S2"), ("x", int)])
    recarray = structured_array.view(np.recarray)
    datas = [basic_dict, structured_array, recarray]
    try:
        import pandas
    except ImportError:
        pass
    else:
        df = pandas.DataFrame(basic_dict)
        datas.append(df)
    for data in datas:
        m = make_matrix(data, 4, [["a"], ["a", "x"]],
                        column_names=["a[a1]", "a[a2]", "a[a1]:x", "a[a2]:x"])
        assert np.allclose(m, [[1, 0, 1, 0],
                               [0, 1, 0, 2],
                               [1, 0, 3, 0],
                               [0, 1, 0, 4]])

def test_data_mismatch():
    test_cases = [
        # Data type mismatch
        ([1, 2, 3], ["a", "b", "c"]),
        ([1, 2, 3], [True, False, True]),
        (["a", "b", "c"], [True, False, True]),
        (C(["a", "b", "c"]), [1, 2, 3]),
        (C(["a", "b", "c"]), [True, False, True]),
        (C(["a", "b", "c"], levels=["c", "b", "a"]), C(["a", "b", "c"])),
        # column number mismatches
        ([[1], [2], [3]], [[1, 1], [2, 2], [3, 3]]),
        ([[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[1, 1], [2, 2], [3, 3]]),
        ]
    setup_predict_only = [
        # This is not an error if both are fed in during make_builders, but it
        # is an error to pass one to make_builders and the other to
        # make_matrices.
        (["a", "b", "c"], ["a", "b", "d"]),
        ]
    termlist = make_termlist(["x"])
    def t_incremental(data1, data2):
        def iter_maker():
            yield {"x": data1}
            yield {"x": data2}
        try:
            builders = make_design_matrix_builders([termlist], iter_maker)
            make_design_matrices(builders, {"x": data1})
            make_design_matrices(builders, {"x": data2})
        except CharltonError:
            pass
        else:
            raise AssertionError
    def t_setup_predict(data1, data2):
        def iter_maker():
            yield {"x": data1}
        builders = make_design_matrix_builders([termlist], iter_maker)
        assert_raises(CharltonError,
                      make_design_matrices, builders, {"x": data2})
    for (a, b) in test_cases:
        t_incremental(a, b)
        t_incremental(b, a)
        t_setup_predict(a, b)
        t_setup_predict(b, a)
    for (a, b) in setup_predict_only:
        t_setup_predict(a, b)
        t_setup_predict(b, a)

    assert_raises(CharltonError,
                  make_matrix, {"x": [1, 2, 3], "y": [1, 2, 3, 4]},
                  2, [["x"], ["y"]])

def test_data_independent_builder():
    data = {"x": [1, 2, 3]}
    def iter_maker():
        yield data

    # If building a formula that doesn't depend on the data at all, we just
    # return a single-row matrix.
    m = make_matrix(data, 0, [], column_names=[])
    assert m.shape == (1, 0)

    m = make_matrix(data, 1, [[]], column_names=["Intercept"])
    assert np.allclose(m, [[1]])

    # Or, if there are other matrices that do depend on the data, we make the
    # data-independent matrices have the same number of rows.
    x_termlist = make_termlist(["x"])

    builders = make_design_matrix_builders([x_termlist, make_termlist()],
                                           iter_maker)
    x_m, null_m = make_design_matrices(builders, data)
    assert np.allclose(x_m, [[1], [2], [3]])
    assert null_m.shape == (3, 0)

    builders = make_design_matrix_builders([x_termlist, make_termlist([])],
                                           iter_maker)
    x_m, intercept_m = make_design_matrices(builders, data)
    assert np.allclose(x_m, [[1], [2], [3]])
    assert np.allclose(intercept_m, [[1], [1], [1]])

def test_same_factor_in_two_matrices():
    data = {"x": [1, 2, 3], "a": ["a1", "a2", "a1"]}
    def iter_maker():
        yield data
    t1 = make_termlist(["x"])
    t2 = make_termlist(["x", "a"])
    builders = make_design_matrix_builders([t1, t2], iter_maker)
    m1, m2 = make_design_matrices(builders, data)
    check_design_matrix(m1, 1, t1, column_names=["x"])
    assert np.allclose(m1, [[1], [2], [3]])
    check_design_matrix(m2, 2, t2, column_names=["x:a[a1]", "x:a[a2]"])
    assert np.allclose(m2, [[1, 0], [0, 2], [3, 0]])

def test_categorical():
    data_strings = {"a": ["a1", "a2", "a1"]}
    data_categ = {"a": C(["a2", "a1", "a2"])}
    def t(data1, data2):
        def iter_maker():
            yield data1
        builders = make_design_matrix_builders([make_termlist(["a"])],
                                               iter_maker)
        make_design_matrices(builders, data2)
    t(data_strings, data_categ)
    t(data_categ, data_strings)

def test_contrast():
    from charlton.contrasts import ContrastMatrix, Sum
    values = ["a1", "a3", "a1", "a2"]
    
    # No intercept in model, full-rank coding of 'a'
    m = make_matrix({"a": C(values)}, 3, [["a"]],
                    column_names=["a[a1]", "a[a2]", "a[a3]"])

    assert np.allclose(m, [[1, 0, 0],
                           [0, 0, 1],
                           [1, 0, 0],
                           [0, 1, 0]])
    
    for s in (Sum, Sum()):
        m = make_matrix({"a": C(values, s)}, 3, [["a"]],
                        column_names=["a[mean]", "a[S.a1]", "a[S.a2]"])
        # Output from R
        assert np.allclose(m, [[1, 1, 0],
                               [1,-1, -1],
                               [1, 1, 0],
                               [1, 0, 1]])
    
    m = make_matrix({"a": C(values, Sum(omit=0))}, 3, [["a"]],
                    column_names=["a[mean]", "a[S.a2]", "a[S.a3]"])
    # Output from R
    assert np.allclose(m, [[1, -1, -1],
                           [1,  0,  1],
                           [1, -1, -1],
                           [1,  1,  0]])

    # Intercept in model, non-full-rank coding of 'a'
    m = make_matrix({"a": C(values)}, 3, [[], ["a"]],
                    column_names=["Intercept", "a[T.a2]", "a[T.a3]"])

    assert np.allclose(m, [[1, 0, 0],
                           [1, 0, 1],
                           [1, 0, 0],
                           [1, 1, 0]])
    
    for s in (Sum, Sum()):
        m = make_matrix({"a": C(values, s)}, 3, [[], ["a"]],
                        column_names=["Intercept", "a[S.a1]", "a[S.a2]"])
        # Output from R
        assert np.allclose(m, [[1, 1, 0],
                               [1,-1, -1],
                               [1, 1, 0],
                               [1, 0, 1]])
    
    m = make_matrix({"a": C(values, Sum(omit=0))}, 3, [[], ["a"]],
                    column_names=["Intercept", "a[S.a2]", "a[S.a3]"])
    # Output from R
    assert np.allclose(m, [[1, -1, -1],
                           [1,  0,  1],
                           [1, -1, -1],
                           [1,  1,  0]])

    # Weird ad hoc less-than-full-rank coding of 'a'
    m = make_matrix({"a": C(values, [[7, 12],
                                     [2, 13],
                                     [8, -1]])},
                    2, [["a"]],
                    column_names=["a[custom0]", "a[custom1]"])
    assert np.allclose(m, [[7, 12],
                           [8, -1],
                           [7, 12],
                           [2, 13]])

    m = make_matrix({"a": C(values, ContrastMatrix([[7, 12],
                                                    [2, 13],
                                                    [8, -1]],
                                                   ["[foo]", "[bar]"]))},
                    2, [["a"]],
                    column_names=["a[foo]", "a[bar]"])
    assert np.allclose(m, [[7, 12],
                           [8, -1],
                           [7, 12],
                           [2, 13]])

    
    
