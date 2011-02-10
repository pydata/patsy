# This file is part of Charlton
# Copyright (C) 2011 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

# This file has the code that figures out how each factor in some given Term
# should be coded. This is complicated by dealing with models with categorical
# factors like:
#   1 + a + a:b
# then technically 'a' (which represents the space of vectors that can be
# produced as linear combinations of the dummy coding of the levels of the
# factor a) is collinear with the intercept, and 'a:b' (which represents the
# space of vectors that can be produced as linear combinations of the dummy
# coding *of a new factor whose levels are the cartesian product of a and b)
# is collinear with both 'a' and the intercept.
#
# In such a case, the rule is that we find some way to code each term so that
# the full space of vectors that it represents *is present in the model* BUT
# there is no collinearity between the different terms. In effect, we have to
# choose a set of vectors that spans everything that that term wants to span,
# *except* that part of the vector space which was already spanned by earlier
# terms.

# This should really be a named tuple, but those don't exist until Python
# 2.6...
class _ExpandedFactor(object):
    def __init__(self, includes_intercept, factor):
        self.includes_intercept = includes_intercept
        self.factor = factor

    def __hash__(self):
        return hash((_ExpandedFactor, self.includes_intercept, self.factor))

    def __eq__(self, other):
        return (isinstance(other, _ExpandedFactor)
                and other.includes_intercept == self.includes_intercept
                and other.factor == self.factor)

    def __repr__(self):
        if self.includes_intercept:
            suffix = "+"
        else:
            suffix = "-"
        return "%r%s" % (self.factor, suffix)

# Importantly, this preserves the order of the input. Both the items inside
# each subset are in the order they were in the original tuple, and the tuples
# are emitted so that they're sorted with respect to their elements position
# in the original tuple.
def _subsets_sorted(tupl):
    def helper(seq):
        if not seq:
            yield ()
        else:
            obj = seq[0]
            for subset in _subsets(seq[1:]):
                yield subset
                yield (obj,) + subset
    # Transform each obj -> (idx, obj) tuple, so that we can later sort them
    # by their position in the original list.
    expanded = list(enumerate(tupl))
    expanded_subsets = list(helper(expanded))
    # This exploits Python's stable sort: we want short before long, and ties
    # broken by natural ordering on the (idx, obj) entries in each subset. So
    # we sort by the latter first, then by the former.
    expanded_subsets.sort()
    expanded_subsets.sort(key=len)
    # And finally, we strip off the idx's:
    for subset in expanded_subsets:
        yield tuple([obj for (idx, obj) in subset])
    
_subsets = _subsets_sorted

def test__subsets():
    assert list(_subsets((1, 2))) == [(), (1,), (2,), (1, 2)]
    assert (list(_subsets((1, 2, 3)))
            == [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)])
    assert len(list(_subsets(range(5)))) == 2 ** 5

# Converts a term into an expanded list of subterms like:
#   a:b  ->  1 + a- + b- + a-:b-
# (where "-" indicates contrast coding.)
def _expand_categorical_part(term, numeric_factors):
    categoricals = tuple([f for f in term.factors if f not in numeric_factors])
    for subset in _subsets_sorted(categoricals):
        yield tuple([_ExpandedFactor(False, factor) for factor in subset])

def test__expand_categorical_part():
    # Exploiting that none of this actually cares whether the 'factors' are
    # valid factor objects....
    from charlton.desc import Term
    t = Term(["c", "b", "a"])
    assert (list(_expand_categorical_part(t, frozenset(["b"])))
            == [(),
                (_ExpandedFactor(False, "c"),), (_ExpandedFactor(False, "a"),),
                (_ExpandedFactor(False, "c"), _ExpandedFactor(False, "a"))])

def _find_combinable_subterms(subterms):
    # The simplification rule is:
    #   subterm + subterm:x-
    # becomes
    #   subterm:x+
    # We simplify greedily from left to right.
    for i, short_subterm in enumerate(subterms):
        for long_subterm in subterms[i + 1:]:
            if len(long_subterm) > len(short_subterm) + 1:
                break
            if len(long_subterm) < len(short_subterm) + 1:
                continue
            diff = set(long_subterm).difference(short_subterm)
            if len(diff) == 1:
                return i, diff.pop()
    return None
            
def _simplify_subterms(subterms):
    while True:
        combinable = _find_combinable_subterms(subterms)
        if combinable is None:
            break
        i, extra_factor = combinable
        assert not extra_factor.includes_intercept
        extra_factor.includes_intercept = True
        subterms.pop(i)

def test__simplify_subterms():
    def expand_abbrevs(l):
        for subterm in l:
            factors = []
            for factor_name in subterm:
                assert factor_name[-1] in ("+", "-")
                factors.append(_ExpandedFactor(factor_name[-1] == "+",
                                               factor_name[:-1]))
            yield factors
    def t(given, expected):
        given = list(expand_abbrevs(given))
        expected = list(expand_abbrevs(expected))
        print "testing if:", given, "->", expected
        _simplify_subterms(given)
        assert given == expected
    t([("a-",)], [("a-",)])
    t([(), ("a-",)], [("a+",)])
    t([(), ("a-",), ("b-",), ("a-", "b-")], [("a+", "b+")])
    t([(), ("a-",), ("a-", "b-")], [("a+",), ("a-", "b-")])
    t([("a-",), ("b-",), ("a-", "b-")], [("b-",), ("a-", "b+")])

def coding_for_categorical(term, numeric_factors, previous_subterms):
    subterms = [subterm
                for subterm in _expand_categorical_part(term, numeric_factors)
                if subterm not in previous_subterms]
    previous_subterms.union(subterms)
    _simplify_subterms(subterms)
    return subterms, set(term.factors).intersection(numeric_factors)
