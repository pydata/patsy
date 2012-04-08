# This file is part of Charlton
# Copyright (C) 2011 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

# The extra functions and such that are made available to formula code by
# default.

__all__ = ["builtins"]

builtins = {}

from charlton.contrasts import Treatment, Poly, Sum, Helmert, Diff

builtins["Treatment"] = Treatment
builtins["Poly"] = Poly
builtins["Sum"] = Sum
builtins["Helmert"] = Helmert
builtins["Diff"] = Diff

def I(x):
    """The identity function. Simply returns its input unchanged.

    Since Charlton's formula parser ignores anything inside a function call
    syntax, this is useful to 'hide' arithmetic operations from it. For
    instance::

      y ~ x1 + x2

    has ``x1`` and ``x2`` as two separate predictors. But in::

      y ~ I(x1 + x2)

    we instead have a single predictor, defined to be the sum of ``x1`` and
    ``x2``."""
    return x

builtins["I"] = I

def test_I():
    assert I(1) == 1
    assert I(None) is None

def Q(name):
    """A way to 'quote' variable names, especially ones that do not otherwise
    meet Python's variable name rules.

    If ``x`` is a variable, ``Q("x")`` returns the value of ``x``. (Note that
    ``Q`` takes the *string* ``"x"``, not the value of ``x`` itself.) This
    works even if instead of ``x``, we have a variable name that would not
    otherwise be legal in Python.

    For example, if you have a column of data named `weight.in.kg`, then you
    can't write::

      y ~ weight.in.kg

    because Python will try to find a variable named ``weight``, that has an
    attribute named ``in``, that has an attribute named ``kg``. (And worse
    yet, ``in`` is a reserved word, which makes this example doubly broken.)
    Instead, write::

      y ~ Q("weight.in.kg")

    and all will be well. Note, though, that this requires embedding a Python
    string inside your formula, which may require some care with your quote
    marks. Some standard options include:

      my_fit_function("y ~ Q('weight.in.kg')", ...)
      my_fit_function('y ~ Q("weight.in.kg")', ...)
      my_fit_function("y ~ Q(\"weight.in.kg\")", ...)

    Note also that ``Q`` is an ordinary Python function, which means that you
    can use it in more complex expressions. For example, this is a legal
    formula::

      y ~ np.sqrt(Q("weight.in.kg"))
    """
    from charlton.eval import EvalEnvironment
    env = EvalEnvironment.capture(1)
    try:
        return env.namespace[name]
    except KeyError:
        raise NameError, "no data named '%s' found" % (name,)

builtins["Q"] = Q

def test_Q():
    a = 1
    assert Q("a") == 1
    assert Q("Q") is Q
    from nose.tools import assert_raises
    assert_raises(NameError, Q, "asdfsadfdsad")
