How formulas work
=================

.. figure:: figures/formula-structure.png
   :align: center
   
   The pieces that make up a formula.

Say we have a formula like::

  y ~ a + a:b + np.log(x)

This overall thing is a **formula**, and it's divided into a left-hand
side, ``y``, and a right-hand side, ``a + a:b +
np.log(x)``. (Sometimes you want a formulas that has no left-hand
side, and you can write that as ``~ x1 + x2`` or even ``x1 + x2``.)
Each side contains a list of **terms** separated by ``+``; on the left
there is one term, ``y``, and on the right, there are three terms,
``a`` and ``a:b`` and ``np.log(x)`` (plus an invisible intercept
term). And finally, each term is the interaction of zero or more
**factors**. A factor is the minimal, indivisible unit that each
formula is built up out of; the factors here are ``y``, ``a``, ``b``,
and ``np.log(x)``. Most of these terms have only one factor -- for
example, the term ``y`` is a kind of trivial interaction between the
factor ``y`` and, well... nothing else. There's only one factor in
that "interaction". The term ``a:b`` is an interaction between two
factors, ``a`` and ``b``. And the intercept term is an interaction
betwee *zero* factors. (This may seem odd, but it turns out that
defining the zero-order interaction to be a column of all ones is very
convenient, just like it turns out to be convenient to define the
product of a zero item list to be ``np.prod([]) == 1``.)

.. warning:: In the context of Charlton, the word **factor** does
   *not* refer specifically to categorical data. What we call a
   "factor" can represent either categorical or numerical data. Think
   of factors like in multiplying factors together, not like in
   factorial design. When we want to refer to categorical data, this
   manual and the Charlton API use the word "categorical".

To make this more concrete, here's how you could construct "by hand"
the same objects that Charlton will construct if given the above
formula::

  env = EvalEnvironment.capture()
  ModelDesc([Term([EvalFactor("y", env)])],
            [Term([]),
             Term([EvalFactor("a", env)]),
             Term([EvalFactor("a", env), EvalFactor("b", env)]),
             Term([EvalFactor("np.log(x)", env)])])

:class:`ModelDesc` represents an overall formula; it just takes two
lists of :class:`Term` objects, representing the left-hand side and
the right-hand side. And each ``Term`` object just takes a list of
factor objects. In this case our factors are of type
:class:`EvalFactor`, which evaluates arbitrary Python code, but in
general any object that implements the factor protocol will do -- see
XX for details.

Of course as a user you never have to actually touch ``ModelDesc``,
``Term``, or ``EvalFactor`` objects by hand -- but it's useful to
know that this lower layer exists in case you ever want to generate a
formula programmatically, and to have an image in your mind of what a
formula really is.

The formula language
--------------------

Now let's talk about exactly how those magic strings are processed.

Since all Charlton models are just sets of terms, you could write any
model just using ``:`` to create interactions, ``+`` to join terms
together into a set, and ``~`` to separate the left-hand side from the
right-hand side.  But for convenience, Charlton also understands a
number of other short-hand operators, and evaluates them all using a
`full-fledged parser
<http://en.wikipedia.org/wiki/Shunting_yard_algorithm>`_ complete with
robust error reporting, etc.

Operators
^^^^^^^^^

The built-in binary operators are:

============  =======================================
``~``         lowest precedence (binds most loosely)
``+``, ``-``
``*``, ``/``
``:``
``**``        highest precedence (binds most tightly)
============  =======================================

Of course, you can override the order of operations using
parentheses. All operations are left-associative (so ``a - b - c`` means
the same as ``(a - b) - c``, not ``a - (b - c)``). Their meanings are as
follows:

``~``
  Separates the left-hand side and right-hand side of a
  formula. Optional; if not present, then the formula is considered to
  contain a right-hand side only.

``+``
  Takes the set of terms given on the left and the set of terms given
  on the right, and returns a set of terms that combines both (i.e.,
  it computes set union). Note that this means that ``a + a`` is just
  ``a``.

``-``
  Takes the set of terms given on the left and removes any terms which
  are given on the right (a set difference operation).

``*``
  ``a * b`` is short-hand for ``a + b + a:b``, and is useful for the
  common case of wanting to include all interactions between a set of
  variables (e.g., standard ANOVA models are of the form ``a * b * c *
  ...``).

``/``
  This one is a bit quirky. ``a / b`` is shorthand for ``a + a:b``, and
  is intended to be useful in cases where you want to fit a standard
  sort of ANOVA model, but ``b`` is nested within ``a``, so ``a*b`` doesn't
  make sense. So far so good. Also, if you have multiple terms on the
  right, then the obvious thing happens: ``a / (b + c)`` is equivalent
  to ``a + a:b + a:c`` (``/`` is "rightward distributive"). *But,* if you
  have multiple terms on the left, then there is a surprising special
  case: ``(a + b)/c`` is equivalent to ``a + b + a:b:c`` (and note that
  this is different from what you'd get out of ``a/c + b/c`` -- ``/``
  is *not* "leftward distributive"). Again, this is motivated by the
  idea of using this for nested variables. It doesn't make sense for
  ``c`` to be nested within both ``a`` and ``b`` separately, unless ``b`` is
  itself nested in ``a`` -- but if that were true, then you'd write
  ``a/b/c`` instead. So if we see ``(a + b)/c``, we decide that ``a`` and
  ``b`` must be independent factors, but that ``c`` is nested within each
  *combination* of levels of ``a`` and ``b``, which is what ``a:b:c`` gives
  us. If this is confusing, then my apologies... the behaviour is
  inherited from S.

``:``
  This takes two sets of terms, and computes the interaction between
  each term on the left and each term on the right. So, for example,
  ``(a + b):(c + d)`` is the same as ``a:c + a:d + b:c +
  b:d``. Calculating the interaction between two terms is also a kind
  of set union operation, but ``:`` takes the union of factors *within*
  two terms, while ``+`` takes the union of two sets of terms. Note that
  this means that ``a:a`` is just ``a``, and ``(a:b):(a:c)`` is the same as
  ``a:b:c``.

``**``
  This takes a set of terms on the left, and an integer *n* on the
  right, and computes the ``*`` of that set of terms with itself *n*
  times. This is useful if you want to compute all interactions up to
  order *n*, but no further. Example::

   (a + b + c + d) ** 3

  is expanded to::

   (a + b + c + d) * (a + b + c + d) * (a + b + c + d)

  Note that an equivalent way to write this particular expression
  would be

   a*b*c*d - a:b:c:d

 (Exercise: why?)

The parser also understands unary ``+`` and ``-``, though they aren't very
useful. ``+`` is a no-op, and ``-`` can only be used in the forms ``-1``
(which means the same as ``0``) and ``-0`` (which means the same as ``1``).

Factors and terms
^^^^^^^^^^^^^^^^^

So that explains how the operators work -- the verbs in the formula
language -- but what about the nouns, the terms like ``y`` and
``np.log(x)`` that are actually picking out bits of your data?

Individual factors are allowed to be arbitrary Python code. Scanning
arbitrary Python code can be quite complicated, but Charlton uses the
official Python tokenizer built into the standard library, so it's
able to do it robustly. There is still a bit of a problem, though,
since Charlton operators like ``+`` are also valid Python
operators. When we see a ``+``, how do we know which interpretation to
use?

The answer is that a Python factor begins whenever we see a token
which

* is not a Charlton operator listed in that table up above, and
* is not a parentheses

And then the factor ends whenever we see a token which

* is a Charlton operator listed in that table up above, and
* it not *enclosed in any kind of parentheses* (where "any kind"
  includes regular, square, and curly brackets)

This will be clearer with an example::

  f(x1 + x2) + x3

First, we see ``f``, which is not an operator or a parentheses, so we
know this string begins with a Python-defined factor. Then we keep
reading from there. The next Charlton operator we see is the ``+`` in
``x1 + x2``... but since at this point we have seen the opening ``(`` but
not the closing ``)``, we ignore it. Eventually we come to the second
``+``, and by this time we have seen the closing parentheses, so we know
that this is the end of the first factor.

One side-effect of this is that if you do want to perform some
arithmetic inside your formula object, you can "hide" it from the
Charlton parser by putting it inside a function call. To make this
more convenient, Charlton provides a builtin function called ``I()``
that simply returns its input. (I.e., it's the Identity function.)
That way you can use ``I(x1 + x2)`` inside a formula to represent the
sum of ``x1`` and ``x2``.

.. note:: We've played a bit fast-and-loose with the distinction
    between factors and terms. Technically, given something like
    ``a:b``, what's happening is first that we create a factor ``a`` and
    then we package it up into a single-factor term. And then we
    create a factor ``b``, and we package it up into a single-factor
    term. And then we evaluate the ``:``, and compute the interaction
    between these two terms.

Intercept handling
^^^^^^^^^^^^^^^^^^

There are two special things about how intercept terms are handled
inside the formula parser.

First, since an intercept term is an interaction of zero factors, we
have no way to write it down using the parts of the language described
so far. Therefore, as a special case, the string "1" is taken to
represent the intercept term.

Second, since intercept terms are almost always desired and
remembering to include them by hand all the time is quite tedious,
they are always included by default in the right-hand side of any
formula. The way this is implemented is exactly as if there is an
invisible ``1 +`` inserted at the beginning of every right-hand side.

Of course, if you don't want an intercept, you can remove it again
just like any other unwanted term, using the ``-`` operator. This
formula has an intercept::

  y ~ x

This formula does not::

  y ~ x - 1

For compatibility with S and R, we also allow the magic terms ``0`` and
``-1`` which represent the "anti-intercept". Adding one of these terms
has exactly the same effect as subtracting the intercept term, and
subtracting one of these terms has exactly the same effect as adding
the intercept term. That means that all of these formulas are
equivalent::

  y ~ x - 1
  y ~ x + -1
  y ~ -1 + x
  y ~ 0 + x
  y ~ x - (-0)

Explore!
^^^^^^^^

The formula language is actually fairly simple once you get the hang
of it, but if you're ever in doubt as to what some construction means,
you can always ask Charlton how it expands.

Here's some code to try out at the Python prompt to get started::

  from charlton import EvalEnvironment, ModelDesc
  env = EvalEnvironment.capture()
  ModelDesc.from_formula("y ~ x", env)
  ModelDesc.from_formula("y ~ x + x + x", env)
  ModelDesc.from_formula("y ~ -1 + x", env)
  ModelDesc.from_formula("~ -1", env)
  ModelDesc.from_formula("y ~ a:b", env)
  ModelDesc.from_formula("y ~ a*b", env)
  ModelDesc.from_formula("y ~ (a + b + c + d) ** 2", env)
  ModelDesc.from_formula("y ~ (a + b)/(c + d)", env)
  ModelDesc.from_formula("f(x1 + x2) + (x + {6: x3, 8 + 1: x4}[3 * i])", env)

From terms to matrices
----------------------

So at this point, you hopefully understand how a string is parsed into
two sets of terms (represented by a :class:`ModelDesc` object holding
:class:`Term` objects).

term ordering

building a formula programmatically

interactions and redundancy
