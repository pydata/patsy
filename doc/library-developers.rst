Using Charlton in your library
==============================

Our goal is to make Charlton the de facto standard for describing
models in Python, regardless of the underlying package in use -- just
as formulas are the standard interface to all R packages. Therefore
we've tried to make it as easy as possible for you to build Charlton
support into your libraries.

Charlton is a good houseguest:

* Pure Python, no compilation necessary.
* Exhaustive tests (>98% statement coverage at time of writing) and
  documentation.
* No dependencies besides numpy (and we even test against numpy 1.2.1,
  as distributed by RHEL 5).
* Tested and supported on every version of Python since 2.4.

So you can be pretty confident that adding a dependency on Charlton
won't create much hassle for your users.

And, of course, the fundamental design is very conservative -- the
formula mini-language in S was first described in Chambers and Hastie
(1992), more than two decades ago, and it's still in heavy use today
in R, which is one of the most popular packages . Many of your users may already be familiar with it.

Using the high-level interface
------------------------------

If you have a function whose signature currently looks like this::

  def mymodel2(X, y, ...):
      ...

or this::

  def mymodel1(X, ...):
      ...

then adding Charlton support is extremely easy (though of course like
any other API change, you may have to deprecate the old interface, or
provide two interfaces in parallel, depending on your situation). Just
write something like::

  def mymodel2_charlton(formula_like, data={}, ...):
      y, X = charlton.dmatrices(formula_like, data, 1)
      ...

or::

  def mymodel1_charlton(formula_like, data={}, ...):
      X = charlton.dmatrix(formula_like, data, 1)
      ...

(See :func:`dmatrices` and :func:`dmatrix` for details.) This won't
force your users to switch to formulas immediately; they can replace
code that looks like this::

  X, y = build_matrices_laboriously()
  result = mymodel2(X, y, ...)
  other_result = mymodel1(X, ...)

with code like this:

  X, y = build_matrices_laboriously()
  result = mymodel2((y, X), data=None, ...)
  other_result = mymodel2(X, data=None, ...)

Of course in the long run they might want to throw away that
:func:`build_matrices_laboriously` function and start using formulas,
but they don't have to do that just to adapt.

Working with metadata
^^^^^^^^^^^^^^^^^^^^^



Predictions
^^^^^^^^^^^



Example
^^^^^^^



Other cool tricks
^^^^^^^^^^^^^^^^^

If you want to compute ANOVAs, then the proper way is to 

Extending the formula syntax
----------------------------

