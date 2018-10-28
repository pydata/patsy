.. _library-developers:

Using Patsy in your library
==============================

.. currentmodule:: patsy

Our goal is to make Patsy the de facto standard for describing
models in Python, regardless of the underlying package in use -- just
as formulas are the standard interface to all R packages. Therefore
we've tried to make it as easy as possible for you to build Patsy
support into your libraries.

Patsy is a good houseguest:

* Pure Python, no compilation necessary.
* Exhaustive tests (>98% statement coverage at time of writing) and
  documentation (you're looking at it).
* No dependencies besides numpy.
* Tested and supported on every version of Python since 2.5. (And 2.4
  probably still works too if you really want it, it's just become too
  hard to keep a working 2.4 environment on the test server.)

So you can be pretty confident that adding a dependency on Patsy
won't create much hassle for your users.

And, of course, the fundamental design is very conservative -- the
formula mini-language in S was first described in Chambers and Hastie
(1992), more than two decades ago. It's still in heavy use today in R,
which is one of the most popular environments for statistical
programming. Many of your users may already be familiar with it. So we
can be pretty certain that it will hold up to real-world usage.

Using the high-level interface
------------------------------

If you have a function whose signature currently looks like this::

  def mymodel2(X, y, ...):
      ...

or this::

  def mymodel1(X, ...):
      ...

then adding Patsy support is extremely easy (though of course like
any other API change, you may have to deprecate the old interface, or
provide two interfaces in parallel, depending on your situation). Just
write something like::

  def mymodel2_patsy(formula_like, data={}, ...):
      y, X = patsy.dmatrices(formula_like, data, 1)
      ...

or::

  def mymodel1_patsy(formula_like, data={}, ...):
      X = patsy.dmatrix(formula_like, data, 1)
      ...

(See :func:`dmatrices` and :func:`dmatrix` for details.) This won't
force your users to switch to formulas immediately; they can replace
code that looks like this::

  X, y = build_matrices_laboriously()
  result = mymodel2(X, y, ...)
  other_result = mymodel1(X, ...)

with code like this::

  X, y = build_matrices_laboriously()
  result = mymodel2((y, X), data=None, ...)
  other_result = mymodel1(X, data=None, ...)

Of course in the long run they might want to throw away that
:func:`build_matrices_laboriously` function and start using formulas,
but they aren't forced to just to start using your new interface.

Working with metadata
^^^^^^^^^^^^^^^^^^^^^

Once you've started using Patsy to handle formulas, you'll probably
want to take advantage of the metadata that Patsy provides, so that
you can display regression coefficients by name and so forth. Design
matrices processed by Patsy always have a ``.design_info``
attribute which contains lots of information about the design: see
:class:`DesignInfo` for details.

Predictions
^^^^^^^^^^^

Another nice feature is making predictions on new data. But this
requires that we can take in new data, and transform it to create a
new `X` matrix. Or if we want to compute the likelihood of our model
on new data, we need both new `X` and `y` matrices.

This is also easily done with Patsy -- first fetch the relevant
:class:`DesignInfo` objects by doing ``input_data.design_info``, and
then pass them to :func:`build_design_matrices` along with the new
data.

Example
^^^^^^^

Here's a simplified class for doing ordinary least-squares regression,
demonstrating the above techniques:

.. warning:: This code has not been validated for numerical
   correctness.

.. literalinclude:: _examples/example_lm.py

And here's how it can be used:

.. ipython:: python
   :suppress:

   with open("_examples/example_lm.py") as f:
       exec(f.read())

.. ipython:: python
   :okwarning:

   from patsy import demo_data
   data = demo_data("x", "y", "a")

   # Old and boring approach (but it still works):
   X = np.column_stack(([1] * len(data["y"]), data["x"]))
   LM((data["y"], X))
   
   # Fancy new way:
   m = LM("y ~ x", data)   
   m
   m.predict({"x": [10, 20, 30]})
   m.loglik(data)
   m.loglik({"x": [10, 20, 30], "y": [-1, -2, -3]})

   # Your users get support for categorical predictors for free:
   LM("y ~ a", data)

   # And variable transformations too:
   LM("y ~ np.log(x ** 2)", data)

Other cool tricks
^^^^^^^^^^^^^^^^^

If you want to compute ANOVAs, then check out
:attr:`DesignInfo.term_name_slices`, :meth:`DesignInfo.slice`.

If you support linear hypothesis tests or otherwise allow your users
to specify linear constraints on model parameters, consider taking
advantage of :meth:`DesignInfo.linear_constraint`.

Extending the formula syntax
----------------------------

The above documentation assumes that you have a relatively simple
model that can be described by one or two matrices (plus whatever
other arguments you take). This covers many of the most popular
models, but it's definitely not sufficient for every model out there.

Internally, Patsy is designed to be very flexible -- for example,
it's quite straightforward to add custom operators to the formula
parser, or otherwise extend the formula evaluation machinery. (Heck,
it only took an hour or two to repurpose it for a totally different
purpose, parsing linear constraints.)  But extending Patsy in a
more fundamental way then this will require just a wee bit more complicated
API than just calling :func:`dmatrices`, and for this initial release,
we've been busy enough getting the basics working that we haven't yet
taken the time to pin down a public extension API we can support.

So, if you want something fancier -- please give us a nudge, it's
entirely likely we can work something out.
