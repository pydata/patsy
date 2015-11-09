.. _categorical-coding:

Coding categorical data
=======================

.. currentmodule:: patsy

Patsy allows great flexibility in how categorical data is coded,
via the function :func:`C`. :func:`C` marks some data as being
categorical (including data which would not automatically be treated
as categorical, such as a column of integers), while also optionally
setting the preferred coding scheme and level ordering.

Let's get some categorical data to work with:

.. ipython:: python

   from patsy import dmatrix, demo_data, ContrastMatrix, Poly
   data = demo_data("a", nlevels=3)
   data

As you know, simply giving Patsy a categorical variable causes it
to be coded using the default :class:`Treatment` coding
scheme. (Strings and booleans are treated as categorical by default.)

.. ipython:: python

   dmatrix("a", data)

We can also alter the level ordering, which is useful for, e.g.,
:class:`Diff` coding:

.. ipython:: python

   l = ["a3", "a2", "a1"]
   dmatrix("C(a, levels=l)", data)

But the default coding is just that -- a default. The easiest
alternative is to use one of the other built-in coding schemes, like
orthogonal polynomial coding:

.. ipython:: python

   dmatrix("C(a, Poly)", data)

There are a number of built-in coding schemes; for details you can
check the :ref:`API reference <categorical-coding-ref>`. But we aren't
restricted to those. We can also provide a custom contrast matrix,
which allows us to produce all kinds of strange designs:

.. ipython:: python

   contrast = [[1, 2], [3, 4], [5, 6]]
   dmatrix("C(a, contrast)", data)
   dmatrix("C(a, [[1], [2], [-4]])", data)

Hmm, those ``[custom0]``, ``[custom1]`` names that Patsy
auto-generated for us are a bit ugly looking. We can attach names to
our contrast matrix by creating a :class:`ContrastMatrix` object, and
make things prettier:

.. ipython:: python

   contrast_mat = ContrastMatrix(contrast, ["[pretty0]", "[pretty1]"])
   dmatrix("C(a, contrast_mat)", data)

And, finally, if we want to get really fancy, we can also define our
own "smart" coding schemes like :class:`Poly`. Just define a class
that has two methods, :meth:`code_with_intercept` and
:meth:`code_without_intercept`. They have identical signatures, taking
a list of levels as their argument and returning a
:class:`ContrastMatrix`. Patsy will automatically choose the
appropriate method to call to produce a full-rank design matrix
without redundancy; see :ref:`redundancy` for the full details on how
Patsy makes this decision.

As an example, here's a simplified version of the built-in
:class:`Treatment` coding object:

.. literalinclude:: _examples/example_treatment.py
                                 
.. ipython:: python
   :suppress:

   with open("_examples/example_treatment.py") as f:
       exec(f.read())

And it can now be used just like the built-in methods:

.. ipython:: python

   # Full rank:
   dmatrix("0 + C(a, MyTreat)", data)
   # Reduced rank:
   dmatrix("C(a, MyTreat)", data)
   # With argument:
   dmatrix("C(a, MyTreat(2))", data)
