Documentation for users
=======================

Here, we describe how to specify a model, and build a model matrix.

Getting started
---------------

If you just want to experiment, try this::

  from charlton import model_matrices
  x = [1, 2, 3, 4, 5]
  y = [4.9, 6.7, 9.0, 10.9, 13.1]
  response, predictors = model_matrices("y ~ x")

Here, :func:`model_matrices` is finding the variables by looking in
the environment where it was called. If you have some sort of
dict-like object that contains your data, you can pass it as a second
argument::

   response, predictors = model_matrices("y ~ x", {"x": x, "y": y})

Of course, any object which supports name-based indexing like
`data["x"]` can be used as the second argument, including numpy
recarrays and pandas DataFrames.

If you prefer to use Charlton only to set up the predictors matrix and
will set up the response vector yourself, you can also use::

  from charlton import model_matrix
  x = [1, 2, 3, 4, 5]
  model_matrix("x")
  model_matrix("x", {"x": x})

Some other things to try::

  a = ["red", "red", "green", "green"]
  b = ["high", "low", "high", "low"]
  model_matrix("x + a")       # Categorical variable
  model_matrix("a:b")         # Interaction of categorical variables
  model_matrix("a + b + a:b") # Separating main effects and interactions
  import numpy as np
  model_matrix("np.log(x)")   # Ad-hoc data transformations
  model_matrix("center(x)")   # Built-in convenience functions

Terminology
-----------



Writing formulas
----------------

The idea in writing a formula is to specify two lists of terms -- one
for the left-hand side (the "response") and one for the right (the
"predictors"). These are separated by a `~`. If for some reason you
don't want to specify any terms on the left-hand side, you can simply
leave it, or omit the `~` altogether. For instance, the following are
strings are legal formulas::

  "y ~ x"
  "~ x"
  "x"

Then on each side of the `~`, there are a variety of operations you
can use to build up a list of terms. Of these, two are
fundamental. The first fundamental operation is `+`, which is to used
to separate distinct terms. For instance, this is how you describe a
regression of `y` on variables `x1`, `x2`, and `x3`::

  "y ~ x1 + x2 + x3"

The second fundamental operation is `:`, which is used to create
interaction terms.

Programmatic API
----------------

Sometimes it's useful to construct model descriptions using Python
code rather than by writing a string (e.g., if you want to use a loop
to add lots and lots of terms). Of course, one could use string
manipulation to construct a formula and then pass it to charlton to be
parsed, but this would be complicated and error prone.


The gory details
----------------

