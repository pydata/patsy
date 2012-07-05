Quickstart
==========

If you prefer to learn by diving in and getting your feet wet, then
here are some cut-and-pasteable examples to play with.

First, let's import stuff and get some data to work with:

.. ipython:: python

   import numpy as np
   import charlton
   from charlton import *
   data = charlton.demo_data("a", "b", "x1", "x2", "y")

:func:`demo_data` gives us a mix of categorical and numerical
variables:

.. ipython:: python

   data

Of course Charlton doesn't much care what sort of object you store
your data in, so long as it can be indexed like a Python dictionary,
``data[varname]``. You may prefer to store your data in a `pandas
<http://pandas.pydata.org>`_ DataFrame, or a numpy record
array... whatever makes you happy.

Now, let's generate design matrices suitable for regressing ``y`` onto
``x1`` and ``x2``.

.. ipython:: python

   dmatrices("y ~ x1 + x2", data)

Notice that an intercept term was automatically added. These are just
ordinary numpy arrays with some extra metadata and a fancy __repr__
method attached, so we can pass them directly to a regression function
like :func:`np.linalg.lstsq`:

.. ipython:: python

   outcome, predictors = dmatrices("y ~ x1 + x2", data)
   betas = np.linalg.lstsq(predictors, outcome)[0].ravel()
   for name, beta in zip(predictors.column_info.column_names, betas):
       print("%s: %s" % (name, beta))

Of course the results aren't very interesting, since this is just
random data.

If you just want the design matrix alone, without the ``y`` values,
use :func:`dmatrix` and leave off the ``y ~`` part at the beginning:

.. ipython:: python

   dmatrix("x1 + x2", data)

We'll do this for the rest of the examples, since seeing the outcome
matrix over and over would get boring.

Usually the intercept is useful, but if we don't want it we can get
rid of it:

.. ipython:: python

   dmatrix("x1 + x2 - 1", data)

We can transform variables using arbitrary Python code:

.. ipython:: python

   dmatrix("x1 + np.log(x2 + 10)", data)

Notice that `np.log` is being pulled out of the environment where
:func:`dmatrix` was called -- if we hadn't done ``import numpy as np``
up above then this wouldn't have worked.

Any variables you've defined are also accessible (and the ``data``
argument is optional):

.. ipython:: python

   new_x2 = data["x2"] * 100
   dmatrix("new_x2")

Charlton has some transformation functions "built in", that are
automatically accessible to your code:

.. ipython:: python

   dmatrix("center(x1) + standardize(x2)", data)

You can see the whole list XX

Arithmetic transformations are also possible, but you'll need to
"protect" them by wrapping them in ``I()``, so that Charlton knows
that you really do want ``+`` to mean addition:

.. ipython:: python

   dmatrix("I(x1 + x2)", data)  # compare to "x1 + x2"

Charlton becomes particularly useful when you have categorical
data. If you use a predictor that has a categorical type (e.g. strings
or bools), it will be automatically coded. Charlton automatically
chooses an appropriate way to code categorical data to avoid
producing a redundant, overdetermined model.

If there is just one categorical variable alone, the default is to
dummy code it:

.. ipython:: python

   dmatrix("0 + a", data)

But if you did that and put the intercept back in, you'd get a
redundant model. So if the intercept is present, Charlton uses
a reduced-rank contrast code (treatment coding by default):

.. ipython:: python

   dmatrix("a", data)

Interactions are also easy -- they represent the cartesian product of
all the factors involved. Here's a dummy coding of each *combination*
of values taken by ``a`` and ``b``:

.. ipython:: python

   dmatrix("0 + a:b", data)

But interactions also know how to use contrast coding to avoid
redundancy. If you have both main effects and interactions in a model,
then Charlton goes from lower-order effects to higher-order effects,
adding in just enough columns to produce a well-defined model. The
result is that each set of columns measures the *additional*
contribution of this effect -- just what you want for a traditional
ANOVA:

.. ipython:: python

   dmatrix("a + b + a:b", data)

Since this is so common, there's a convenient short-hand:

.. ipython:: python

   dmatrix("a*b", data)

Of course you can use other coding schemes too (or even define your
own). Here's orthogonal polynomial coding:

.. ipython:: python

   dmatrix("C(c, Poly)", {"c": ["c1", "c1", "c2", "c2", "c3", "c3"]})

You can even write interactions between categorical and numerical
variables. Here we fit two different slope coefficients for ``x1``;
one for the ``a1`` group, and one for the ``a2`` group:

.. ipython:: python

   dmatrix("a:x1", data)

The same redundancy avoidance code works here, so if you'd rather have
treatment-coded slopes (one slope for ``a1``, and a second for the
difference between the ``a1`` and ``a2`` group slopes), then you can
request it like this:

.. ipython:: python

   # compare to the difference between "0 + a" and "1 + a"
   dmatrix("x1 + a:x1", data)

And more complex expressions work too:

.. ipython:: python

   dmatrix("C(a, Poly):center(x1)", data)
