Quickstart
==========

.. currentmodule:: patsy

If you prefer to learn by diving in and getting your feet wet, then
here are some cut-and-pasteable examples to play with.

First, let's import stuff and get some data to work with:

.. ipython:: python

   import numpy as np
   from patsy import dmatrices, dmatrix, demo_data
   data = demo_data("a", "b", "x1", "x2", "y", "z column")

:func:`demo_data` gives us a mix of categorical and numerical
variables:

.. ipython:: python

   data

Of course Patsy doesn't much care what sort of object you store
your data in, so long as it can be indexed like a Python dictionary,
``data[varname]``. You may prefer to store your data in a `pandas
<http://pandas.pydata.org>`_ DataFrame, or a numpy
record array... whatever makes you happy.

Now, let's generate design matrices suitable for regressing ``y`` onto
``x1`` and ``x2``.

.. ipython:: python

   dmatrices("y ~ x1 + x2", data)

The return value is a Python tuple containing two DesignMatrix
objects, the first representing the left-hand side of our formula, and
the second representing the right-hand side. Notice that an intercept
term was automatically added to the right-hand side. These are just
ordinary numpy arrays with some extra metadata and a fancy __repr__
method attached, so we can pass them directly to a regression function
like :func:`np.linalg.lstsq`:

.. ipython:: python
   :okwarning:

   outcome, predictors = dmatrices("y ~ x1 + x2", data)
   betas = np.linalg.lstsq(predictors, outcome)[0].ravel()
   for name, beta in zip(predictors.design_info.column_names, betas):
       print("%s: %s" % (name, beta))

Of course the resulting numbers aren't very interesting, since this is just
random data.

If you just want the design matrix alone, without the ``y`` values,
use :func:`dmatrix` and leave off the ``y ~`` part at the beginning:

.. ipython:: python

   dmatrix("x1 + x2", data)

We'll use dmatrix for the rest of the examples, since seeing the
outcome matrix over and over would get boring. This matrix's metadata
is stored in an extra attribute called ``.design_info``, which is a
:class:`DesignInfo` object you can explore at your leisure:

.. ipython::

   In [0]: d = dmatrix("x1 + x2", data)

   @verbatim
   In [0]: d.design_info.<TAB>
   d.design_info.builder              d.design_info.slice
   d.design_info.column_name_indexes  d.design_info.term_name_slices
   d.design_info.column_names         d.design_info.term_names
   d.design_info.describe             d.design_info.term_slices
   d.design_info.linear_constraint    d.design_info.terms

Usually the intercept is useful, but if we don't want it we can get
rid of it:

.. ipython:: python

   dmatrix("x1 + x2 - 1", data)

We can transform variables using arbitrary Python code:

.. ipython:: python

   dmatrix("x1 + np.log(x2 + 10)", data)

Notice that ``np.log`` is being pulled out of the environment where
:func:`dmatrix` was called -- ``np.log`` is accessible because we did
``import numpy as np`` up above. Any functions or variables that you
could reference when calling :func:`dmatrix` can also be used inside
the formula passed to :func:`dmatrix`. For example:

.. ipython:: python

   new_x2 = data["x2"] * 100
   dmatrix("new_x2")

Patsy has some transformation functions "built in", that are
automatically accessible to your code:

.. ipython:: python

   dmatrix("center(x1) + standardize(x2)", data)

See :mod:`patsy.builtins` for a complete list of functions made
available to formulas. You can also define your own transformation
functions in the ordinary Python way:

.. ipython:: python

   def double(x):
       return 2 * x

   dmatrix("x1 + double(x1)", data)

.. currentmodule:: patsy.builtins

This flexibility does create problems in one case, though -- because
we interpret whatever you write in-between the ``+`` signs as Python
code, you do in fact have to write valid Python code. And this can be
tricky if your variable names have funny characters in them, like
whitespace or punctuation. Fortunately, patsy has a builtin
"transformation" called :func:`Q` that lets you "quote" such
variables:

.. ipython::

   In [1]: weird_data = demo_data("weird column!", "x1")

   # This is an error...
   @verbatim
   In [2]: dmatrix("weird column! + x1", weird_data)
   [...]
   PatsyError: error tokenizing input (maybe an unclosed string?)
       weird column! + x1
                   ^

   # ...but this works:
   In [3]: dmatrix("Q('weird column!') + x1", weird_data)

:func:`Q` even plays well with other transformations:

.. ipython:: python

   dmatrix("double(Q('weird column!')) + x1", weird_data)

Arithmetic transformations are also possible, but you'll need to
"protect" them by wrapping them in :func:`I()`, so that Patsy knows
that you really do want ``+`` to mean addition:

.. ipython:: python

   dmatrix("I(x1 + x2)", data)  # compare to "x1 + x2"

.. currentmodule:: patsy

Note that while Patsy goes to considerable efforts to take in data
represented using different Python data types and convert them into a
standard representation, all this work happens *after* any
transformations you perform as part of your formula. So, for example,
if your data is in the form of numpy arrays, "+" will perform
element-wise addition, but if it is in standard Python lists, it will
perform concatenation:

.. ipython:: python

   dmatrix("I(x1 + x2)", {"x1": np.array([1, 2, 3]), "x2": np.array([4, 5, 6])})
   dmatrix("I(x1 + x2)", {"x1": [1, 2, 3], "x2": [4, 5, 6]})

Patsy becomes particularly useful when you have categorical
data. If you use a predictor that has a categorical type (e.g. strings
or bools), it will be automatically coded. Patsy automatically
chooses an appropriate way to code categorical data to avoid
producing a redundant, overdetermined model.

If there is just one categorical variable alone, the default is to
dummy code it:

.. ipython:: python

   dmatrix("0 + a", data)

But if you did that and put the intercept back in, you'd get a
redundant model. So if the intercept is present, Patsy uses
a reduced-rank contrast code (treatment coding by default):

.. ipython:: python

   dmatrix("a", data)

The ``T.`` notation is there to remind you that these columns are
treatment coded.

Interactions are also easy -- they represent the cartesian product of
all the factors involved. Here's a dummy coding of each *combination*
of values taken by ``a`` and ``b``:

.. ipython:: python

   dmatrix("0 + a:b", data)

But interactions also know how to use contrast coding to avoid
redundancy. If you have both main effects and interactions in a model,
then Patsy goes from lower-order effects to higher-order effects,
adding in just enough columns to produce a well-defined model. The
result is that each set of columns measures the *additional*
contribution of this effect -- just what you want for a traditional
ANOVA:

.. ipython:: python

   dmatrix("a + b + a:b", data)

Since this is so common, there's a convenient short-hand:

.. ipython:: python

   dmatrix("a*b", data)

Of course you can use :ref:`other coding schemes
<categorical-coding-ref>` too (or even :ref:`define your own
<categorical-coding>`). Here's :class:`orthogonal polynomial coding
<Poly>`:

.. ipython:: python

   dmatrix("C(c, Poly)", {"c": ["c1", "c1", "c2", "c2", "c3", "c3"]})

You can even write interactions between categorical and numerical
variables. Here we fit two different slope coefficients for ``x1``;
one for the ``a1`` group, and one for the ``a2`` group:

.. ipython:: python

   dmatrix("a:x1", data)

The same redundancy avoidance code works here, so if you'd rather have
treatment-coded slopes (one slope for the ``a1`` group, and a second
for the difference between the ``a1`` and ``a2`` group slopes), then
you can request it like this:

.. ipython:: python

   # compare to the difference between "0 + a" and "1 + a"
   dmatrix("x1 + a:x1", data)

And more complex expressions work too:

.. ipython:: python

   dmatrix("C(a, Poly):center(x1)", data)
