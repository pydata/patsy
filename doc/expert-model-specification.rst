.. _expert-model-specification:

Model specification for experts and computers
=============================================

.. currentmodule:: patsy

While the formula language is great for interactive model-fitting and
exploratory data analysis, there are times when we want a different or
more systematic interface for creating design matrices. If you ever
find yourself writing code that pastes together bits of strings to
create a formula, then stop! And read this chapter.

Our first option, of course, is that we can go ahead and write some
code to construct our design matrices directly, just like we did in
the old days. Since this is supported directly by :func:`dmatrix` and
:func:`dmatrices`, it also works with any third-party library
functions that use Patsy internally. Just pass in an array_like or
a tuple ``(y_array_like, X_array_like)`` in place of the formula.

.. ipython:: python

   from patsy import dmatrix
   X = [[1, 10], [1, 20], [1, -2]]
   dmatrix(X)

By using a :class:`DesignMatrix` with :class:`DesignInfo` attached, we
can also specify custom names for our custom matrix (or even term
slices and so forth), so that we still get the nice output and such
that Patsy would otherwise provide:

.. ipython:: python

   from patsy import DesignMatrix, DesignInfo
   design_info = DesignInfo(["Intercept!", "Not intercept!"])
   X_dm = DesignMatrix(X, design_info)
   dmatrix(X_dm)

Or if all we want to do is to specify column names, we could also just
use a :class:`pandas.DataFrame`:

.. ipython:: python

   import pandas
   df = pandas.DataFrame([[1, 10], [1, 20], [1, -2]],
                         columns=["Intercept!", "Not intercept!"])
   dmatrix(df)

However, there is also a middle ground between pasting together
strings and going back to putting together design matrices out of
string and baling wire. Patsy has a straightforward Python
interface for representing the result of parsing formulas, and you can
use it directly. This lets you keep Patsy's normal advantages --
handling of categorical data and interactions, predictions, term
tracking, etc. -- while using a nice high-level Python API. An example
of somewhere this might be useful is if, say, you had a GUI with a
tick box next to each variable in your data set, and wanted to
construct a formula containing all the variables that had been
checked, and letting Patsy deal with categorical data handling. Or
this would be the approach you'd take for doing stepwise regression,
where you need to programatically add and remove terms.

Whatever your particular situation, the strategy is this:

#. Construct some factor objects (probably using :class:`LookupFactor` or
   :class:`EvalFactor`
#. Put them into some :class:`Term` objects,
#. Put the :class:`Term` objects into two lists, representing the
   left- and right-hand side of your formula,
#. And then wrap the whole thing up in a :class:`ModelDesc`.

(See :ref:`formulas` if you need a refresher on what each of these
things are.)

.. ipython:: python

   import numpy as np
   from patsy import (ModelDesc, EvalEnvironment, Term, EvalFactor,
                      LookupFactor, demo_data, dmatrix)
   data = demo_data("a", "x")

   # LookupFactor takes a dictionary key:
   a_lookup = LookupFactor("a")
   # EvalFactor takes arbitrary Python code:
   x_transform = EvalFactor("np.log(x ** 2)")
   # First argument is empty list for dmatrix; we would need to put
   # something there if we were calling dmatrices.
   desc = ModelDesc([],
                    [Term([a_lookup]),
                     Term([x_transform]),
                     # An interaction:
                     Term([a_lookup, x_transform])])
   # Create the matrix (or pass 'desc' to any statistical library
   # function that uses patsy.dmatrix internally):
   dmatrix(desc, data)

Notice that no intercept term is included. Implicit intercepts are a
feature of the formula parser, not the underlying representation. If you
want an intercept, include the constant :const:`INTERCEPT` in your
list of terms (which is just sugar for ``Term([])``).

.. note::

   Another option is to just pass your term lists directly to
   :func:`design_matrix_builders`, and skip the :class:`ModelDesc`
   entirely -- all of the highlevel API functions like :func:`dmatrix`
   accept :class:`DesignMatrixBuilder` objects as well as
   :class:`ModelDesc` objects.

Example: say our data has 100 different numerical columns that we want
to include in our design -- and we also have a few categorical
variables with a more complex interaction structure. Here's one
solution:

.. literalinclude:: _examples/add_predictors.py

.. ipython:: python
   :suppress:

   with open("_examples/add_predictors.py") as f:
       exec(f.read())

.. ipython:: python

   extra_predictors = ["x%s" % (i,) for i in range(10)]
   desc = add_predictors("np.log(y) ~ a*b + c:d", extra_predictors)
   desc.describe()

The factor protocol
-------------------

If :class:`LookupFactor` and :class:`EvalFactor` aren't enough for
you, then you can define your own factor class.

The full interface looks like this:

.. class:: factor_protocol

    .. method:: name()

       This must return a short string describing this factor. It will
       be used to create column names, among other things.

    .. attribute:: origin

       A :class:`patsy.Origin` if this factor has one; otherwise, just
       set it to None.

    .. method:: __eq__(obj)
                __ne__(obj)
                __hash__()

       If your factor will ever contain categorical data or
       participate in interactions, then it's important to make sure
       you've defined :meth:`~object.__eq__` and
       :meth:`~object.__ne__` and that your type is
       :term:`hashable`. These methods will determine which factors
       Patsy considers equal for purposes of redundancy elimination.

    .. method:: memorize_passes_needed(state, eval_env)

       Return the number of passes through the data that this factor
       will need in order to set up any :ref:`stateful-transforms`.

       If you don't want to support stateful transforms, just return
       0. In this case, :meth:`memorize_chunk` and
       :meth:`memorize_finish` will never be called.

       `state` is an (initially) empty dict which is maintained by the
       builder machinery, and that we can do whatever we like with. It
       will be passed back in to all memorization and evaluation
       methods.

       `eval_env` is an :class:`EvalEnvironment` object, describing
       the Python environment where the factor is being evaluated.

    .. method:: memorize_chunk(state, which_pass, data)

       Called repeatedly with each 'chunk' of data produced by the
       `data_iter_maker` passed to :func:`design_matrix_builders`.

       `state` is the state dictionary. `which_pass` will be zero on
       the first pass through the data, and eventually reach the
       value you returned from :meth:`memorize_passes_needed`, minus
       one.

       Return value is ignored.

    .. method:: memorize_finish(state, which_pass)

       Called once after each pass through the data.

       Return value is ignored.

    .. method:: eval(state, data)

       Evaluate this factor on the given `data`. Return value should
       ideally be a 1-d or 2-d array or :func:`Categorical` object,
       but this will be checked and converted as needed.

In addition, factor objects should be pickleable/unpickleable, so as
to allow models containing them to be pickled/unpickled. (Or, if for
some reason your factor objects are *not* safely pickleable, you
should consider giving them a `__getstate__` method which raises an
error, so that any users which attempt to pickle a model containing
your factors will get a clear failure immediately, instead of only
later when they try to unpickle.)

.. warning:: Do not store evaluation-related state in
   attributes of your factor object! The same factor object may
   appear in two totally different formulas, or if you have two
   factor objects which compare equally, then only one may be
   executed, and which one this is may vary randomly depending
   on how :func:`build_design_matrices` is called! Use only the
   `state` dictionary for storing state.

The lifecycle of a factor object therefore looks like:

#. Initialized.
#. :meth:`memorize_passes_needed` is called.
#. ``for i in range(passes_needed):``

   #. :meth:`memorize_chunk` is called one or more times
   #. :meth:`memorize_finish` is called

#. :meth:`eval` is called zero or more times.

Alternative formula implementations
-----------------------------------

Even if you hate Patsy's formulas all together, to the extent that
you're going to go and implement your own competing mechanism for
defining formulas, you can still Patsy-based
interfaces. Unfortunately, this isn't *quite* as clean as we'd like,
because for now there's no way to define a custom
:class:`DesignMatrixBuilder`. So you do still have to go through
Patsy's formula-building machinery. But, this machinery simply
passes numerical data through unchanged, so in extremis you can:

* Define a special factor object that simply defers to your existing
  machinery
* Define the magic ``__patsy_get_model_desc__`` method on your
  formula object. :func:`dmatrix` and friends check for the presence
  of this method on any object that is passed in, and if found, it is
  called (passing in the :class:`EvalEnvironment`), and expected to
  return a :class:`ModelDesc`. And your :class:`ModelDesc` can, of
  course, include your special factor object(s).

Put together, it looks something like this:

.. code-block:: python

  class MyAlternativeFactor(object):
      # A factor object that simply returns the design 
      def __init__(self, alternative_formula, side):
          self.alternative_formula = alternative_formula
          self.side = side

      def name(self):
          return self.side

      def memorize_passes_needed(self, state):
          return 0

      def eval(self, state, data):
          return self.alternative_formula.get_matrix(self.side, data)

  class MyAlternativeFormula(object):
      ...

      def __patsy_get_model_desc__(self, eval_env):
          return ModelDesc([Term([MyAlternativeFactor(self, side="left")])],
                           [Term([MyAlternativeFactor(self, side="right")])],


  my_formula = MyAlternativeFormula(...)
  dmatrix(my_formula, data)

The only downside to this approach is that you can't control the names
of individual columns. (A workaround would be to create multiple terms
each with its own factor that returns a different pieces of your
overall matrix.) If this is a problem for you, though, then let's talk
-- we can probably work something out.
