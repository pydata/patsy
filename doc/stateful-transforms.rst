.. _stateful-transforms:

Stateful transforms
===================

.. currentmodule:: patsy

There's a subtle problem that sometimes bites people when working with
formulas. Suppose that I have some numerical data called ``x``, and I
would like to center it before fitting. The obvious way would be to
write::

   y ~ I(x - np.mean(x))  # BROKEN! Don't do this!

or, even better we could package it up into a function:

.. ipython:: python

   def naive_center(x):  # BROKEN! don't use!
       x = np.asarray(x)
       return x - np.mean(x)

and then write our formula like::

   y ~ naive_center(x)

Why is this a bad idea? Let's set up an example.

.. ipython:: python

   import numpy as np
   from patsy import dmatrix, build_design_matrices, incr_dbuilder
   data = {"x": [1, 2, 3, 4]}

Now we can build a design matrix and see what we get:

.. ipython:: python

   mat = dmatrix("naive_center(x)", data)
   mat

Those numbers look correct, and in fact they are correct. If all we're
going to do with this model is call :func:`dmatrix` once, then
everything is fine -- which is what makes this problem so insidious.

Often we want to do more with a model than this. For instance, we
might find some new data, and want to feed it into our model to make
predictions. To do this, though, we first need to reapply the same
transformation, like so:

.. ipython:: python

   new_data = {"x": [5, 6, 7, 8]}
   # Broken!
   build_design_matrices([mat.design_info], new_data)[0]

So it's clear what's happened here -- Patsy has centered the new
data, just like it centered the old data. But if you think about what
this means statistically, it makes no sense. According to this, the
new data point where x is 5 will behave exactly like the old data
point where x is 1, because they both produce the same input to the
actual model.

The problem is what it means to apply "the same transformation". Here,
what we really want to do is to subtract the mean *of the original
data* from the new data.

Patsy's solution is called a *stateful transform*. These look like
ordinary functions, but they perform a bit of magic to remember the
state of the original data, and use it in transforming new data.
Several useful stateful transforms are included out of the box,
including one called :func:`center`.

Using :func:`center` instead of :func:`naive_center` produces the same
correct result for our original matrix. It's used in exactly the same
way:

.. ipython:: python

   fixed_mat = dmatrix("center(x)", data)
   fixed_mat

But if we then feed in our new data, we also get out the correct result:

.. ipython:: python

   # Correct!
   build_design_matrices([fixed_mat.design_info], new_data)[0]

Another situation where we need some stateful transform magic is when
we are working with data that is too large to fit into memory at
once. To handle such cases, Patsy allows you to set up a design
matrix while working our way incrementally through the data. But if we
use :func:`naive_center` when building a matrix incrementally, then it
centers each *chunk* of data, not the data as a whole. (Of course,
depending on how your data is distributed, this might end up being
just similar enough for you to miss the problem until it's too late.)

.. ipython:: python

   data_chunked = [{"x": data["x"][:2]},
                   {"x": data["x"][2:]}]
   dinfo = incr_dbuilder("naive_center(x)", lambda: iter(data_chunked))
   # Broken!
   np.row_stack([build_design_matrices([dinfo], chunk)[0]
                 for chunk in data_chunked])

But if we use the proper stateful transform, this just works:

.. ipython:: python

   dinfo = incr_dbuilder("center(x)", lambda: iter(data_chunked))
   # Correct!
   np.row_stack([build_design_matrices([dinfo], chunk)[0]
                 for chunk in data_chunked])

.. note::

   Under the hood, the way this works is that :func:`incr_dbuilder`
   iterates through the data once to calculate the mean, and then we
   use :func:`build_design_matrices` to iterate through it a second
   time creating our design matrix. While taking two passes through a
   large data set may be slow, there's really no other way to
   accomplish what the user asked for. The good news is that Patsy is
   smart enough to make only the minimum number of passes
   necessary. For example, in our example with :func:`naive_center`
   above, :func:`incr_dbuilder` would not have done a full pass
   through the data at all. And if you have multiple stateful
   transforms in the same formula, then Patsy will process them in
   parallel in a single pass.

And, of course, we can use the resulting :class:`DesignInfo` object
for prediction as well:

.. ipython:: python

   # Correct!
   build_design_matrices([dinfo], new_data)[0]

In fact, Patsy's stateful transform handling is clever enough that
it can support arbitrary mixing of stateful transforms with other
Python code. E.g., if :func:`center` and :func:`spline` were both
stateful transforms, then even a silly a formula like this will be
handled 100% correctly::

  y ~ I(spline(center(x1)) + center(x2))

However, it isn't perfect -- there are two things you have to be
careful of. Let's put them in red:

.. warning::

   If you are unwise enough to ignore this section, write a function
   like ``naive_center`` above, and use it in a formula, then Patsy will
   not notice. If you use that formula with :func:`incr_dbuilders` or
   for predictions, then you will just silently get the wrong
   results. We have a plan to detect such cases, but it isn't
   implemented yet (and in any case can never be 100% reliable). So be
   careful!

.. warning::

   Even if you do use a "real" stateful transform like :func:`center`
   or :func:`standardize`, still have to make sure that Patsy can
   "see" that you are using such a transform. Currently the rule is
   that you must access the stateful transform function using a
   simple, bare variable reference, without any dots or other
   lookups::

     dmatrix("y ~ center(x)", data)  # okay
     asdf = patsy.center
     dmatrix("y ~ asdf(x)", data)  # okay
     dmatrix("y ~ patsy.center(x)", data)  # BROKEN! DON'T DO THIS!
     funcs = {"center": patsy.center}
     dmatrix("y ~ funcs['center'](x)", data)  # BROKEN! DON'T DO THIS!

Builtin stateful transforms
---------------------------

There are a number of builtin stateful transforms beyond
:func:`center`; see :ref:`stateful transforms
<stateful-transforms-list>` in the API reference for a complete list.

.. _stateful-transform-protocol:

Defining a stateful transform
-----------------------------

You can also easily define your own stateful transforms. The first
step is to define a class which fulfills the stateful transform
protocol. The lifecycle of a stateful transform object is as follows:

#. An instance of your type will be constructed.
#. :meth:`memorize_chunk` will be called one or more times.
#. :meth:`memorize_finish` will be called once.
#. :meth:`transform` will be called one or more times, on either the
   same or different data to what was initially passed to
   :meth:`memorize_chunk`. You can trust that any non-data arguments
   will be identical between calls to :meth:`memorize_chunk` and
   :meth:`transform`.

And here are the methods and call signatures you need to define:

.. class:: stateful_transform_protocol

  .. method:: __init__()
     :noindex:

     It must be possible to create an instance of the class by calling
     the constructor with no arguments.

  .. method:: memorize_chunk(*args, **kwargs)

     Update any internal state, based on the data passed into
     `memorize_chunk`.

  .. method:: memorize_finish()

     Do any housekeeping you want to do between the last call to
     :meth:`memorize_chunk` and the first call to
     :meth:`transform`. For example, if you are computing some summary
     statistic that cannot be done incrementally, then your
     :meth:`memorize_chunk` method might just store the data that's
     passed in, and then :meth:`memorize_finish` could compute the
     summary statistic and delete the stored data to free up the
     associated memory.

  .. method:: transform(*args, **kwargs)

     This method should transform the input data passed to it. It
     should be deterministic, and it should be "point-wise", in the
     sense that when passed an array it performs an independent
     transformation on each data point that is not affected by any
     other data points passed to :meth:`transform`.

Then once you have created your class, pass it to
:func:`stateful_transform` to create a callable stateful transform
object suitable for use inside or outside formulas.

Here's a simple example of how you might implement a working version
of :func:`center` (though it's less robust and featureful than the
real builtin)::

  class MyExampleCenter(object):
      def __init__(self):
          self._total = 0
          self._count = 0
          self._mean = None

      def memorize_chunk(self, x):
          self._total += np.sum(x)
          self._count += len(x)

      def memorize_finish(self):
          self._mean = self.total * 1. / self._count

      def transform(self, x):
          return x - self._mean

  my_example_center = patsy.stateful_transform(MyExampleCenter)
  print(my_example_center(np.array([1, 2, 3])))

But of course, if you come up with any useful ones, please let us know
so we can incorporate them into patsy itself!
