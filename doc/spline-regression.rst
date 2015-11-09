.. _spline-regression:

Spline regression
=================

.. currentmodule:: patsy

.. ipython:: python
   :suppress:

   import numpy as np
   from patsy import dmatrix, build_design_matrices

Patsy offers a set of specific stateful transforms (for more details about
stateful transforms see :ref:`stateful-transforms`) that you can use in
formulas to generate splines bases and express non-linear fits.

General B-splines
-----------------

B-spline bases can be generated with the :func:`bs` stateful
transform. The spline bases returned by :func:`bs` are designed to be
compatible with those produced by the R ``bs`` function.
The following code illustrates a typical basis and the resulting spline:

.. ipython:: python

   import matplotlib.pyplot as plt
   plt.title("B-spline basis example (degree=3)");
   x = np.linspace(0., 1., 100)
   y = dmatrix("bs(x, df=6, degree=3, include_intercept=True) - 1", {"x": x})
   # Define some coefficients
   b = np.array([1.3, 0.6, 0.9, 0.4, 1.6, 0.7])
   # Plot B-spline basis functions (colored curves) each multiplied by its coeff
   plt.plot(x, y*b);
   @savefig basis-bspline.png align=center
   # Plot the spline itself (sum of the basis functions, thick black curve)
   plt.plot(x, np.dot(y, b), color='k', linewidth=3);

In the following example we first set up our B-spline basis using some data and
then make predictions on a new set of data:

.. ipython:: python

   data = {"x": np.linspace(0., 1., 100)}
   design_matrix = dmatrix("bs(x, df=4)", data)

   new_data = {"x": [0.1, 0.25, 0.9]}
   build_design_matrices([design_matrix.design_info], new_data)[0]


:func:`bs` can produce B-spline bases of arbitrary degrees -- e.g.,
``degree=0`` will give produce piecewise-constant functions,
``degree=1`` will produce piecewise-linear functions, and the default
``degree=3`` produces cubic splines. The next section describes more
specialized functions for producing different types of cubic splines.


Natural and cyclic cubic regression splines
-------------------------------------------

Natural and cyclic cubic regression splines are provided through the stateful
transforms :func:`cr` and :func:`cc` respectively. Here the spline is
parameterized directly using its values at the knots. These splines were designed
to be compatible with those found in the R package
`mgcv <http://cran.r-project.org/web/packages/mgcv/index.html>`_
(these are called *cr*, *cs* and *cc* in the context of *mgcv*), but
can be used with any model.

.. warning::
   Note that the compatibility with *mgcv* applies only to the **generation of
   spline bases**: we do not implement any kind of *mgcv*-compatible penalized
   fitting process.
   Thus these spline bases can be used to precisely reproduce
   predictions from a model previously fitted with *mgcv*, or to serve as
   building blocks for other regression models (like OLS).

Here are some illustrations of typical natural and cyclic spline bases:

.. ipython:: python

   plt.title("Natural cubic regression spline basis example");
   y = dmatrix("cr(x, df=6) - 1", {"x": x})
   # Plot natural cubic regression spline basis functions (colored curves) each multiplied by its coeff
   plt.plot(x, y*b);
   @savefig basis-crspline.png align=center
   # Plot the spline itself (sum of the basis functions, thick black curve)
   plt.plot(x, np.dot(y, b), color='k', linewidth=3);


.. ipython:: python

   plt.title("Cyclic cubic regression spline basis example");
   y = dmatrix("cc(x, df=6) - 1", {"x": x})
   # Plot cyclic cubic regression spline basis functions (colored curves) each multiplied by its coeff
   plt.plot(x, y*b);
   @savefig basis-ccspline.png align=center
   # Plot the spline itself (sum of the basis functions, thick black curve)
   plt.plot(x, np.dot(y, b), color='k', linewidth=3);


In the following example we first set up our spline basis using same data as for
the B-spline example above and then make predictions on a new set of data:

.. ipython:: python

   design_matrix = dmatrix("cr(x, df=4, constraints='center')", data)
   new_design_matrix = build_design_matrices([design_matrix.design_info], new_data)[0]
   new_design_matrix
   np.asarray(new_design_matrix)

Note that in the above example 5 knots are actually used to achieve 4 degrees
of freedom since a centering constraint is requested.

Note that the API is different from *mgcv*:

* In patsy one can specify the number of degrees of freedom directly (actual number of
  columns of the resulting design matrix) whereas in *mgcv* one has to specify
  the number of knots to use. For instance, in the case of cyclic regression splines (with no
  additional constraints) the actual degrees of freedom is the number of knots
  minus one.
* In patsy one can specify inner knots as well as lower and upper exterior knots
  which can be useful for cyclic spline for instance.
* In *mgcv* a centering/identifiability constraint is automatically computed and
  absorbed in the resulting design matrix.
  The purpose of this is to ensure that if ``b`` is the array of *initial* parameters
  (corresponding to the *initial* unconstrained design matrix ``dm``), our
  model is centered, ie ``np.mean(np.dot(dm, b))`` is zero.
  We can rewrite this as ``np.dot(c, b)`` being zero with ``c`` a 1-row
  constraint matrix containing the mean of each column of ``dm``.
  Absorbing this constraint in the *final* design matrix means that we rewrite the model
  in terms of *unconstrained* parameters (this is done through a QR-decomposition
  of the constraint matrix). Those unconstrained parameters have the property
  that when projected back into the initial parameters space (let's call ``b_back``
  the result of this projection), the constraint
  ``np.dot(c, b_back)`` being zero is automatically verified.
  In patsy one can choose between no
  constraint, a centering constraint like *mgcv* (``'center'``) or a user provided
  constraint matrix.


Tensor product smooths
----------------------

Smooths of several covariates can be generated through a tensor product of
the bases of marginal univariate smooths. For these marginal smooths one can
use the above defined splines as well as user defined smooths provided they
actually transform input univariate data into some kind of smooth functions
basis producing a 2-d array output with the ``(i, j)`` element corresponding
to the value of the ``j`` th basis function at the ``i`` th data point.
The tensor product stateful transform is called :func:`te`.

.. note::
   The implementation of this tensor product is compatible with *mgcv* when
   considering only cubic regression spline marginal smooths, which means that
   generated bases will match those produced by *mgcv*.
   Recall that we do not implement any kind of *mgcv*-compatible penalized
   fitting process.

In the following code we show an example of tensor product basis functions
used to represent a smooth of two variables ``x1`` and ``x2``. Note how
marginal spline bases patterns can be observed on the x and y contour projections:

.. ipython::

   In [10]: from matplotlib import cm

   In [20]: from mpl_toolkits.mplot3d.axes3d import Axes3D

   In [30]: x1 = np.linspace(0., 1., 100)

   In [40]: x2 = np.linspace(0., 1., 100)

   In [50]: x1, x2 = np.meshgrid(x1, x2)

   In [60]: df = 3

   In [70]: y = dmatrix("te(cr(x1, df), cc(x2, df)) - 1",
      ....:            {"x1": x1.ravel(), "x2": x2.ravel(), "df": df})
      ....:

   In [80]: print y.shape

   In [90]: fig = plt.figure()

   In [100]: fig.suptitle("Tensor product basis example (2 covariates)");

   In [110]: for i in range(df * df):
      .....:     ax = fig.add_subplot(df, df, i + 1, projection='3d')
      .....:     yi = y[:, i].reshape(x1.shape)
      .....:     ax.plot_surface(x1, x2, yi, rstride=4, cstride=4, alpha=0.15)
      .....:     ax.contour(x1, x2, yi, zdir='z', cmap=cm.coolwarm, offset=-0.5)
      .....:     ax.contour(x1, x2, yi, zdir='y', cmap=cm.coolwarm, offset=1.2)
      .....:     ax.contour(x1, x2, yi, zdir='x', cmap=cm.coolwarm, offset=-0.2)
      .....:     ax.set_xlim3d(-0.2, 1.0)
      .....:     ax.set_ylim3d(0, 1.2)
      .....:     ax.set_zlim3d(-0.5, 1)
      .....:     ax.set_xticks([0, 1])
      .....:     ax.set_yticks([0, 1])
      .....:     ax.set_zticks([-0.5, 0, 1])
      .....:

   @savefig basis-tesmooth.png align=center
   In [120]: fig.tight_layout()

Following what we did for univariate splines in the preceding sections, we will
now set up a 3-d smooth basis using some data and then make predictions on a
new set of data:

.. ipython:: python

   data = {"x1": np.linspace(0., 1., 100),
           "x2": np.linspace(0., 1., 100),
           "x3": np.linspace(0., 1., 100)}
   design_matrix = dmatrix("te(cr(x1, df=3), cr(x2, df=3), cc(x3, df=3), constraints='center')",
                           data)
   new_data = {"x1": [0.1, 0.2],
               "x2": [0.2, 0.3],
               "x3": [0.3, 0.4]}
   new_design_matrix = build_design_matrices([design_matrix.design_info], new_data)[0]
   new_design_matrix
   np.asarray(new_design_matrix)
