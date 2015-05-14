Changes
=======

.. currentmodule:: patsy

v0.4.0
------

* Formulas (more precisely, :class:`EvalFactor` objects) now only
  keep a reference to the variables required from their environment
  instead of the whole environment when the formula was defined.

* Incompatible change: :class:`EvalFactor` does not take an
  ``eval_env`` argument anymore.

* Incompatible change: the :func:`design_matrix_builders` function now
  requires an ``eval_env`` as an additional argument.

v0.3.0
------

.. image:: https://zenodo.org/badge/4175/njsmith/zs.png
   :target: http://dx.doi.org/10.5281/zenodo.11445

|
* New stateful transforms for computing natural and cylic cubic
  splines with constraints, and tensor spline bases with
  constraints. (Thanks to `@broessli <https://github.com/broessli>`_
  and GDF Suez for contributing this code.)

* Dropped support for Python 2.5 and earlier.

* Switched to using a single source tree for both Python 2 and Python
  3.

* Added a fast-path to skip NA detection for inputs with boolean
  dtypes (thanks to Matt Davis for patch).

* Incompatible change: Sometimes when building a design matrix for a
  formula that does not depend on the data in any way, like ``"1 ~
  1"``, we have no way to determine how many rows the resulting matrix
  should have. In previous versions of patsy, when this occurred we
  simply returned a matrix with 1 row. In 0.3.0+, we instead refuse to
  guess, and raise an error.

  Note that because of the next change listed, this situation occurs
  less frequently in 0.3.0 than in previous versions.

* If the ``data`` argument to :func:`build_design_matrices` (or
  derived functions like :func:`dmatrix`, :func:`dmatrices`) is a
  :class:`pandas.DataFrame`, then we now check its number of rows and
  index, and insist that the output design matrices match. This also
  means that if ``data`` is a DataFrame, then the error described in
  the first bullet above cannot occur -- we will simply return a
  column of 1s that is the same size as the input dataframe.

* Worked around some more limitations in py2exe/py2app and friends.

v0.2.1
------

.. image:: https://zenodo.org/badge/doi/10.5281/zenodo.11447.png
   :target: http://dx.doi.org/10.5281/zenodo.11447

|
* Fixed a nasty bug in missing value handling where, if missing values
  were present, ``dmatrix(..., result_type="dataframe")`` would always
  crash, and ``dmatrices("y ~ 1")`` would produce left- and right-hand
  side matrices that had different numbers of rows. (As far as I can
  tell, this bug could not possibly cause incorrect results, only
  crashes, since it always involved the creation of matrices with
  incommensurate shapes. Therefore there is no need to worry about the
  accuracy of any analyses that were successfully performed with
  v0.2.0.)
* Modified ``patsy/__init__.py`` to work around limitations in
  py2exe/py2app/etc.

v0.2.0
------

.. image:: https://zenodo.org/badge/doi/10.5281/zenodo.11448.png
   :target: http://dx.doi.org/10.5281/zenodo.11448

|
Warnings:

* The lowest officially supported Python version is now 2.5. So far as
  I know everything still works with Python 2.4, but as everyone else
  has continued to drop support for 2.4, testing on 2.4 has become so
  much trouble that I've given up.

New features:

* New support for automatically detecting and (optionally) removing
  missing values (see :class:`NAAction`).
* New stateful transform for B-spline regression:
  :func:`bs`. (Requires scipy.)
* Added a core API to make it possible to run predictions on only a
  subset of model terms. (This is particularly useful for
  e.g. plotting the isolated effect of a single fitted spline term.)
  See :meth:`DesignMatrixBuilder.subset`.
* :class:`LookupFactor` now allows users to mark variables as
  categorical directly.
* :class:`pandas.Categorical` objects are now recognized as
  representing categorical data and handled appropriately.
* Better error reporting for exceptions raised by user code inside
  formulas. We now, whenever possible, tag the generated exception
  with information about which factor's code raised it, and use this
  information to give better error reporting.
* :meth:`EvalEnvironment.capture` now takes a `reference` argument,
  to make it easier to implement new :func:`dmatrix`-like functions.

Other: miscellaneous doc improvements and bug fixes.

v0.1.0
------

.. image:: https://zenodo.org/badge/doi/10.5281/zenodo.11449.png
   :target: http://dx.doi.org/10.5281/zenodo.11449

|
First public release.
