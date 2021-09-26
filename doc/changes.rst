Changes
=======

.. currentmodule:: patsy

All Patsy releases are archived at Zenodo:

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.592075.svg
   :target: https://doi.org/10.5281/zenodo.592075

v0.5.2
------

* Fix some deprecation warnings associated with importing from the `collections`
  module (rather than `collections.abc`) in Python 3.7+.

v0.5.1
------

* The Python 3.6.7 and 3.7.1 point releases changed the standard
  tokenizer module in a way that broke patsy. Updated patsy to work
  with these point releases. (See `#131
  <https://github.com/pydata/patsy/pull/131>`__ for details.)


v0.5.0
------

* Dropped support for Python 2.6 and 3.3.
* Update to keep up with ``pandas`` API changes
* More consistent handling of degenerate linear constraints in
  :meth:`DesignInfo.linear_constraint` (`#89
  <https://github.com/pydata/patsy/issues/89>`__)
* Fix a crash in ``DesignMatrix.__repr__`` when ``shape[0] == 0``


v0.4.1
------

New features:

* On Python 2, accept ``unicode`` strings containing only ASCII
  characters as valid formula descriptions in
  the high-level formula API (:func:`dmatrix` and friends). This is
  intended as a convenience for people using Python 2 with ``from
  __future__ import unicode_literals``. (See :ref:`py2-versus-py3`.)

Bug fixes:

* Accept ``long`` as a valid integer type in the new
  :class:`DesignInfo` classes. In particular this fixes errors that
  arise on 64-bit Windows builds (where ``ndarray.shape`` contains
  ``long`` objects), like ``ValueError: For numerical factors,
  num_columns must be an int.``

* Fix deprecation warnings encountered with numpy 1.10


v0.4.0
------

Incompatible changes:

* :class:`EvalFactor` and :meth:`ModelDesc.from_formula` no longer
  take an ``eval_env`` argument.

* The :func:`design_matrix_builders` function and the
  :meth:`factor_protocol.memorize_passes_needed` method now require an
  ``eval_env`` as an additional argument.

* The :class:`DesignInfo` constructor's arguments have totally
  changed. In addition to the changes needed to support the new
  features below, we no longer support "shim" DesignInfo objects that
  have non-trivial term specifications. This was only included in the
  first place to provide a compatibility hook for competing formula
  libraries; four years later, no such libraries have shown up. If one
  does, we can re-add it, but I'm not going to bother maintaining it
  in the mean time...

* Dropped support for Python 3.2.

Other changes:

* Patsy now supports Pandas's new (version 0.15 or later) categorical
  objects.

* Formulas (or more precisely, :class:`EvalFactor` objects) now only
  keep a reference to the variables required from their environment
  instead of the whole environment where the formula was
  defined. (Thanks to Christian Hudon.)

* :class:`DesignInfo` has new attributes
  :attr:`DesignInfo.factor_infos` and :attr:`DesignInfo.term_codings`
  which provide detailed metadata about how each factor and term is
  encoded.

* As a result of the above changes, the split between
  :class:`DesignInfo` and :class:`DesignMatrixBuilder` is no longer
  necessary; :class:`DesignMatrixBuiler` has been eliminated. So for
  example, :func:`design_matrix_builders` now returns a list of
  :class:`DesignInfo` objects, and you can now pass
  :class:`DesignInfo` objects directly to any function for building
  design matrices. For compatibility, :class:`DesignInfo` continues to
  provide ``.builder`` and ``.design_info`` attributes, so that old
  code should continue to work; however, these attributes are
  deprecated.

* Ensured that attempting to pickle most Patsy objects raises an
  error. This has never been supported, and the interesting cases
  failed in any case, but now we're taking a more systematic
  approach. (Soon we will add real, supported pickling support.)

* Fixed a bug when running under ``python -OO``.


v0.3.0
------

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

First public release.
