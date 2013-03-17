Changes
=======

.. currentmodule:: patsy

v0.2.0
------

New features:

* New support for automatically detecting and (optionally) removing
  missing values (see :class:`NAAction`).
* Recognize :class:`pandas.Categorical` objects as categorical data
  and handle them appropriately.
* Better error reporting for exceptions raised by user code inside
  formulas. We now, whenever possible, tag the generated exception
  with information about which factor's code raised it.

Other: Misc. doc improvements and bug fixes.

v0.1.0
------
  First public release.
