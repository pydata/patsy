Overview
========

.. warning::
  These docs are very much a work in progress, to the point that they
  may contain incorrect information. Sorry. For now you're probably
  better off looking at the source, but we're working on it!

Introduction
------------

:mod:`charlton` is a Python package for describing statistical models
and building design matrices. It is closely inspired by the 'formula'
mini-language used in `R <http://www.r-project.org/>`_ and `S
<https://secure.wikimedia.org/wikipedia/en/wiki/S_programming_language>`_. The
name is in honor of `Val Charlton
<http://www.wimbledon.arts.ac.uk/35174.htm>`_, who `built models
<http://www.imdb.com/name/nm0153313/>`_ for Monty Python.

For instance, if we have some variable ``y``, and we want to regress it
against some other variables ``x``, ``a``, ``b``, and the `interaction
<https://secure.wikimedia.org/wikipedia/en/wiki/Interaction_%28statistics%29>`_
of ``a`` and ``b``, then we simply write::

  y ~ x + a + b + a:b

and Charlton takes care of building appropriate matrices. Furthermore,
it:

* Allows data transformations to be specified using arbitrary Python
  code: instead of ``x``, we could have written ``log(x)``, ``(x >
  0)``, or even ``log(x) if x > 1e-5 else log(1e-5)``,
* Provides a range of convenient options for coding `categorical
  <https://secure.wikimedia.org/wikipedia/en/wiki/Level_of_measurement#Nominal_scale>`_
  variables, including automatic detection and removal of
  redundancies,
* Knows how to apply 'the same' transformation used on original data
  to new data, even for tricky transformations like centering or
  standardization (critical if you want to use your model to make
  predictions),
* Has an incremental mode to handle data sets which are too large to
  fit into memory at one time, and
* Features a rich and extensible API for integration into statistical
  packages.

What Charlton *won't* do is, well, statistics --- it just lets you
describe models in general terms. It doesn't know or care whether you
ultimately want to do linear regression, time-series analysis, or fit
a forest of `decision trees
<https://secure.wikimedia.org/wikipedia/en/wiki/Decision_tree_learning>`_,
and it certainly won't do any of those things for you. But if you're
using a statistical package that requires you to provide a raw model
matrix, then you can use Charlton to painlessly construct that model
matrix; and if you're the author of a statistics package, then I hope
you'll consider integrating Charlton as part of your front-end.

Charlton's goal is to become the standard high-level interface to
describing statistical models in Python, regardless of what particular
model or library is being used underneath.

Download
--------

The current release may be downloaded from the Python Package index at

  http://pypi.python.org/pypi/charlton/

Or the latest *development version* may be found in our `Git
repository <https://github.com/charlton/charlton>`_::

  git clone git://github.com/charlton/charlton.git

Requirements
------------

Installing :mod:`charlton` requires:

* `Python <http://python.org/>`_ (version 2.4 or later; Python 3 is
  not yet supported)
* `NumPy <http://numpy.scipy.org/>`_

Installation
------------

If you have ``pip`` installed, then a simple ::

  pip install --upgrade charlton

should get you the latest version. Otherwise, download and unpack the
source distribution, and then run ::

  python setup.py install

Contact
-------

Post your suggestions and questions directly to the `pydata mailing
list <https://groups.google.com/group/pydata>`_
(pydata@googlegroups.com), or to our `bug tracker
<https://github.com/charlton/charlton/issues>`_. You could also
contact `Nathaniel J. Smith <mailto:njs@pobox.com>`_ directly, but
really the mailing list is almost always a better bet, because more
people will see your query and others will be able to benefit from any
answers you get.
