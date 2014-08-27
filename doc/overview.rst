Overview
========

  |epigraph|_

  .. |epigraph| replace:: *"It's only a model."*

  .. _epigraph: https://en.wikipedia.org/wiki/Patsy_%28Monty_Python%29

:mod:`patsy` is a Python package for describing statistical models
(especially linear models, or models that have a linear component)
and building design matrices. It is closely inspired by and compatible
with the `formula <http://cran.r-project.org/doc/manuals/R-intro.html#Formulae-for-statistical-models>`_ mini-language used in `R
<http://www.r-project.org/>`_ and `S
<https://secure.wikimedia.org/wikipedia/en/wiki/S_programming_language>`_.

For instance, if we have some variable `y`, and we want to regress it
against some other variables `x`, `a`, `b`, and the `interaction
<https://secure.wikimedia.org/wikipedia/en/wiki/Interaction_%28statistics%29>`_
of `a` and `b`, then we simply write::

  patsy.dmatrices("y ~ x + a + b + a:b", data)

and Patsy takes care of building appropriate matrices. Furthermore,
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
  fit into memory at one time,
* Provides a language for symbolic, human-readable specification of
  linear constraint matrices,
* Has a thorough test suite (>97% statement coverage) and solid
  underlying theory, allowing it to correctly handle corner cases that
  even R gets wrong, and
* Features a simple API for integration into statistical packages.

What Patsy *won't* do is, well, statistics --- it just lets you
describe models in general terms. It doesn't know or care whether you
ultimately want to do linear regression, time-series analysis, or fit
a forest of `decision trees
<https://secure.wikimedia.org/wikipedia/en/wiki/Decision_tree_learning>`_,
and it certainly won't do any of those things for you --- it just
gives a high-level language for describing which factors you want your
underlying model to take into account. It's not suitable for
implementing arbitrary non-linear models from scratch; for that,
you'll be better off with something like `Theano
<http://deeplearning.net/software/theano/>`_, `SymPy
<http://sympy.org/>`_, or just plain Python. But if you're using a
statistical package that requires you to provide a raw model matrix,
then you can use Patsy to painlessly construct that model matrix; and
if you're the author of a statistics package, then I hope you'll
consider integrating Patsy as part of your front-end.

Patsy's goal is to become the standard high-level interface to
describing statistical models in Python, regardless of what particular
model or library is being used underneath.

Download
--------

The current release may be downloaded from the Python Package index at

  http://pypi.python.org/pypi/patsy/

Or the latest *development version* may be found in our `Git
repository <https://github.com/pydata/patsy>`_::

  git clone git://github.com/pydata/patsy.git

Requirements
------------

Installing :mod:`patsy` requires:

* `Python <http://python.org/>`_ (version 2.6, 2.7, or 3.3+)
* `Six <https://pypi.python.org/pypi/six>`_
* `NumPy <http://numpy.scipy.org/>`_

Installation
------------

If you have ``pip`` installed, then a simple ::

  pip install --upgrade patsy

should get you the latest version. Otherwise, download and unpack the
source distribution, and then run ::

  python setup.py install

Contact
-------

Post your suggestions and questions directly to the `pydata mailing
list <https://groups.google.com/group/pydata>`_
(pydata@googlegroups.com, `gmane archive
<http://news.gmane.org/gmane.comp.python.pydata>`_), or to our `bug
tracker <https://github.com/pydata/patsy/issues>`_. You could also
contact `Nathaniel J. Smith <mailto:njs@pobox.com>`_ directly, but
really the mailing list is almost always a better bet, because more
people will see your query and others will be able to benefit from any
answers you get.

License
-------

2-clause BSD. See the file `LICENSE.txt
<https://github.com/pydata/patsy/blob/master/LICENSE.txt>`_ for details.

Users
-----

We currently know of the following projects using Patsy to provide a
high-level interface to their statistical code:

* `Statsmodels <http://statsmodels.sourceforge.net/>`_
* `PyMC3 <https://github.com/pymc-devs/pymc/tree/pymc3/>`_ (`tutorial <http://twiecki.github.io/blog/2013/09/12/bayesian-glms-1/>`_)
* `HDDM <https://github.com/hddm-devs/hddm>`_
* `rERPy <https://github.com/rerpy/rerpy>`_
* `UrbanSim <https://github.com/synthicity/urbansim>`_

If you'd like your project to appear here, see our documentation for
:ref:`library developers <library-developers>`!
