``charlton`` API reference
==========================

This is a complete reference for everything you get when you `import
charlton`.

.. module:: charlton

.. ipython:: python
   :suppress:

   from charlton import *

Basic API
---------

.. autofunction:: dmatrices
.. autofunction:: dmatrix

.. autofunction:: ddataframes
.. autofunction:: ddataframe

.. autofunction:: incr_dbuilders
.. autofunction:: incr_dbuilder

.. autoexception:: CharltonError
   :members:

Design matrices
---------------

.. autoclass:: DesignInfo

   Example:

   .. ipython:: python

      mat = dmatrix("a + x", demo_data("a", "x", nlevels=3))
      di = mat.design_info

   .. attribute:: column_names

      The names of each column, represented as a list of strings in
      the proper order. Guaranteed to exist.

      .. ipython:: python

         di.column_names

   .. attribute:: column_name_indexes

      An :class:`~collections.OrderedDict` mapping column names (as
      strings) to column indexes (as integers). Guaranteed to exist
      and to be sorted from low to high.

      .. ipython:: python

         di.column_name_indexes

   .. attribute:: term_names

      The names of each term, represented as a list of strings in
      the proper order. Guaranteed to exist. There is a one-to-many
      relationship between columns and terms -- each term generates
      one or more columns.

      .. ipython:: python

         di.term_names

   .. attribute:: term_name_slices

      An :class:`~collections.OrderedDict` mapping term names (as
      strings) to Python :func:`slice` objects indicating which
      columns correspond to each term. Guaranteed to exist. The slices
      are guaranteed to be sorted from left to right and to cover the
      whole range of columns with no overlaps or gaps.

      .. ipython:: python

         di.term_name_slices

   .. attribute:: terms

      A list of :class:`Term` objects representing each term. May be
      None, for example if a user passed in a plain preassembled
      design matrix rather than using the Charlton machinery.

      .. ipython:: python

         di.terms

   .. attribute:: term_slices

      An :class:`~collections.OrderedDict` mapping :class:`Term`
      objects to Python :func:`slice` objects indicating which columns
      correspond to which terms. Like :attr:`terms`, this may be None.

      .. ipython:: python

         di.term_slices

   .. attribute:: builder

      A :class:`DesignMatrixBuilder` object that can be used to
      generate more design matrices of this type (e.g. for
      prediction). May be None.

   A number of convenience methods are also provided that take
   advantage of the above metadata:

   .. automethod:: describe

   .. automethod:: linear_constraint

   .. automethod:: slice

   .. automethod:: from_array

.. autoclass:: DesignMatrix

   .. automethod:: __new__

Convenience utilities
---------------------

.. autofunction:: balanced

.. autofunction:: demo_data

Stateful transforms
-------------------

.. autofunction:: center

.. autofunction:: standardize

.. function:: scale(x, center=True, rescale=True, ddof=0)

   An alias for :func:`standardize`, for R compatibility.

.. autofunction:: stateful_transform

Handling categorical data
-------------------------

.. autofunction:: C

.. autoclass:: Treatment
.. autoclass:: Diff
.. autoclass:: Poly
.. autoclass:: Sum
.. autoclass:: Helmert

.. autoclass:: ContrastMatrix

.. autoclass:: Categorical

Working with formulas
---------------------

.. autoclass:: Term

.. data:: INTERCEPT

.. autoclass:: LookupFactor

.. autoclass:: EvalFactor

.. autoclass:: ModelDesc

Working with the Python execution environment
---------------------------------------------

.. autoclass:: EvalEnvironment
   :members:

Building design matrices
------------------------

.. autofunction:: design_matrix_builders

.. autoclass:: DesignMatrixBuilder

.. autofunction:: build_design_matrices

Linear constraints
------------------

.. autoclass:: LinearConstraint

Origin tracking
---------------

.. autoclass:: Origin
   :members:
