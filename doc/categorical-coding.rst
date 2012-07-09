.. categorical-coding:

Coding categorical data
=======================



.. ipython:: python

   from charlton import dmatrix, demo_data, ContrastMatrix
   from charlton.builtins import *
   data = demo_data("a", nlevels=3)
   data

   dmatrix("a", data)

   contrast = [[1, 2], [3, 4], [5, 6]]
   dmatrix("C(a, contrast)", data)

   dmatrix("C(a, [[1], [2], [-4]])", data)

   contrast_mat = ContrastMatrix(contrast, ["foo", "bar"])
   dmatrix("C(a, contrast_mat)", data)

   dmatrix("C(a, Poly)", data)

For a full list of built-in coding schemes, see the :ref:`API
reference <categorical-coding-ref>`.

You can also define your own 
A simplified version of the built-in :class:`Treatment` coding object::

   class MyTreatment(object):
       def __init__(self, reference=0):
           self.reference = reference

       def code_with_intercept(self, levels):
           return ContrastMatrix(np.eye(len(levels)),
                                 ["[%s]" % (level,) for level in levels])

       def code_without_intercept(self, levels):
           eye = np.eye(len(levels) - 1)
           contrasts = np.vstack((eye[:self.reference, :],
                                  np.zeros((1, len(levels) - 1)),
                                  eye[self.reference:, :]))
           return ContrastMatrix(contrasts,
                                 ["[T.%s]" % (level,) for level in
                                  levels[:reference] + levels[reference + 1:]])
                                 
And it can now be used like::

   dmatrix("C(a, MyTreatment)", balanced(a=3))
   dmatrix("C(a, MyTreatment(2))", balanced(a=3))
