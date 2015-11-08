.. _py2-versus-py3:

Python 2 versus Python 3
========================

.. currentmodule:: patsy

The biggest difference between Python 2 and Python 3 is in their
string handling, and this is particularly relevant to Patsy since
it parses user input. We follow a simple rule: input to Patsy
should always be of type ``str``. That means that on Python 2, you
should pass byte-strings (not unicode), and on Python 3, you should
pass unicode strings (not byte-strings). Similarly, when Patsy
passes text back (e.g. :attr:`DesignInfo.column_names`), it's always
in the form of a ``str``.

In addition to this being the most convenient for users (you never
need to use any b"weird" u"prefixes" when writing a formula string),
it's actually a necessary consequence of a deeper change in the Python
language: in Python 2, Python code itself is represented as
byte-strings, and that's the only form of input accepted by the
:mod:`tokenize` module. On the other hand, Python 3's tokenizer and
parser use unicode, and since Patsy processes Python code, it has
to follow suit.

There is one exception to this rule: on Python 2, as a convenience for
those using ``from __future__ import unicode_literals``, the
high-level API functions :func:`dmatrix`, :func:`dmatrices`,
:func:`incr_dbuilders`, and :func:`incr_dbuilder` do accept
``unicode`` strings -- BUT these unicode string objects are still
required to contain only ASCII characters; if they contain any
non-ASCII characters then an error will be raised. If you really need
non-ASCII in your formulas, then you should consider upgrading to
Python 3. Low-level APIs like :meth:`ModelDesc.from_formula` continue
to insist on ``str`` objects only.
