Python 2 versus Python 3
========================

.. currentmodule:: patsy

The biggest difference between Python 2 and Python 3 is in their
string handling, and this is particularly relevant to Patsy since
it parses user input. We follow a simple rule: input to Patsy
should always be of type `str`. That means that on Python 2, you
should pass byte-strings (not unicode), and on Python 3, you should
pass unicode strings (not byte-strings). Similarly, when Patsy
passes text back (e.g. :attr:`DesignInfo.column_names`), it's always
in the form of a `str`.

In addition to this being the most convenient for users (you never
need to use any b"weird" u"prefixes" when writing a formula string),
it's actually a necessary consequence of a deeper change in the Python
language: in Python 2, Python code itself is represented as
byte-strings, and that's the only form of input accepted by the
:mod:`tokenize` module. On the other hand, Python 3's tokenizer and
parser use unicode, and since Patsy processes Python code, it has
to follow suit.
