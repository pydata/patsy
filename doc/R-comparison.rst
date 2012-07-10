.. _R-comparison:

Differences between R and Patsy formulas
===========================================

.. currentmodule:: patsy

Patsy has a very high degree of compatibility with R. Almost any
formula you would use in R will also work in Patsy -- with a few
caveats.

.. note:: All R quirks described herein were last verified with R
   2.15.0.

Differences from R:

- Most obviously, we both support using arbitrary code to perform
  variable transformations, but in Patsy this code is written in
  Python, not R.

- Patsy has no ``%in%``. In R, ``a %in% b`` is identical to
  ``b:a``. Patsy only supports the ``b:a`` version of this syntax.

- In Patsy, only ``**`` can be used for exponentiation. In R, both
  ``^`` and ``**`` can be used for exponentiation, i.e., you can write
  either ``(a + b)^2`` or ``(a + b)**2``.  In Patsy (as in Python
  generally), only ``**`` indicates exponentiation; ``^`` is ignored
  by the parser (and if present, will be interpreted as a call to the
  Python binary XOR operator).

- In Patsy, the left-hand side of a formula uses the same
  evaluation rules as the right-hand side. In R, the left hand side is
  treated as R code, so a formula like ``y1 + y2 ~ x1 + x2`` actually
  regresses the *sum* of ``y1`` and ``y2`` onto the *set of
  predictors* ``x1`` and ``x2``. In Patsy, the only difference
  between the left-hand side and the right-hand side is that there is
  no automatic intercept added to the left-hand side. (In this regard
  Patsy is similar to the R enhanced formula package `Formula
  <http://cran.r-project.org/web/packages/Formula/index.html>`_.)

- Patsy produces a different column ordering for formulas involving
  numeric predictors.  In R, there are two rules for term ordering:
  first, lower-order interactions are sorted before higher-order
  interactions, and second, interactions of the same order are listed
  in whatever order they appeared in the formula. In Patsy, we add
  another rule: terms are first grouped together based on which
  numeric factors they include. Then within each group, we use the
  same ordering as R.

- Patsy has more rigorous handling of the presence or absence of
  the intercept term. In R, the rules for when deciding whether to
  include an intercept are somewhat idiosyncratic and can ignore
  things like parentheses. To understand the difference, first
  consider the formula ``a + (b - a)``. In both Patsy and R, we
  first evaluate the ``(b - a)`` part; since there is no ``a`` term to
  remove, this simplifies to just ``b``. We then evaluate ``a + b``:
  the end result is a model which contains an ``a`` term in it.

  Now consider the formula ``1 + (b - 1)``. In Patsy, this is
  analogous to the case above: first ``(b - 1)`` is reduced to just ``b``,
  and then ``1 + b`` produces a model with intercept included. In R, the
  parentheses are ignored, and ``1 + (b - 1)`` gives a model that does
  *not* include the intercept.

  This can be slightly more confusing when it comes to the implicit
  intercept term. In Patsy, this is handled exactly as if the
  right-hand side of each formula has an invisible ``"1 +"`` inserted at
  the beginning. Therefore in Patsy, these formulas are different::

    # Python:
    dmatrices("y ~ b - 1")   # equivalent to 1 + b - 1: no intercept
    dmatrices("y ~ (b - 1)") # equivalent to 1 + (b - 1): has intercept

  In R, these two formulas are equivalent.

- Patsy has a more accurate algorithm for deciding whether to use a
  full- or reduced-rank coding scheme for categorical factors. There
  are two situations in which R's coding algorithm for categorical
  variables can become confused and produce over- or under-specified
  model matrices. Patsy, so far as we are aware, produces correctly
  specified matrices in all cases. It's unlikely that you'll run into
  these in actual usage, but they're worth mentioning. To illustrate,
  let's define ``a`` and ``b`` as categorical predictors, each with 2
  levels:

  .. code-block:: rconsole

    # R:
    > a <- factor(c("a1", "a1", "a2", "a2"))
    > b <- factor(c("b1", "b2", "b1", "b2"))

  .. ipython:: python
     :suppress:

     a = ["a1", "a1", "a2", "a2"]
     b = ["b1", "b2", "b1", "b2"]
     from patsy import dmatrix

  The first problem occurs for formulas like ``1 + a:b``. This produces
  a model matrix with rank 4, just like many other formulas that
  include ``a:b``, such as ``0 + a:b``, ``1 + a + a:b``, and ``a*b``:

  .. code-block:: rconsole

    # R:
    > qr(model.matrix(~ 1 + a:b))$rank
    [1] 4
  
  However, the matrix produced for this formula has 5 columns, meaning
  that it contains redundant overspecification:

  .. code-block:: rconsole

    # R:
    > mat <- model.matrix(~ 1 + a:b)
    > ncol(mat)
    [1] 5

  The underlying problem is that R's algorithm does not pay attention
  to 'non-local' redundancies -- it will adjust its coding to avoid a
  redundancy between two terms of degree-n, or a term of degree-n and
  one of degree-(n+1), but it is blind to a redundancy between a term
  of degree-n and one of degree-(n+2), as we have here.

  Patsy's algorithm has no such limitation:

  .. ipython:: python

    # Python:
    a = ["a1", "a1", "a2", "a2"]
    b = ["b1", "b2", "b1", "b2"]
    mat = dmatrix("1 + a:b")
    mat.shape[1]

  To produce this result, it codes ``a:b`` uses the same columns that
  would be used to code ``b + a:b`` in the formula ``"1 + b + a:b"``.

  The second problem occurs for formulas involving numeric
  predictors. Effectively, when determining coding schemes, R assumes
  that all factors are categorical. So for the formula ``0 + a:c +
  a:b``, R will notice that if it used a full-rank coding for the ``c``
  and ``b`` factors, then both terms would be collinear with ``a``, and
  thus each other. Therefore, it encodes ``c`` with a full-rank
  encoding, and uses a reduced-rank encoding for ``b``. (And the ``0 +``
  lets it avoid the previous bug.) So far, so good.

  But now consider the formula ``0 + a:x + a:b``, where ``x`` is
  numeric. Here, ``a:x`` and ``a:b`` will not be collinear, even if we do
  use a full-rank encoding for ``b``. Therefore, we *should* use a
  full-rank encoding for ``b``, and produce a model matrix with 6
  columns. But in fact, R gives us only 4:
  
  .. code-block:: rconsole

    # R:
    > x <- c(1, 2, 3, 4)
    > mat <- model.matrix(~ 0 + a:x + a:b)
    > ncol(mat)
    [1] 4

  The problem is that it cannot tell the difference between ``0 + a:x +
  a:b`` and ``0 + a:c + a:b``: it uses the same coding for both, whether
  it's appropriate or not.

  (The alert reader might wonder whether this bug could be triggered
  by a simpler formula, like ``0 + x + b``. It turns out that R's code
  ``do_modelmatrix`` function has a special-case where for first-order
  interactions only, it *will* peek at the type of the data before
  deciding on a coding scheme.)

  Patsy always checks whether each factor is categorical or numeric
  before it makes coding decisions, and thus handles this case
  correctly:

  .. ipython:: python

    # Python:
    x = [1, 2, 3, 4]
    mat = dmatrix("0 + a:x + a:b")
    mat.shape[1]
