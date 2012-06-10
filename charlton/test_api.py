# This file is part of Charlton
# Copyright (C) 2012 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

# Exhaustive end-to-end tests of the top-level API.

from nose.tools import assert_raises
from charlton.api import *
from charlton.test_build import assert_full_rank, make_test_factors

# End-to-end tests need to include:
# - numerical and categorical factors
# - categorical: string, integer, bool, random-python-objects
# - numerical: integer, float
# - user-specified coding
# - transformations from the environment
# - depth= argument, pulling variables out of environment
# - order dependence:
#     of terms (following numericalness, interaction order, and, written order)
#     of factors within a term
# - with and without response variable
# - incremental building with nested stateful transforms
# - use of builtins
# - test I(a / b) varies depending on __future__ state of caller

# XX what term ordering *do* we want?
