# This file is part of Patsy
# Copyright (C) 2012 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

# This file contains compatibility code for supporting old versions of Python
# and numpy. (If we can concentrate it here, hopefully it'll make it easier to
# get rid of weird hacks once we drop support for old versions).

##### Numpy

import os

# To force use of the compat code, set this env var to a non-empty value:
optional_dep_ok = not os.environ.get("PATSY_AVOID_OPTIONAL_DEPENDENCIES")

##### Python standard library

# The Python license requires that all derivative works contain a "brief
# summary of the changes made to Python". Both for license compliance, and for
# our own sanity, therefore, please add a note at the top of any snippets you
# add here explaining their provenance, any changes made, and what versions of
# Python require them:


# 'raise from' available in Python 3+
import sys
from patsy import PatsyError


def call_and_wrap_exc(msg, origin, f, *args, **kwargs):
    try:
        return f(*args, **kwargs)
    except Exception as e:
        new_exc = PatsyError("%s: %s: %s" % (msg, e.__class__.__name__, e), origin)
        raise new_exc from e
