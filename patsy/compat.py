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

# The *_indices functions were added in numpy 1.4
import numpy as np
if optional_dep_ok and hasattr(np, "triu_indices"):
    from numpy import triu_indices
    from numpy import tril_indices
    from numpy import diag_indices
else:
    def triu_indices(n):
        return np.triu(np.ones((n, n))).nonzero()
    def tril_indices(n):
        return np.tril(np.ones((n, n))).nonzero()
    def diag_indices(n):
        return (np.arange(n), np.arange(n))

##### Python standard library

# The Python license requires that all derivative works contain a "brief
# summary of the changes made to Python". Both for license compliance, and for
# our own sanity, therefore, please add a note at the top of any snippets you
# add here explaining their provenance, any changes made, and what versions of
# Python require them:

# Copied unchanged from Python 2.7.3's re.py module; all I did was add the
# import statements at the top.
# This code seems to be included in Python 2.5+.
import re
if optional_dep_ok and hasattr(re, "Scanner"):
    Scanner = re.Scanner
else:
    import sre_parse
    import sre_compile
    class Scanner:
        def __init__(self, lexicon, flags=0):
            from sre_constants import BRANCH, SUBPATTERN
            self.lexicon = lexicon
            # combine phrases into a compound pattern
            p = []
            s = sre_parse.Pattern()
            s.flags = flags
            for phrase, action in lexicon:
                p.append(sre_parse.SubPattern(s, [
                    (SUBPATTERN, (len(p)+1, sre_parse.parse(phrase, flags))),
                    ]))
            s.groups = len(p)+1
            p = sre_parse.SubPattern(s, [(BRANCH, (None, p))])
            self.scanner = sre_compile.compile(p)
        def scan(self, string):
            result = []
            append = result.append
            match = self.scanner.scanner(string).match
            i = 0
            while 1:
                m = match()
                if not m:
                    break
                j = m.end()
                if i == j:
                    break
                action = self.lexicon[m.lastindex-1][1]
                if hasattr(action, '__call__'):
                    self.match = m
                    action = action(self, m.group())
                if action is not None:
                    append(action)
                i = j
            return result, string[i:]

# functools available in Python 2.5+
# This is just a cosmetic thing, so don't bother emulating it if we don't
# have it.
def compat_wraps(f1):
    def do_wrap(f2):
        return f2
    return do_wrap
if optional_dep_ok:
    try:
        from functools import wraps
    except ImportError:
        wraps = compat_wraps
else:
    wraps = compat_wraps

# collections.Mapping available in Python 2.6+
import collections
if optional_dep_ok and hasattr(collections, "Mapping"):
    Mapping = collections.Mapping
else:
    Mapping = dict

# OrderedDict is only available in Python 2.7+. compat_ordereddict.py has
# comments at the top.
import collections
if optional_dep_ok and hasattr(collections, "OrderedDict"):
    from collections import OrderedDict
else:
    from patsy.compat_ordereddict import OrderedDict

# 'raise from' available in Python 3+
import sys
from patsy import PatsyError
def call_and_wrap_exc(msg, origin, f, *args, **kwargs):
    try:
        return f(*args, **kwargs)
    except Exception as e:
        if sys.version_info[0] >= 3:
            new_exc = PatsyError("%s: %s: %s"
                                 % (msg, e.__class__.__name__, e),
                                 origin)
            # Use 'exec' to hide this syntax from the Python 2 parser:
            exec("raise new_exc from e")
        else:
            # In python 2, we just let the original exception escape -- better
            # than destroying the traceback. But if it's a PatsyError, we can
            # at least set the origin properly.
            if isinstance(e, PatsyError):
                e.set_origin(origin)
            raise
