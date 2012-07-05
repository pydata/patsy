# This file is part of Charlton
# Copyright (C) 2011-2012 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

"""charlton is a Python package for describing statistical models and building
design matrices. It is closely inspired by the 'formula' mini-language used in
R and S."""

# Do this first, to make it easy to check for warnings while testing:
import os
if os.environ.get("CHARLTON_FORCE_NO_WARNINGS"):
    import warnings
    warnings.filterwarnings("error", module="^charlton")
    del warnings
del os

import charlton.origin

class CharltonError(Exception):
    """This is the main error type raised by Charlton functions.

    In addition to the usual Python exception features, you can pass a second
    argument to this function specifying the origin of the error; this is
    included in any error message, and used to help the user locate errors
    arising from malformed formulas. This second argument should be an
    :class:`Origin` object, or else an arbitrary object with a ``.origin``
    attribute. (If it is neither of these things, then it will simply be
    ignored.)
    """
    def __init__(self, message, origin=None):
        Exception.__init__(self, message)
        # This is the special sphinx attribute-docstring syntax:
        #: The error message (without origin information). Use ``str(exc)``
        #: rather than ``exc.message`` if you want to display origin
        #: information.
        self.message = message
        if hasattr(origin, "origin"):
            origin = origin.origin
        if not isinstance(origin, charlton.origin.Origin):
            origin = None
        #: The :class:`Origin` of the offending object (or ``None``).
        self.origin = origin
        
    def __str__(self):
        if self.origin is None:
            return self.message
        else:
            return ("%s\n%s"
                    % (self.message, self.origin.caretize(indent=4)))


# 'from charlton import *' gives you a minimal API designed specifically for
# interactive use:
__all__ = ["CharltonError", "dmatrix", "dmatrices"]

# We make a richer API available for explicit use. To see what exactly is
# exported, check each module's __all__.
from charlton.highlevel import *
from charlton.build import *
from charlton.categorical import *
from charlton.constraint import *
from charlton.contrasts import *
from charlton.desc import *
from charlton.design_matrix import *
from charlton.eval import *
from charlton.origin import *
from charlton.state import *
from charlton.user_util import *
# XX FIXME: we aren't exporting any of the explicit parsing interface
# yet. Need to figure out how to do that.
