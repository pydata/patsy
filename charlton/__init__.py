# This file is part of Charlton
# Copyright (C) 2011-2012 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

# Do this first, to make it easy to check for warnings while testing:
import os
if os.environ.get("CHARLTON_FORCE_NO_WARNINGS"):
    import warnings
    warnings.filterwarnings("error", module="^charlton")
    del warnings
del os

import charlton.origin

class CharltonError(Exception):
    def __init__(self, message, origin=None):
        Exception.__init__(self, message)
        self.message = message
        if hasattr(origin, "origin"):
            origin = origin.origin
        if not isinstance(origin, charlton.origin.Origin):
            origin = None
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
