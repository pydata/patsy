# This file is part of Charlton
# Copyright (C) 2011 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

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

