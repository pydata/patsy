# This file is part of Charlton
# Copyright (C) 2011 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

# The core 'origin' tracking system. This point of this is to have machinery
# so if some object is ultimately derived from some portion of a string (e.g.,
# a formula), then we can keep track of that, and use it to give proper error
# messages.

__all__ = ["CharltonErrorWithOrigin", "Origin", "StringWithOrigin"]

from charlton import CharltonError

class CharltonErrorWithOrigin(CharltonError):
    def __init__(self, message, origin):
        Exception.__init__(self, message, origin)
        self.message = message
        if hasattr(origin, "origin"):
            origin = origin.origin
        if not isinstance(origin, Origin):
            origin = None
        self.origin = origin
        
    def __str__(self):
        if self.origin is None:
            return self.message
        else:
            return ("%s\n%s"
                    % (self.message, self.origin.caretize(indent=4)))

class Origin(object):
    def __init__(self, code, start, end):
        self.code = code
        self.start = start
        self.end = end

    def combine(cls, origins):
        codes = set([o.code for o in origins])
        assert len(codes) == 1
        start = min([o.start for o in origins])
        end = max([o.end for o in origins])
        return cls(codes.pop(), start, end)
    combine = classmethod(combine)

    def __eq__(self, other):
        return (isinstance(other, Origin)
                and self.code == other.code
                and self.start == other.start
                and self.end == other.end)

    def __hash__(self):
        return hash((Origin, self.code, self.start, self.end))

    # NB: This does not include a trailing newline
    def caretize(self, indent=0):
        return ("%s%s\n%s%s%s" 
                % (" " * indent,
                   self.code,
                   " " * indent,
                   " " * self.start,
                   "^" * (self.end - self.start)))

    def __repr__(self):
        return "<Origin %s->%s<-%s (%s-%s)>" % (
            self.code[:self.start],
            self.code[self.start:self.end],
            self.code[self.end:],
            self.start, self.end)

def test_Origin():
    o1 = Origin("012345", 2, 4)
    o2 = Origin("012345", 4, 5)
    assert o1.caretize() == "012345\n  ^^"
    assert o2.caretize() == "012345\n    ^"
    o3 = Origin.combine([o1, o2])
    assert o3.code == "012345"
    assert o3.start == 2
    assert o3.end == 5
    assert o3.caretize(indent=2) == "  012345\n    ^^^"
    assert o3 == Origin("012345", 2, 5)

class StringWithOrigin(str):
    def __new__(cls, value, origin):
        self = str.__new__(cls, value)
        self.origin = origin
        return self

    def __repr__(self):
        return "<%s %s>" % (self.__class__.__name__, str.__repr__(self))
