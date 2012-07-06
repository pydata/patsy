#!/usr/bin/env python

import sys
import re
from os.path import dirname, abspath

root = dirname(dirname(abspath(__file__)))
charlton_ref = root + "/doc/API-reference.rst"

autodoc_re = re.compile("^\.\. auto.*:: (.*)")
def autodocumented(rst_path):
    documented = set()
    for line in open(rst_path):
        match = autodoc_re.match(line.strip())
        if match:
            documented.add(match.group(1))
    return documented

import charlton

documented = autodocumented(charlton_ref)
#print(documented)
missed = [export for export in charlton.__all__ if export not in documented]
if missed:
    print("MISSING DOCS: %s" % (missed,))
    sys.exit(1)
else:
    print("Reference docs look good!")
    sys.exit(0)
