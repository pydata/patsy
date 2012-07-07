#!/usr/bin/env python

# NB: this currently works on both Py2 and Py3, and should be kept that way.

import sys
import re
from os.path import dirname, abspath

root = dirname(dirname(abspath(__file__)))
charlton_ref = root + "/doc/API-reference.rst"

doc_re = re.compile("^\.\. .*:: ([^\(]*)")
def _documented(rst_path):
    documented = set()
    for line in open(rst_path):
        match = doc_re.match(line.strip())
        if match:
            documented.add(match.group(1))
    return documented

try:
    import charlton
except ImportError:
    sys.path.append(root)
    import charlton

documented = _documented(charlton_ref)
#print(documented)
missed = [export for export in charlton.__all__ if export not in documented]
if missed:
    print("MISSING DOCS:")
    for m in missed:
        print("  %s" % (m,))
    sys.exit(1)
else:
    print("Reference docs appear to be complete.")
    sys.exit(0)
