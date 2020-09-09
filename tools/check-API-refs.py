#!/usr/bin/env python

# NB: this currently works on both Py2 and Py3, and should be kept that way.

import sys
import re
from os.path import dirname, abspath

root = dirname(dirname(abspath(__file__)))
patsy_ref = root + "/doc/API-reference.rst"

doc_re = re.compile(r"^\.\. (.*):: ([^\(]*)")


def _documented(rst_path):
    documented = set()
    with open(rst_path) as rst_file:
        for line in rst_file:
            match = doc_re.match(line.rstrip())
            if match:
                directive = match.group(1)
                symbol = match.group(2)
                if directive not in ["module", "ipython"]:
                    documented.add(symbol)
        return documented


try:
    import patsy
except ImportError:
    sys.path.append(root)
    import patsy

documented = set(_documented(patsy_ref))
# print(documented)
exported = set(patsy.__all__)
missed = exported.difference(documented)
extra = documented.difference(exported)
if missed:
    print("DOCS MISSING FROM %s:" % (patsy_ref,))
    for m in sorted(missed):
        print("  %s" % (m,))
if extra:
    print("EXTRA DOCS IN %s:" % (patsy_ref,))
    for m in sorted(extra):
        print("  %s" % (m,))

if missed or extra:
    sys.exit(1)
else:
    print("Reference docs look good.")
    sys.exit(0)
