# This file is part of Patsy
# Copyright (C) 2011-2014 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

# This file must be kept very simple, because it is consumed from several
# places -- it is imported by patsy/__init__.py, execfile'd by setup.py, etc.

# We use a simple scheme:
#   1.0.0 -> 1.0.0-dev -> 1.1.0 -> 1.1.0-dev
# where the -dev versions are never released into the wild, they're just what
# we stick into the VCS in between releases.
#
# This is compatible with PEP 440:
#   http://legacy.python.org/dev/peps/pep-0440/
# in a slightly abusive way -- PEP 440 provides no guidance on what version
# number to use for *unreleased* versions, so we use an "integrator suffix",
# which is intended to be used for things like Debian's locally patched
# version, and is not allowed on public index servers. Which sounds about
# right, actually... Crucially, PEP 440 says that "foo-bar" sorts *after*
# "foo", which is what we want for a dev version. (Compare to "foo.dev0",
# which sorts *before* "foo".)

__version__ = "0.3.0"
