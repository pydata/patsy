#!/usr/bin/env python

import sys
from setuptools import setup

DESC = """A Python package for describing statistical models and for
building design matrices."""

LONG_DESC = open("README.rst").read()

# defines __version__
exec(open("patsy/version.py").read())

setup(
    name="patsy",
    version=__version__,
    description=DESC,
    long_description=LONG_DESC,
    author="Nathaniel J. Smith",
    author_email="njs@pobox.com",
    license="2-clause BSD",
    packages=["patsy"],
    url="https://github.com/pydata/patsy",
    install_requires=["numpy"],
    classifiers =
      [ "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        ],
)
