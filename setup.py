#!/usr/bin/env python

from setuptools import setup

DESC = (
    "A Python package for describing statistical models and for "
    "building design matrices."
)

LONG_DESC = open("README.md").read()

# defines __version__
exec(open("patsy/version.py").read())

setup(
    name="patsy",
    version=__version__,  # noqa: F821
    description=DESC,
    long_description=LONG_DESC,
    long_description_content_type="text/markdown",
    author="Nathaniel J. Smith",
    author_email="njs@pobox.com",
    license="2-clause BSD",
    packages=["patsy"],
    url="https://github.com/pydata/patsy",
    install_requires=[
        # Possibly we need an even newer numpy than this, but we definitely
        # need at least 1.4 for triu_indices
        "numpy >= 1.4",
    ],
    extras_require={
        "test": ["pytest", "pytest-cov", "scipy"],
    },
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 6 - Mature",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering",
    ],
)
