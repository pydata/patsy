#!/usr/bin/env python

from setuptools import setup

DESC = """A Python library for describing statistical models and for
building model matrices."""

setup(
    name="charlton",
    version="0.0+dev",
    description=DESC,
    author="Nathaniel J. Smith",
    author_email="njs@pobox.com",
    license="2-clause BSD",
    packages=["charlton"],
    url="https://github.com/charlton/charlton",
    )
