# Patsy

**Notice:** `patsy` is no longer under active development. As of August 2021,
Matthew Wardrop (@matthewwardrop) and Tomás Capretto (@tomicapretto) have taken
on responsibility from Nathaniel Smith (@njsmith) for keeping the lights on, but
no new feature development is planned. The spiritual successor of this project
is [Formulaic](https://github.com/matthewwardrop/formulaic), and we
recommend those interested in new feature development contribute there. Those
whose use-cases continue to be met by `patsy` can continue using this package
with increased confidence that things will continue to work as is for the
foreseeable future.

---

Patsy is a Python library for describing statistical models
(especially linear models, or models that have a linear component) and
building design matrices. Patsy brings the convenience of [R](http://www.r-project.org/) "formulas" to Python.

[![PyPI - Version](https://img.shields.io/pypi/v/patsy.svg)](https://pypi.org/project/spec-classes/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/patsy.svg)
![https://patsy.readthedocs.io/](https://img.shields.io/badge/docs-read%20now-blue.svg)
![PyPI - Status](https://img.shields.io/pypi/status/patsy.svg)
![https://travis-ci.org/pydata/patsy](https://travis-ci.org/pydata/patsy.png?branch=master)
![https://coveralls.io/r/pydata/patsy?branch=master](https://coveralls.io/repos/pydata/patsy/badge.png?branch=master)
![https://doi.org/10.5281/zenodo.592075](https://zenodo.org/badge/DOI/10.5281/zenodo.592075.svg)

- **Documentation:** <https://patsy.readthedocs.io/>
- **Downloads:** <http://pypi.python.org/pypi/patsy/>
- **Code and issues:** <https://github.com/pydata/patsy>
- **Mailing list:** <pydata@googlegroups.com> (<http://groups.google.com/group/pydata>)


## Dependencies

  * Python (2.6, 2.7, or 3.3+)
  * six
  * numpy
  * Optional:
    * pytest/pytest-cov: needed to run tests
    * scipy: needed for spline-related functions like ``bs``

## Installation
  ``pip install patsy`` (or, for traditionalists: ``python setup.py install``)

## License

2-clause BSD, see LICENSE.txt for details.
