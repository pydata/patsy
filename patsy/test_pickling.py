from __future__ import print_function

import six
from six.moves import cPickle as pickle

import os
import shutil

from patsy import EvalFactor, EvalEnvironment, VarLookupDict

import numpy as np
from patsy.state import center, scale, standardize
from patsy.categorical import C
from patsy.splines import bs
from patsy.desc import Term, ModelDesc
from patsy.mgcv_cubic_splines import cc, te, cr
from patsy.contrasts import ContrastMatrix
from patsy.constraint import LinearConstraint
from patsy.missing import NAAction
from patsy.origin import Origin


PICKE_TESTCASES_ROOTDIR = os.path.join(os.path.dirname(__file__), '..', 'pickle_testcases')


class _MockFactor(object):
    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __hash__(self):
        return hash((_MockFactor, str(self._name)))

f1 = _MockFactor("a")
f2 = _MockFactor("b")

pickling_testcases = {
    "evalfactor_simple": EvalFactor("a+b"),
    "varlookupdict_simple": VarLookupDict([{"a": 1}, {"a": 2, "b": 3}]),
    "evalenv_simple": EvalEnvironment([{"a": 1}]),
    "evalenv_transform_center": EvalEnvironment([{'center': center}]),
    "evalenv_transform_scale": EvalEnvironment([{'scale': scale}]),
    "evalenv_transform_standardize": EvalEnvironment([{
            'standardize': standardize
    }]),
    "evalenv_transform_catgorical": EvalEnvironment([{'C': C}]),
    "evalenv_transform_bs": EvalEnvironment([{'cs': bs}]),
    "evalenv_transform_te": EvalEnvironment([{'te': te}]),
    "evalenv_transform_cr": EvalEnvironment([{'cs': cr}]),
    "evalenv_transform_cc": EvalEnvironment([{'cc': cc}]),
    "evalenv_pickle": EvalEnvironment([{'np': np}]),
    "term": Term([1, 2, 1]),
    "contrast_matrix": ContrastMatrix([[1, 0], [0, 1]], ["a", "b"]),
    "linear_constraint": LinearConstraint(["a"], [[0]]),
    "model_desc": ModelDesc([Term([]), Term([f1])],
                            [Term([f1]), Term([f1, f2])]),
    "na_action": NAAction(NA_types=["NaN", "None"]),
    "origin": Origin("012345", 2, 5)
    }


def test_pickling_same_version_roundtrips():
    for obj in six.itervalues(pickling_testcases):
        yield (check_pickling_same_version_roundtrips, obj)


def check_pickling_same_version_roundtrips(obj):
        assert obj == pickle.loads(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL))


def test_pickling_old_versions_still_work():
    for (dirpath, dirnames, filenames) in os.walk(PICKE_TESTCASES_ROOTDIR):
        for fname in filenames:
            if os.path.splitext(fname)[1] == '.pickle':
                yield check_pickling_old_versions_still_work, os.path.join(dirpath, fname)


def check_pickling_old_versions_still_work(pickle_filename):
    testcase_name = os.path.splitext(os.path.basename(pickle_filename))[0]
    with open(pickle_filename, 'rb') as f:
        # When adding features to a class, it will happen that there is no
        # way to make an instance of that version version of that class
        # equal to any instance of a previous version. How do we handle
        # that?
        # Maybe adding a minimum version requirement to each test?
        assert pickling_testcases[testcase_name] == pickle.load(f)


def test_unpickling_future_gives_sensible_error_msg():
    # TODO How would we go about testing this?
    pass


def create_pickles(version):
    # TODO Add options to overwrite pickles directory, with force=True
    # during development.
    # TODO Add safety check that said force=True option will still give an
    # error when trying to remove pickles for a released version, by
    # comparing the version argument here with patsy.__version__.
    pickle_testcases_dir = os.path.join(PICKE_TESTCASES_ROOTDIR, version)
    if os.path.exists(pickle_testcases_dir):
        raise OSError("{} already exists. Aborting.".format(pickle_testcases_dir))
    pickle_testcases_tempdir = pickle_testcases_dir + "_inprogress"
    os.mkdir(pickle_testcases_tempdir)

    try:
        for name, obj in six.iteritems(pickling_testcases):
            with open(os.path.join(pickle_testcases_tempdir, "{}.pickle".format(name)), "wb") as f:
                pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    except Exception:
        print("Exception during creation of pickles for {}. " \
              "Removing partially created directory.".format(version))
        shutil.rmtree(pickle_testcases_tempdir)
        raise
    finally:
        os.rename(pickle_testcases_tempdir, pickle_testcases_dir)
    print("Successfully created pickle testcases for new version {}.".format(version))

if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(description="Create and save pickle testcases for a new version of patsy.")
    arg_parser.add_argument("version", help="The version of patsy for which to save a new set of pickle testcases.")
    args = arg_parser.parse_args()

    create_pickles(args.version)
