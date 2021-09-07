from __future__ import print_function

import six
from six.moves import cPickle as pickle

import os
import shutil

from patsy import EvalFactor, EvalEnvironment

import numpy as np
from patsy.eval import VarLookupDict
from patsy.state import center, scale, standardize
from patsy.categorical import C
from patsy.splines import bs
from patsy.desc import Term, ModelDesc, _MockFactor
from patsy.mgcv_cubic_splines import cc, te, cr
from patsy.contrasts import ContrastMatrix
from patsy.constraint import LinearConstraint
from patsy.missing import NAAction
from patsy.origin import Origin
from patsy.design_info import SubtermInfo
from patsy.util import assert_pickled_equals


PICKLE_TESTCASES_ROOTDIR = os.path.join(os.path.dirname(__file__), '..',
                                        'pickle_testcases')

f1 = _MockFactor("a")
f2 = _MockFactor("b")

cm = ContrastMatrix(np.ones((2, 2)), ["[1]", "[2]"])
si = SubtermInfo(["a", "x"], {"a": cm}, 4)


def _unwrap_stateful_function(function, *args, **kwargs):
    obj = function.__patsy_stateful_transform__()
    obj.memorize_chunk(*args, **kwargs)
    obj.memorize_finish()
    return (obj, args, kwargs)


pickling_testcases = {
    "evalfactor_simple": EvalFactor("a+b"),
    "term": Term([1, 2, 1]),
    "contrast_matrix": ContrastMatrix([[1, 0], [0, 1]], ["a", "b"]),
    "subterm_info": si,
    "linear_constraint": LinearConstraint(["a"], [[0]]),
    "model_desc": ModelDesc([Term([]), Term([f1])],
                            [Term([f1]), Term([f1, f2])]),
    "na_action": NAAction(NA_types=["NaN", "None"]),
    "origin": Origin("012345", 2, 5),
    "transform_center": _unwrap_stateful_function(center,
                                                  np.arange(10, 20, 0.1)),
    "transform_standardize_norescale": _unwrap_stateful_function(
        standardize,
        np.arange(10, 20, 0.1),
    ),
    "transform_standardize_rescale": _unwrap_stateful_function(
        standardize,
        np.arange(10, 20, 0.1),
        rescale=True
    ),
    "transform_bs_df3": _unwrap_stateful_function(
        bs,
        np.arange(10, 20, 0.1),
        df=3
    ),
    "transform_bs_knots_13_15_17": _unwrap_stateful_function(
        bs,
        np.arange(10, 20, 0.1),
        knots=[13, 15, 17]
    ),
    "transform_cc_df3": _unwrap_stateful_function(
        cc,
        np.arange(10, 20, 0.1),
        df=3
    ),
    "transform_cc_knots_13_15_17": _unwrap_stateful_function(
        cc,
        np.arange(10, 20, 0.1),
        knots=[13, 15, 17]
    ),
    "transform_cr_df3": _unwrap_stateful_function(
        cr,
        np.arange(10, 20, 0.1),
        df=3
    ),
    "transform_cr_knots_13_15_17": _unwrap_stateful_function(
        cr,
        np.arange(10, 20, 0.1),
        knots=[13, 15, 17]
    ),
    "transform_te_cr5": _unwrap_stateful_function(
        te,
        cr(np.arange(10, 20, 0.1), df=5)
    ),
    "transform_te_cr5_center": _unwrap_stateful_function(
        te,
        cr(np.arange(10, 20, 0.1), df=5),
        constraint='center'
    ),
    }


def test_pickling_same_version_roundtrips():
    for obj in six.itervalues(pickling_testcases):
        if isinstance(obj, tuple):
            yield (check_pickling_same_version_roundtrips, obj[0])
        else:
            yield (check_pickling_same_version_roundtrips, obj)


def check_pickling_same_version_roundtrips(obj):
        pickled_obj = pickle.loads(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL))
        assert_pickled_equals(obj, pickled_obj)


def test_pickling_old_versions_still_work():
    for (dirpath, dirnames, filenames) in os.walk(PICKLE_TESTCASES_ROOTDIR):
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
        obj = pickling_testcases[testcase_name]
        if isinstance(obj, tuple):
            assert_pickled_equals(pickling_testcases[testcase_name][0],
                                  pickle.load(f))
        else:
            assert_pickled_equals(pickling_testcases[testcase_name],
                                  pickle.load(f))


def test_pickling_transforms():
    for obj in six.itervalues(pickling_testcases):
        if isinstance(obj, tuple):
            obj, args, kwargs = obj
            yield (check_pickling_transforms, obj, args, kwargs)


def check_pickling_transforms(obj, args, kwargs):
    pickled_obj = pickle.loads(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL))
    np.testing.assert_allclose(obj.transform(*args, **kwargs),
                               pickled_obj.transform(*args, **kwargs))


def test_unpickling_future_gives_sensible_error_msg():
    # TODO How would we go about testing this?
    pass


def create_pickles(version):
    # TODO Add options to overwrite pickles directory, with force=True
    # during development.
    # TODO Add safety check that said force=True option will still give an
    # error when trying to remove pickles for a released version, by
    # comparing the version argument here with patsy.__version__.
    pickle_testcases_dir = os.path.join(PICKLE_TESTCASES_ROOTDIR, version)
    if os.path.exists(pickle_testcases_dir):
        raise OSError("{} already exists. Aborting.".format(pickle_testcases_dir))
    pickle_testcases_tempdir = pickle_testcases_dir + "_inprogress"
    os.mkdir(pickle_testcases_tempdir)

    try:
        for name, obj in six.iteritems(pickling_testcases):
            if isinstance(obj, tuple):
                obj = obj[0]
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
