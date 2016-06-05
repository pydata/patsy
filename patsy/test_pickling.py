from __future__ import print_function

import six
from six.moves import cPickle as pickle

import os
import shutil

from patsy import EvalFactor

PICKE_TESTCASES_ROOTDIR = os.path.join(os.path.dirname(__file__), '..', 'pickle_testcases')

pickling_testcases = {
    "evalfactor_simple": EvalFactor("a+b"),
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
    with open(pickle_filename, 'rb') as f:
        testcase_name = os.path.splitext(os.path.basename(pickle_filename))[0]
        # When adding features to a class, it will happen that there is no
        # way to make an instance of that version version of that class
        # equal to any instance of a previous version. How do we handle
        # that?
        # Maybe adding a minimum version requirement to each test?
        assert pickling_testcases[testcase_name] == pickle.load(f)

def test_unpickling_future_gives_sensible_error_msg():
    # TODO How do we do this? And how do we test it then?
    pass

def create_pickles(version):
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
        print("Exception during creation of pickles for {}. Removing directory.".format(version))
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
