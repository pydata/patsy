import six
from six.moves import cPickle as pickle

import os

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
        assert pickling_testcases[testcase_name] == pickle.load(f)

def test_unpickling_future_gives_sensible_error_msg():
    # TODO How do we do this? And how do we test it then?
    pass

def save_pickle_testcases(version):
    pickle_testcases_dir = os.path.join(PICKE_TESTCASES_ROOTDIR, version)
    # Fails if the directory already exists, which is what we want here
    # since we don't want to overwrite testcases accidentally.
    os.mkdir(pickle_testcases_dir)

    for name, obj in six.iteritems(pickling_testcases):
        with open(os.path.join(pickle_testcases_dir, '{}.pickle'.format(name)), 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    import argparse

    # Should we use a "create-pickles" sub-command to make things clearer?
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("version", help="The version of patsy for which to save a new set of pickle testcases.")
    args = arg_parser.parse_args()

    save_pickle_testcases(args.version)
