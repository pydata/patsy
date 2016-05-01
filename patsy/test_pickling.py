from six.moves import cPickle as pickle

from patsy.eval import EvalFactor
from patsy.version import __version__


objects_to_test = [
    ("EvalFactor('a+b', 'mars')", {
        "0.4.1+dev": "ccopy_reg\n_reconstructor\np1\n(cpatsy.eval\nEvalFactor\np2\nc__builtin__\nobject\np3\nNtRp4\n(dp5\nS\'code\'\np6\nS\'a + b\'\np7\nsS\'origin\'\np8\nS\'mars\'\np9\nsS\'version\'\np10\nS\'0.4.1+dev\'\np11\nsb."
    })
    ]

def test_pickling_roundtrips():
    for obj_code, pickled_history in objects_to_test:
        obj = eval(obj_code)
        print pickle.dumps(obj).encode('string-escape')
        assert obj == pickle.loads(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL))
        for version, pickled in pickled_history.items():
            assert pickle.dumps(obj) == pickled
