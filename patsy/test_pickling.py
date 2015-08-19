from six.moves import cPickle as pickle

from patsy import EvalFactor

stuff = [
    EvalFactor("a+b"),
    ]

def test_pickling_roundtrips():
    for obj in stuff:
        assert obj == pickle.loads(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL))

def test_unpickling_future_gives_sensible_error_msg():
    pass

# Entrypoint: python -m patsy.test_pickling ...

if __name__ == "__main__":
    # TODO Save pickle. Make sure it's running from the right directory, so
    # the pickles are saved in the right place.

    
