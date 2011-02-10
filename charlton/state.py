# This file is part of Charlton
# Copyright (C) 2011 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

# Stateful transform protocol:
#   def __init__(self):
#       pass
#   def memorize_chunk(self, input_data):
#       return None
#   def memorize_finish(self):
#       return None
#   def transform(self, input_data):
#       return output_data

__all__ = ["builtin_stateful_transforms"]

import numpy as np
from charlton.util import atleast_2d_column_default, wide_dtype_for

builtin_stateful_transforms = {}

from charlton.categorical import CategoricalTransform, ContrastTransform
builtin_stateful_transforms["categorical"] = CategoricalTransform
builtin_stateful_transforms["categ"] = CategoricalTransform
builtin_stateful_transforms["contrast"] = ContrastTransform
# R compatibility:
builtin_stateful_transforms["C"] = ContrastTransform

# class NonIncrementalStatefulTransform(object):
#     def __init__(self):
#         self._data = []
    
#     def memorize_chunk(self, input_data, *args, **kwargs):
#         self._data.append(input_data)
#         self._args = _args
#         self._kwargs = kwargs

#     def memorize_finish(self):
#         all_data = np.row_stack(self._data)
#         del self._data
#         del self._args
#         del self._kwargs
#         self.memorize_all(all_data, *self._args, **self._kwargs)

#     def memorize_all(self, input_data, *args, **kwargs):
#         raise NotImplementedError

#     def transform(self, input_data, *args, **kwargs):
#         raise NotImplementedError

# class QuantileEstimatingTransform(NonIncrementalStatefulTransform):
#     def memorize_all(self, input_data, *args, **kwargs):
        
def _test_stateful(cls, input, output, *args, **kwargs):
    test_cases = [
        # List input, one chunk
        ([input], output),
        # Scalar input, many chunks
        (input, output),
        # List input, many chunks:
        ([[n] for n in input], output),
        # 0-d array input, many chunks:
        ([np.array(n) for n in input], output),
        # 1-d array input, one chunk:
        ([np.array(input)], output),
        # 1-d array input, many chunks:
        ([np.array([n]) for n in input], output),
        # 2-d array input, one chunk:
        ([np.column_stack((input, input[::-1]))],
         np.column_stack((output, output[::-1]))),
        # 2-d array input, many chunks:
        ([np.array([[input[i], input[-i-1]]]) for i in xrange(len(input))],
         np.column_stack((output, output[::-1]))),
        ]
    for input_obj, output_obj in test_cases:
        print input_obj
        t = cls()
        for input_chunk in input_obj:
            t.memorize_chunk(input_chunk, *args, **kwargs)
        t.memorize_finish()
        all_outputs = [t.transform(input_chunk, *args, **kwargs)
                       for input_chunk in input_obj]
        all_output1 = np.row_stack(all_outputs)
        assert all_output1.shape[0] == len(input)
        output_obj_reshaped = np.asarray(output_obj).reshape(all_output1.shape)
        assert np.allclose(all_output1, output_obj_reshaped)
        if np.asarray(input_obj[0]).ndim == 0:
            all_input = np.array(input_obj)
        elif np.asarray(input_obj[0]).ndim == 1:
            # Don't use row_stack, because that would turn this into a 1xn
            # matrix:
            all_input = np.concatenate(input_obj)
        else:
            all_input = np.row_stack(input_obj)
        all_output2 = t.transform(all_input, *args, **kwargs)
        assert np.allclose(all_output2, output_obj_reshaped)
    
class Center(object):
    def __init__(self):
        self._sum = None
        self._count = 0

    def memorize_chunk(self, x):
        x = atleast_2d_column_default(x)
        self._count += x.shape[0]
        this_total = np.sum(x, 0, dtype=wide_dtype_for(x))
        # This is to handle potentially multi-column x's:
        if self._sum is None:
            self._sum = this_total
        else:
            self._sum += this_total

    def memorize_finish(self):
        pass

    def transform(self, x):
        # XX: this probably returns very wide floating point data, which is
        # perhaps not what we desire -- should the mean be cast down to the
        # input data's width? (well, not if the input data is integer, but you
        # know what I mean.)
        return atleast_2d_column_default(x) - (self._sum / self._count)

builtin_stateful_transforms["center"] = Center

def test_Center():
    _test_stateful(Center, [1, 2, 3], [-1, 0, 1])
    _test_stateful(Center, [1, 2, 1, 2], [-0.5, 0.5, -0.5, 0.5])
    _test_stateful(Center,
                   [1.3, -10.1, 7.0, 12.0],
                   [-1.25, -12.65, 4.45, 9.45])


# See:
#   http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#On-line_algorithm
# or page 232 of Knuth vol. 3 (3rd ed.).
class Standardize(object):
    def __init__(self):
        self.current_n = 0
        self.current_mean = None
        self.current_M2 = None

    def memorize_chunk(self, x, center=True, rescale=True, ddof=0):
        x = atleast_2d_column_default(x)
        if self.current_mean is None:
            self.current_mean = np.zeros(x.shape[1], dtype=wide_dtype_for(x))
            self.current_M2 = np.zeros(x.shape[1], dtype=wide_dtype_for(x))
        # XX this can surely be vectorized but I am feeling lazy:
        for i in xrange(x.shape[0]):
            self.current_n += 1
            delta = x[i, :] - self.current_mean
            self.current_mean += delta / self.current_n
            self.current_M2 += delta * (x[i, :] - self.current_mean)

    def memorize_finish(self):
        pass

    def transform(self, x, center=True, rescale=True, ddof=0):
        x = atleast_2d_column_default(x)
        if np.issubdtype(x.dtype, np.integer):
            x = np.array(x, dtype=float)
        else:
            x = np.array(x)
        if center:
            x -= self.current_mean
        if rescale:
            x /= np.sqrt(self.current_M2 / (self.current_n - ddof))
        return x

builtin_stateful_transforms["standardize"] = Standardize
# R compatibility:
builtin_stateful_transforms["scale"] = Standardize

def test_Standardize():
    _test_stateful(Standardize, [1, -1], [1, -1])
    _test_stateful(Standardize, [12, 10], [1, -1])
    _test_stateful(Standardize,
                   [12, 11, 10],
                   [np.sqrt(3./2), 0, -np.sqrt(3./2)])

    _test_stateful(Standardize,
                   [12.0, 11.0, 10.0],
                   [np.sqrt(3./2), 0, -np.sqrt(3./2)])

    _test_stateful(Standardize,
                   [12.0+0j, 11.0, 10.0],
                   [np.sqrt(3./2), 0, -np.sqrt(3./2)])

    _test_stateful(Standardize, [1, -1], [np.sqrt(2)/2, -np.sqrt(2)/2],
                   ddof=1)

    _test_stateful(Standardize,
                   range(20),
                   list((np.arange(20) - 9.5) / 5.7662812973353983),
                   ddof=0)
    _test_stateful(Standardize,
                   range(20),
                   list((np.arange(20) - 9.5) / 5.9160797830996161),
                   ddof=1)
    _test_stateful(Standardize,
                   range(20),
                   list((np.arange(20) - 9.5)),
                   rescale=False, ddof=1)
    _test_stateful(Standardize,
                   range(20),
                   list(np.arange(20) / 5.9160797830996161),
                   center=False, ddof=1)
    _test_stateful(Standardize,
                   range(20),
                   range(20),
                   center=False, rescale=False, ddof=1)
    
