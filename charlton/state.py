# This file is part of Patsy
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

# BETTER WAY: always run the first row of data through the builder alone, and
# check that it gives the same output row as when running the whole block of
# data through at once. This gives us the same information, but it's robust
# against people writing their own centering functions.

# QUESTION: right now we refuse to even fit a model that contains a
# my_transform(x)-style function. Maybe we should allow it to be fit (with a
# warning), and only disallow making predictions with it? Need to revisit this
# question once it's clearer what exactly our public API will look like,
# because right now I'm not sure how to tell whether we are being called for
# fitting versus being called for prediction.

import numpy as np
from patsy.util import (atleast_2d_column_default,
                           asarray_or_pandas, pandas_friendly_reshape,
                           wide_dtype_for)
from patsy.compat import wraps

# These are made available in the patsy.* namespace
__all__ = ["stateful_transform",
           "center", "standardize", "scale",
           ]

def stateful_transform(class_):
    """Create a stateful transform callable object from a class that fulfills
    the :ref:`stateful transform protocol <stateful-transform-protocol>`.
    """
    @wraps(class_)
    def stateful_transform_wrapper(*args, **kwargs):
        transform = class_()
        transform.memorize_chunk(*args, **kwargs)
        transform.memorize_finish()
        return transform.transform(*args, **kwargs)
    stateful_transform_wrapper.__patsy_stateful_transform__ = class_
    return stateful_transform_wrapper

# class NonIncrementalStatefulTransform(object):
#     def __init__(self):
#         self._data = []
#    
#     def memorize_chunk(self, input_data, *args, **kwargs):
#         self._data.append(input_data)
#         self._args = _args
#         self._kwargs = kwargs
#
#     def memorize_finish(self):
#         all_data = np.row_stack(self._data)
#         args = self._args
#         kwargs = self._kwargs
#         del self._data
#         del self._args
#         del self._kwargs
#         self.memorize_all(all_data, *args, **kwargs)
#
#     def memorize_all(self, input_data, *args, **kwargs):
#         raise NotImplementedError
#
#     def transform(self, input_data, *args, **kwargs):
#         raise NotImplementedError
#
# class QuantileEstimatingTransform(NonIncrementalStatefulTransform):
#     def memorize_all(self, input_data, *args, **kwargs):
        
def _test_stateful(cls, input, output, *args, **kwargs):
    input = np.asarray(input)
    output = np.asarray(output)
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
    from patsy.util import have_pandas
    if have_pandas:
        import pandas
        pandas_type = (pandas.Series, pandas.DataFrame)
        pandas_index = np.linspace(0, 1, num=len(input))
        output_series = pandas.Series(output, index=pandas_index)
        input_2d = np.column_stack((input, input[::-1]))
        output_2d = np.column_stack((output, output[::-1]))
        output_dataframe = pandas.DataFrame(output_2d, index=pandas_index)
        test_cases += [
            # Series input, one chunk
            ([pandas.Series(input, index=pandas_index)], output_series),
            # Series input, many chunks
            ([pandas.Series([x], index=[idx])
              for (x, idx) in zip(input, pandas_index)],
             output_series),
            # DataFrame input, one chunk
            ([pandas.DataFrame(input_2d, index=pandas_index)], output_dataframe),
            # DataFrame input, many chunks
            ([pandas.DataFrame([input_2d[i, :]], index=[pandas_index[i]])
              for i in xrange(len(input))],
             output_dataframe),
            ]
    for input_obj, output_obj in test_cases:
        print input_obj
        t = cls()
        for input_chunk in input_obj:
            t.memorize_chunk(input_chunk, *args, **kwargs)
        t.memorize_finish()
        all_outputs = []
        for input_chunk in input_obj:
            output_chunk = t.transform(input_chunk, *args, **kwargs)
            assert output_chunk.ndim == np.asarray(input_chunk).ndim
            all_outputs.append(output_chunk)
        if have_pandas and isinstance(all_outputs[0], pandas_type):
            all_output1 = pandas.concat(all_outputs)
            assert np.array_equal(all_output1.index, pandas_index)
        elif all_outputs[0].ndim == 0:
            all_output1 = np.array(all_outputs)
        elif all_outputs[0].ndim == 1:
            all_output1 = np.concatenate(all_outputs)
        else:
            all_output1 = np.row_stack(all_outputs)
        assert all_output1.shape[0] == len(input)
        # output_obj_reshaped = np.asarray(output_obj).reshape(all_output1.shape)
        # assert np.allclose(all_output1, output_obj_reshaped)
        assert np.allclose(all_output1, output_obj)
        if np.asarray(input_obj[0]).ndim == 0:
            all_input = np.array(input_obj)
        elif have_pandas and isinstance(input_obj[0], pandas_type):
            # handles both Series and DataFrames
            all_input = pandas.concat(input_obj)
        elif np.asarray(input_obj[0]).ndim == 1:
            # Don't use row_stack, because that would turn this into a 1xn
            # matrix:
            all_input = np.concatenate(input_obj)
        else:
            all_input = np.row_stack(input_obj)
        all_output2 = t.transform(all_input, *args, **kwargs)
        if have_pandas and isinstance(input_obj[0], pandas_type):
            assert np.array_equal(all_output2.index, pandas_index)
        assert all_output2.ndim == all_input.ndim
        assert np.allclose(all_output2, output_obj)
    
class Center(object):
    """center(x)

    A stateful transform that centers input data, i.e., subtracts the mean.

    If input has multiple columns, centers each column separately.

    Equivalent to ``standardize(x, rescale=False)``
    """
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
        x = asarray_or_pandas(x)
        # This doesn't copy data unless our input is a DataFrame that has
        # heterogenous types. And in that case we're going to be munging the
        # types anyway, so copying isn't a big deal.
        x_arr = np.asarray(x)
        if np.issubdtype(x_arr.dtype, np.integer):
            dt = float
        else:
            dt = x_arr.dtype
        mean_val = np.asarray(self._sum / self._count, dtype=dt)
        centered = atleast_2d_column_default(x, preserve_pandas=True) - mean_val
        return pandas_friendly_reshape(centered, x.shape)

center = stateful_transform(Center)

def test_Center():
    _test_stateful(Center, [1, 2, 3], [-1, 0, 1])
    _test_stateful(Center, [1, 2, 1, 2], [-0.5, 0.5, -0.5, 0.5])
    _test_stateful(Center,
                   [1.3, -10.1, 7.0, 12.0],
                   [-1.25, -12.65, 4.45, 9.45])


def test_stateful_transform_wrapper():
    assert np.allclose(center([1, 2, 3]), [-1, 0, 1])
    assert np.allclose(center([1, 2, 1, 2]), [-0.5, 0.5, -0.5, 0.5])
    assert center([1.0, 2.0, 3.0]).dtype == np.dtype(float)
    assert (center(np.array([1.0, 2.0, 3.0], dtype=np.float32)).dtype
            == np.dtype(np.float32))
    assert center([1, 2, 3]).dtype == np.dtype(float)

    from patsy.util import have_pandas
    if have_pandas:
        import pandas
        s = pandas.Series([1, 2, 3], index=["a", "b", "c"])
        df = pandas.DataFrame([[1, 2], [2, 4], [3, 6]],
                              columns=["x1", "x2"],
                              index=[10, 20, 30])
        s_c = center(s)
        assert isinstance(s_c, pandas.Series)
        assert np.array_equal(s_c.index, ["a", "b", "c"])
        assert np.allclose(s_c, [-1, 0, 1])
        df_c = center(df)
        assert isinstance(df_c, pandas.DataFrame)
        assert np.array_equal(df_c.index, [10, 20, 30])
        assert np.array_equal(df_c.columns, ["x1", "x2"])
        assert np.allclose(df_c, [[-1, -2], [0, 0], [1, 2]])
        

# See:
#   http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#On-line_algorithm
# or page 232 of Knuth vol. 3 (3rd ed.).
class Standardize(object):
    """standardize(x, center=True, rescale=True, ddof=0)

    A stateful transform that standardizes input data, i.e. it subtracts the
    mean and divides by the sample standard deviation.

    Either centering or rescaling or both can be disabled by use of keyword
    arguments. The `ddof` argument controls the delta degrees of freedom when
    computing the standard deviation (cf. :func:`numpy.std`). The default of
    ``ddof=0`` produces the maximum likelihood estimate; use ``ddof=1`` if you
    prefer the square root of the unbiased estimate of the variance.

    If input has multiple columns, standardizes each column separately.

    .. note:: This function computes the mean and standard deviation using a
       memory-efficient online algorithm, making it suitable for use with
       large incrementally processed data-sets.
    """
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
        # XX: this forces all inputs to double-precision real, even if the
        # input is single- or extended-precision or complex. But I got all
        # tangled up in knots trying to do that without breaking something
        # else (e.g. by requiring an extra copy).
        x = asarray_or_pandas(x, copy=True, dtype=float)
        x_2d = atleast_2d_column_default(x, preserve_pandas=True)
        if center:
            x_2d -= self.current_mean
        if rescale:
            x_2d /= np.sqrt(self.current_M2 / (self.current_n - ddof))
        return pandas_friendly_reshape(x_2d, x.shape)

standardize = stateful_transform(Standardize)
# R compatibility:
scale = standardize

def test_Standardize():
    _test_stateful(Standardize, [1, -1], [1, -1])
    _test_stateful(Standardize, [12, 10], [1, -1])
    _test_stateful(Standardize,
                   [12, 11, 10],
                   [np.sqrt(3./2), 0, -np.sqrt(3./2)])

    _test_stateful(Standardize,
                   [12.0, 11.0, 10.0],
                   [np.sqrt(3./2), 0, -np.sqrt(3./2)])

    # XX: see the comment in Standardize.transform about why this doesn't
    # work:
    # _test_stateful(Standardize,
    #                [12.0+0j, 11.0+0j, 10.0],
    #                [np.sqrt(3./2)+0j, 0, -np.sqrt(3./2)])

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
    
