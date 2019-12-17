"""
Copyright (C) 2019  Patrick Schwab, ETH Zurich

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
 of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
import numpy as np
import collections


class Validation(object):
    @staticmethod
    def check_dataset(X, y):
        if X is None or y is None:
            raise ValueError("Dataset __X__ or __y__ are None.")

        if len(X) != len(y):
            raise ValueError("__X__ and __y__ must be of the same length. "
                             "Your __X__ length was = " + str(len(X)) +
                             ". Your __y__ length was = " + str(len(y)))

    @staticmethod
    def flatten(arr):
        for x in arr:
            if isinstance(x, collections.Iterable) and not isinstance(x, (str, bytes)):
                for sub_x in Validation.flatten(x):
                    yield sub_x
            else:
                yield x

    @staticmethod
    def check_downsample_factors_at_initialisation(downsample_factors):
        original_downsample_factors = downsample_factors
        if not isinstance(downsample_factors, collections.Sequence):
            downsample_factors = (downsample_factors,)

        for factor in downsample_factors:
            if not isinstance(factor, int):
                raise ValueError("__downsample_factors__ must be an integer or a tuple of integers. "
                                 "Found: " + str(original_downsample_factors) +
                                 ", offending factor: " + str(factor) + ".")

            if factor < 1:
                raise ValueError("__downsample_factors__ must be a positive integer or a tuple of positive integers. "
                                 "Found: " + str(original_downsample_factors) +
                                 ", offending factor: " + str(factor) + ".")

    @staticmethod
    def check_is_positive_integer_greaterequals_1(value, var_name="value"):
        if not isinstance(value, int):
            raise ValueError("__{}__ must be an integer. Found: {}.".format(var_name, value))

        if value < 1:
            raise ValueError("__{}__ must be greater or equal to 1. Found: {}.".format(var_name, value))

    @staticmethod
    def check_is_fraction(value, var_name="value"):
        if not (value <= 1.0 and value >= 0.0):
            raise ValueError("__{}__ must be between 0.0 and 1.0 (inclusive). Found: {}.".format(var_name, value))

    @staticmethod
    def get_at_level(target_len, outer_x, depth):
        if depth == 0:
            return len(outer_x) == target_len
        else:
            return list(map(lambda child: Validation.get_at_level(target_len, child, depth - 1), outer_x))

    @staticmethod
    def get_input_dimension(X):
        if X is None:
            raise ValueError("Dataset __X__ must not be None.")

        if len(X) == 0:
            raise ValueError("Dataset __X__ must contain more than 0 samples.")

        num_samples = len(X)
        level = 1
        probe = X[0]
        input_dim = []

        probe_dims = np.array(probe).shape
        if len(probe_dims) == 0:
            raise ValueError("Dataset __X__ must have at least two dimensions, e.g. (num_samples, num_features). "
                             "If your dataset has only one feature use np.expand_dims(x, axis=-1) to set its shape "
                             "to (num_samples, 1).")

        while len(probe_dims) > 1 and \
              (probe_dims[0] == 0 or isinstance(probe[0], collections.Sequence) or isinstance(probe[0], np.ndarray)):
            is_fixed_size = np.all(list(Validation.flatten(Validation.get_at_level(len(probe), X, level))))
            if is_fixed_size:
                input_dim += [len(probe)]
            else:
                input_dim += [None]
            level += 1
            probe = probe[0]
            probe_dims = np.array(probe).shape

        input_dim += [probe_dims[0]]
        input_dim = tuple(input_dim)
        return num_samples, input_dim

    @staticmethod
    def get_output_dimension(y):
        if y is None or len(y) == 0:
            raise ValueError("y must contain more than 0 samples.")

        # Must be (num_samples,) for regression and (num_samples, num_classes) for classification
        # NOTE: binary classification must be (num_samples, 2) to differentiate from regression.
        is_array = isinstance(y[0], collections.Sequence) or isinstance(y[0], np.ndarray)
        output_dim = len(y[0]) if is_array else 1
        return output_dim

    @staticmethod
    def get_full_input_shape(num_samples, input_dim):
        if not isinstance(input_dim, collections.Sequence):
            input_dim = (input_dim,)

        full_shape = (num_samples,) + input_dim
        return full_shape

    @staticmethod
    def get_attribution_shape(X):
        # (n, p)          -> (n, p)
        # (n, t, p)       -> (n, t, 1)
        # (n, x, y, c)    -> (n, x, y, 1)
        # (n, x, y, z, c) -> (n, x, y, z, 1)
        num_samples, input_dim = Validation.get_input_dimension(X)
        ret_val = Validation.get_attribution_shape_from_input_shape(num_samples, input_dim)
        return ret_val

    @staticmethod
    def get_attribution_shape_from_input_shape(num_samples, input_dim):
        target_shape = Validation.get_full_input_shape(num_samples, input_dim)

        if len(input_dim) >= 2:
            # Summarise attribution over channels.
            target_shape = target_shape[:-1] + (1,)

        return target_shape

    @staticmethod
    def is_variable_length(x):
        is_var_len = (isinstance(x, np.ndarray) and len(x) > 0 and len(x.shape) == 1 and len(x[0]) > 1) or \
                     (isinstance(x, list) and len(x) > 0 and
                      not np.allclose(list(map(len, x)), len(x[0]), rtol=0, atol=0))
        return is_var_len
