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


class NumpyInterface(object):
    @staticmethod
    def copy(val):
        return np.copy(val)

    @staticmethod
    def shape(val):
        return val.shape

    @staticmethod
    def int_shape(val):
        return NumpyInterface.shape(val)

    @staticmethod
    def prod(val):
        return np.prod(val)

    @staticmethod
    def reshape(val, shape):
        return val.reshape(shape)

    @staticmethod
    def arange(val):
        return np.arange(val)

    @staticmethod
    def pad(val, pad_width, mode, constant_values):
        return np.pad(val, pad_width, mode, constant_values=constant_values)

    @staticmethod
    def as_int(val):
        if isinstance(val, np.ndarray):
            return val.astype(int)
        else:
            return int(val)

    @staticmethod
    def expand_dims(val, axis):
        return np.expand_dims(val, axis=axis)

    @staticmethod
    def sum(val, axis, keepdims=np._NoValue):
        return np.sum(val, axis=axis, keepdims=keepdims)

    @staticmethod
    def abs(val):
        return np.abs(val)

    @staticmethod
    def multiply(val1, val2):
        return np.multiply(val1, val2)

    @staticmethod
    def cast(val, type):
        return val.astype(type)

    @staticmethod
    def cumsum(val, axis, reverse=False):
        if reverse:
            return np.cumsum(val[::-1], axis=axis)[::-1]
        else:
            return np.cumsum(val, axis=axis)

    @staticmethod
    def argmax(val, axis):
        return np.argmax(val, axis)

    @staticmethod
    def one_hot(a, depth, on_value=1., off_value=0., dtype=float, axis=-1):
        b = np.ones((a.size, depth), dtype=dtype) * off_value
        b[np.arange(a.size), a.reshape((-1,))] = on_value
        b = np.reshape(b, a.shape[:-1] + (depth,))
        return b

    @staticmethod
    def clip(val, low, high):
        clipped = np.clip(val, low, high)
        return clipped

    @staticmethod
    def greater(val1, val2):
        clipped = np.greater(val1, val2)
        return clipped

    @staticmethod
    def sign(val):
        result = np.sign(val)
        return result

    @staticmethod
    def maximum(val1, val2):
        return np.maximum(val1, val2)

    @staticmethod
    def epsilon():
        return np.finfo(float).eps

    @staticmethod
    def log(val):
        return np.log(val)

    @staticmethod
    def stack(val, axis=None):
        return np.stack(val, axis=axis)

    @staticmethod
    def squeeze(val, axis=None):
        return np.squeeze(val, axis=axis)

    @staticmethod
    def stop_gradient(val):
        return val

    @staticmethod
    def mean(val, axis=None):
        return np.mean(val, axis=axis)

    @staticmethod
    def expand_dims(val, axis=None):
        return np.expand_dims(val, axis=axis)

    @staticmethod
    def kullback_leibler_divergence(y_true, y_pred):
        y_true = NumpyInterface.clip(y_true, NumpyInterface.epsilon(), 1.0)
        y_pred = NumpyInterface.clip(y_pred, NumpyInterface.epsilon(), 1.0)
        ret_val = NumpyInterface.sum(y_true * NumpyInterface.log(y_true / y_pred), axis=-1)
        return ret_val

    @staticmethod
    def binary_crossentropy(y_true, y_pred):
        y_true = y_true[..., -1]
        y_pred = y_pred[..., -1]
        values = -(y_true * np.log(y_pred + np.finfo(float).eps) + (1-y_true)*np.log(1 - y_pred + np.finfo(float).eps))
        return values

    @staticmethod
    def constant(value):
        return value
