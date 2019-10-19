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
import tensorflow as tf


class TensorflowInterface(object):
    @staticmethod
    def copy(val):
        return tf.identity(val)

    @staticmethod
    def shape(val):
        return tf.shape(val)

    @staticmethod
    def int_shape(val):
        return tf.keras.backend.int_shape(val)

    @staticmethod
    def prod(val):
        return tf.reduce_prod(val)

    @staticmethod
    def reshape(val, shape):
        return tf.reshape(val, shape)

    @staticmethod
    def arange(val):
        return tf.range(val)

    @staticmethod
    def pad(val, pad_width, mode, constant_values):
        return tf.pad(val, pad_width, mode, constant_values=constant_values)

    @staticmethod
    def as_int(val):
        return tf.cast(val, tf.int32)

    @staticmethod
    def expand_dims(val, axis):
        return tf.expand_dims(val, axis)

    @staticmethod
    def sum(val, axis, keepdims=None):
        return tf.reduce_sum(val, axis, keepdims=keepdims)

    @staticmethod
    def abs(val):
        return tf.abs(val)

    @staticmethod
    def multiply(val1, val2):
        return tf.multiply(val1, val2)

    @staticmethod
    def cast(val, type):
        return tf.cast(val, type)

    @staticmethod
    def cumsum(val, axis, reverse=False):
        return tf.cumsum(val, axis=axis, reverse=reverse)

    @staticmethod
    def argmax(val, axis):
        return tf.argmax(val, axis)

    @staticmethod
    def one_hot(indices, depth, on_value=1., off_value=0., dtype=float, axis=-1):
        return tf.one_hot(indices, depth=depth, on_value=on_value, off_value=off_value, dtype=dtype, axis=axis)

    @staticmethod
    def clip(val, low, high):
        clipped = tf.clip_by_value(val, low, high)
        return clipped

    @staticmethod
    def greater(val1, val2):
        clipped = tf.greater(val1, val2)
        return clipped

    @staticmethod
    def sign(val):
        result = tf.sign(val)
        return result

    @staticmethod
    def maximum(val1, val2):
        return tf.maximum(val1, val2)

    @staticmethod
    def epsilon():
        return tf.keras.backend.epsilon()

    @staticmethod
    def log(val):
        return tf.math.log(val)

    @staticmethod
    def stack(val, axis=None):
        return tf.stack(val, axis=axis)

    @staticmethod
    def squeeze(val, axis=None):
        return tf.keras.backend.squeeze(val, axis=axis)

    @staticmethod
    def stop_gradient(val):
        return tf.stop_gradient(val)

    @staticmethod
    def mean(val, axis=None):
        return tf.reduce_mean(val, axis=axis)

    @staticmethod
    def expand_dims(val, axis=None):
        return tf.expand_dims(val, axis=axis)

    @staticmethod
    def kullback_leibler_divergence(y_true, y_pred):
        return tf.keras.losses.kullback_leibler_divergence(y_true, y_pred)

    @staticmethod
    def constant(value):
        return tf.constant(value)
