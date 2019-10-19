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
from __future__ import print_function

import tensorflow as tf
from cxplain.backend.tf_math_interface import TensorflowInterface


def safe_evaluate_custom_loss_function(loss_function, y_true, y_pred, math_ops):
    ret_val = loss_function(y_true, y_pred)

    if len(math_ops.int_shape(ret_val)) == 2 and math_ops.int_shape(ret_val)[-1] == 1:
        # Value returned by __loss_function__ should be of shape (num_samples,)
        ret_val = math_ops.squeeze(ret_val, axis=-1)

    if len(math_ops.int_shape(ret_val)) == 0 or (
            math_ops.int_shape(ret_val)[0] is not None and
            math_ops.int_shape(y_true)[0] is not None and
            math_ops.int_shape(ret_val)[0] != math_ops.int_shape(y_true)[0]
    ):
        raise ValueError("Your custom loss function must return a scalar for each pair of y_pred and y_true values. "
                         "Please ensure that your loss function does not, for example, average over all samples, "
                         "as it would then return only one scalar value "
                         "independently of the number of samples passed.")

    return ret_val


def get_delta_errors(y_true, all_but_one_auxiliary_outputs, error_with_all_features,
                     loss_function, log_transform, math_ops):
    return get_delta_errors_fixed_size(y_true, all_but_one_auxiliary_outputs, error_with_all_features,
                                       loss_function, log_transform, math_ops)


def get_delta_errors_fixed_size(y_true, all_but_one_auxiliary_outputs, error_with_all_features,
                                loss_function, log_transform, math_ops):
    delta_errors = []
    for all_but_one_auxiliary_output in all_but_one_auxiliary_outputs:
        error_without_one_feature = safe_evaluate_custom_loss_function(
            loss_function, y_true, all_but_one_auxiliary_output, math_ops
        )

        # The error without the feature is an indicator as to how potent the left-out feature is as a predictor.
        delta_error = math_ops.maximum(error_without_one_feature - error_with_all_features, math_ops.epsilon())
        if log_transform:
            delta_error = math_ops.log(1 + delta_error)
        delta_errors.append(delta_error)
    delta_errors = math_ops.stack(delta_errors, axis=-1)
    return delta_errors


def calculate_delta_errors(y_true, auxiliary_outputs, all_but_one_auxiliary_outputs,
                           loss_function, log_transform=False, math_ops=TensorflowInterface):
    error_with_all_features = safe_evaluate_custom_loss_function(loss_function, y_true, auxiliary_outputs, math_ops)

    delta_errors = get_delta_errors(y_true, all_but_one_auxiliary_outputs, error_with_all_features,
                                    loss_function, log_transform, math_ops)

    shape = math_ops.int_shape(delta_errors)
    if shape is not None and len(shape) > 2:
        delta_errors = math_ops.squeeze(delta_errors, axis=-2)
    delta_errors /= (math_ops.sum(delta_errors, axis=-1, keepdims=True))

    # Ensure correct format.
    delta_errors = math_ops.clip(delta_errors, math_ops.epsilon(), 1.0)

    # NOTE: Without stop_gradient back-propagation would attempt to optimise the error_variance
    # instead of/in addition to the distance between attention weights and Granger-causal attributions,
    # which is not desired.
    delta_errors = math_ops.stop_gradient(delta_errors)
    return delta_errors


def causal_loss(y_true, y_pred, attention_weights, auxiliary_outputs, all_but_one_auxiliary_outputs,
                loss_function, math_ops=TensorflowInterface):
    delta_errors = calculate_delta_errors(y_true,
                                          auxiliary_outputs,
                                          all_but_one_auxiliary_outputs,
                                          loss_function,
                                          math_ops=math_ops)

    # Ensure correct format.
    attention_weights = math_ops.clip(attention_weights, math_ops.epsilon(), 1.0)

    if len(attention_weights.shape) == 3:
        attention_weights = math_ops.squeeze(attention_weights, axis=-1)

    return math_ops.mean(math_ops.kullback_leibler_divergence(delta_errors, attention_weights))
