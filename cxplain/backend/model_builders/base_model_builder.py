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
import six
import numpy as np
import collections
import tensorflow as tf
from functools import partial
import tensorflow.keras.backend as K
from abc import ABCMeta, abstractmethod
from tensorflow.python.keras.models import Model
from cxplain.backend.validation import Validation
from cxplain.backend.causal_loss import causal_loss
from cxplain.backend.masking.masking_util import MaskingUtil
from tensorflow.python.keras.backend import resize_images, resize_volumes
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers import Input, Dense, Flatten, Lambda, Reshape


@six.add_metaclass(ABCMeta)
class BaseModelBuilder(object):
    def __init__(self, callbacks=list([]), early_stopping_patience=12,
                 batch_size=64, num_epochs=100, validation_fraction=0.1, shuffle=True,
                 learning_rate=0.0001, optimizer=None, verbose=0):
        self.batch_size = batch_size
        Validation.check_is_positive_integer_greaterequals_1(num_epochs, var_name="num_epochs")
        self.num_epochs = num_epochs
        Validation.check_is_fraction(validation_fraction, var_name="validation_fraction")
        self.validation_fraction = validation_fraction
        self.shuffle = shuffle
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.verbose = verbose
        self.callbacks = callbacks
        self.early_stopping_patience = early_stopping_patience

    @abstractmethod
    def build(self, input_layer):
        """
        :param input_layer: input layer of the explanation model; shape=(None, num_input_features)
        :return: last layer of the explanation model; will be flattened
        """
        raise NotImplementedError()

    def build_explanation_model(self, input_dim, output_dim, loss, downsample_factors=(1,)):
        num_indices, num_channels, steps, downsampling_factor =\
            MaskingUtil.get_input_constants(input_dim, downsample_factors)

        if downsampling_factor != 1 and num_indices is None:
            raise ValueError("Attribution downsampling is not supported for variable length inputs. "
                             "Please pad your data samples to the same size to use downsampling.")

        input_shape = (input_dim,) if not isinstance(input_dim, collections.Sequence) else input_dim
        input_layer = Input(shape=input_shape)
        last_layer = self.build(input_layer)

        if num_indices is None:
            last_layer = Dense(1, activation="linear")(last_layer)
            last_layer = Flatten()(last_layer)  # None * None outputs
            last_layer = Lambda(K.softmax, output_shape=K.int_shape(last_layer))(last_layer)
        else:
            last_layer = Flatten()(last_layer)
            last_layer = Dense(num_indices, activation="softmax")(last_layer)

        # Prepare extra inputs for causal loss.
        all_auxiliary_outputs = Input(shape=(output_dim,), name="all")
        all_but_one_auxiliary_outputs_input = Input(shape=(num_indices, output_dim), name="all_but_one")

        if num_indices is not None:
            all_but_one_auxiliary_outputs = Lambda(lambda x: tf.unstack(x, axis=1))(all_but_one_auxiliary_outputs_input)
            if K.int_shape(all_but_one_auxiliary_outputs_input)[1] == 1:
                all_but_one_auxiliary_outputs = [all_but_one_auxiliary_outputs]
        else:
            all_but_one_auxiliary_outputs = all_but_one_auxiliary_outputs_input

        causal_loss_fun = partial(causal_loss,
                                  attention_weights=last_layer,
                                  auxiliary_outputs=all_auxiliary_outputs,
                                  all_but_one_auxiliary_outputs=all_but_one_auxiliary_outputs,
                                  loss_function=loss)
        causal_loss_fun.__name__ = "causal_loss"

        if downsampling_factor != 1:
            last_layer = Reshape(tuple(steps) + (1,))(last_layer)

            if len(steps) == 1:
                # Add a dummy dimension to enable usage of __resize_images__.
                last_layer = Reshape(tuple(steps) + (1, 1))(last_layer)
                last_layer = Lambda(lambda x: resize_images(x,
                                                            height_factor=downsample_factors[0],
                                                            width_factor=1,
                                                            data_format="channels_last"))(last_layer)
            elif len(steps) == 2:
                last_layer = Lambda(lambda x: resize_images(x,
                                                            height_factor=downsample_factors[0],
                                                            width_factor=downsample_factors[1],
                                                            data_format="channels_last"))(last_layer)
            elif len(steps) == 3:
                last_layer = Lambda(lambda x: resize_volumes(x,
                                                             depth_factor=downsample_factors[0],
                                                             height_factor=downsample_factors[1],
                                                             width_factor=downsample_factors[2],
                                                             data_format="channels_last"))(last_layer)
            else:
                raise ValueError("Attribution maps of larger dimensionality than 3D data are not currently supported. "
                                 "Requested output dim was: {}.".format(len(steps)))

            attribution_shape = Validation.get_attribution_shape_from_input_shape(num_samples=1,
                                                                                  input_dim=input_dim)[1:]
            collapsed_attribution_shape = (int(np.prod(attribution_shape)),)
            last_layer = Reshape(collapsed_attribution_shape)(last_layer)

            # Re-normalise to sum = 1 after resizing (sum = __downsampling_factor__ after resizing).
            last_layer = Lambda(lambda x: x / float(downsampling_factor))(last_layer)

        # We must connect all inputs to the output to bypass a bug in model saving in tf < 1.15.0rc0.
        # For easier handling when calling .fit(), we transform all outputs to be the same shape.
        # See https://github.com/tensorflow/tensorflow/pull/30244
        all_but_one_same_shape_output_layer = Lambda(lambda x: x[:, 0, :])(all_but_one_auxiliary_outputs_input)

        model = Model(inputs=[input_layer,
                              all_auxiliary_outputs,
                              all_but_one_auxiliary_outputs_input],
                      outputs=[last_layer, all_auxiliary_outputs, all_but_one_same_shape_output_layer])
        model = self.compile_model(model, main_losses=[causal_loss_fun, "mse", "mse"], loss_weights=[1.0]*3,
                                   learning_rate=self.learning_rate, optimizer=self.optimizer)

        prediction_model = Model(input_layer, last_layer)
        return model, prediction_model

    def compile_model(self, model, learning_rate=0.001, optimizer=None, loss_weights=list([1.0]),
                      main_losses=list(["mse"]), metrics={}, gradient_clipping_threshold=100):
        losses = main_losses

        if optimizer == "rmsprop":
            opt = tf.compat.v1.train.RMSPropOptimizer(learning_rate=learning_rate)
        elif optimizer == "sgd":
            opt = tf.compat.v1.train.MomentumOptimizer(learning_rate=learning_rate, use_nesterov=True, momentum=0.9)
        else:
            opt = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

        model.compile(loss=losses,
                      loss_weights=loss_weights,
                      optimizer=opt,
                      metrics=metrics)
        return model

    def fit(self, model, precomputed, y, model_filepath):
        callbacks = [
            ModelCheckpoint(filepath=model_filepath,
                            save_best_only=True,
                            save_weights_only=True),
            EarlyStopping(patience=self.early_stopping_patience)
        ] + self.callbacks

        # Perform an initial model save so that one version of the model is always saved
        # even if model fitting or check-pointing fails.
        model.save_weights(model_filepath)

        history = model.fit(x=precomputed,
                            # We must feed two extra outputs due to a bug in TensorFlow < 1.15.0rc0 that would not
                            # allow saving models without connecting all inputs to output nodes.
                            # See https://github.com/tensorflow/tensorflow/pull/30244
                            y=[y, y, y],
                            batch_size=self.batch_size,
                            shuffle=self.shuffle,
                            validation_split=self.validation_fraction,
                            epochs=self.num_epochs,
                            verbose=self.verbose,
                            callbacks=callbacks)

        # Restore to best encountered model.
        model.load_weights(model_filepath)
        return history

    def evaluate(self, model, X, y, sample_weight=None):
        # We must feed two extra outputs due to a bug in TensorFlow < 1.15.0rc0 that would not
        # allow saving models without connecting all inputs to output nodes.
        # See https://github.com/tensorflow/tensorflow/pull/30244
        return_value = model.evaluate(x=X,
                                      y=[y, y, y],
                                      sample_weight=sample_weight,
                                      verbose=self.verbose)
        return return_value

    def predict(self, model, X):
        return_value = model.predict(X)
        return return_value
