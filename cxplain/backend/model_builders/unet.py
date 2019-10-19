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
import tensorflow.keras.backend as K
from tensorflow.python.keras.regularizers import L1L2
from cxplain.backend.model_builders.base_model_builder import BaseModelBuilder
from tensorflow.python.keras.layers import UpSampling2D, BatchNormalization, Dropout, Conv2D, Activation, \
    MaxPooling2D, concatenate, Reshape


class UNetModelBuilder(BaseModelBuilder):
    def __init__(self, downsample_factors, num_layers=2, num_units=64, activation="relu", with_bn=False,
                 p_dropout=0.0, l2_weight=0.0, with_bias=True, skip_last_dense=False,
                 callbacks=list([]), early_stopping_patience=12,
                 batch_size=64, num_epochs=100, validation_fraction=0.1, shuffle=True,
                 learning_rate=0.0001, optimizer=None, verbose=0):
        super(UNetModelBuilder, self).__init__(callbacks, early_stopping_patience, batch_size, num_epochs,
                                               validation_fraction, shuffle, learning_rate, optimizer, verbose)
        self.downsample_factors = downsample_factors
        self.num_layers = num_layers
        self.num_units = num_units
        self.activation = activation
        self.with_bn = with_bn
        self.with_bias = with_bias
        self.p_dropout = p_dropout
        self.l2_weight = l2_weight
        self.skip_last_dense = skip_last_dense
        self.num_output_channels = 1

    def build(self, input_layer):
        downsampling_factor = int(np.prod(self.downsample_factors))
        last_layer = input_layer

        input_shape = K.int_shape(last_layer)
        if len(input_shape) == 3:
            # Add channel dimension if not already present.
            last_layer = Reshape(input_shape[1:] + (1,))(last_layer)

        per_stage_before_pool = []
        for layer_idx in range(self.num_layers + 1):
            cur_num_units = int(np.rint(self.num_units*2**layer_idx))
            last_layer = Conv2D(cur_num_units, 3,
                                padding='same',
                                kernel_initializer='he_normal',
                                kernel_regularizer=L1L2(l2=self.l2_weight),
                                bias_regularizer=L1L2(l2=self.l2_weight),
                                use_bias=not self.with_bn and self.with_bias)(last_layer)
            if self.with_bn:
                last_layer = BatchNormalization(beta_regularizer=L1L2(l2=self.l2_weight),
                                                gamma_regularizer=L1L2(l2=self.l2_weight))(last_layer)
            last_layer = Activation(self.activation)(last_layer)
            last_layer = Conv2D(cur_num_units, 3,
                                padding='same',
                                kernel_initializer='he_normal',
                                kernel_regularizer=L1L2(l2=self.l2_weight),
                                bias_regularizer=L1L2(l2=self.l2_weight),
                                use_bias=not self.with_bn and self.with_bias)(last_layer)
            if self.with_bn:
                last_layer = BatchNormalization(beta_regularizer=L1L2(l2=self.l2_weight),
                                                gamma_regularizer=L1L2(l2=self.l2_weight))(last_layer)
            last_layer = Activation(self.activation)(last_layer)
            per_stage_before_pool.append(last_layer)

            if layer_idx != self.num_layers:  # Last layer doesn't require max pooling.
                last_layer = MaxPooling2D(pool_size=(2, 2))(last_layer)

            if self.p_dropout != 0.0:
                last_layer = Dropout(self.p_dropout)(last_layer)

        start_idx = 0 if downsampling_factor == 1 else int(np.log2(self.downsample_factors[0]))
        for layer_idx in reversed(range(start_idx, self.num_layers)):
            cur_num_units = int(np.rint(self.num_units*2**layer_idx))

            last_layer = UpSampling2D(size=(2, 2))(last_layer)
            last_layer = Conv2D(cur_num_units, 2,
                                padding='same', kernel_initializer='he_normal',
                                kernel_regularizer=L1L2(l2=self.l2_weight),
                                bias_regularizer=L1L2(l2=self.l2_weight),
                                use_bias=not self.with_bn and self.with_bias)(last_layer)
            if self.with_bn:
                last_layer = BatchNormalization(beta_regularizer=L1L2(l2=self.l2_weight),
                                                gamma_regularizer=L1L2(l2=self.l2_weight))(last_layer)
            last_layer = Activation(self.activation)(last_layer)
            last_layer = concatenate([per_stage_before_pool[layer_idx], last_layer], axis=3)
            last_layer = Conv2D(cur_num_units, 3,
                                padding='same', kernel_initializer='he_normal',
                                kernel_regularizer=L1L2(l2=self.l2_weight),
                                bias_regularizer=L1L2(l2=self.l2_weight),
                                use_bias=not self.with_bn and self.with_bias)(last_layer)
            if self.with_bn:
                last_layer = BatchNormalization(beta_regularizer=L1L2(l2=self.l2_weight),
                                                gamma_regularizer=L1L2(l2=self.l2_weight))(last_layer)
            last_layer = Activation(self.activation)(last_layer)
            last_layer = Conv2D(cur_num_units, 3,
                                padding='same', kernel_initializer='he_normal',
                                kernel_regularizer=L1L2(l2=self.l2_weight),
                                bias_regularizer=L1L2(l2=self.l2_weight),
                                use_bias=not self.with_bn and self.with_bias)(last_layer)
            if self.with_bn:
                last_layer = BatchNormalization(beta_regularizer=L1L2(l2=self.l2_weight),
                                                gamma_regularizer=L1L2(l2=self.l2_weight))(last_layer)
            last_layer = Activation(self.activation)(last_layer)

        last_layer = Conv2D(self.num_output_channels, 3,
                            activation="linear" if self.skip_last_dense else self.activation,
                            padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=L1L2(l2=self.l2_weight),
                            bias_regularizer=L1L2(l2=self.l2_weight),
                            use_bias=not self.with_bn and self.with_bias)(last_layer)
        return last_layer
