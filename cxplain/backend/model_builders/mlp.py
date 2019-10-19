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
from tensorflow.python.keras.layers import Dense, BatchNormalization, Dropout
from cxplain.backend.model_builders.base_model_builder import BaseModelBuilder


class MLPModelBuilder(BaseModelBuilder):
    def __init__(self, num_layers=2, num_units=64, activation="relu", with_bn=False, p_dropout=0.0,
                 callbacks=list([]), early_stopping_patience=12,
                 batch_size=64, num_epochs=100, validation_fraction=0.1, shuffle=True,
                 learning_rate=0.0001, optimizer=None, verbose=0):
        super(MLPModelBuilder, self).__init__(callbacks, early_stopping_patience, batch_size, num_epochs,
                                              validation_fraction, shuffle, learning_rate, optimizer, verbose)
        self.num_layers = num_layers
        self.num_units = num_units
        self.activation = activation
        self.with_bn = with_bn
        self.p_dropout = p_dropout

    def build(self, input_layer):
        last_layer = input_layer
        for _ in range(self.num_layers):
            last_layer = Dense(self.num_units, activation=self.activation)(last_layer)
            if self.with_bn:
                last_layer = BatchNormalization()(last_layer)
            if not np.isclose(self.p_dropout, 0):
                last_layer = Dropout(self.p_dropout)(last_layer)
        return last_layer
