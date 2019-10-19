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

import unittest
from cxplain.util.test_util import TestUtil
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble.forest import RandomForestRegressor
from cxplain import MLPModelBuilder, ZeroMasking, CXPlain
from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.python.keras.losses import categorical_crossentropy


class TestUncertainty(unittest.TestCase):
    def test_boston_housing_valid(self):
        (x_train, y_train), (x_test, y_test) = TestUtil.get_boston_housing()
        explained_model = RandomForestRegressor(n_estimators=64, max_depth=5, random_state=1)
        explained_model.fit(x_train, y_train)

        model_builder = MLPModelBuilder(num_layers=2, num_units=32, activation="relu", p_dropout=0.2, verbose=0,
                                        batch_size=32, learning_rate=0.001, num_epochs=3, early_stopping_patience=128)
        masking_operation = ZeroMasking()
        loss = mean_squared_error

        for num_models in [2, 5, 10]:
            explainer = CXPlain(explained_model, model_builder, masking_operation, loss, num_models=num_models)

            explainer.fit(x_train, y_train)
            eval_score = explainer.score(x_test, y_test)
            train_score = explainer.get_last_fit_score()
            median, confidence = explainer.predict(x_test, confidence_level=0.95)

            self.assertTrue(median.shape == x_test.shape)
            self.assertTrue(confidence.shape == x_test.shape + (2,))

            # Flatten predictions for iteration below.
            median = median.reshape((len(x_test), -1))
            confidence = confidence.reshape((len(x_test), -1, 2))

            for sample_idx in range(len(x_test)):
                for feature_idx in range(len(x_test[sample_idx])):
                    self.assertTrue(confidence[sample_idx][feature_idx][0] <=
                                    median[sample_idx][feature_idx] <=
                                    confidence[sample_idx][feature_idx][1])
                    self.assertTrue(confidence[sample_idx][feature_idx][0] >= 0)
                    self.assertTrue(confidence[sample_idx][feature_idx][1] >= 0)

    def test_boston_housing_confidence_level_invalid(self):
        (x_train, y_train), (x_test, y_test) = TestUtil.get_boston_housing()
        explained_model = RandomForestRegressor(n_estimators=64, max_depth=5, random_state=1)
        explained_model.fit(x_train, y_train)

        model_builder = MLPModelBuilder(num_layers=2, num_units=32, activation="relu", p_dropout=0.2, verbose=0,
                                        batch_size=32, learning_rate=0.001, num_epochs=3, early_stopping_patience=128)
        masking_operation = ZeroMasking()
        loss = mean_squared_error

        num_models = 2
        explainer = CXPlain(explained_model, model_builder, masking_operation, loss, num_models=num_models)

        explainer.fit(x_train, y_train)

        invalid_confidence_levels = [1.01, -0.5, -0.01]

        for confidence_level in invalid_confidence_levels:
            with self.assertRaises(ValueError):
                explainer.predict(x_test, confidence_level=confidence_level)

    def test_mnist_valid(self):
        num_subsamples = 100
        (x_train, y_train), (x_test, y_test) = TestUtil.get_mnist(flattened=False, num_subsamples=num_subsamples)

        explained_model = MLPClassifier(solver='lbfgs', alpha=1e-5,
                                        hidden_layer_sizes=(64, 32), random_state=1)
        explained_model.fit(x_train.reshape((len(x_train), -1)), y_train)

        model_builder = MLPModelBuilder(num_layers=2, num_units=64, activation="relu", p_dropout=0.2, verbose=0,
                                        batch_size=256, learning_rate=0.001, num_epochs=3,
                                        early_stopping_patience=128)
        masking_operation = ZeroMasking()
        loss = categorical_crossentropy

        downsample_factors = [(2, 2), (4, 4), (4, 7), (7, 4), (7, 7)]
        for downsample_factor in downsample_factors:
            explainer = CXPlain(explained_model, model_builder, masking_operation, loss, num_models=2,
                                downsample_factors=downsample_factor, flatten_for_explained_model=True)

            explainer.fit(x_train, y_train)
            eval_score = explainer.score(x_test, y_test)
            train_score = explainer.get_last_fit_score()
            median, confidence = explainer.predict(x_test, confidence_level=0.95)

            self.assertTrue(median.shape == x_test.shape)
            self.assertTrue(confidence.shape == x_test.shape[:-1] + (2,))

            # Flatten predictions for iteration below.
            median = median.reshape((len(x_test), -1))
            confidence = confidence.reshape((len(x_test), -1, 2))

            for sample_idx in range(len(x_test)):
                for feature_idx in range(len(x_test[sample_idx])):
                    self.assertTrue(confidence[sample_idx][feature_idx][0] <=
                                    median[sample_idx][feature_idx] <=
                                    confidence[sample_idx][feature_idx][1])
                    self.assertTrue(confidence[sample_idx][feature_idx][0] >= 0)
                    self.assertTrue(confidence[sample_idx][feature_idx][1] >= 0)

    def test_mnist_confidence_levels_valid(self):
        num_subsamples = 100
        (x_train, y_train), (x_test, y_test) = TestUtil.get_mnist(flattened=False, num_subsamples=num_subsamples)

        explained_model = MLPClassifier(solver='lbfgs', alpha=1e-5,
                                        hidden_layer_sizes=(64, 32), random_state=1)
        explained_model.fit(x_train.reshape((len(x_train), -1)), y_train)

        model_builder = MLPModelBuilder(num_layers=2, num_units=64, activation="relu", p_dropout=0.2, verbose=0,
                                        batch_size=256, learning_rate=0.001, num_epochs=3,
                                        early_stopping_patience=128)
        masking_operation = ZeroMasking()
        loss = categorical_crossentropy

        confidence_levels = [0.0, 1.0, 1.01, -0.01]
        for confidence_level in confidence_levels:
            downsample_factor = (2, 2)
            explainer = CXPlain(explained_model, model_builder, masking_operation, loss, num_models=2,
                                downsample_factors=downsample_factor, flatten_for_explained_model=True)

            explainer.fit(x_train, y_train)

            with self.assertRaises(ValueError):
                _ = explainer.predict(x_test, confidence_level=confidence_level)

    def test_cifar10_valid(self):
        num_subsamples = 100
        (x_train, y_train), (x_test, y_test) = TestUtil.get_cifar10(flattened=False, num_subsamples=num_subsamples)

        explained_model = MLPClassifier(solver='lbfgs', alpha=1e-5,
                                        hidden_layer_sizes=(64, 32), random_state=1)
        explained_model.fit(x_train.reshape((len(x_train), -1)), y_train)

        model_builder = MLPModelBuilder(num_layers=2, num_units=64, activation="relu", p_dropout=0.2, verbose=0,
                                        batch_size=256, learning_rate=0.001, num_epochs=3, early_stopping_patience=128)
        masking_operation = ZeroMasking()
        loss = categorical_crossentropy

        # TODO: Test uneven downsample factors.
        downsample_factors = [(2, 2), (4, 4), (4, 8), (8, 4), (8, 8)]
        for downsample_factor in downsample_factors:
            explainer = CXPlain(explained_model, model_builder, masking_operation, loss, num_models=2,
                                downsample_factors=downsample_factor, flatten_for_explained_model=True)

            explainer.fit(x_train, y_train)
            eval_score = explainer.score(x_test, y_test)
            train_score = explainer.get_last_fit_score()
            median, confidence = explainer.predict(x_test, confidence_level=0.95)

            self.assertTrue(median.shape == x_test.shape[:-1] + (1,))
            self.assertTrue(confidence.shape == x_test.shape[:-1] + (2,))

            # Flatten predictions for iteration below.
            median = median.reshape((len(x_test), -1))
            confidence = confidence.reshape((len(x_test), -1, 2))

            for sample_idx in range(len(x_test)):
                for feature_idx in range(len(x_test[sample_idx])):
                    self.assertTrue(confidence[sample_idx][feature_idx][0] <=
                                    median[sample_idx][feature_idx] <=
                                    confidence[sample_idx][feature_idx][1])
                    self.assertTrue(confidence[sample_idx][feature_idx][0] >= 0)
                    self.assertTrue(confidence[sample_idx][feature_idx][1] >= 0)


if __name__ == '__main__':
    unittest.main()
