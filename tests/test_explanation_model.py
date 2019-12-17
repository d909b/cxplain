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

import shutil
import tempfile
import unittest
import numpy as np
from os.path import join
from sklearn.pipeline import Pipeline
from cxplain.util.test_util import TestUtil
from sklearn.neural_network import MLPClassifier
from tensorflow.python.keras.layers import Input
from cxplain.util.count_vectoriser import CountVectoriser
from sklearn.feature_extraction.text import TfidfTransformer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.ensemble.forest import RandomForestRegressor, RandomForestClassifier
from tensorflow.python.keras.losses import categorical_crossentropy, mean_squared_error, binary_crossentropy
from cxplain import CXPlain, MLPModelBuilder, ZeroMasking, WordDropMasking, RNNModelBuilder, UNetModelBuilder


class TestExplanationModel(unittest.TestCase):
    def test_boston_housing_valid(self):
        (x_train, y_train), (x_test, y_test) = TestUtil.get_boston_housing()
        explained_model = RandomForestRegressor(n_estimators=64, max_depth=5, random_state=1)
        explained_model.fit(x_train, y_train)

        model_builder = MLPModelBuilder(num_layers=2, num_units=32, activation="relu", p_dropout=0.2, verbose=0,
                                        batch_size=32, learning_rate=0.001, num_epochs=2, early_stopping_patience=128)
        masking_operation = ZeroMasking()
        loss = mean_squared_error
        explainer = CXPlain(explained_model, model_builder, masking_operation, loss)

        explainer.fit(x_train, y_train)
        self.assertEqual(explainer.prediction_model.output_shape, (None, np.prod(x_test.shape[1:])))

        eval_score = explainer.score(x_test, y_test)
        train_score = explainer.get_last_fit_score()
        median = explainer.predict(x_test)
        self.assertTrue(median.shape == x_test.shape)

    def test_boston_housing_no_fit_invalid(self):
        (x_train, y_train), (x_test, y_test) = TestUtil.get_boston_housing()
        explained_model = RandomForestRegressor(n_estimators=64, max_depth=5, random_state=1)
        explained_model.fit(x_train, y_train)

        model_builder = MLPModelBuilder(num_layers=2, num_units=32, activation="relu", p_dropout=0.2, verbose=0,
                                        batch_size=32, learning_rate=0.001, num_epochs=2, early_stopping_patience=128)
        masking_operation = ZeroMasking()
        loss = mean_squared_error
        explainer = CXPlain(explained_model, model_builder, masking_operation, loss)

        with self.assertRaises(AssertionError):
            explainer.predict(x_test, y_test)

        with self.assertRaises(AssertionError):
            explainer.score(x_test, y_test)

    def test_boston_housing_load_save_valid(self):
        (x_train, y_train), (x_test, y_test) = TestUtil.get_boston_housing()
        explained_model = RandomForestRegressor(n_estimators=64, max_depth=5, random_state=1)
        explained_model.fit(x_train, y_train)

        model_builder = MLPModelBuilder(num_layers=2, num_units=32, activation="relu", p_dropout=0.2, verbose=0,
                                        batch_size=32, learning_rate=0.001, num_epochs=2, early_stopping_patience=128)
        masking_operation = ZeroMasking()
        loss = mean_squared_error

        num_models_settings = [1, 2]
        for num_models in num_models_settings:
            explainer = CXPlain(explained_model, model_builder, masking_operation, loss,
                                num_models=num_models)

            explainer.fit(x_train, y_train)
            median_1 = explainer.predict(x_test)

            tmp_dir_name = tempfile.mkdtemp()
            explainer.save(tmp_dir_name)

            with self.assertRaises(ValueError):
                explainer.save(tmp_dir_name, overwrite=False)

            explainer.save(tmp_dir_name, overwrite=True)
            explainer.load(tmp_dir_name)
            median_2 = explainer.predict(x_test)

            self.assertTrue(np.array_equal(median_1, median_2))

            shutil.rmtree(tmp_dir_name)  # Cleanup.

    def test_mnist_valid(self):
        num_subsamples = 100
        (x_train, y_train), (x_test, y_test) = TestUtil.get_mnist(flattened=False, num_subsamples=num_subsamples)

        explained_model = MLPClassifier(solver='lbfgs', alpha=1e-5,
                                        hidden_layer_sizes=(64, 32), random_state=1)
        explained_model.fit(x_train.reshape((len(x_train), -1)), y_train)

        model_builder = MLPModelBuilder(num_layers=2, num_units=64, activation="relu", p_dropout=0.2, verbose=0,
                                        batch_size=256, learning_rate=0.001, num_epochs=2, early_stopping_patience=128)
        masking_operation = ZeroMasking()
        loss = categorical_crossentropy

        downsample_factors = [(2, 2), (4, 4), (4, 7), (7, 4), (7, 7)]
        for downsample_factor in downsample_factors:
            explainer = CXPlain(explained_model, model_builder, masking_operation, loss,
                                downsample_factors=downsample_factor, flatten_for_explained_model=True)

            explainer.fit(x_train, y_train)

            self.assertEqual(explainer.prediction_model.output_shape, (None, np.prod(x_test.shape[1:])))

            eval_score = explainer.score(x_test, y_test)
            train_score = explainer.get_last_fit_score()
            median = explainer.predict(x_test)
            self.assertTrue(median.shape == x_test.shape)

    def test_mnist_unet_valid(self):
        num_subsamples = 100
        (x_train, y_train), (x_test, y_test) = TestUtil.get_mnist(flattened=False, num_subsamples=num_subsamples)

        explained_model = MLPClassifier(solver='lbfgs', alpha=1e-5,
                                        hidden_layer_sizes=(64, 32), random_state=1)
        explained_model.fit(x_train.reshape((len(x_train), -1)), y_train)
        masking_operation = ZeroMasking()
        loss = categorical_crossentropy

        downsample_factors = [(2, 2), (4, 4), (4, 7), (7, 4), (7, 7)]
        with_bns = [True if i % 2 == 0 else False for i in range(len(downsample_factors))]
        for downsample_factor, with_bn in zip(downsample_factors, with_bns):
            model_builder = UNetModelBuilder(downsample_factor, num_layers=2, num_units=64, activation="relu",
                                             p_dropout=0.2, verbose=0, batch_size=256, learning_rate=0.001,
                                             num_epochs=2, early_stopping_patience=128, with_bn=with_bn)

            explainer = CXPlain(explained_model, model_builder, masking_operation, loss,
                                downsample_factors=downsample_factor, flatten_for_explained_model=True)

            explainer.fit(x_train, y_train)
            eval_score = explainer.score(x_test, y_test)
            train_score = explainer.get_last_fit_score()
            median = explainer.predict(x_test)
            self.assertTrue(median.shape == x_test.shape)

    def test_nlp_padded_valid(self):
        num_words = 1024
        (x_train, y_train), (x_test, y_test) = TestUtil.get_random_variable_length_dataset(max_value=num_words)

        explained_model = RandomForestClassifier(n_estimators=64, max_depth=5, random_state=1)

        counter = CountVectoriser(num_words)
        tfidf_transformer = TfidfTransformer()

        explained_model = Pipeline([('counts', counter),
                                    ('tfidf', tfidf_transformer),
                                    ('model', explained_model)])
        explained_model.fit(x_train, y_train)

        model_builder = RNNModelBuilder(embedding_size=num_words, with_embedding=True,
                                        num_layers=2, num_units=32, activation="relu", p_dropout=0.2, verbose=0,
                                        batch_size=32, learning_rate=0.001, num_epochs=2, early_stopping_patience=128)
        masking_operation = WordDropMasking()
        loss = binary_crossentropy
        explainer = CXPlain(explained_model, model_builder, masking_operation, loss)

        x_train = pad_sequences(x_train, padding="post", truncating="post", dtype=int)
        x_test = pad_sequences(x_test, padding="post", truncating="post", dtype=int, maxlen=x_train.shape[1])

        explainer.fit(x_train, y_train)
        eval_score = explainer.score(x_test, y_test)
        train_score = explainer.get_last_fit_score()
        median = explainer.predict(x_test)
        self.assertTrue(median.shape == x_test.shape)

    def test_imdb_padded_valid(self):
        num_samples = 32
        num_words = 1024
        (x_train, y_train), (x_test, y_test) = TestUtil.get_imdb(word_dictionary_size=num_words,
                                                                 num_subsamples=num_samples)

        explained_model = RandomForestClassifier(n_estimators=64, max_depth=5, random_state=1)

        counter = CountVectoriser(num_words)
        tfidf_transformer = TfidfTransformer()

        explained_model = Pipeline([('counts', counter),
                                    ('tfidf', tfidf_transformer),
                                    ('model', explained_model)])
        explained_model.fit(x_train, y_train)

        model_builder = RNNModelBuilder(embedding_size=num_words, with_embedding=True,
                                        num_layers=2, num_units=32, activation="relu", p_dropout=0.2, verbose=0,
                                        batch_size=32, learning_rate=0.001, num_epochs=2, early_stopping_patience=128)
        masking_operation = WordDropMasking()
        loss = binary_crossentropy
        explainer = CXPlain(explained_model, model_builder, masking_operation, loss)

        x_train = pad_sequences(x_train, padding="post", truncating="post", dtype=int)
        x_test = pad_sequences(x_test, padding="post", truncating="post", dtype=int, maxlen=x_train.shape[1])

        explainer.fit(x_train, y_train)
        eval_score = explainer.score(x_test, y_test)
        train_score = explainer.get_last_fit_score()
        median = explainer.predict(x_test)
        self.assertTrue(median.shape == x_test.shape)

    def test_nlp_erroneous_rnn_args_invalid(self):
        num_words = 1024
        (x_train, y_train), (x_test, y_test) = TestUtil.get_random_variable_length_dataset(max_value=num_words)

        explained_model = RandomForestClassifier(n_estimators=64, max_depth=5, random_state=1)

        counter = CountVectoriser(num_words)
        tfidf_transformer = TfidfTransformer()

        explained_model = Pipeline([('counts', counter),
                                    ('tfidf', tfidf_transformer),
                                    ('model', explained_model)])
        explained_model.fit(x_train, y_train)

        with self.assertRaises(ValueError):
            _ = RNNModelBuilder(with_embedding=True, verbose=0)  # Must also specify the embedding_size argument.

        model_builder = RNNModelBuilder(embedding_size=num_words, with_embedding=True, verbose=0)

        input_layer = Input(shape=(10, 2))
        with self.assertRaises(ValueError):
            model_builder.build(input_layer)

        input_layer = Input(shape=(10, 3))
        with self.assertRaises(ValueError):
            model_builder.build(input_layer)

    def test_nlp_not_padded_invalid(self):
        num_words = 1024
        (x_train, y_train), (_, _) = TestUtil.get_random_variable_length_dataset(max_value=num_words)

        explained_model = RandomForestClassifier(n_estimators=64, max_depth=5, random_state=1)

        counter = CountVectoriser(num_words)
        tfidf_transformer = TfidfTransformer()

        explained_model = Pipeline([('counts', counter),
                                    ('tfidf', tfidf_transformer),
                                    ('model', explained_model)])
        explained_model.fit(x_train, y_train)

        model_builder = RNNModelBuilder(embedding_size=num_words, with_embedding=True,
                                        num_layers=2, num_units=32, activation="relu", p_dropout=0.2, verbose=0,
                                        batch_size=32, learning_rate=0.001, num_epochs=2, early_stopping_patience=128)
        masking_operation = WordDropMasking()
        loss = binary_crossentropy
        explainer = CXPlain(explained_model, model_builder, masking_operation, loss)

        with self.assertRaises(ValueError):
            explainer.fit(x_train, y_train)

    def test_time_series_valid(self):
        num_samples = 1024
        fixed_length = 99
        (x_train, y_train), (x_test, y_test) = TestUtil.get_random_fixed_length_dataset(num_samples=num_samples,
                                                                                        fixed_length=fixed_length)

        model_builder = RNNModelBuilder(with_embedding=False, num_layers=2, num_units=32,
                                        activation="relu", p_dropout=0.2, verbose=0,
                                        batch_size=32, learning_rate=0.001, num_epochs=2,
                                        early_stopping_patience=128)

        explained_model = MLPClassifier()
        explained_model.fit(x_train.reshape((-1, np.prod(x_train.shape[1:]))), y_train)

        masking_operation = ZeroMasking()
        loss = binary_crossentropy
        explainer = CXPlain(explained_model, model_builder, masking_operation, loss,
                            flatten_for_explained_model=True)

        explainer.fit(x_train, y_train)
        eval_score = explainer.score(x_test, y_test)
        train_score = explainer.get_last_fit_score()
        median = explainer.predict(x_test)
        self.assertTrue(median.shape == x_test.shape)

    @staticmethod
    def make_at_tmp(file_name):
        tmp_dir = tempfile.mkdtemp()
        file_path = join(tmp_dir, file_name)
        with open(file_path, "w") as fp:
            fp.writelines("empty\n")
        return tmp_dir

    def test_overwrite_single_model_invalid(self):
        (x_train, y_train), (x_test, y_test) = TestUtil.get_boston_housing()

        model_builder = MLPModelBuilder()
        explained_model = RandomForestRegressor(n_estimators=64, max_depth=5, random_state=1)
        explained_model.fit(x_train, y_train)
        masking_operation = ZeroMasking()
        loss = binary_crossentropy
        explainer = CXPlain(explained_model, model_builder, masking_operation, loss)

        file_names = [
            CXPlain.get_config_file_name(),
            CXPlain.get_explained_model_file_name(".pkl"),
            CXPlain.get_loss_pkl_file_name(),
            CXPlain.get_model_builder_pkl_file_name(),
            CXPlain.get_masking_operation_pkl_file_name()
        ]

        # Test with untrained explanation model.
        for file_name in file_names:
            tmp_dir = TestExplanationModel.make_at_tmp(file_name)
            with self.assertRaises(ValueError):
                explainer.save(tmp_dir, overwrite=False)

        # Test with trained explanation model.
        explainer.fit(x_train, y_train)

        file_names = [
            CXPlain.get_config_file_name(),
            CXPlain.get_explained_model_file_name(".pkl"),
            CXPlain.get_loss_pkl_file_name(),
            CXPlain.get_model_builder_pkl_file_name(),
            CXPlain.get_masking_operation_pkl_file_name(),
            CXPlain.get_prediction_model_h5_file_name(),
        ]
        for file_name in file_names:
            tmp_dir = TestExplanationModel.make_at_tmp(file_name)
            with self.assertRaises(ValueError):
                explainer.save(tmp_dir, overwrite=False)

    def test_overwrite_ensemble_model_invalid(self):
        (x_train, y_train), (x_test, y_test) = TestUtil.get_boston_housing()

        model_builder = MLPModelBuilder()
        explained_model = RandomForestRegressor(n_estimators=64, max_depth=5, random_state=1)
        explained_model.fit(x_train, y_train)
        masking_operation = ZeroMasking()
        loss = binary_crossentropy
        num_models = 5
        explainer = CXPlain(explained_model, model_builder, masking_operation, loss,
                            num_models=num_models)

        file_names = [
            CXPlain.get_config_file_name(),
            CXPlain.get_explained_model_file_name(".pkl"),
            CXPlain.get_loss_pkl_file_name(),
            CXPlain.get_model_builder_pkl_file_name(),
            CXPlain.get_masking_operation_pkl_file_name()
        ]

        # Test with untrained explanation model.
        for file_name in file_names:
            tmp_dir = TestExplanationModel.make_at_tmp(file_name)
            with self.assertRaises(ValueError):
                explainer.save(tmp_dir, overwrite=False)

        # Test with trained explanation model.
        explainer.fit(x_train, y_train)

        file_names = [
            CXPlain.get_config_file_name(),
            CXPlain.get_explained_model_file_name(".pkl"),
            CXPlain.get_loss_pkl_file_name(),
            CXPlain.get_model_builder_pkl_file_name(),
            CXPlain.get_masking_operation_pkl_file_name()
        ] + [CXPlain.get_prediction_model_h5_file_name(i) for i in range(num_models)]

        for file_name in file_names:
            tmp_dir = TestExplanationModel.make_at_tmp(file_name)
            with self.assertRaises(ValueError):
                explainer.save(tmp_dir, overwrite=False)


if __name__ == '__main__':
    unittest.main()
