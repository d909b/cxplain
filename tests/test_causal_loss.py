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
import numpy as np
from sklearn.pipeline import Pipeline
from cxplain.util.test_util import TestUtil
from cxplain.util.count_vectoriser import CountVectoriser
from sklearn.feature_extraction.text import TfidfTransformer
from cxplain.backend.masking.zero_masking import ZeroMasking
from cxplain.backend.numpy_math_interface import NumpyInterface
from cxplain.backend.masking.word_drop_masking import WordDropMasking
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from cxplain.backend.causal_loss import calculate_delta_errors, causal_loss


class TestCausalLoss(unittest.TestCase):
    def setUp(self):
        np.random.seed(909)

    def test_causal_loss_simple(self):
        models = TestUtil.get_classification_models()

        batch_size = 32
        num_samples = 1024
        x, y = TestUtil.get_test_dataset_with_one_oracle_feature(num_samples=num_samples)

        for explained_model in models:
            TestUtil.fit_proxy(explained_model, x, y)
            masking = ZeroMasking()
            _, y_pred, all_y_pred_imputed = masking.get_predictions_after_masking(explained_model, x, y,
                                                                                  batch_size=batch_size,
                                                                                  downsample_factors=(1,),
                                                                                  flatten=True)
            auxiliary_outputs = y_pred
            all_but_one_auxiliary_outputs = all_y_pred_imputed
            all_but_one_auxiliary_outputs = TestUtil.split_auxiliary_outputs_on_feature_dim(
                all_but_one_auxiliary_outputs
            )

            delta_errors = calculate_delta_errors(y,
                                                  auxiliary_outputs,
                                                  all_but_one_auxiliary_outputs,
                                                  NumpyInterface.binary_crossentropy,
                                                  math_ops=NumpyInterface)

            # Ensure correct delta error dimensionality.
            self.assertEqual(delta_errors.shape, (num_samples, x.shape[-1]))

            # Feature at index 0 should be the most important for __explained_model__'s predictions
            # - if the model converged correctly.
            self.assertEqual(np.argmax(np.sum(delta_errors, axis=0)), 0)

    def test_causal_loss_broken_loss_function(self):
        explained_model = TestUtil.get_classification_models()[0]

        batch_size = 32
        num_samples = 1024
        x, y = TestUtil.get_test_dataset_with_one_oracle_feature(num_samples=num_samples)

        TestUtil.fit_proxy(explained_model, x, y)
        masking = ZeroMasking()
        _, y_pred, all_y_pred_imputed = masking.get_predictions_after_masking(explained_model, x, y,
                                                                              batch_size=batch_size,
                                                                              downsample_factors=(1,),
                                                                              flatten=True)
        auxiliary_outputs = y_pred
        all_but_one_auxiliary_outputs = all_y_pred_imputed
        all_but_one_auxiliary_outputs = TestUtil.split_auxiliary_outputs_on_feature_dim(
            all_but_one_auxiliary_outputs
        )

        def broken_loss(y_true, y_pred):
            return np.mean(NumpyInterface.binary_crossentropy(y_true, y_pred), axis=0)

        with self.assertRaises(ValueError):
            _ = calculate_delta_errors(y,
                                       auxiliary_outputs,
                                       all_but_one_auxiliary_outputs,
                                       broken_loss,
                                       math_ops=NumpyInterface)

    def test_causal_loss_duplicate_feature(self):
        models = TestUtil.get_classification_models()

        batch_size = 32
        num_samples = 1024
        x, y = TestUtil.get_test_dataset_with_two_oracle_features(num_samples=num_samples)

        for explained_model in models:
            TestUtil.fit_proxy(explained_model, x, y)
            masking = ZeroMasking()
            _, y_pred, all_y_pred_imputed = masking.get_predictions_after_masking(explained_model, x, y,
                                                                                  batch_size=batch_size,
                                                                                  downsample_factors=(1,),
                                                                                  flatten=True)
            auxiliary_outputs = y_pred
            all_but_one_auxiliary_outputs = all_y_pred_imputed
            all_but_one_auxiliary_outputs = TestUtil.split_auxiliary_outputs_on_feature_dim(
                all_but_one_auxiliary_outputs
            )

            delta_errors = calculate_delta_errors(y,
                                                  auxiliary_outputs,
                                                  all_but_one_auxiliary_outputs,
                                                  NumpyInterface.binary_crossentropy,
                                                  math_ops=NumpyInterface)

            # Ensure correct delta error dimensionality.
            self.assertEqual(delta_errors.shape, (num_samples, x.shape[-1]))

            # Ensure both input oracles receive the same importance.
            self.assertTrue(np.allclose(delta_errors[:, 0], delta_errors[:, 1], atol=0.1, rtol=0.1))

    def test_causal_loss_confounded_input(self):
        models = TestUtil.get_classification_models()

        batch_size = 32
        num_samples = 1024
        x, y = TestUtil.get_test_dataset_with_confounded_input(num_samples=num_samples)

        for explained_model in models:
            TestUtil.fit_proxy(explained_model, x, y)
            masking = ZeroMasking()
            _, y_pred, all_y_pred_imputed = masking.get_predictions_after_masking(explained_model, x, y,
                                                                                  batch_size=batch_size,
                                                                                  downsample_factors=(1,),
                                                                                  flatten=True)
            auxiliary_outputs = y_pred
            all_but_one_auxiliary_outputs = all_y_pred_imputed
            all_but_one_auxiliary_outputs = TestUtil.split_auxiliary_outputs_on_feature_dim(
                all_but_one_auxiliary_outputs
            )

            delta_errors = calculate_delta_errors(y,
                                                  auxiliary_outputs,
                                                  all_but_one_auxiliary_outputs,
                                                  NumpyInterface.binary_crossentropy,
                                                  math_ops=NumpyInterface)

            # Ensure correct delta error dimensionality.
            self.assertEqual(delta_errors.shape, (num_samples, x.shape[-1]))

            # Ensure both input oracles receive (roughly) the same importance upon convergence.
            self.assertTrue(np.abs(np.diff(np.sum(delta_errors, axis=0) / float(num_samples))) < 0.1)

    def test_causal_loss_padded_input(self):
        models = TestUtil.get_classification_models()

        batch_size = 32
        num_samples = 1024
        num_words = 1024

        (x_train, y_train), (x_test, y_test) = \
            TestUtil.get_random_variable_length_dataset(num_samples=num_samples, max_value=num_words)
        x, y = np.concatenate([x_train, x_test], axis=0), np.concatenate([y_train, y_test], axis=0)

        self.assertEqual(x.shape[0], num_samples)

        for explained_model in models:
            counter = CountVectoriser(num_words)
            tfidf_transformer = TfidfTransformer()

            explained_model = Pipeline([('counts', counter),
                                        ('tfidf', tfidf_transformer),
                                        ('model', explained_model)])
            TestUtil.fit_proxy(explained_model, x, y)
            masking = WordDropMasking()

            x = pad_sequences(x, padding="post", truncating="post", dtype=int)

            _, y_pred, all_y_pred_imputed = masking.get_predictions_after_masking(explained_model, x, y,
                                                                                  batch_size=batch_size,
                                                                                  downsample_factors=(1,),
                                                                                  flatten=False)
            auxiliary_outputs = y_pred
            all_but_one_auxiliary_outputs = all_y_pred_imputed
            all_but_one_auxiliary_outputs = TestUtil.split_auxiliary_outputs_on_feature_dim(
                all_but_one_auxiliary_outputs
            )

            delta_errors = calculate_delta_errors(y,
                                                  auxiliary_outputs,
                                                  all_but_one_auxiliary_outputs,
                                                  NumpyInterface.binary_crossentropy,
                                                  math_ops=NumpyInterface)

            # Ensure correct delta error dimensionality.
            self.assertEqual(delta_errors.shape, (num_samples, x.shape[1]))


if __name__ == '__main__':
    unittest.main()
