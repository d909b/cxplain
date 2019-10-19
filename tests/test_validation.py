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
from cxplain.util.test_util import TestUtil
from cxplain.backend.validation import Validation
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


class TestValidation(unittest.TestCase):
    def setUp(self):
        np.random.seed(909)

    def test_input_shape_tabular_valid(self):
        test_num_samples = [1, 2, 1024]
        test_num_features = [1, 2, 1024]
        for num_samples in test_num_samples:
            for num_features in test_num_features:
                x = np.random.random_sample(size=(num_samples, num_features))
                n, input_dim = Validation.get_input_dimension(x)
                self.assertEqual(n, num_samples)
                self.assertEqual(input_dim, (num_features,))

    def test_input_shape_invalid_none(self):
        with self.assertRaises(ValueError):
            Validation.get_input_dimension(None)

    def test_input_shape_invalid_1dim(self):
        with self.assertRaises(ValueError):
            Validation.get_input_dimension([1])
        with self.assertRaises(ValueError):
            Validation.get_input_dimension([1, 2, 3])
        with self.assertRaises(ValueError):
            Validation.get_input_dimension([None])

    def test_input_shape_time_series_fixed_valid(self):
        test_num_samples = [1, 2, 1024]
        test_num_lens = [1, 2, 256]
        test_num_features = [1, 2, 1024]
        for num_samples in test_num_samples:
            for ts_length in test_num_lens:
                for num_features in test_num_features:
                    x = np.random.random_sample(size=(num_samples, ts_length, num_features))
                    n, input_dim = Validation.get_input_dimension(x)
                    self.assertEqual(n, num_samples)
                    self.assertEqual(input_dim, (ts_length, num_features))

    def test_input_shape_time_series_variable_valid(self):
        test_num_samples = [2, 3, 1024]
        test_num_lens = [1, 2, 256]
        test_num_features = [1, 2, 1024]
        for num_samples in test_num_samples:
            for num_features in test_num_features:
                x = [np.random.random_sample(size=(test_num_lens[i % len(test_num_lens)], num_features))
                     for i in range(num_samples)]
                n, input_dim = Validation.get_input_dimension(x)
                self.assertEqual(n, num_samples)
                self.assertEqual(input_dim, (None, num_features))

    def test_input_shape_image_fixed_valid(self):
        test_num_samples = [1, 2, 1024]
        test_num_rows = [1, 2, 256]
        test_num_cols = [1, 2, 256]
        test_num_channels = [1, 2, 3]
        for num_samples in test_num_samples:
            for rows in test_num_rows:
                for cols in test_num_cols:
                    for num_channels in test_num_channels:
                        x = np.random.random_sample(size=(num_samples, rows, cols, num_channels))
                        n, input_dim = Validation.get_input_dimension(x)
                        self.assertEqual(n, num_samples)
                        self.assertEqual(input_dim, (rows, cols, num_channels))

    def test_input_shape_image_variable_valid(self):
        test_num_samples = [2, 3, 1024]
        test_num_lens = [2, 3, 256]
        test_num_features = [1, 2, 3]
        for num_samples in test_num_samples:
            for num_features in test_num_features:
                x = [np.random.random_sample(size=(test_num_lens[i % len(test_num_lens)],
                                                   test_num_lens[(i + 1) % len(test_num_lens)], num_features))
                     for i in range(num_samples)]
                n, input_dim = Validation.get_input_dimension(x)
                self.assertEqual(n, num_samples)
                self.assertEqual(input_dim, (None, None, num_features))

    def test_input_shape_volume_fixed_valid(self):
        test_num_samples = [1, 2, 128]
        test_num_voxels = [1, 2, 64]
        test_num_channels = [1, 2, 3]
        for num_samples in test_num_samples:
            for rows in test_num_voxels:
                for cols in test_num_voxels:
                    for depth in test_num_voxels:
                        for num_channels in test_num_channels:
                            x = np.random.random_sample(size=(num_samples, rows, cols, depth, num_channels))
                            n, input_dim = Validation.get_input_dimension(x)
                            self.assertEqual(n, num_samples)
                            self.assertEqual(input_dim, (rows, cols, depth, num_channels))

    def test_input_shape_volume_variable_valid(self):
        test_num_samples = [2, 3, 128]
        test_num_lens = [2, 3, 64]
        test_num_features = [1, 2, 3]
        for num_samples in test_num_samples:
            for num_features in test_num_features:
                x = [np.random.random_sample(size=(test_num_lens[i % len(test_num_lens)],
                                                   test_num_lens[(i + 1) % len(test_num_lens)],
                                                   test_num_lens[(i + 2) % len(test_num_lens)],
                                                   num_features))
                     for i in range(num_samples)]
                n, input_dim = Validation.get_input_dimension(x)
                self.assertEqual(n, num_samples)
                self.assertEqual(input_dim, (None, None, None, num_features))

    def test_is_variable_length_list_true(self):
        (x, _), _ = TestUtil.get_random_variable_length_dataset(max_value=1024)
        return_value = Validation.is_variable_length(x)
        self.assertEqual(return_value, True)

    def test_is_variable_length_ndarray_true(self):
        (x, _), _ = TestUtil.get_random_variable_length_dataset(max_value=1024)
        x = np.array(x)
        return_value = Validation.is_variable_length(x)
        self.assertEqual(return_value, True)

    def test_is_variable_length_padded_false(self):
        (x, _), _ = TestUtil.get_random_variable_length_dataset(max_value=1024)
        x = pad_sequences(x, padding="post", truncating="post", dtype=int)
        return_value = Validation.is_variable_length(x)
        self.assertEqual(return_value, False)

    def test_get_attribution_shape_multi_channel(self):
        num_samples, intermediary_dimensions, num_channels = [1, 2, 100], [0, 1, 2, 3], [0, 1, 2, 3]

        for samples in num_samples:
            for num_dims in intermediary_dimensions:
                for channels in num_channels:
                    source_size = (samples,) + (2,)*num_dims
                    if channels != 0:
                        source_size += (channels,)

                    data = np.random.normal(0, 1, size=source_size)

                    if num_dims == 0 and channels == 0:
                        with self.assertRaises(ValueError):
                            Validation.get_attribution_shape(data)
                        continue
                    else:
                        attribution_shape = Validation.get_attribution_shape(data)

                    if len(source_size) >= 3:
                        adjusted_source_size = source_size[:-1] + (1,)
                        self.assertEqual(attribution_shape, adjusted_source_size)
                    else:
                        self.assertEqual(attribution_shape, source_size)

    def test_check_downsample_factors_at_initialisation(self):
        with self.assertRaises(ValueError):
            Validation.check_downsample_factors_at_initialisation((-1,))

        with self.assertRaises(ValueError):
            Validation.check_downsample_factors_at_initialisation(-1)

        with self.assertRaises(ValueError):
            Validation.check_downsample_factors_at_initialisation(1.1)

        with self.assertRaises(ValueError):
            Validation.check_downsample_factors_at_initialisation(-1.1)

        with self.assertRaises(ValueError):
            Validation.check_downsample_factors_at_initialisation((3.3, 2.2))

        Validation.check_downsample_factors_at_initialisation((3, 2, 1))

    def test_check_is_positive_integer_greaterequals_1(self):
        with self.assertRaises(ValueError):
            Validation.check_is_positive_integer_greaterequals_1(-1)

        with self.assertRaises(ValueError):
            Validation.check_is_positive_integer_greaterequals_1(1.1)

        with self.assertRaises(ValueError):
            Validation.check_is_positive_integer_greaterequals_1(-1.1)

        with self.assertRaises(ValueError):
            Validation.check_is_positive_integer_greaterequals_1(0)

        Validation.check_is_positive_integer_greaterequals_1(1)
        Validation.check_is_positive_integer_greaterequals_1(2)

    def test_check_is_fraction(self):
        with self.assertRaises(ValueError):
            Validation.check_is_fraction(-1.0)

        with self.assertRaises(ValueError):
            Validation.check_is_fraction(1.01)

        with self.assertRaises(ValueError):
            Validation.check_is_fraction(-0.01)

        Validation.check_is_fraction(1.0)
        Validation.check_is_fraction(0.0)
        Validation.check_is_fraction(0.00000001)
        Validation.check_is_fraction(1.0 - 0.00000001)


if __name__ == '__main__':
    unittest.main()
