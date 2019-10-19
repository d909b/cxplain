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
from cxplain.backend.masking.masking_util import MaskingUtil


class TestMasking(unittest.TestCase):
    def setUp(self):
        np.random.seed(909)

    def test_extract_downsample_factors_invalid_length(self):
        with self.assertRaises(ValueError):
            MaskingUtil.extract_downsample_factors((2, 2), 3)
        with self.assertRaises(ValueError):
            MaskingUtil.extract_downsample_factors((1, 1, 1, 1), 3)

    def test_extract_downsample_factors_valid_length(self):
        ret_val = MaskingUtil.extract_downsample_factors((2,), 2)
        self.assertEqual(ret_val, (2, 2))

        ret_val = MaskingUtil.extract_downsample_factors(1, 3)
        self.assertEqual(ret_val, (1, 1, 1))

        ret_val = MaskingUtil.extract_downsample_factors((1, 2, 3), 3)
        self.assertEqual(ret_val, (1, 2, 3))

    def test_get_input_constants(self):
        test_cases_is = [(1,), (2,), (2,),
                         (1, 1), (2, 2), (2, 2), (4, 4), (4, 4), (4, 2),
                         (2, 2, 2), (2, 2, 2), (3, 1, 3)]
        test_cases_df = [(1,), (1,), (2,),
                         (1, 1), (1, 1), (2, 2), (1, 4), (4, 1), (4, 2),
                         (1, 1, 1), (2, 2, 2), (1, 1, 1)]
        expected_steps = [(1,), (2,), (1,),
                          (1, 1), (2, 2), (1, 1), (4, 1), (1, 4), (1, 1),
                          (2, 2, 2), (1, 1, 1), (3, 1, 3)]

        for shape, factors, expected in zip(test_cases_is, test_cases_df, expected_steps):
            shape = shape + (1,)  # Add channel dim.
            num_indices, num_channels, steps, downsampling_factor = \
                MaskingUtil.get_input_constants(input_dim=shape,
                                                downsample_factors=factors)

            self.assertEqual((num_indices, num_channels, downsampling_factor),
                             (int(np.prod(expected)), 1, int(np.prod(factors))))

            self.assertTrue(np.array_equal(np.array(expected), steps))
    
    def test_get_ith_mask_invalid(self):
        with self.assertRaises(ValueError):
            MaskingUtil.get_ith_mask(i=0, image_shape=(1,), downsample_factors=(1,))

        with self.assertRaises(ValueError):
            MaskingUtil.get_ith_mask(i=0, image_shape=(1,)*5, downsample_factors=(1,))


if __name__ == '__main__':
    unittest.main()
