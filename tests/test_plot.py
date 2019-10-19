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

import tempfile
import unittest
import numpy as np
from os import path
from cxplain.visualisation.plot import Plot


class TestPlots(unittest.TestCase):
    def setUp(self):
        np.random.seed(909)

    def test_check_plot_input_shapes_valid(self):
        x = np.random.normal(0, 1, size=(10, 5, 5, 3))
        attribution = np.random.normal(0, 1, size=(10, 5, 5, 1))
        confidence = np.random.normal(0, 1, size=(10, 5, 5, 1, 2))
        Plot.check_plot_input(x, attribution, confidence)

    def test_check_plot_input_attribution_shape_invalid(self):
        x = np.random.normal(0, 1, size=(10, 5, 5, 3))
        attribution = np.random.normal(0, 1, size=(10, 5, 5, 2))
        confidence = None
        with self.assertRaises(ValueError):
            Plot.check_plot_input(x, attribution, confidence)

    def test_check_plot_input_confidence_shape_invalid(self):
        x = np.random.normal(0, 1, size=(10, 5, 5, 3))
        attribution = np.random.normal(0, 1, size=(10, 5, 5, 1))
        confidence = np.random.normal(0, 1, size=(10, 5, 5, 1))
        with self.assertRaises(ValueError):
            Plot.check_plot_input(x, attribution, confidence)

        confidence = np.random.normal(0, 1, size=(10, 5, 5, 3))
        with self.assertRaises(ValueError):
            Plot.check_plot_input(x, attribution, confidence)

    def test_plot_attribution_1d_invalid(self):
        x = np.random.normal(0, 1, size=(1, 1))
        attribution = np.random.normal(0, 1, size=(1, 1))
        confidence = np.random.normal(0, 1, size=(1, 1, 2))

        with self.assertRaises(ValueError):
            Plot.plot_attribution_1d(x, attribution, confidence,
                                     run_without_gui=True)

        x = np.random.normal(0, 1, size=(1,))
        with self.assertRaises(ValueError):
            Plot.plot_attribution_1d(x, attribution, confidence,
                                     run_without_gui=True)

        attribution = np.random.normal(0, 1, size=(1,))
        with self.assertRaises(ValueError):
            Plot.plot_attribution_1d(x, attribution, confidence,
                                     run_without_gui=True)

        confidence = np.random.normal(0, 1, size=(1, 2))
        Plot.plot_attribution_1d(x, attribution, confidence,
                                 run_without_gui=True)

    def test_plot_attribution_1d_valid(self):
        x = np.random.normal(0, 1, size=(1,))
        attribution = np.random.normal(0, 1, size=(1,))
        confidence = np.random.normal(0, 1, size=(1, 2))

        tmp_file = tempfile.NamedTemporaryFile()
        Plot.plot_attribution_1d(x, attribution, confidence,
                                 run_without_gui=True, filepath=tmp_file.name)

        self.assertTrue(path.isfile(tmp_file.name) and path.getsize(tmp_file.name) > 0)

    def test_plot_attribution_2d_valid(self):
        x = np.random.normal(0, 1, size=(2, 2, 3))
        attribution = np.random.normal(0, 1, size=(2, 2, 1))
        confidence = np.random.normal(0, 1, size=(2, 2, 1, 2))

        tmp_file = tempfile.NamedTemporaryFile()
        Plot.plot_attribution_2d(x, attribution, confidence,
                                 run_without_gui=True, filepath=tmp_file.name)

        self.assertTrue(path.isfile(tmp_file.name) and path.getsize(tmp_file.name) > 0)


if __name__ == '__main__':
    unittest.main()
