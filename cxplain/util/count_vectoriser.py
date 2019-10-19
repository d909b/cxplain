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
from sklearn.base import BaseEstimator


class CountVectoriser(BaseEstimator):
    def __init__(self, num_words):
        self.num_words = num_words

    def fit(self, raw_documents, y=None):
        self.fit_transform(raw_documents)
        return self

    def fit_transform(self, raw_documents, y=None):
        x = self.transform(raw_documents)
        return x

    def transform(self, raw_documents):
        x = CountVectoriser.to_counts(self.num_words, raw_documents)
        return x

    @staticmethod
    def to_counts(num_words, data):
        from collections import Counter

        num_samples = len(data)
        ret_val = np.zeros(shape=(num_samples, num_words), dtype=np.int8)
        for i, sample in enumerate(data):
            cur_data = map(lambda x: x[0], data[i])
            counts = Counter(cur_data)

            # Insert into sparse matrix.
            for k, v in counts.items():
                ret_val[i, k] = v
        return ret_val
