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
from functools import partial
from cxplain.backend.validation import Validation
from cxplain.backend.masking.base_masking import BaseMasking
from cxplain.backend.masking.masking_util import MaskingUtil


class WordDropMasking(BaseMasking):
    def __init__(self):
        super(WordDropMasking, self).__init__()

    def get_predictions_after_masking(self, explained_model, X, y, downsample_factors=(1,), batch_size=64,
                                      flatten=False):
        Validation.check_dataset(X, y)

        num_batches = int(np.ceil(len(X) / float(batch_size)))

        all_outputs = []
        for batch_idx in range(num_batches):
            x_batch = X[batch_idx*batch_size:(batch_idx+1)*batch_size]
            y_pred = MaskingUtil.get_prediction(explained_model, x_batch, flatten=flatten)

            x_imputed = []
            for x_i in x_batch:
                x_curr = []
                for j in range(len(x_i)):
                    x_i_imputed_j = np.concatenate([x_i[:j], x_i[j+1:]], axis=0)  # Drop entry at each index j.
                    x_curr.append(x_i_imputed_j)
                x_imputed.append(x_curr)

            all_y_pred_imputed = []
            for j, x_imputed_curr in enumerate(x_imputed):
                if len(x_imputed_curr) == 0:
                    y_pred_imputed = y_pred[j].reshape((1, -1))
                else:
                    y_pred_imputed = MaskingUtil.get_prediction(explained_model, x_imputed_curr, flatten=flatten)
                all_y_pred_imputed.append(y_pred_imputed)

            all_outputs.append((x_batch, y_pred, all_y_pred_imputed))

        all_outputs = [np.concatenate(list(map(partial(lambda x, dim: x[dim], dim=dim), all_outputs)))
                       for dim in range(len(all_outputs[0]))]

        return all_outputs
