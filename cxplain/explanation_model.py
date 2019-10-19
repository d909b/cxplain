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
import os
import six
import numpy as np
from sklearn.base import BaseEstimator
from abc import ABCMeta, abstractmethod
from cxplain.backend.validation import Validation


@six.add_metaclass(ABCMeta)
class CXPlain(BaseEstimator):
    def __init__(self, explained_model, model_builder, masking_operation, loss,
                 downsample_factors=(1,), num_models=1):
        super(CXPlain, self).__init__()
        self.explained_model = explained_model
        self.model_builder = model_builder
        self.masking_operation = masking_operation
        self.loss = loss
        self.last_masked_data = None
        self.prediction_model = None

        Validation.check_is_positive_integer_greaterequals_1(num_models, var_name="num_models")
        self.num_models = num_models

        Validation.check_downsample_factors_at_initialisation(downsample_factors)
        self.downsample_factors = downsample_factors

    @staticmethod
    def get_config_file_name():
        return "cxplain_config.json"

    @staticmethod
    def get_model_builder_pkl_file_name():
        return "cxplain_model_builder.pkl"

    @staticmethod
    def get_masking_operation_pkl_file_name():
        return "cxplain_masking_operation.pkl"

    @staticmethod
    def get_loss_pkl_file_name():
        return "cxplain_loss.pkl"

    @staticmethod
    def get_explained_model_file_name(file_extension):
        return "cxplain_explained_model" + file_extension

    @staticmethod
    def get_model_h5_name_with_base_name(base_name, index):
        if index is None:
            return base_name + ".h5"
        else:
            return base_name + "_{}.h5".format(index)

    @staticmethod
    def get_model_h5_file_name(index=None):
        base_name = "cxplain_model"
        return CXPlain.get_model_h5_name_with_base_name(base_name, index)

    @staticmethod
    def get_prediction_model_h5_file_name(index=None):
        base_name = "cxplain_prediction_model"
        return CXPlain.get_model_h5_name_with_base_name(base_name, index)

    def get_config(self, directory, model_serialiser):
        if self.prediction_model is None:
            prediction_model = None
        else:
            if self.num_models == 1:
                prediction_model = os.path.join(directory, CXPlain.get_prediction_model_h5_file_name())
            else:
                prediction_model = [os.path.join(directory, CXPlain.get_prediction_model_h5_file_name(i))
                                    for i in range(self.num_models)]

        config = {
            "model_builder": os.path.join(directory, CXPlain.get_model_builder_pkl_file_name()),
            "masking_operation": os.path.join(directory, CXPlain.get_masking_operation_pkl_file_name()),
            "loss": os.path.join(directory, CXPlain.get_loss_pkl_file_name()),
            "downsample_factors": self.downsample_factors,
            "num_models": self.num_models,
            "prediction_model": prediction_model,
            "explained_model": os.path.join(directory, CXPlain.get_explained_model_file_name(
                model_serialiser.get_file_extension())
            ),
        }
        return config

    def get_masked_data(self):
        return self.last_masked_data

    @abstractmethod
    def _build_model(self, X, y):
        raise NotImplementedError()

    @abstractmethod
    def _fit_single(self, model, X, y, masked_data=None):
        raise NotImplementedError()

    def fit(self, X, y, masked_data=None):
        """
        Fits the CXPlain instance to a dataset (X, y) consisting of samples __X__ and ground truth labels __y__.

        :param X: The data samples to train on. The first dimension must be the number of samples.
        :param y: The ground truth labels to train on. The first dimension must be the number of samples.
        :param masked_data: An array of precomputed masked data as can be obtained from __get_masked_data__.
                            If None, the masked data is computed. If set, the precomputed masked data is used for
                            scoring and computation of the masked data is skipped (Optional, default: None).
        :return: This CXPlain instance (for call chaining).
        """
        self._build_model(X, y)

        if self.num_models == 1:
            self._fit_single(self.model, X, y, masked_data=masked_data)
        else:
            for model in self.model:
                self._fit_single(model, X, y, masked_data=masked_data)
                masked_data = self.get_masked_data()
        return self

    @abstractmethod
    def _predict_single(self, model, X):
        raise NotImplementedError()

    def _predict_multiple(self, X, confidence_level=None):
        if self.prediction_model is None or \
           not isinstance(self.prediction_model, list) or \
           len(self.prediction_model) == 0:
            raise AssertionError("Prediction model must be initialised with an ensemble of models "
                                 "when calling __predict__. Did you forget to __fit__ the explanation models?")

        predictions = [self._predict_single(model, X) for model in self.prediction_model]
        median = np.median(predictions, axis=0)

        if confidence_level is not None:
            alpha = 1.0 - confidence_level
            half_alpha = 0.5 * alpha
            start_quantile = np.percentile(predictions, half_alpha*100, axis=0)
            end_quantile = np.percentile(predictions, (1. - half_alpha)*100, axis=0)

            return median, np.concatenate([np.expand_dims(start_quantile, axis=-1),
                                           np.expand_dims(end_quantile, axis=-1)],
                                          axis=-1)
        else:
            return median

    def predict(self, X, confidence_level=None):
        """
        Estimates the importance of the inputs in __X__ towards the __self.explained_model__'s decision.
        Provides confidence intervals if __confidence_level__ is not None.

        :param X: The data samples to be evaluated. The first dimension must be the number of samples.
        :param confidence_level: The confidence level used to report the confidence intervals, i.e. a
                                 confidence level of 0.95 would indicate that you wish to obtain the
                                 0.025 and 0.975 quantiles of the output distribution. If None,
                                 no confidence is returned. The CXPlain instance must have been
                                 initialised with __num_models__ > 1 in order to be able to
                                 compute confidence intervals. (Optional, default: None).
        :return: (i) An array of predictions that estimate the importace of each input feature in __X__
                 based on the sample data __X__. The first dimension of the returned array will be the sample dimension
                 and it will match that of __X__, if confidence_level is None,
                 or
                 (ii) a tuple of two entries with the first entry being the predictions and the second entry being
                 the confidence interval (CI) for each provided feature importance estimate reported in the first entry.
                 The last dimension of the confidence interval reported is (2,) and the entries are
                 (CI lower bound, CI upper bound) if confidence_level is not None
        :exception AssertionError Thrown if __predict__ was called without first fitting the explanation model
                                  using __fit__.
        :exception ValueError Thrown if the value of __confidence_level__ was not in the range [0, 1].

        """
        if self.prediction_model is None:
            raise AssertionError("Model must be initialised when calling __predict__. "
                                 "Did you forget to __fit__ the explanation model?")

        if confidence_level is not None and \
                (confidence_level <= 0.0 or confidence_level >= 1.0 or \
                 np.isclose(confidence_level, 0.) or \
                 np.isclose(confidence_level, 1.)):
            raise ValueError("The __confidence_level__ must be a value between 0 (exclusive) and 1 (exclusive).")

        if self.num_models == 1:
            ret_val = self._predict_single(self.prediction_model, X)
        else:
            ret_val = self._predict_multiple(X, confidence_level=confidence_level)

        target_shape = Validation.get_attribution_shape(X)

        if len(target_shape) >= 4:
            confidence_shape = target_shape[:-1] + (2,)
        else:
            confidence_shape = target_shape + (2,)

        if isinstance(ret_val, tuple):
            ret_val = ret_val[0].reshape(target_shape), ret_val[1].reshape(confidence_shape)
        else:
            ret_val = ret_val.reshape(target_shape)

        return ret_val

    def explain(self, X, confidence_level=0.95):
        """
        Convenience function. Same functionality as __predict__.
        """
        return self.predict(X, confidence_level=confidence_level)

    @abstractmethod
    def score(self, X, y, sample_weight=None):
        raise NotImplementedError()
