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
import json
import pickle
import numpy as np
from cxplain.explanation_model import CXPlain
from cxplain.backend.validation import Validation
from tensorflow.python.keras.losses import categorical_crossentropy
from cxplain.backend.serialisation.tf_model_serialisation import TensorFlowModelSerialiser
from cxplain.backend.serialisation.pickle_model_serialisation import PickleModelSerialiser


class TensorflowCXPlain(CXPlain):
    def __init__(self, explained_model, model_builder, masking_operation, loss=categorical_crossentropy,
                 model_filepath=None, flatten_for_explained_model=False, num_models=1, downsample_factors=(1,)):
        """
        Initialises a causal explanation model (CXPlain) using the TensorFlow backend.

        :param explained_model: The model to be explained. Must have a __explained_model.predict()__ function that
                                returns the explained_model's prediction y for a given input sample X. (Required)
        :param model_builder: A cxplain.backend.model_builders.base_model_builder instance that builds the explanation
                              model that will be used to explain the __explained_model__. (Required)
        :param masking_operation: The masking operation that will be used to estimate the importance of removing a
                                  certain input feature from the set of available features. (Required)
        :param loss: The loss function by which the reduction in prediction error associated with removing a certain
                     input feature will be measured. Typically the mean squared error for regression tasks and
                     the cross entropy for classification tasks. (Required)
        :param model_filepath: The filepath the trained explanation model will be saved to.
                               If None, a temporary filepath will be created using tempfile.NamedTemporaryFile.
                               (Optional, default: None)
        :param flatten_for_explained_model: Whether or not input samples X should be flattened to shape
                                            (num_samples, num_features) before being passed to
                                            __explained_model.predict()__. (Optional, default: False)
        :param num_models: The number of explanation models to train in a bootstrap resampled ensemble. A value greater
                           than 1 is required to obtain uncertainty estimates for feature importance scores. A higher
                           number of models increases the quality of uncertainty estimates, but also requires linearly
                           more computation time. (Optional, default: 1, range: [1, inf)
        :param downsample_factors: Defines the downsample factor(s) to be applied at each axis of the output attribution
                                   map, i.e. a value of (2, 2) would downsample the output attribution map of an
                                   image with shape (28, 28) to (14, 14). Expects either a single integer value or a
                                   tuple of integer values on the range [1, inf). A value of 1 indicates no
                                   downsampling. Downsampling significantly decreases the computation and memory
                                   required to train a CXPlain model, but also reduces the output resolution.
                                   For performance reasons, it is recommended to avoid very high output attribution map
                                   resolutions. (Optional, default: 1, range: [1, inf))
        :exception ValueError Thrown if (i) __num_models__ is not an integer, or not in the range [1, inf).
                                        (ii) __downsample_factors__ is not a tuple or single integer  in range [1, inf)
        """
        super(TensorflowCXPlain, self).__init__(explained_model, model_builder, masking_operation, loss,
                                                downsample_factors, num_models)

        self.model_filepath = model_filepath
        self.model = None
        self.prediction_model = None
        self.last_fit_score = None
        self.flatten_for_explained_model = flatten_for_explained_model

    def get_config(self, directory, model_serialiser):
        """
        Serialises the configuration required to re-build the given CXPlain instance at a later point into a
        python dictionary.

        :param directory: The directory the configuration and binary files will be saved to. (Required)
        :param model_serialiser: A custom serialiser that will be used to serialise the explained model
                                 binary. (Required)
        :return: A python dictionary containing all the configuration details required for re-building the given
                 CXPlain instance.
        """
        config = super(TensorflowCXPlain, self).get_config(directory, model_serialiser)

        if self.num_models == 1:
            model = os.path.join(directory, CXPlain.get_model_h5_file_name())
        else:
            model = [os.path.join(directory, CXPlain.get_model_h5_file_name(i))
                     for i in range(self.num_models)]

        config["flatten_for_explained_model"] = self.flatten_for_explained_model
        config["model_filepath"] = self.model_filepath
        config["last_fit_score"] = self.last_fit_score
        config["model"] = model
        return config

    @staticmethod
    def _clean_output_dims(expected_output_dim, masked_data):
        # Ensure binary outputs only have one output feature, not two.
        if expected_output_dim == 1:
            explained_model_output_dim_y_pred = masked_data[1].shape[-1]
            if explained_model_output_dim_y_pred == 2:
                masked_data[1] = masked_data[1][..., 0:1]
            elif explained_model_output_dim_y_pred > 2:
                raise ValueError("Expected explained model predictions to have one or two output dimensions "
                                 "but got " + str(explained_model_output_dim_y_pred) +
                                 "dims in y_pred.")

            explained_model_output_dim_imputed = masked_data[2].shape[-1]
            if explained_model_output_dim_imputed == 2:
                masked_data[2] = masked_data[2][..., 0:1]
            elif explained_model_output_dim_imputed > 2:
                raise ValueError("Expected explained model predictions to have one or two output dimensions "
                                 "but got " + str(explained_model_output_dim_imputed) +
                                 "dims in y_pred_imputed.")

        return masked_data

    def _build_single(self, input_dim, output_dim):
        model, prediction_model = \
            self.model_builder.build_explanation_model(input_dim=input_dim,
                                                       output_dim=output_dim,
                                                       loss=self.loss,
                                                       downsample_factors=self.downsample_factors)
        return model, prediction_model

    def _build_ensemble(self, input_dim, output_dim):
        model, prediction_model = [], []
        for _ in range(self.num_models):
            cur_model, cur_prediction_model = self._build_single(input_dim=input_dim, output_dim=output_dim)
            model.append(cur_model)
            prediction_model.append(cur_prediction_model)
        return model, prediction_model

    def _build_model(self, X, y):
        Validation.check_dataset(X, y)

        if Validation.is_variable_length(X):
            raise ValueError("Variable length inputs to CXPlain are currently not supported.")

        n, p = Validation.get_input_dimension(X)
        output_dim = Validation.get_output_dimension(y)

        if self.model is None:
            if self.num_models == 1:
                build_fun = self._build_single
            else:
                build_fun = self._build_ensemble

            self.model, self.prediction_model = build_fun(input_dim=p, output_dim=output_dim)

    def _fit_single(self, model, X, y, masked_data=None):
        Validation.check_dataset(X, y)

        if len(X) != 0:
            # Pre-compute target outputs if none are passed.
            if masked_data is None:
                output_dim = Validation.get_output_dimension(y)
                masked_data = self.masking_operation.get_predictions_after_masking(self.explained_model, X, y,
                                                                                   batch_size=
                                                                                   self.model_builder.batch_size,
                                                                                   downsample_factors=
                                                                                   self.downsample_factors,
                                                                                   flatten=
                                                                                   self.flatten_for_explained_model)

                masked_data = TensorflowCXPlain._clean_output_dims(output_dim, masked_data)

            self.last_masked_data = masked_data

            if self.model_filepath is None:
                from tempfile import NamedTemporaryFile
                model_filepath = NamedTemporaryFile(delete=False).name
            else:
                model_filepath = self.model_filepath

            self.last_history = self.model_builder.fit(model, masked_data, y, model_filepath)
        return self

    def get_last_fit_score(self):
        """
        Retrieves the minimal loss observed during the last call to __fit__.

        :return: The minimal loss observed during the last call to __fit__.
        :exception AssertionError Thrown if __fit__ has not been called yet.
        """
        if self.last_history is None:
            raise AssertionError("__last_history__ is None. You must call __fit__ before getting the last fit score"
                                 "using __get_last_fit_score__.")

        best_idx = np.argmin(self.last_history.history["val_loss"])
        ret_val = {k: v[best_idx] for k, v in self.last_history.history.items()}
        return ret_val

    def _predict_single(self, model, X):
        return self.model_builder.predict(model, X)

    def _score_single(self, model, X, y, sample_weight):
        return_value = self.model_builder.evaluate(model, X, y, sample_weight)

        if isinstance(return_value, list):
            return_value = return_value[0]
        return return_value

    def score(self, X, y, sample_weight=None, masked_data=None):
        """
        Evaluates the performance, in terms of causal loss, of the current CXPlain model

        :param X: The data samples to be evaluated. The first dimension must be the number of samples. (Required)
        :param y: The ground truth labels to be compared to. The first dimension must be the number of
                  samples. (Required)
        :param sample_weight: The sample weight to apply to the samples in X during evaluation. The first dimension
                              must be the number of samples and it must match that of __X__ and __y__.
                              If None, equal weihting is used (Optional, default: None).
        :param masked_data: An array of precomputed masked data as can be obtained from __get_masked_data__.
                            If None, the masked data is computed. If set, the precomputed masked data is used for
                            scoring and computation of the masked data is skipped (Optional, default: None).
        :return: Score results as returned by self.model_builder.evaluate(model, X, y, sample_weight) either
                 (i) as a single score result if __num_models__ = 1 or as a list of score results
                 if __num_models__ is greater than 1.
        :exception AssertionError Thrown if the explanation model has not been fitted using __fit__ yet.
        """
        if self.model is None:
            raise AssertionError("Model must be initialised when calling __predict__. "
                                 "Did you forget to __fit__ the explanation model?")

        output_dim = Validation.get_output_dimension(y)
        if masked_data is None:
            masked_data = self.masking_operation.get_predictions_after_masking(self.explained_model, X, y,
                                                                               batch_size=
                                                                               self.model_builder.batch_size,
                                                                               downsample_factors=
                                                                               self.downsample_factors,
                                                                               flatten=
                                                                               self.flatten_for_explained_model)
            masked_data = TensorflowCXPlain._clean_output_dims(output_dim, masked_data)

        self.last_masked_data = masked_data

        if self.num_models == 1:
            return_value = self._score_single(self.model, masked_data, y, sample_weight)
        else:
            return_value = [self._score_single(model, masked_data, y, sample_weight) for model in self.model]
        return return_value

    def save(self, directory_path, overwrite=False, custom_model_saver=PickleModelSerialiser()):
        """
        Serialise a CXPlain instance to disk for reloading at a later point.

        If overwrite is set to True, the following files in __directory_path__ will be overwritten:
        - cxplain_config.json
        - cxplain_model_builder.pkl
        - cxplain_masking_operation.pkl
        - cxplain_loss.pkl
        - cxplain_explained_model + file extension
        - cxplain_model.h5 or cxplain_model_{model_index}.h5 for ensembles
        - cxplain_prediction_model.h5 or cxplain_prediction_model_{model_index}.h5 for ensembles

        :param directory_path: The directory path to which you wish to write the CXPlain instance. (Required)
        :param overwrite: Whether or not to overwrite any potentially already existing CXPlain instances
                          in __directory_path__. (Optional, default: False)
        :param custom_model_saver: A cxplain.backend.serialisation.ModelSerialiser instance that defines how the
                                   __self.explained_model__ will be serialised to disk. Uses pickle serialisation
                                   by default. Note not all models should be serialised via pickle, e.g. TensorFlow
                                   models. Use a cxplain.backend.serialisation.TensorFlowModelSerialiser for
                                   TensorFlow.Keras models.
                                   (Optional, default: A cxplain.backend.serialisation.PickleModelSerialiser)
        :exception ValueError Thrown if the __directory_path__ already contains a saved CXPlain instance and
                              __overwrite__ was not set to __True__.

        """
        directory_path = os.path.abspath(directory_path)
        if not os.path.exists(directory_path):
            os.mkdir(directory_path)

        already_exists_exception_message = "__directory_path__ already contains a saved CXPlain instance and" \
                                           " __overwrite__ was set to __False__. Conflicting file: {}"

        config_file_name = CXPlain.get_config_file_name()
        config_file_path = os.path.join(directory_path, config_file_name)
        if os.path.exists(config_file_path) and not overwrite:
            raise ValueError(already_exists_exception_message.format(config_file_path))
        else:
            with open(config_file_path, "w") as fp:
                json.dump(self.get_config(directory_path, custom_model_saver), fp)

        if self.prediction_model is not None:
            tensorflow_serialiser = TensorFlowModelSerialiser()
            if self.num_models > 1:
                for i, model in enumerate(self.prediction_model):
                    model_file_path = os.path.join(directory_path, CXPlain.get_prediction_model_h5_file_name(i))
                    if os.path.exists(model_file_path) and not overwrite:
                        raise ValueError(already_exists_exception_message.format(model_file_path))
                    else:
                        tensorflow_serialiser.save(model, model_file_path)
                for i, model in enumerate(self.model):
                    model_file_path = os.path.join(directory_path, CXPlain.get_model_h5_file_name(i))
                    if os.path.exists(model_file_path) and not overwrite:
                        raise ValueError(already_exists_exception_message.format(model_file_path))
                    else:
                        tensorflow_serialiser.save(model, model_file_path)
            else:
                model_file_path = os.path.join(directory_path, CXPlain.get_prediction_model_h5_file_name())
                if os.path.exists(model_file_path) and not overwrite:
                    raise ValueError(already_exists_exception_message.format(model_file_path))
                else:
                    tensorflow_serialiser.save(self.prediction_model, model_file_path)

                model_file_path = os.path.join(directory_path, CXPlain.get_model_h5_file_name())
                if os.path.exists(model_file_path) and not overwrite:
                    raise ValueError(already_exists_exception_message.format(model_file_path))
                else:
                    tensorflow_serialiser.save(self.model, model_file_path)

        explained_model_path = os.path.join(directory_path, CXPlain.get_explained_model_file_name(
            custom_model_saver.get_file_extension())
        )

        if os.path.exists(explained_model_path) and not overwrite:
            raise ValueError(already_exists_exception_message.format(explained_model_path))
        else:
            custom_model_saver.save(self.explained_model, explained_model_path)

        model_builder_path = os.path.join(directory_path, CXPlain.get_model_builder_pkl_file_name())
        if os.path.exists(model_builder_path) and not overwrite:
            raise ValueError(already_exists_exception_message.format(model_builder_path))
        else:
            with open(model_builder_path, "wb") as fp:
                pickle.dump(self.model_builder, fp)

        masking_path = os.path.join(directory_path, CXPlain.get_masking_operation_pkl_file_name())
        if os.path.exists(masking_path) and not overwrite:
            raise ValueError(already_exists_exception_message.format(masking_path))
        else:
            with open(masking_path, "wb") as fp:
                pickle.dump(self.masking_operation, fp)

        loss_path = os.path.join(directory_path, CXPlain.get_loss_pkl_file_name())
        if os.path.exists(loss_path) and not overwrite:
            raise ValueError(already_exists_exception_message.format(loss_path))
        else:
            with open(loss_path, "wb") as fp:
                pickle.dump(self.loss, fp)

    @staticmethod
    def load(directory_path, custom_model_loader=PickleModelSerialiser()):
        """
        Deserialise a CXPlain instance from disk.

        :param directory_path: The directory path to which the CXPlain instance had been written. (Required)
        :param custom_model_loader: A cxplain.backend.serialisation.ModelSerialiser instance that defines how the
                                    __self.explained_model__ will be deserialised from disk. Uses pickle serialisation
                                    by default. Note not all models should be deserialised via pickle, e.g. TensorFlow
                                    models. Use a cxplain.backend.serialisation.TensorFlowModelSerialiser for
                                    TensorFlow.Keras models.
                                    (Optional, default: A cxplain.backend.serialisation.PickleModelSerialiser)
        :return: The deserialised CXPlain instance.
        :exception AssertionError Thrown if the CXPlain instance at __directory_path__ is malformed.
        """
        config_file_name = CXPlain.get_config_file_name()
        config_file_path = os.path.join(directory_path, config_file_name)
        with open(config_file_path, "r") as fp:
            config = json.load(fp)

        with open(config["model_builder"], "rb") as model_builder_fp:
            model_builder = pickle.load(model_builder_fp)

        with open(config["masking_operation"], "rb") as masking_operation_fp:
            masking_operation = pickle.load(masking_operation_fp)

        with open(config["loss"], "rb") as loss_fp:
            loss = pickle.load(loss_fp)

        downsample_factors = config["downsample_factors"]
        flatten_for_explained_model = config["flatten_for_explained_model"]
        last_fit_score = config["last_fit_score"]
        model_filepath = config["model_filepath"]
        num_models = config["num_models"]
        prediction_model = config["prediction_model"]
        model = config["model"]

        if not isinstance(prediction_model, list) and num_models > 1 or num_models < 1:
            raise AssertionError("CXPlain instance at __directory_path__ was malformed.")

        if prediction_model is not None:
            tensorflow_serialiser = TensorFlowModelSerialiser()
            if num_models == 1:
                prediction_model = tensorflow_serialiser.load(prediction_model)
                model = tensorflow_serialiser.load(model)
            else:
                prediction_model = [tensorflow_serialiser.load(cur_model) for cur_model in prediction_model]
                model = [tensorflow_serialiser.load(cur_model) for cur_model in model]

        explained_model = custom_model_loader.load(config["explained_model"])

        instance = TensorflowCXPlain(explained_model, model_builder, masking_operation, loss,
                                     flatten_for_explained_model=flatten_for_explained_model,
                                     model_filepath=model_filepath,
                                     downsample_factors=downsample_factors,
                                     num_models=num_models)
        instance.prediction_model = prediction_model
        instance.model = model
        instance.last_fit_score = last_fit_score
        return instance
