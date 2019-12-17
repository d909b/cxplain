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
import tensorflow as tf
from nose.tools import nottest
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler


@nottest
class TestUtil(object):
    @staticmethod
    def get_classification_models():
        models = [
            LogisticRegression(random_state=1),
            RandomForestClassifier(n_estimators=64, max_depth=5, random_state=1),
        ]
        return models

    @staticmethod
    def fit_proxy(explained_model, x, y):
        if isinstance(explained_model, LogisticRegression):
            y_cur = np.argmax(y, axis=-1)
        else:
            y_cur = y
        explained_model.fit(x, y_cur)

    @staticmethod
    def split_auxiliary_outputs_on_feature_dim(all_but_one_auxiliary_outputs):
        return list(map(np.squeeze, np.split(all_but_one_auxiliary_outputs,
                                             all_but_one_auxiliary_outputs.shape[1],
                                             axis=1)))

    @staticmethod
    def get_boston_housing(x_scaler=StandardScaler(), y_scaler=StandardScaler()):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data()

        if x_scaler is not None:
            x_scaler.fit(x_train)
            x_train = x_scaler.transform(x_train)
            x_test = x_scaler.transform(x_test)

        if y_scaler is not None:
            y_train = np.expand_dims(y_train, axis=-1)
            y_test = np.expand_dims(y_test, axis=-1)
            y_scaler.fit(y_train)
            y_train = y_scaler.transform(y_train)
            y_test = y_scaler.transform(y_test)
            y_train = y_train[:, 0]
            y_test = y_test[:, 0]

        return (x_train, y_train), (x_test, y_test)

    @staticmethod
    def get_boston_housing_feature_names():
        feature_names = [
            "CRIM - per capita crime rate by town",
            "ZN - proportion of residential land zoned for lots over 25,000 sq.ft.",
            "INDUS - proportion of non-retail business acres per town.",
            "CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)",
            "NOX - nitric oxides concentration (parts per 10 million)",
            "RM - average number of rooms per dwelling",
            "AGE - proportion of owner-occupied units built prior to 1940",
            "DIS - weighted distances to five Boston employment centres",
            "RAD - index of accessibility to radial highways",
            "TAX - full-value property-tax rate per $10,000",
            "PTRATIO - pupil-teacher ratio by town",
            "B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town",
            "LSTAT - % lower status of the population",
            "MEDV - Median value of owner-occupied homes in $1000's"
        ]
        return feature_names

    @staticmethod
    def get_imdb(word_dictionary_size=1024, num_subsamples=None):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=word_dictionary_size)

        x_train = np.array([np.expand_dims(xx, axis=-1) for xx in x_train])
        x_test = np.array([np.expand_dims(xx, axis=-1) for xx in x_test])

        assert len(x_train[0].shape) == len(x_test[0].shape) == 2
        assert x_train[0].shape[-1] == x_test[0].shape[-1] == 1

        if num_subsamples is not None:
            x_train = x_train[:num_subsamples]
            y_train = y_train[:num_subsamples]
            x_test = x_test[:num_subsamples]
            y_test = y_test[:num_subsamples]

        return (x_train, y_train), (x_test, y_test)

    @staticmethod
    def get_imdb_word_dictionary(index_from=3):
        word_to_id = tf.keras.datasets.imdb.get_word_index()
        word_to_id = {k: (v + index_from) for k, v in word_to_id.items()}
        word_to_id["<PAD>"] = 0
        word_to_id["<START>"] = 1
        word_to_id["<UNK>"] = 2
        word_to_id["<UNUSED>"] = 3

        id_to_word = {value: key for key, value in word_to_id.items()}
        return id_to_word

    @staticmethod
    def imdb_dictionary_indidces_to_words(indices, index_from=3):
        id_to_word = TestUtil.get_imdb_word_dictionary(index_from=index_from)
        indices = np.squeeze(indices)
        return [id_to_word[idx] for idx in indices]

    @staticmethod
    def get_mnist(flattened=False, num_subsamples=None):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        x_train, x_test = x_train.astype(float), x_test.astype(float)
        original_shape = x_train.shape[1:]
        x_train = x_train.reshape((len(x_train), -1))
        x_test = x_test.reshape((len(x_test), -1))

        x_train /= 255.
        x_test /= 255.

        if not flattened:
            x_train = x_train.reshape((len(x_train),) + original_shape)
            x_test = x_test.reshape((len(x_test),) + original_shape)

        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)

        y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

        if num_subsamples is not None:
            x_train = x_train[:num_subsamples]
            y_train = y_train[:num_subsamples]
            x_test = x_test[:num_subsamples]
            y_test = y_test[:num_subsamples]

        return (x_train, y_train), (x_test, y_test)

    @staticmethod
    def get_cifar10(x_scaler=MinMaxScaler(), flattened=False, num_subsamples=None):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        x_train, x_test = x_train.astype(float), x_test.astype(float)
        original_shape = x_train.shape[1:]
        x_train = x_train.reshape((len(x_train), -1))
        x_test = x_test.reshape((len(x_test), -1))

        if x_scaler is not None:
            x_scaler.fit(x_train)
            x_train = x_scaler.transform(x_train)
            x_test = x_scaler.transform(x_test)

        if not flattened:
            x_train = x_train.reshape((len(x_train),) + original_shape)
            x_test = x_test.reshape((len(x_test),) + original_shape)

        y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

        if num_subsamples is not None:
            x_train = x_train[:num_subsamples]
            y_train = y_train[:num_subsamples]
            x_test = x_test[:num_subsamples]
            y_test = y_test[:num_subsamples]

        return (x_train, y_train), (x_test, y_test)

    @staticmethod
    def get_test_dataset_with_one_oracle_feature(num_samples, num_classes=2):
        y = np.random.randint(0, 2, num_samples)
        x = np.column_stack([y, np.zeros_like(y), np.zeros_like(y), np.zeros_like(y)])
        y = tf.keras.utils.to_categorical(y, num_classes=num_classes)
        return x, y

    @staticmethod
    def get_test_dataset_with_two_oracle_features(num_samples, num_classes=2):
        y = np.random.randint(0, 2, num_samples)
        x = np.column_stack([y, y])
        y = tf.keras.utils.to_categorical(y, num_classes=num_classes)
        return x, y

    @staticmethod
    def get_test_dataset_with_two_oracle_features_plus_one_random_noise(num_samples, num_classes=2):
        y = np.random.randint(0, 2, num_samples)
        x = np.column_stack([y, y, np.random.random_sample(num_samples)])
        y = tf.keras.utils.to_categorical(y, num_classes=num_classes)
        return x, y

    @staticmethod
    def get_test_dataset_with_confounded_input(num_samples, num_classes=2):
        y = np.random.randint(0, 2, num_samples)

        confounder = np.random.random_sample(num_samples)
        x1 = 2*confounder - 2*np.random.normal(0, 0.1, num_samples)
        x2 = -1.25*confounder + 2*np.random.normal(0, 0.1, num_samples)

        x = np.column_stack([x1, x2])
        y = tf.keras.utils.to_categorical(y, num_classes=num_classes)
        return x, y

    @staticmethod
    def get_random_variable_length_dataset(num_samples=256, shape=(None, 1),
                                           test_set_fraction=0.2, num_classes=2,
                                           max_variable_len=100, max_value=1024):
        test_set_fraction = np.clip(test_set_fraction, 0, 1)
        num_test_samples = int(np.ceil(test_set_fraction*num_samples))
        num_train_samples = num_samples - num_test_samples

        def replace_none_with_random(the_shape):
            out_shape = []
            for dim in the_shape:
                if dim is None:
                    out_shape += [np.random.randint(0, max_variable_len)]
                else:
                    out_shape += [dim]
            return tuple(out_shape)

        x = [np.random.randint(0, max_value, size=replace_none_with_random(shape)) for _ in range(num_samples)]
        y = np.array([np.random.randint(0, num_classes) for _ in range(num_samples)])

        x_train = x[:num_train_samples]
        x_test = x[num_train_samples:]
        y_train = y[:num_train_samples]
        y_test = y[num_train_samples:]

        if num_classes != 2:
            y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
            y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

        return (x_train, y_train), (x_test, y_test)

    @staticmethod
    def get_random_fixed_length_dataset(num_samples=256, shape=(None, 1), std_dev=15.0,
                                        test_set_fraction=0.2, num_classes=2, fixed_length=100):
        test_set_fraction = np.clip(test_set_fraction, 0, 1)
        num_test_samples = int(np.ceil(test_set_fraction*num_samples))
        num_train_samples = num_samples - num_test_samples

        def replace_none_with_fixed(the_shape):
            out_shape = []
            for dim in the_shape:
                if dim is None:
                    out_shape += [fixed_length]
                else:
                    out_shape += [dim]
            return tuple(out_shape)

        x = np.array([np.random.normal(0, std_dev, size=replace_none_with_fixed(shape)) for _ in range(num_samples)])
        y = np.array([np.random.randint(0, num_classes) for _ in range(num_samples)])

        x_train = x[:num_train_samples]
        x_test = x[num_train_samples:]
        y_train = y[:num_train_samples]
        y_test = y[num_train_samples:]

        if num_classes != 2:
            y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
            y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

        return (x_train, y_train), (x_test, y_test)
