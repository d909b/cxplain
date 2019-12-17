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
import itertools
import collections
import numpy as np
from cxplain.backend.validation import Validation
from cxplain.backend.numpy_math_interface import NumpyInterface


class MaskingUtil(object):
    @staticmethod
    def get_input_constants(input_dim, downsample_factors):
        if not isinstance(input_dim, collections.Sequence):
            input_dim = (input_dim,)

        if len(input_dim) > 1:
            num_channels = input_dim[-1]
            image_shape = input_dim[:-1]
        else:
            num_channels = 1
            image_shape = input_dim

        downsampling_factor = int(np.prod(downsample_factors))

        if downsampling_factor != 1:
            # Upsample the attention weights to match the input dimension.
            num_indices, steps = MaskingUtil.get_num_indices(image_shape, downsample_factors)
        else:
            has_variable_dimension = np.any(list(map(lambda x: x is None, image_shape)))
            if has_variable_dimension:
                num_indices = None
            else:
                num_indices = int(np.prod(image_shape))

            steps = image_shape
        return num_indices, num_channels, steps, downsampling_factor

    @staticmethod
    def get_num_indices(image_shape, downsample_factors):
        downsample_factors = MaskingUtil.extract_downsample_factors(downsample_factors,
                                                                    expected_length=len(image_shape))
        steps = np.ones((len(image_shape)))
        for idx, (delta, max) in enumerate(zip(downsample_factors, image_shape)):
            steps[idx] = int(np.ceil(max / float(delta)))
        return int(np.prod(steps)), steps

    @staticmethod
    def extract_downsample_factors(downsample_factors, expected_length):
        if not isinstance(downsample_factors, collections.Sequence):
            downsample_factors = (downsample_factors,)

        if len(downsample_factors) == 1:
            return (downsample_factors[0],)*expected_length
        else:
            if len(downsample_factors) != expected_length:
                raise ValueError(
                    "Dimension missmatch - downsample factors shoud match the dimension of the input data."
                    "Expected length: " + str(expected_length) + ", found: " + str(len(downsample_factors)) + "."
                )
            return downsample_factors

    @staticmethod
    def get_ith_mask1d(i, image_shape, downsample_factors, math_ops=NumpyInterface):
        dt = MaskingUtil.extract_downsample_factors(downsample_factors, expected_length=1)
        max_t = image_shape[0]
        mask = np.ones(downsample_factors)
        x_steps = int(np.ceil(max_t / float(dt)))
        index_array = np.arange(x_steps).astype(int)
        offset_t = index_array[i]
        mask = math_ops.pad(mask,
                            [(offset_t * dt, (x_steps - offset_t - 1) * dt)],
                            'constant', constant_values=0)
        return mask[:max_t]

    @staticmethod
    def get_ith_mask2d(i, image_shape, downsample_factors, math_ops=NumpyInterface):
        dx, dy = MaskingUtil.extract_downsample_factors(downsample_factors, expected_length=2)
        max_x, max_y = image_shape[:2]
        mask = np.ones(downsample_factors)
        x_steps, y_steps = int(np.ceil(max_x / float(dx))), int(np.ceil(max_y / float(dy)))
        index_array = np.array(list(itertools.product(np.arange(x_steps), np.arange(y_steps)))).astype(int)
        offset_x, offset_y = index_array[i]
        mask = math_ops.pad(mask,
                            [(offset_x * dx, (x_steps - offset_x - 1) * dx),
                             (offset_y * dy, (y_steps - offset_y - 1) * dy)],
                            'constant', constant_values=0)
        return mask[:max_x, :max_y]

    @staticmethod
    def get_ith_mask3d(i, image_shape, downsample_factors, math_ops=NumpyInterface):
        dx, dy, dz = MaskingUtil.extract_downsample_factors(downsample_factors, expected_length=3)
        max_x, max_y, max_z = image_shape[:3]
        mask = np.ones(downsample_factors)

        x_steps, y_steps, z_steps = int(np.ceil(max_x / float(dx))), \
                                    int(np.ceil(max_y / float(dy))), \
                                    int(np.ceil(max_z / float(dz))),

        index_array = np.array(list(itertools.product(np.arange(x_steps),
                                                      np.arange(y_steps),
                                                      np.arange(z_steps)))).astype(int)
        offset_x, offset_y, offset_z = index_array[i]
        mask = math_ops.pad(mask,
                            [(offset_x * dx, (x_steps - offset_x - 1) * dx),
                             (offset_y * dy, (y_steps - offset_y - 1) * dy),
                             (offset_z * dz, (z_steps - offset_z - 1) * dz)],
                            'constant', constant_values=0)
        return mask[:max_x, :max_y, :max_z]

    @staticmethod
    def get_ith_mask(i, image_shape, downsample_factors, math_ops=NumpyInterface):
        if len(image_shape) == 1:
            raise ValueError("Masking is not supported for inputs that are one dimensional.")
        elif len(image_shape) == 2:
            return MaskingUtil.get_ith_mask1d(i, image_shape, downsample_factors, math_ops)
        elif len(image_shape) == 3:
            return MaskingUtil.get_ith_mask2d(i, image_shape, downsample_factors, math_ops)
        elif len(image_shape) == 4:
            return MaskingUtil.get_ith_mask3d(i, image_shape, downsample_factors, math_ops)
        else:
            raise ValueError("Masking is currently not supported for inputs that are higher dimensional than 3D.")

    @staticmethod
    def predict_proxy(model, x):
        if hasattr(model, "predict_proba"):
            result = model.predict_proba(x)

            # TODO: Handle multi-class outputs
            if isinstance(result, list) and len(result) == 2:
                result = result[-1]

                assert np.allclose(np.argmax(result, axis=-1),
                                   np.argmax(model.predict(x), axis=-1),
                                   rtol=0, atol=0)
            return result
        else:
            return model.predict(x)

    @staticmethod
    def get_x_imputed(x, downsample_factors, math_ops):
        input_dim = np.array(x[0]).shape
        num_indices, num_channels, _, downsampling_factor =\
            MaskingUtil.get_input_constants(input_dim, downsample_factors)

        if num_indices is None:
            raise ValueError("Variable length inputs are currently not supported for ZeroMasking.")

        all_x_imputed = []
        for i in range(num_indices):
            def predict_with_i_imputed(x, index):
                x_imputed = math_ops.copy(x)
                original_shape = math_ops.shape(x_imputed)
                target_shape = (original_shape[0], math_ops.as_int(math_ops.prod(original_shape[1:])))

                if downsampling_factor == 1:
                    needs_reshape = len(original_shape) > 2
                    if needs_reshape:
                        x_imputed = math_ops.reshape(x_imputed, target_shape)

                    x_imputed[:, index] = 0

                    if needs_reshape:
                        x_imputed = math_ops.reshape(x_imputed, original_shape)
                else:
                    full_shape = Validation.get_full_input_shape(original_shape[0], input_dim)
                    mask = MaskingUtil.get_ith_mask(index, input_dim, downsample_factors, math_ops=math_ops)
                    x_imputed = math_ops.reshape(x_imputed, full_shape)

                    inverted_mask = (mask - 1.) * -1.
                    x_imputed = math_ops.multiply(x_imputed,
                                                  math_ops.expand_dims(math_ops.cast(inverted_mask, float),
                                                                       axis=-1))

                    # Transform to target shape = (num_samples, num_transformed_features).
                    x_imputed = math_ops.reshape(x_imputed, target_shape)

                return x_imputed

            current_x_imputed = predict_with_i_imputed(x, index=i)
            all_x_imputed.append(current_x_imputed)
        return all_x_imputed

    @staticmethod
    def get_prediction(model, x, flatten=False):
        if flatten:
            x = np.reshape(x, (len(x), -1))

        pred = MaskingUtil.predict_proxy(model, x).astype(np.float32)
        if len(pred.shape) < 2:
            pred = np.expand_dims(pred, axis=-1)
        return pred
