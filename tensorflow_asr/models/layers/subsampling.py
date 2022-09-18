# Copyright 2020 Huy Le Nguyen (@usimarit)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf

from tensorflow_asr.models.layers.blurpool import BlurPool1D, BlurPool2D
from tensorflow_asr.utils import math_util, shape_util


class TimeReduction(tf.keras.layers.Layer):
    def __init__(
        self,
        factor: int,
        name: str = "TimeReduction",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.time_reduction_factor = factor

    def padding(
        self,
        time,
    ):
        new_time = tf.math.ceil(time / self.time_reduction_factor) * self.time_reduction_factor
        return tf.cast(new_time, dtype=tf.int32) - time

    def call(
        self,
        inputs,
        **kwargs,
    ):
        shape = shape_util.shape_list(inputs)
        outputs = tf.pad(inputs, [[0, 0], [0, self.padding(shape[1])], [0, 0]])
        outputs = tf.reshape(outputs, [shape[0], -1, shape[-1] * self.time_reduction_factor])
        return outputs


class VggSubsampling(tf.keras.layers.Layer):
    def __init__(
        self,
        filters: tuple or list = (32, 64),
        kernel_size: int or list or tuple = 3,
        pool_size: int or list or tuple = 2,
        strides: int or list or tuple = 2,
        padding: str = "same",
        activation: str = "relu",
        kernel_regularizer=None,
        bias_regularizer=None,
        name="VggSubsampling",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters[0],
            kernel_size=kernel_size,
            strides=1,
            padding=padding,
            name=f"{name}_conv_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activation=activation,
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=filters[0],
            kernel_size=kernel_size,
            strides=1,
            padding=padding,
            name=f"{name}_conv_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activation=activation,
        )
        self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=strides, padding=padding, name=f"{name}_maxpool_1")
        self.conv3 = tf.keras.layers.Conv2D(
            filters=filters[1],
            kernel_size=kernel_size,
            strides=1,
            padding=padding,
            name=f"{name}_conv_3",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activation=activation,
        )
        self.conv4 = tf.keras.layers.Conv2D(
            filters=filters[1],
            kernel_size=kernel_size,
            strides=1,
            padding=padding,
            name=f"{name}_conv_4",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activation=activation,
        )
        self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=strides, padding=padding, name=f"{name}_maxpool_2")
        self.time_reduction_factor = self.maxpool1.pool_size[0] * self.maxpool2.pool_size[0]

    def call(
        self,
        inputs,
        training=False,
        **kwargs,
    ):
        outputs = self.conv1(inputs, training=training)
        outputs = self.conv2(outputs, training=training)
        outputs = self.maxpool1(outputs, training=training)

        outputs = self.conv3(outputs, training=training)
        outputs = self.conv4(outputs, training=training)
        outputs = self.maxpool2(outputs, training=training)

        return math_util.merge_two_last_dims(outputs)


class Conv2dSubsampling(tf.keras.layers.Layer):
    def __init__(
        self,
        filters: int,
        strides: list or tuple or int = 2,
        kernel_size: int or list or tuple = 3,
        padding: str = "same",
        activation: str = "relu",
        kernel_regularizer=None,
        bias_regularizer=None,
        name="Conv2dSubsampling",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            name=f"{name}_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activation=activation,
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            name=f"{name}_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activation=activation,
        )
        self.time_reduction_factor = self.conv1.strides[0] * self.conv2.strides[0]

    def call(
        self,
        inputs,
        training=False,
        **kwargs,
    ):
        outputs = self.conv1(inputs, training=training)
        outputs = self.conv2(outputs, training=training)
        return math_util.merge_two_last_dims(outputs)


class Conv1dSubsampling(tf.keras.layers.Layer):
    def __init__(
        self,
        filters: int,
        strides: int = 2,
        kernel_size: int = 3,
        padding: str = "causal",
        activation: str = "relu",
        kernel_regularizer=None,
        bias_regularizer=None,
        name="Conv1dSubsampling",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.conv1 = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            name=f"{name}_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activation=activation,
        )
        self.conv2 = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            name=f"{name}_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activation=activation,
        )
        self.time_reduction_factor = self.conv1.strides[0] * self.conv2.strides[0]

    def call(
        self,
        inputs,
        training=False,
        **kwargs,
    ):
        outputs = math_util.merge_two_last_dims(inputs)
        outputs = self.conv1(outputs, training=training)
        outputs = self.conv2(outputs, training=training)
        return outputs


class Conv2dBlurPoolSubsampling(tf.keras.layers.Layer):
    def __init__(
        self,
        filters: int,
        strides: int = 2,
        kernel_size: int = 3,
        conv_padding: str = "same",
        pool_padding: str = "reflect",
        activation: str = "relu",
        kernel_regularizer=None,
        bias_regularizer=None,
        name="Conv2dBlurPoolSubsampling",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding=conv_padding,
            name=f"{name}_conv_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activation=activation,
        )
        self.blur_pool_1 = BlurPool2D(
            filters=filters, kernel_size=kernel_size, strides=strides, padding=pool_padding, name=f"{name}_blur_pool_1"
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding=conv_padding,
            name=f"{name}_conv_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activation=activation,
        )
        self.blur_pool_2 = BlurPool2D(
            filters=filters, kernel_size=kernel_size, strides=strides, padding=pool_padding, name=f"{name}_blur_pool_2"
        )
        self.time_reduction_factor = self.blur_pool_1.strides * self.blur_pool_2.strides

    def call(
        self,
        inputs,
        training=False,
        **kwargs,
    ):
        outputs = self.conv1(inputs, training=training)
        outputs = self.blur_pool_1(outputs)
        outputs = self.conv2(outputs, training=training)
        outputs = self.blur_pool_2(outputs)
        return math_util.merge_two_last_dims(outputs)


class Conv1dBlurPoolSubsampling(tf.keras.layers.Layer):
    def __init__(
        self,
        filters: int,
        strides: int = 2,
        kernel_size: int = 3,
        conv_padding: str = "causal",
        pool_padding: str = "reflect",
        activation: str = "relu",
        kernel_regularizer=None,
        bias_regularizer=None,
        name="Conv1dBlurPoolSubsampling",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.conv1 = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding=conv_padding,
            name=f"{name}_conv_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activation=activation,
        )
        self.blur_pool_1 = BlurPool1D(
            filters=filters, kernel_size=kernel_size, strides=strides, padding=pool_padding, name=f"{name}_blur_pool_1"
        )
        self.conv2 = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding=conv_padding,
            name=f"{name}_conv_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activation=activation,
        )
        self.blur_pool_2 = BlurPool1D(
            filters=filters, kernel_size=kernel_size, strides=strides, padding=pool_padding, name=f"{name}_blur_pool_2"
        )
        self.time_reduction_factor = self.blur_pool_1.strides * self.blur_pool_2.strides

    def call(
        self,
        inputs,
        training=False,
        **kwargs,
    ):
        outputs = math_util.merge_two_last_dims(inputs)
        outputs = self.conv1(outputs, training=training)
        outputs = self.blur_pool_1(outputs)
        outputs = self.conv2(outputs, training=training)
        outputs = self.blur_pool_2(outputs)
        return outputs
