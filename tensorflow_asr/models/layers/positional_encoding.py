# pylint: disable=attribute-defined-outside-init
# Copyright 2022 Huy Le Nguyen (@usimarit)
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

from tensorflow_asr.utils import shape_util


def compute_sinusoid_position_encoding(
    batch_size,
    max_length,
    dmodel,
    dtype=tf.float32,
):
    # length of sequence is the second last dimension of the inputs
    position = tf.cast(tf.range(max_length), dtype)
    min_freq = tf.cast(1 / 10000.0, dtype=dtype)
    timescales = tf.pow(min_freq, tf.cast(2 * (tf.range(dmodel) // 2), dtype) / tf.cast(dmodel, dtype))
    angles = tf.expand_dims(position, 1) * tf.expand_dims(timescales, 0)
    # even indices are sine, odd are cosine
    cos_mask = tf.cast(tf.range(dmodel) % 2, dtype)
    sin_mask = 1 - cos_mask
    # embedding shape is [seq_length, hidden_size]
    positional_encodings = tf.sin(angles) * sin_mask + tf.cos(angles) * cos_mask
    return tf.tile(positional_encodings[None, :, :], [batch_size, 1, 1])


class RelativePositionalEncoding(tf.keras.layers.Layer):
    def build(self, input_shape):
        dmodel = input_shape[-1]
        if dmodel % 2 != 0:
            raise ValueError("dmodel must be even")
        super().build(input_shape)
        self._pos_encoding = None
        self._center_pos = None
        self._max_length = input_shape[1]
        if input_shape[1] is not None:
            self._compute_pos_encoding(batch_size=input_shape[0], max_length=input_shape[1], dmodel=input_shape[2])

    def _compute_pos_encoding(self, batch_size, max_length, dmodel):
        self._pos_encoding = compute_sinusoid_position_encoding(batch_size=batch_size, max_length=max_length, dmodel=dmodel, dtype=tf.float32)
        self._center_pos = tf.shape(self._pos_encoding)[1] // 2 + 1
        self._max_length = max_length

    def call(self, inputs, cache_length=0):
        batch_size, max_length, dmodel = shape_util.shape_list(inputs)
        max_length += cache_length
        if self._pos_encoding is None:
            self._compute_pos_encoding(batch_size=batch_size, max_length=max_length, dmodel=dmodel)
        tf.cond(
            tf.not_equal(self._max_length, max_length),
            true_fn=lambda: self._compute_pos_encoding(batch_size=batch_size, max_length=max_length, dmodel=dmodel),
            false_fn=lambda: None,
        )
        start_pos = self._center_pos - max_length
        end_pos = self._center_pos + max_length - 1
        pos_encoding = tf.slice(self._pos_encoding, [0, start_pos, 0], [-1, end_pos, -1])
        return pos_encoding
