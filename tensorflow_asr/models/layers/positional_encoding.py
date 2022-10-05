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

from tensorflow_asr.utils import shape_util


def compute_relative_position_encoding(
    input_shape,
    dtype=tf.float32,
):
    batch_size, max_length, dmodel = input_shape[0], input_shape[1], input_shape[2]
    pos_seq = tf.range(max_length - 1, -1, -1.0, dtype=tf.float32)
    inv_freq = 1 / (10000 ** (tf.range(0, dmodel, 2.0, dtype=tf.float32) / dmodel))
    sinusoid_inp = tf.einsum("i,j->ij", pos_seq, inv_freq)
    sinusoid_inp = tf.cast(sinusoid_inp, dtype=dtype)
    # # Sin cos will be [max_length, size // 2]
    # # we add 0 between numbers by using padding and reshape
    # sin = tf.pad(tf.expand_dims(tf.sin(pe_matrix[:, 0::2]), -1), [[0, 0], [0, 0], [0, 1]], mode="CONSTANT", constant_values=0)
    # sin = tf.reshape(sin, [max_length, dmodel])
    # cos = tf.pad(tf.expand_dims(tf.cos(pe_matrix[:, 1::2]), -1), [[0, 0], [0, 0], [1, 0]], mode="CONSTANT", constant_values=0)
    # cos = tf.reshape(cos, [max_length, dmodel])
    # # Then add sin and cos, which results in [max_length, dmodel]
    # pe_matrix = tf.add(sin, cos)
    pe = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)
    pe = tf.tile(pe[None, :, :], [batch_size, 1, 1])  # [B, max_length, dmodel]
    return pe


class PositionalEncoding(tf.keras.layers.Layer):
    def build(self, input_shape):
        dmodel = input_shape[-1]
        if dmodel % 2 != 0:
            raise ValueError("dmodel must be even")
        super().build(input_shape)

    def call(self, inputs, relative_position_encoding=None):
        if relative_position_encoding is None:
            pos_encoding = compute_relative_position_encoding(shape_util.shape_list(inputs), dtype=inputs.dtype)
        else:
            pos_encoding = relative_position_encoding
        pos_encoding = tf.cast(pos_encoding, dtype=inputs.dtype)
        outputs = tf.add(inputs, pos_encoding)
        return outputs
