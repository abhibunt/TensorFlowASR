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


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, scalar: float = 10000.0, name="positional_encoding", **kwargs):
        super().__init__(name=name, **kwargs)
        self.scalar = tf.convert_to_tensor(scalar, dtype=tf.float32)

    def _create_encoding_matrix(self, batch_size, max_length, dmodel):
        pos = tf.expand_dims(tf.range(max_length - 1, -1, -1.0, dtype=tf.float32), axis=1)
        index = tf.expand_dims(tf.range(0, dmodel, dtype=tf.float32), axis=0)
        pe_matrix = pos * (1 / tf.pow(self.scalar, (2 * (index // 2)) / dmodel))
        # Sin cos will be [max_length, size // 2]
        # we add 0 between numbers by using padding and reshape
        sin = tf.pad(tf.expand_dims(tf.sin(pe_matrix[:, 0::2]), -1), [[0, 0], [0, 0], [0, 1]], mode="CONSTANT", constant_values=0)
        sin = tf.reshape(sin, [max_length, dmodel])
        cos = tf.pad(tf.expand_dims(tf.cos(pe_matrix[:, 1::2]), -1), [[0, 0], [0, 0], [1, 0]], mode="CONSTANT", constant_values=0)
        cos = tf.reshape(cos, [max_length, dmodel])
        # Then add sin and cos, which results in [max_length, dmodel]
        pe_matrix = tf.add(sin, cos)
        pe_matrix = tf.repeat(tf.expand_dims(pe_matrix, axis=0), batch_size, axis=0)  # [B, max_length, dmodel]
        return pe_matrix

    def call(self, inputs, inputs_length):
        batch_size, max_length, dmodel = shape_util.shape_list(inputs)
        pe_matrix = self._create_encoding_matrix(batch_size=batch_size, max_length=max_length, dmodel=dmodel)
        pe_matrix_mask = tf.sequence_mask(inputs_length, maxlen=max_length, dtype=pe_matrix.dtype)
        pe_matrix_mask = tf.repeat(tf.expand_dims(pe_matrix_mask, axis=-1), dmodel, axis=-1)
        pe_matrix = tf.multiply(pe_matrix, pe_matrix_mask)
        pe_matrix = tf.cast(pe_matrix, dtype=inputs.dtype)
        return pe_matrix
