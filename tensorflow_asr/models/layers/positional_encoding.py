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

    def build(self, input_shape):
        output_shape, _ = input_shape
        dmodel = output_shape[-1]
        assert dmodel % 2 == 0, f"Input last dim must be even: {dmodel}"

    def _create_encoding_matrix(self, sequence_length, max_length, dmodel):
        pos = tf.expand_dims(tf.range(sequence_length - 1, -1, -1.0, dtype=tf.float32), axis=1)
        index = tf.expand_dims(tf.range(0, dmodel, dtype=tf.float32), axis=0)
        pe_matrix = pos * (1 / tf.pow(self.scalar, (2 * (index // 2)) / dmodel))
        # Sin cos will be [sequence_length, size // 2]
        # we add 0 between numbers by using padding and reshape
        sin = tf.pad(tf.expand_dims(tf.sin(pe_matrix[:, 0::2]), -1), [[0, 0], [0, 0], [0, 1]], mode="CONSTANT", constant_values=0)
        sin = tf.reshape(sin, [sequence_length, dmodel])
        cos = tf.pad(tf.expand_dims(tf.cos(pe_matrix[:, 1::2]), -1), [[0, 0], [0, 0], [1, 0]], mode="CONSTANT", constant_values=0)
        cos = tf.reshape(cos, [sequence_length, dmodel])
        # Then add sin and cos, which results in [sequence_length, dmodel]
        pe_matrix = tf.add(sin, cos)
        # pad to [max_length, dmodel]
        pe_matrix = tf.pad(pe_matrix, [[0, max_length - sequence_length], [0, 0]], mode="CONSTANT", constant_values=0)
        return pe_matrix

    def call(self, inputs):
        outputs, inputs_length = inputs
        _, max_length, dmodel = shape_util.shape_list(outputs)
        pe_matrix = tf.vectorized_map(
            fn=lambda x: self._create_encoding_matrix(x, max_length=max_length, dmodel=dmodel),
            elems=inputs_length,
            warn=False,
        )
        pe_matrix = tf.cast(pe_matrix, dtype=outputs.dtype)
        pe = tf.add(outputs, pe_matrix)
        return pe
