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

# class PositionalEncoding(tf.keras.layers.Layer):
#     def __init__(self, scalar: float = 10000.0, name="positional_encoding", **kwargs):
#         super().__init__(name=name, **kwargs)
#         self.scalar = tf.convert_to_tensor(scalar, dtype=tf.float32)

#     def build(self, input_shape):
#         output_shape, _ = input_shape
#         dmodel = output_shape[-1]
#         assert dmodel % 2 == 0, f"Input last dim must be even: {dmodel}"

#     def _create_encoding_matrix(self, batch_size, max_length, dmodel):
#         pos = tf.expand_dims(tf.range(max_length - 1, -1, -1.0, dtype=tf.float32), axis=1)
#         index = tf.expand_dims(tf.range(0, dmodel, dtype=tf.float32), axis=0)
#         pe_matrix = pos * (1 / tf.pow(self.scalar, (2 * (index // 2)) / dmodel))
#         # Sin cos will be [max_length, size // 2]
#         # we add 0 between numbers by using padding and reshape
#         sin = tf.pad(tf.expand_dims(tf.sin(pe_matrix[:, 0::2]), -1), [[0, 0], [0, 0], [0, 1]], mode="CONSTANT", constant_values=0)
#         sin = tf.reshape(sin, [max_length, dmodel])
#         cos = tf.pad(tf.expand_dims(tf.cos(pe_matrix[:, 1::2]), -1), [[0, 0], [0, 0], [1, 0]], mode="CONSTANT", constant_values=0)
#         cos = tf.reshape(cos, [max_length, dmodel])
#         # Then add sin and cos, which results in [max_length, dmodel]
#         pe_matrix = tf.add(sin, cos)
#         pe_matrix = tf.repeat(tf.expand_dims(pe_matrix, axis=0), batch_size, axis=0)  # [B, max_length, dmodel]
#         return pe_matrix

#     def call(self, inputs):
#         outputs, inputs_length = inputs
#         batch_size, max_length, dmodel = shape_util.shape_list(outputs)
#         pe_matrix = self._create_encoding_matrix(batch_size=batch_size, max_length=max_length, dmodel=dmodel)
#         pe_matrix_mask = tf.sequence_mask(inputs_length, maxlen=max_length, dtype=pe_matrix.dtype)
#         pe_matrix_mask = tf.repeat(tf.expand_dims(pe_matrix_mask, axis=-1), dmodel, axis=-1)
#         pe_matrix = tf.multiply(pe_matrix, pe_matrix_mask)
#         pe_matrix = tf.cast(pe_matrix, dtype=outputs.dtype)
#         pe = tf.add(outputs, pe_matrix)
#         return pe


def compute_positional_encoding(
    hidden_size,
    pos_seq,
    batch_size=None,
):
    inv_freq = 1.0 / (10000.0 ** (tf.range(0, hidden_size, 2.0) / hidden_size))
    sinusoid_input = tf.einsum("i,d->id", tf.cast(pos_seq, dtype=inv_freq.dtype), inv_freq)
    relative_position_encoding = tf.concat([tf.sin(sinusoid_input), tf.cos(sinusoid_input)], -1)
    relative_position_encoding = relative_position_encoding[None, :, :]
    if batch_size is not None:
        relative_position_encoding = tf.tile(relative_position_encoding, [batch_size, 1, 1])
    return relative_position_encoding


def compute_attention_positional_encoding(
    hidden_size,
    total_length,
    seq_length,
    attention_type="bi",
    batch_size=None,
    clamp_length=-1,
    dtype=tf.float32,
):
    """Computes the relative position encoding.
    Args:
      attention_type: str, the attention type. Can be "uni" (directional) or "bi" (directional).
      hidden_size: int, the hidden size.
      batch_size: int, the batch size.
      total_length: int, the sequence length added to the memory length.
      seq_length: int, the length of each sequence.
      clamp_length: int, clamp all relative distances larger than clamp_length. -1 means no clamping.
      dtype: the dtype of the encoding.
    Returns:
      A Tensor, representing the position encoding.
    """
    freq_seq = tf.range(0, hidden_size, 2.0)
    if dtype is not None and dtype != tf.float32:
        freq_seq = tf.cast(freq_seq, dtype=dtype)

    if attention_type == "bi":
        beg, end = total_length, -seq_length
    elif attention_type == "uni":
        beg, end = total_length, -1
    else:
        raise ValueError(f"Unknown `attention_type` {attention_type}.")

    forward_position_sequence = tf.range(beg, end, -1.0)
    if dtype is not None and dtype != tf.float32:
        forward_position_sequence = tf.cast(forward_position_sequence, dtype=dtype)
    if clamp_length > 0:
        forward_position_sequence = tf.clip_by_value(forward_position_sequence, -clamp_length, clamp_length)

    relative_position_encoding = compute_positional_encoding(hidden_size, forward_position_sequence, batch_size)
    return relative_position_encoding


class RelativeAttentionPositionEncoding(tf.keras.layers.Layer):
    def __init__(
        self,
        attention_type="bi",
        clamp_length=-1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._attention_type = attention_type
        self._clamp_length = clamp_length

    def call(self, inputs, inputs_length):
        _, max_length, dmodel = shape_util.shape_list(inputs)
        list_inputs_length = tf.unstack(inputs_length, axis=0)
        list_rel_pe = tf.nest.map_structure(
            lambda x: compute_attention_positional_encoding(
                hidden_size=dmodel,
                total_length=max_length,
                seq_length=x,
                attention_type=self._attention_type,
                dtype=inputs.dtype,
            ),
            list_inputs_length,
        )
        return tf.concat(list_rel_pe, 0)
