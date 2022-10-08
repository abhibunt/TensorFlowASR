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


import math

import tensorflow as tf
from keras.layers import EinsumDense, MultiHeadAttention
from keras.layers.multi_head_attention import _build_proj_equation, _get_output_shape

from tensorflow_asr.utils import shape_util


def _rel_shift(x):
    x = tf.transpose(x, [2, 3, 0, 1])  # BHTS -> TSBH
    x_shape = tf.shape(x)

    x = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])
    x = tf.reshape(x, [x_shape[1] + 1, x_shape[0], x_shape[2], x_shape[3]])
    x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
    x = tf.reshape(x, x_shape)

    x = tf.transpose(x, [2, 3, 0, 1])  # TSBH -> BHTS
    return x


def compute_causal_mask(query, value=None):
    """Computes a causal mask (e.g., for masked self-attention layers).
    For example, if query and value both contain sequences of length 4,
    this function returns a boolean `Tensor` equal to:
    ```
    [[[True,  False, False, False],
      [True,  True,  False, False],
      [True,  True,  True,  False],
      [True,  True,  True,  True]]]
    ```
    Args:
      query: query `Tensor` of shape `(B, T, ...)`.
      value: value `Tensor` of shape `(B, S, ...)` (optional, defaults to
      query).
    Returns:
      mask: a boolean `Tensor` of shape [1, T, S] containing a lower
            triangular matrix of shape [T, S].
    """
    q_seq_length = tf.shape(query)[1]
    v_seq_length = q_seq_length if value is None else tf.shape(value)[1]
    return tf.linalg.band_part(tf.ones((1, q_seq_length, v_seq_length), tf.bool), -1, 0)  # creates a lower triangular matrix


def compute_self_attention_mask(inputs, inputs_length, use_causal_mask=False):
    """
    Returns
    ```
    [[[True, True, True, False],
      [True, True, True, False],
      [True, True, True, False],
      [False, False, False, False]]]
    ```
    """
    _, max_length, _ = shape_util.shape_list(inputs)
    mask = tf.sequence_mask(inputs_length, maxlen=max_length)
    attention_mask = mask[:, :, None] & mask[:, None, :]
    if use_causal_mask:
        attention_mask = attention_mask & compute_causal_mask(attention_mask)
    return attention_mask


class MultiHeadRelativeAttention(MultiHeadAttention):
    def __init__(
        self,
        num_heads,
        key_dim,
        value_dim=None,
        dropout=0.0,
        use_bias=True,
        output_shape=None,
        attention_axes=None,
        kernel_initializer="variance_scaling",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=value_dim,
            dropout=dropout,
            use_bias=use_bias,
            output_shape=output_shape,
            attention_axes=attention_axes,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )

    def _build_from_signature(self, query, value, key=None):
        super()._build_from_signature(query=query, value=value, key=key)
        if hasattr(value, "shape"):
            value_shape = tf.TensorShape(value.shape)
        else:
            value_shape = value
        if key is None:
            key_shape = value_shape
        elif hasattr(key, "shape"):
            key_shape = tf.TensorShape(key.shape)
        else:
            key_shape = key

        common_kwargs = dict(
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
        )

        with tf.init_scope():  # pylint: disable=not-context-manager
            einsum_equation, _, output_rank = _build_proj_equation(key_shape.rank - 1, bound_dims=1, output_dims=2)
            self._encoding_dense = EinsumDense(
                einsum_equation,
                output_shape=_get_output_shape(output_rank - 1, [self._num_heads, self._key_dim]),
                bias_axes=None,
                name="encoding",
                **common_kwargs,
            )

    def _compute_attention(
        self,
        query,
        key,
        value,
        position,
        content_attention_bias,
        positional_attention_bias,
        attention_mask=None,
        training=None,
    ):
        content_attention = tf.einsum(self._dot_product_equation, key, query + content_attention_bias)
        positional_attention = tf.einsum(self._dot_product_equation, position, query + positional_attention_bias)
        positional_attention = _rel_shift(positional_attention)
        attention_sum = content_attention + positional_attention

        attention_scores = tf.multiply(attention_sum, 1.0 / math.sqrt(float(self._key_dim)))
        attention_scores = self._masked_softmax(attention_scores, attention_mask)
        attention_output = self._dropout_layer(attention_scores, training=training)
        attention_output = tf.einsum(self._combine_equation, attention_output, value)  # BTNH

        return attention_output, attention_scores

    def call(
        self,
        query,
        value,
        relative_position_encoding,
        content_attention_bias,
        positional_attention_bias,
        key=None,
        state=None,
        attention_mask=None,
        return_attention_scores=False,
        training=None,
    ):
        if not self._built_from_signature:
            self._build_from_signature(query, value, key=key)
        if key is None:
            key = value
        if state is not None and state.shape.ndims > 1:
            value = tf.concat([state, value], 1)
            key = tf.concat([state, key], 1)

        # `query` = BTNH
        query = self._query_dense(query)
        # `key` = BSNH
        key = self._key_dense(key)
        # `value` = BSNH
        value = self._value_dense(value)
        # `position` = BLNH
        position = self._encoding_dense(relative_position_encoding)
        # `attention_output` = BTNH
        attention_output, attention_scores = self._compute_attention(
            query=query,
            key=key,
            value=value,
            position=position,
            content_attention_bias=content_attention_bias,
            positional_attention_bias=positional_attention_bias,
            attention_mask=attention_mask,
            training=training,
        )
        # `attention_output` = [B, T, output_shape]
        attention_output = self._output_dense(attention_output)

        if return_attention_scores:
            return attention_output, attention_scores
        return attention_output
