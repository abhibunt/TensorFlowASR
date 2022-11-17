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

import typing

import tensorflow as tf

from tensorflow_asr.utils import math_util


def _rel_shift(x):
    x = tf.transpose(x, perm=[2, 3, 0, 1])  # BHNM -> NMBH
    x_shape = tf.shape(x)

    x = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])  # shift on position time dimension M
    x = tf.reshape(x, [x_shape[1] + 1, x_shape[0], x_shape[2], x_shape[3]])
    x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
    x = tf.reshape(x, x_shape)

    x = tf.transpose(x, perm=[2, 3, 0, 1])  # NMBH -> BHNM
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


def compute_self_attention_mask(max_length, inputs_length, mem_length=None, use_causal_mask=False):
    """
    Returns
    ```
    [[[True, True, True, False],
      [True, True, True, False],
      [True, True, True, False],
      [False, False, False, False]]]
    ```
    """
    qmask = tf.sequence_mask(inputs_length, maxlen=max_length)
    vmask = tf.sequence_mask(inputs_length + mem_length, maxlen=max_length + mem_length) if mem_length is not None else qmask
    attention_mask = qmask[:, :, None] & vmask[:, None, :]
    if use_causal_mask:
        attention_mask = attention_mask & compute_causal_mask(qmask, value=vmask)
    return attention_mask


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        head_size,
        output_size: int = None,
        dropout: float = 0.0,
        use_projection_bias: bool = True,
        return_attn_coef: bool = False,
        kernel_initializer: typing.Union[str, typing.Callable] = "variance_scaling",
        kernel_regularizer: typing.Union[str, typing.Callable] = None,
        kernel_constraint: typing.Union[str, typing.Callable] = None,
        bias_initializer: typing.Union[str, typing.Callable] = "zeros",
        bias_regularizer: typing.Union[str, typing.Callable] = None,
        bias_constraint: typing.Union[str, typing.Callable] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if output_size is not None and output_size < 1:
            raise ValueError("output_size must be a positive number")

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        self.head_size = head_size
        self.num_heads = num_heads
        self.output_size = output_size
        self.use_projection_bias = use_projection_bias
        self.return_attn_coef = return_attn_coef

        self.dropout = tf.keras.layers.Dropout(dropout, name="dropout")
        self._droput_rate = dropout

    def build(self, input_shape):
        num_query_features = input_shape[0][-1]
        num_key_features = input_shape[1][-1]
        num_value_features = input_shape[2][-1] if len(input_shape) > 2 else num_key_features
        output_size = self.output_size if self.output_size is not None else num_value_features
        self.query_kernel = self.add_weight(
            name="query_kernel",
            shape=[self.num_heads, num_query_features, self.head_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.key_kernel = self.add_weight(
            name="key_kernel",
            shape=[self.num_heads, num_key_features, self.head_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.value_kernel = self.add_weight(
            name="value_kernel",
            shape=[self.num_heads, num_value_features, self.head_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.projection_kernel = self.add_weight(
            name="projection_kernel",
            shape=[self.num_heads, self.head_size, output_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_projection_bias:
            self.projection_bias = self.add_weight(
                name="projection_bias",
                shape=[output_size],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.projection_bias = None

    def call_qkv(self, query, key, value):
        # verify shapes
        if key.shape[-2] != value.shape[-2]:
            raise ValueError("the number of elements in 'key' must be equal to the same as the number of elements in 'value'")
        # Linear transformations
        query = tf.einsum("BNI,HIO->BNHO", query, self.query_kernel)
        key = tf.einsum("BMI,HIO->BMHO", key, self.key_kernel)
        value = tf.einsum("BMI,HIO->BMHO", value, self.value_kernel)

        return query, key, value

    def call_attention(self, query, key, value, logits, training=False, attention_mask=None):
        # attention_mask with shape [B, Tquery, Tkey] with 1 is for positions we want to attend, 0 for masked
        if attention_mask is not None:
            if len(attention_mask.shape) < 2:
                raise ValueError("'mask' must have at least 2 dimensions")
            if query.shape[-3] != attention_mask.shape[-2]:
                raise ValueError("mask's second to last dimension must be equal to the number of elements in 'query'")
            if key.shape[-3] != attention_mask.shape[-1]:
                raise ValueError("mask's last dimension must be equal to the number of elements in 'key'")

        # apply mask
        if attention_mask is not None:  # possibly expand on the head dimension so broadcasting works
            if len(attention_mask.shape) != len(logits.shape):
                attention_mask = tf.expand_dims(attention_mask, -3)
            attn_coef = math_util.masked_fill(logits, attention_mask, value=math_util.large_compatible_negative(logits.dtype))
            attn_coef = tf.nn.softmax(attn_coef)
        else:
            attn_coef = tf.nn.softmax(logits)

        # attention dropout
        attn_coef_dropout = self.dropout(attn_coef, training=training)

        # attention * value
        multihead_output = tf.einsum("BHNM,BMHI->BNHI", attn_coef_dropout, value)

        # Run the outputs through another linear projection layer. Recombining heads
        # is automatically done.
        output = tf.einsum("BNHI,HIO->BNO", multihead_output, self.projection_kernel)

        if self.projection_bias is not None:
            output += self.projection_bias

        return output, attn_coef

    def call(self, inputs, mems=None, training=False, attention_mask=None):
        query, key, value = inputs

        if mems is not None:
            key = tf.concat([tf.cast(mems, dtype=key.dtype), key], 1)
            value = tf.concat([tf.cast(mems, dtype=value.dtype), value], 1)

        query, key, value = self.call_qkv(query, key, value)

        # Scale dot-product, doing the division to either query or key
        # instead of their product saves some computation
        depth = tf.constant(self.head_size, dtype=query.dtype)
        query /= tf.sqrt(depth)

        # Calculate dot product attention
        logits = tf.einsum("BNHO,BMHO->BHNM", query, key)

        output, attn_coef = self.call_attention(query, key, value, logits, training=training, attention_mask=attention_mask)

        if self.return_attn_coef:
            return output, attn_coef
        return output

    def compute_output_shape(self, input_shape):
        num_value_features = input_shape[2][-1] if len(input_shape) > 2 else input_shape[1][-1]
        output_size = self.output_size if self.output_size is not None else num_value_features

        output_shape = input_shape[0][:-1] + (output_size,)

        if self.return_attn_coef:
            num_query_elements = input_shape[0][-2]
            num_key_elements = input_shape[1][-2]
            attn_coef_shape = input_shape[0][:-2] + (
                self.num_heads,
                num_query_elements,
                num_key_elements,
            )
            return output_shape, attn_coef_shape

        return output_shape


class MultiHeadRelativeAttention(MultiHeadAttention):
    def build(self, input_shape):
        num_pos_features = input_shape[-1][-1]
        self.pos_kernel = self.add_weight(
            name="pos_kernel",
            shape=[self.num_heads, num_pos_features, self.head_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.pos_bias_u = self.add_weight(
            name="content_attention_bias",
            shape=[self.num_heads, self.head_size],
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
        )
        self.pos_bias_v = self.add_weight(
            name="positional_attention_bias",
            shape=[self.num_heads, self.head_size],
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
        )
        super().build(input_shape[:-1])

    def call(self, inputs, mems=None, training=False, attention_mask=None):
        query, key, value, pos = inputs

        if mems is not None:
            key = tf.concat([tf.cast(mems, dtype=key.dtype), key], 1)
            value = tf.concat([tf.cast(mems, dtype=value.dtype), value], 1)

        query, key, value = self.call_qkv(query, key, value)

        pos = tf.einsum("BMI,HIO->BMHO", pos, self.pos_kernel)

        query_with_u = query + self.pos_bias_u
        query_with_v = query + self.pos_bias_v

        logits_with_u = tf.einsum("BNHO,BMHO->BHNM", query_with_u, key)
        logits_with_v = tf.einsum("BNHO,BMHO->BHNM", query_with_v, pos)
        logits_with_v = _rel_shift(logits_with_v)

        logits = logits_with_u + logits_with_v

        depth = tf.constant(self.head_size, dtype=logits.dtype)
        logits /= tf.sqrt(depth)

        output, attn_coef = self.call_attention(query, key, value, logits, training=training, attention_mask=attention_mask)

        if self.return_attn_coef:
            return output, attn_coef
        return output
