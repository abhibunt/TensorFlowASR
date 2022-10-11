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

try:
    from keras.layers.multi_head_attention import _build_proj_equation, _get_output_shape
except ImportError:
    from keras.layers.attention.multi_head_attention import _build_proj_equation, _get_output_shape

from tensorflow_asr.utils import shape_util


def _rel_shift(x):
    x = tf.transpose(x, perm=[2, 3, 0, 1])  # BNTR -> TRBN
    x_shape = tf.shape(x)

    x = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])  # shift on position time dimension
    x = tf.reshape(x, [x_shape[1] + 1, x_shape[0], x_shape[2], x_shape[3]])
    x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
    x = tf.reshape(x, x_shape)

    x = tf.transpose(x, perm=[2, 3, 0, 1])  # TRBN -> BNTR
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
    [[[True, True, True, True],
      [True, True, True, True],
      [True, True, True, True],
      [False, False, False, False]]]
    ```
    """
    _, max_length, _ = shape_util.shape_list(inputs)
    mask = tf.sequence_mask(inputs_length, maxlen=max_length)
    attention_mask = tf.tile(mask[:, :, None], [1, 1, max_length])  # BTS
    if use_causal_mask:
        attention_mask = attention_mask & compute_causal_mask(attention_mask)
    return attention_mask


class MultiHeadRelativeAttention(MultiHeadAttention):
    """A multi-head attention layer with relative attention + position encoding.
    This layer shares the same input/output projections as the common
    `tf.keras.layers.MultiHeadAttention` layer.
    When it calculates attention logits, position encoding is projected to form
    relative keys. The logits are composed by shifted relative logits and content
    logits.
    **Note: This layer is currently experimental.
    Attributes:
      kernel_initializer: The kernel initializer. Defaults to variance_scaling.
    Call args:
      query: Query `Tensor` of shape `[B, T, dim]`.
      value: Value `Tensor` of shape `[B, S, dim]`.
      content_attention_bias: Bias `Tensor` for content based attention of shape
        `[num_heads, dim]`.
      positional_attention_bias: Bias `Tensor` for position based attention of
        shape `[num_heads, dim]`.
      key: Optional key `Tensor` of shape `[B, S, dim]`. If not given, will use
        `value` for both `key` and `value`, which is the most common case.
      relative_position_encoding: Relative positional encoding `Tensor` of shape
        `[B, L, dim]`.
      segment_matrix: Optional `Tensor` representing segmentation IDs used in
        XLNet of shape `[B, S, S + M]`.
      segment_encoding: Optional `Tensor` representing the segmentation encoding
        as used in XLNet of shape `[2, num_heads, dim]`.
      segment_attention_bias: Optional trainable bias parameter added to the query
        had when calculating the segment-based attention score used in XLNet of
        shape `[num_heads, dim]`.
      state: Optional `Tensor` of shape `[B, M, E]` where M is the length of the
        state or memory. If passed, this is also attended over as in Transformer
        XL.
      attention_mask: A boolean mask of shape `[B, T, S]` that prevents attention
        to certain positions.
    """

    def __init__(
        self,
        kernel_initializer="variance_scaling",
        **kwargs,
    ):
        super().__init__(kernel_initializer=kernel_initializer, **kwargs)

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
        segment_matrix=None,
        segment_encoding=None,
        segment_attention_bias=None,
        attention_mask=None,
        training=None,
    ):
        """Computes the attention.
        This function defines the computation inside `call` with projected
        multihead Q, K, V, R inputs.
        Args:
          query: Projected query `Tensor` of shape `[B, T, N, key_dim]`.
          key: Projected key `Tensor` of shape `[B, S + M, N, key_dim]`.
          value: Projected value `Tensor` of shape `[B, S + M, N, key_dim]`.
          position: Projected position `Tensor` of shape `[B, L, N, key_dim]`.
          content_attention_bias: Trainable bias parameter added to the query head
            when calculating the content-based attention score.
          positional_attention_bias: Trainable bias parameter added to the query
            head when calculating the position-based attention score.
          segment_matrix: Optional `Tensor` representing segmentation IDs used in
            XLNet.
          segment_encoding: Optional trainable `Tensor` representing the
            segmentation encoding as used in XLNet.
          segment_attention_bias: Optional trainable bias parameter added to the
            query had when calculating the segment-based attention score used in
            XLNet.
          attention_mask: (default None) Optional mask that is added to attention
            logits. If state is not None, the mask source sequence dimension should
            extend M.
        Returns:
          attention_output: Multi-headed output of attention computation of shape
            `[B, S, N, key_dim]`.
        """
        content_attention = tf.einsum(self._dot_product_equation, key, query + content_attention_bias)  # BSNH,BTNH->BNTS
        positional_attention = tf.einsum(self._dot_product_equation, position, query + positional_attention_bias)  # BRNH,BTNH->BNTR
        positional_attention = _rel_shift(positional_attention)

        if segment_matrix is not None:
            segment_attention = tf.einsum("bind,snd->bnis", query + segment_attention_bias, segment_encoding)
            target_shape = tf.shape(positional_attention)
            segment_attention = tf.where(
                tf.broadcast_to(tf.expand_dims(segment_matrix, 1), target_shape),
                tf.broadcast_to(segment_attention[:, :, :, 1:], target_shape),
                tf.broadcast_to(segment_attention[:, :, :, :1], target_shape),
            )
            attention_sum = content_attention + positional_attention + segment_attention
        else:
            attention_sum = content_attention + positional_attention

        attention_scores = tf.multiply(attention_sum, 1.0 / math.sqrt(float(self._key_dim)))

        attention_scores = self._masked_softmax(attention_scores, attention_mask)

        attention_output = self._dropout_layer(attention_scores, training=training)

        attention_output = tf.einsum(self._combine_equation, attention_output, value)  # BNTS,BVNH->BTNH
        return attention_output

    def call(
        self,
        query,
        value,
        relative_position_encoding,
        content_attention_bias,
        positional_attention_bias,
        key=None,
        segment_matrix=None,
        segment_encoding=None,
        segment_attention_bias=None,
        state=None,
        attention_mask=None,
        training=None,
    ):
        """Compute multi-head relative attention over inputs.
        Size glossary:
          * Number of heads (H): the number of attention heads.
          * Value size (V): the size of each value embedding per head.
          * Key size (K): the size of each key embedding per head. Equally, the size
            of each query embedding per head. Typically K <= V.
          * Batch dimensions (B).
          * Query (target) attention axes shape (T).
          * Value (source) attention axes shape (S), the rank must match the target.
          * Encoding length (L): The relative positional encoding length.
        Args:
          query: attention input.
          value: attention input.
          content_attention_bias: A trainable bias parameter added to the query head
            when calculating the content-based attention score.
          positional_attention_bias: A trainable bias parameter added to the query
            head when calculating the position-based attention score.
          key: attention input.
          relative_position_encoding: relative positional encoding for key and
            value.
          segment_matrix: Optional `Tensor` representing segmentation IDs used in
            XLNet.
          segment_encoding: Optional `Tensor` representing the segmentation encoding
            as used in XLNet.
          segment_attention_bias: Optional trainable bias parameter added to the
            query had when calculating the segment-based attention score used in
            XLNet.
          state: (default None) optional state. If passed, this is also attended
            over as in TransformerXL.
          attention_mask: (default None) Optional mask that is added to attention
            logits. If state is not None, the mask source sequence dimension should
            extend M.
        Returns:
          attention_output: The result of the computation, of shape [B, T, E],
            where `T` is for target sequence shapes and `E` is the query input last
            dimension if `output_shape` is `None`. Otherwise, the multi-head outputs
            are projected to the shape specified by `output_shape`.
        """
        if not self._built_from_signature:
            self._build_from_signature(query, value, key=key)
        if key is None:
            key = value
        if state is not None and state.shape.ndims > 1:
            value = tf.concat([state, value], 1)
            key = tf.concat([state, key], 1)

        # `query` = [B, T, N ,H]
        query = self._query_dense(query)

        # `key` = [B, S + M, N, H]
        key = self._key_dense(key)

        # `value` = [B, S + M, N, H]
        value = self._value_dense(value)

        # `position` = [B, L, N, H]
        position = self._encoding_dense(relative_position_encoding)

        attention_output = self._compute_attention(
            query=query,
            key=key,
            value=value,
            position=position,
            content_attention_bias=content_attention_bias,
            positional_attention_bias=positional_attention_bias,
            segment_matrix=segment_matrix,
            segment_encoding=segment_encoding,
            segment_attention_bias=segment_attention_bias,
            attention_mask=attention_mask,
            training=training,
        )

        # `attention_output` = [B, S, N, H]
        attention_output = self._output_dense(attention_output)

        return attention_output
