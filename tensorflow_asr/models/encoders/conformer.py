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

from tensorflow_asr.models.activations.glu import GLU
from tensorflow_asr.models.layers.depthwise_conv1d import DepthwiseConv1D
from tensorflow_asr.models.layers.multihead_attention import (
    MultiHeadAttention,
    MultiHeadRelativeAttention,
    compute_self_attention_mask,
)
from tensorflow_asr.models.layers.positional_encoding import compute_relative_position_encoding
from tensorflow_asr.models.layers.subsampling import (
    Conv1dBlurPoolSubsampling,
    Conv1dSubsampling,
    Conv2dBlurPoolSubsampling,
    Conv2dSubsampling,
    VggBlurPoolSubsampling,
    VggSubsampling,
)
from tensorflow_asr.utils import math_util, shape_util

L2 = tf.keras.regularizers.l2(1e-6)


class FFModule(tf.keras.layers.Layer):
    r"""
    architecture::
      input
      /   \
      |   ln(.)                   # input_dim
      |   fflayer(.)              # 4 * input_dim
      |   swish(.)
      |   dropout(.)
      |   fflayer(.)              # input_dim
      |   dropout(.)
      |   * 1/2
      \   /
        +
        |
      output
    """

    def __init__(
        self,
        input_dim,
        dropout=0.0,
        fc_factor=0.5,
        kernel_regularizer=L2,
        bias_regularizer=L2,
        name="ff_module",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.fc_factor = fc_factor
        self.ln = tf.keras.layers.LayerNormalization(
            name="ln",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
        )
        self.ffn1 = tf.keras.layers.Dense(
            4 * input_dim,
            name="dense_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.swish = tf.keras.layers.Activation(tf.nn.swish, name="swish_activation")
        self.do1 = tf.keras.layers.Dropout(dropout, name="dropout_1")
        self.ffn2 = tf.keras.layers.Dense(
            input_dim,
            name="dense_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.do2 = tf.keras.layers.Dropout(dropout, name="dropout_2")
        self.res_add = tf.keras.layers.Add(name="add")

    def call(self, inputs, training=False):
        outputs = self.ln(inputs, training=training)
        outputs = self.ffn1(outputs, training=training)
        outputs = self.swish(outputs)
        outputs = self.do1(outputs, training=training)
        outputs = self.ffn2(outputs, training=training)
        outputs = self.do2(outputs, training=training)
        outputs = self.res_add([inputs, self.fc_factor * outputs])
        return outputs


class MHSAModule(tf.keras.layers.Layer):
    r"""
    architecture::
      input
      /   \
      |   ln(.)                   # input_dim
      |   mhsa(.)                 # head_size = dmodel // num_heads
      |   dropout(.)
      \   /
        +
        |
      output
    """

    def __init__(
        self,
        dmodel,
        head_size,
        num_heads,
        dropout=0.0,
        mha_type="relmha",
        kernel_regularizer=L2,
        bias_regularizer=L2,
        name="mhsa_module",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.ln = tf.keras.layers.LayerNormalization(
            name="ln",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
        )
        if mha_type == "relmha":
            self.mha = MultiHeadRelativeAttention(
                num_heads=num_heads,
                head_size=head_size,
                output_size=dmodel,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name="mhsa",
            )
        elif mha_type == "mha":
            self.mha = MultiHeadAttention(
                num_heads=num_heads,
                head_size=head_size,
                output_size=dmodel,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name="mhsa",
            )
        else:
            raise ValueError("mha_type must be either 'mha' or 'relmha'")
        self.do = tf.keras.layers.Dropout(dropout, name="dropout")
        self.res_add = tf.keras.layers.Add(name="add")
        self.mha_type = mha_type

    def call(
        self,
        inputs,
        relative_position_encoding=None,
        training=False,
        attention_mask=None,
    ):
        outputs = self.ln(inputs, training=training)
        if self.mha_type == "relmha":
            outputs = self.mha([outputs, outputs, outputs, relative_position_encoding], training=training, attention_mask=attention_mask)
        else:
            outputs = self.mha([outputs, outputs, outputs], training=training, attention_mask=attention_mask)
        outputs = self.do(outputs, training=training)
        outputs = self.res_add([inputs, outputs])
        return outputs


class ConvModule(tf.keras.layers.Layer):
    r"""
    architecture::
      input
      /   \
      |   ln(.)                   # input_dim
      |   fflayer(.)              # 2 * input_dim
      |    |
      |   glu(.)                  # input_dim
      |   depthwise_conv_1d(.)
      |   bnorm(.)
      |   swish(.)
      |    |
      |   fflayer(.)
      |   dropout(.)
      \   /
        +
        |
      output
    """

    def __init__(
        self,
        input_dim,
        kernel_size=32,
        dropout=0.0,
        depth_multiplier=1,
        padding="same",
        kernel_regularizer=L2,
        bias_regularizer=L2,
        name="conv_module",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.ln = tf.keras.layers.LayerNormalization(
            name="ln",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
        )
        self.pw_conv_1 = tf.keras.layers.Dense(
            2 * input_dim,
            name="pw_conv_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.glu = GLU(axis=-1, name="glu_activation")
        self.dw_conv = DepthwiseConv1D(
            kernel_size=kernel_size,
            strides=1,
            padding=padding,
            name="dw_conv",
            depth_multiplier=depth_multiplier,
            depthwise_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.bn = tf.keras.layers.BatchNormalization(
            name="bn",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
        )
        self.swish = tf.keras.layers.Activation(tf.nn.swish, name="swish_activation")
        self.pw_conv_2 = tf.keras.layers.Dense(
            input_dim,
            name="pw_conv_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.do = tf.keras.layers.Dropout(dropout, name="dropout")
        self.res_add = tf.keras.layers.Add(name="add")

    def call(self, inputs, training=False):
        outputs = self.ln(inputs, training=training)
        outputs = self.pw_conv_1(outputs, training=training)
        outputs = self.glu(outputs)
        outputs = self.dw_conv(outputs, training=training)
        outputs = self.bn(outputs, training=training)
        outputs = self.swish(outputs)
        outputs = self.pw_conv_2(outputs, training=training)
        outputs = self.do(outputs, training=training)
        outputs = self.res_add([inputs, outputs])
        return outputs


class ConformerBlock(tf.keras.layers.Layer):
    """
    architecture::
      x = x + 1/2 * FFN(x)
      x = x + MHSA(x)
      x = x + Lconv(x)
      x = x + 1/2 * FFN(x)
      y = ln(x)
    """

    def __init__(
        self,
        input_dim,
        dropout=0.0,
        fc_factor=0.5,
        head_size=36,
        num_heads=4,
        mha_type="relmha",
        kernel_size=32,
        depth_multiplier=1,
        padding="same",
        kernel_regularizer=L2,
        bias_regularizer=L2,
        name="conformer_block",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.ffm1 = FFModule(
            input_dim=input_dim,
            dropout=dropout,
            fc_factor=fc_factor,
            name="ff_module_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.mhsam = MHSAModule(
            dmodel=input_dim,
            head_size=head_size,
            num_heads=num_heads,
            dropout=dropout,
            mha_type=mha_type,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="mhsa_module",
        )
        self.convm = ConvModule(
            input_dim=input_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            name="conv_module",
            depth_multiplier=depth_multiplier,
            padding=padding,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.ffm2 = FFModule(
            input_dim=input_dim,
            dropout=dropout,
            fc_factor=fc_factor,
            name="ff_module_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.ln = tf.keras.layers.LayerNormalization(
            name="ln",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=kernel_regularizer,
        )

    def call(
        self,
        inputs,
        relative_position_encoding=None,
        training=False,
        attention_mask=None,
    ):
        outputs = self.ffm1(inputs, training=training)
        outputs = self.mhsam(
            outputs,
            relative_position_encoding=relative_position_encoding,
            training=training,
            attention_mask=attention_mask,
        )
        outputs = self.convm(outputs, training=training)
        outputs = self.ffm2(outputs, training=training)
        outputs = self.ln(outputs, training=training)
        return outputs


class ConformerEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        subsampling,
        dmodel=144,
        num_blocks=16,
        mha_type="relmha",
        head_size=36,
        num_heads=4,
        kernel_size=32,
        depth_multiplier=1,
        padding="causal",
        fc_factor=0.5,
        dropout=0.0,
        kernel_regularizer=L2,
        bias_regularizer=L2,
        name="conformer_encoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        subsampling_name = subsampling.pop("type", "conv2d")
        if subsampling_name == "vgg":
            subsampling_class = VggSubsampling
        elif subsampling_name == "vgg_blurpool":
            subsampling_class = VggBlurPoolSubsampling
        elif subsampling_name == "conv2d":
            subsampling_class = Conv2dSubsampling
        elif subsampling_name == "conv1d":
            subsampling_class = Conv1dSubsampling
        elif subsampling_name == "conv2d_blurpool":
            subsampling_class = Conv2dBlurPoolSubsampling
        elif subsampling_name == "conv1d_blurpool":
            subsampling_class = Conv1dBlurPoolSubsampling
        else:
            raise ValueError("subsampling must be either 'vgg', 'vgg_blurpool', 'conv2d', 'conv1d', 'conv2d_blurpool', 'conv1d_blurpool'")

        self.conv_subsampling = subsampling_class(
            **subsampling,
            name="subsampling",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )

        self.linear = tf.keras.layers.Dense(
            dmodel,
            name="linear",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.do = tf.keras.layers.Dropout(dropout, name="dropout")
        self._mha_type = mha_type

        self.conformer_blocks = []
        for i in range(num_blocks):
            conformer_block = ConformerBlock(
                input_dim=dmodel,
                dropout=dropout,
                fc_factor=fc_factor,
                head_size=head_size,
                num_heads=num_heads,
                mha_type=mha_type,
                kernel_size=kernel_size,
                depth_multiplier=depth_multiplier,
                padding=padding,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"block_{i}",
            )
            self.conformer_blocks.append(conformer_block)

    def call(self, inputs, training=False):
        outputs, inputs_length = inputs
        outputs = self.conv_subsampling(outputs, training=training)
        inputs_length = math_util.get_reduced_length(inputs_length, self.conv_subsampling.time_reduction_factor)
        outputs = self.linear(outputs, training=training)
        outputs = self.do(outputs, training=training)
        # attention_mask = compute_self_attention_mask(outputs, inputs_length)
        if self._mha_type == "relmha":
            relative_position_encoding = compute_relative_position_encoding(shape_util.shape_list(outputs), dtype=outputs.dtype)
        else:
            relative_position_encoding = None
        for cblock in self.conformer_blocks:
            outputs = cblock(
                outputs,
                relative_position_encoding=relative_position_encoding,
                training=training,
                # attention_mask=attention_mask,
            )
        return outputs, inputs_length
