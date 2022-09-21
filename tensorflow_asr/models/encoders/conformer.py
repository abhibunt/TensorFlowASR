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
from tensorflow_asr.models.layers.multihead_attention import MultiHeadAttention, RelPositionMultiHeadAttention
from tensorflow_asr.models.layers.positional_encoding import PositionalEncoding, PositionalEncodingConcat
from tensorflow_asr.models.layers.subsampling import (
    Conv1dBlurPoolSubsampling,
    Conv1dSubsampling,
    Conv2dBlurPoolSubsampling,
    Conv2dSubsampling,
    VggBlurPoolSubsampling,
    VggSubsampling,
)

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
            name=f"{name}_ln",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
            dtype=tf.float32,  # Use float32 in layernorm for numeric stability.
        )
        self.ffn1 = tf.keras.layers.Dense(
            4 * input_dim,
            name=f"{name}_dense_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.swish = tf.keras.layers.Activation(tf.nn.swish, name=f"{name}_swish_activation")
        self.do1 = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout_1")
        self.ffn2 = tf.keras.layers.Dense(
            input_dim,
            name=f"{name}_dense_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.do2 = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout_2")
        self.res_add = tf.keras.layers.Add(name=f"{name}_add")

    def call(
        self,
        inputs,
        training=False,
        **kwargs,
    ):
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
            name=f"{name}_ln",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
            dtype=tf.float32,  # Use float32 in layernorm for numeric stability.
        )
        if mha_type == "relmha":
            self.mha = RelPositionMultiHeadAttention(
                name=f"{name}_mhsa",
                head_size=head_size,
                num_heads=num_heads,
                output_size=dmodel,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
            )
        elif mha_type == "mha":
            self.mha = MultiHeadAttention(
                name=f"{name}_mhsa",
                head_size=head_size,
                num_heads=num_heads,
                output_size=dmodel,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
            )
        else:
            raise ValueError("mha_type must be either 'mha' or 'relmha'")
        self.do = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")
        self.res_add = tf.keras.layers.Add(name=f"{name}_add")
        self.mha_type = mha_type

    def call(
        self,
        inputs,
        training=False,
        mask=None,
        **kwargs,
    ):
        inputs, pos = inputs  # pos is positional encoding
        outputs = self.ln(inputs, training=training)
        if self.mha_type == "relmha":
            outputs = self.mha([outputs, outputs, outputs, pos], training=training, mask=mask)
        else:
            outputs = outputs + pos
            outputs = self.mha([outputs, outputs, outputs], training=training, mask=mask)
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
            name=f"{name}_ln",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
            dtype=tf.float32,  # Use float32 in layernorm for numeric stability.
        )
        self.pw_conv_1 = tf.keras.layers.Dense(
            2 * input_dim,
            name=f"{name}_pw_conv_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.glu = GLU(axis=-1, name=f"{name}_glu_activation")
        self.dw_conv = DepthwiseConv1D(
            kernel_size=kernel_size,
            strides=1,
            padding=padding,
            name=f"{name}_dw_conv",
            depth_multiplier=depth_multiplier,
            depthwise_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.bn = tf.keras.layers.BatchNormalization(
            name=f"{name}_bn",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
        )
        self.swish = tf.keras.layers.Activation(tf.nn.swish, name=f"{name}_swish_activation")
        self.pw_conv_2 = tf.keras.layers.Dense(
            input_dim,
            name=f"{name}_pw_conv_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.do = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")
        self.res_add = tf.keras.layers.Add(name=f"{name}_add")

    def call(
        self,
        inputs,
        training=False,
        **kwargs,
    ):
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
            name=f"{name}_ff_module_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.mhsam = MHSAModule(
            mha_type=mha_type,
            dmodel=input_dim,
            head_size=head_size,
            num_heads=num_heads,
            dropout=dropout,
            name=f"{name}_mhsa_module",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.convm = ConvModule(
            input_dim=input_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            name=f"{name}_conv_module",
            depth_multiplier=depth_multiplier,
            padding=padding,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.ffm2 = FFModule(
            input_dim=input_dim,
            dropout=dropout,
            fc_factor=fc_factor,
            name=f"{name}_ff_module_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.ln = tf.keras.layers.LayerNormalization(
            name=f"{name}_ln",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=kernel_regularizer,
            dtype=tf.float32,  # Use float32 in layernorm for numeric stability.
        )

    def call(
        self,
        inputs,
        training=False,
        mask=None,
        **kwargs,
    ):
        inputs, pos = inputs  # pos is positional encoding
        outputs = self.ffm1(inputs, training=training, **kwargs)
        outputs = self.mhsam([outputs, pos], training=training, mask=mask, **kwargs)
        outputs = self.convm(outputs, training=training, **kwargs)
        outputs = self.ffm2(outputs, training=training, **kwargs)
        outputs = self.ln(outputs, training=training)
        return outputs


class ConformerEncoder(tf.keras.Model):
    def __init__(
        self,
        subsampling,
        positional_encoding="sinusoid",
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
        if subsampling_name == "vgg_blurpool":
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
            raise ValueError(
                "subsampling must be either 'vgg', 'vgg_blurpool', 'conv2d', 'conv1d', 'conv2d_blurpool', 'conv1d_blurpool'"
            )

        self.conv_subsampling = subsampling_class(
            **subsampling,
            name=f"{name}_subsampling",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )

        if positional_encoding == "sinusoid":
            self.pe = PositionalEncoding(name=f"{name}_pe")
        elif positional_encoding == "sinusoid_v2":
            self.pe = PositionalEncoding(alpha=2, beta=0, name=f"{name}_pe")
        elif positional_encoding == "sinusoid_concat":
            self.pe = PositionalEncodingConcat(name=f"{name}_pe")
        elif positional_encoding == "sinusoid_concat_v2":
            self.pe = PositionalEncodingConcat(alpha=2, beta=-1, name=f"{name}_pe")
        elif positional_encoding == "subsampling":
            self.pe = tf.keras.layers.Activation("linear", name=f"{name}_pe")
        else:
            raise ValueError(
                "positional_encoding must be either 'sinusoid', \
                'sinusoid_concat', 'sinusoid_v2', 'sinusoid_concat_v2' or 'subsampling'"
            )

        self.linear = tf.keras.layers.Dense(
            dmodel,
            name=f"{name}_linear",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.do = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")

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
                name=f"{name}_block_{i}",
            )
            self.conformer_blocks.append(conformer_block)

    def call(
        self,
        inputs,
        training=False,
        mask=None,
        **kwargs,
    ):
        # input with shape [B, T, V1, V2]
        outputs = self.conv_subsampling(inputs, training=training)
        outputs = self.linear(outputs, training=training)
        pe = self.pe(outputs)
        outputs = self.do(outputs, training=training)
        for cblock in self.conformer_blocks:
            outputs = cblock([outputs, pe], training=training, mask=mask, **kwargs)
        return outputs
