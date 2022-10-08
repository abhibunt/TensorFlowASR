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

from tensorflow_asr.models.ctc.base_ctc import CtcModel
from tensorflow_asr.models.encoders.conformer import L2, ConformerEncoder
from tensorflow_asr.models.layers.base_layer import Layer


class ConformerDecoder(Layer):
    def __init__(
        self,
        vocab_size: int,
        rnn_units: int = 512,
        rnn_implementation: int = 2,
        rnn_unroll: bool = False,
        kernel_regularizer=L2,
        bias_regularizer=L2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._vocab_size = vocab_size
        self.lstm = tf.keras.layers.LSTM(
            units=rnn_units,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="lstm",
            return_sequences=True,
            return_state=False,
            implementation=rnn_implementation,
            unroll=rnn_unroll,
        )
        self.vocab = tf.keras.layers.Dense(
            vocab_size,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="logits",
        )

    def call(self, inputs, training=False):
        logits, logits_length = inputs
        maxlen = tf.shape(logits)[1]
        mask = tf.sequence_mask(logits_length, maxlen=maxlen)
        logits = self.lstm(logits, training=training, mask=mask)
        logits = self.vocab(logits, training=training)
        return logits, logits_length

    def compute_output_shape(self, input_shape):
        logits_shape, logits_length_shape = input_shape
        outputs_shape = logits_shape[:-1] + (self._vocab_size,)
        return tuple(outputs_shape), tuple(logits_length_shape)


class Conformer(CtcModel):
    def __init__(
        self,
        vocab_size: int,
        encoder_subsampling: dict,
        encoder_dmodel: int = 144,
        encoder_num_blocks: int = 16,
        encoder_head_size: int = 36,
        encoder_num_heads: int = 4,
        encoder_mha_type: str = "relmha",
        encoder_use_attention_mask: bool = False,
        encoder_kernel_size: int = 32,
        encoder_depth_multiplier: int = 1,
        encoder_padding: str = "same",
        encoder_fc_factor: float = 0.5,
        encoder_dropout: float = 0,
        encoder_trainable: bool = True,
        decoder_rnn_units: int = 512,
        decoder_rnn_implementation: int = 2,
        decoder_rnn_unroll: bool = False,
        kernel_regularizer=L2,
        bias_regularizer=L2,
        name: str = "conformer",
        **kwargs,
    ):
        super().__init__(
            encoder=ConformerEncoder(
                subsampling=encoder_subsampling,
                dmodel=encoder_dmodel,
                num_blocks=encoder_num_blocks,
                head_size=encoder_head_size,
                num_heads=encoder_num_heads,
                mha_type=encoder_mha_type,
                use_attention_mask=encoder_use_attention_mask,
                kernel_size=encoder_kernel_size,
                depth_multiplier=encoder_depth_multiplier,
                padding=encoder_padding,
                fc_factor=encoder_fc_factor,
                dropout=encoder_dropout,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                trainable=encoder_trainable,
                name="encoder",
            ),
            decoder=ConformerDecoder(
                vocab_size=vocab_size,
                rnn_units=decoder_rnn_units,
                rnn_implementation=decoder_rnn_implementation,
                rnn_unroll=decoder_rnn_unroll,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name="decoder",
            ),
            name=name,
            **kwargs,
        )
        self.dmodel = encoder_dmodel
        self.time_reduction_factor = self.encoder.conv_subsampling.time_reduction_factor
