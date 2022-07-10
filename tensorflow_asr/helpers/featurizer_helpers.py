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

from tensorflow_asr.configs.config import Config
from tensorflow_asr.featurizers import speech_featurizers, text_featurizers

logger = tf.get_logger()


def prepare_featurizers(
    config: Config,
    subwords: bool = False,
    sentence_piece: bool = False,
    wordpiece: bool = True,
):
    speech_featurizer = speech_featurizers.TFSpeechFeaturizer(config.speech_config)
    if sentence_piece:
        logger.info("Loading SentencePiece model ...")
        text_featurizer = text_featurizers.SentencePieceFeaturizer(config.decoder_config)
    elif subwords:
        logger.info("Loading subwords ...")
        text_featurizer = text_featurizers.SubwordFeaturizer(config.decoder_config)
    elif wordpiece:
        logger.info("Loading wordpiece ...")
        text_featurizer = text_featurizers.WordPieceFeaturizer(config.decoder_config)
    else:
        logger.info("Use characters ...")
        text_featurizer = text_featurizers.CharFeaturizer(config.decoder_config)
    return speech_featurizer, text_featurizer
