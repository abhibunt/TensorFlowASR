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

import fire

from tensorflow_asr.utils import env_util

env_util.setup_environment()

from tensorflow_asr.configs.config import Config
from tensorflow_asr.datasets.asr_dataset import ASRDataset
from tensorflow_asr.helpers import featurizer_helpers
from tensorflow_asr.utils.file_util import preprocess_paths


def main(
    *transcripts,
    stage: str = "train",
    config_file: str = None,
    wordpiece: bool = True,
    sentence_piece: bool = False,
    subwords: bool = False,
    metadata: str = None,
):
    transcripts = preprocess_paths(transcripts)

    config = Config(config_file)

    speech_featurizer, text_featurizer = featurizer_helpers.prepare_featurizers(
        config=config, subwords=subwords, sentence_piece=sentence_piece, wordpiece=wordpiece
    )

    dataset = ASRDataset(
        data_paths=transcripts,
        speech_featurizer=speech_featurizer,
        text_featurizer=text_featurizer,
        stage=stage,
        shuffle=False,
    )

    dataset.update_metadata(metadata)


if __name__ == "__main__":
    fire.Fire(main)
