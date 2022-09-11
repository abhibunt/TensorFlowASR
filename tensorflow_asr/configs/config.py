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

from typing import Union

from tensorflow_asr.augmentations.augmentation import Augmentation
from tensorflow_asr.utils import file_util


class DecoderConfig:
    def __init__(self, config: dict = None):
        if not config:
            config = {}
        self.blank_index = config.pop("blank_index", 0)
        self.unknown_token = config.pop("unknown_token", "[PAD]")
        self.unknown_index = config.pop("unknown_index", self.blank_index)

        self.beam_width = config.pop("beam_width", 0)
        self.norm_score = config.pop("norm_score", True)
        self.lm_config = config.pop("lm_config", {})

        self.vocabulary = file_util.preprocess_paths(config.pop("vocabulary", None))
        self.vocab_size = config.pop("vocab_size", 1000)
        self.max_token_length = config.pop("max_token_length", 50)
        self.max_unique_chars = config.pop("max_unique_chars", None)
        self.num_iterations = config.pop("num_iterations", 4)
        self.reserved_tokens = config.pop("reserved_tokens", None)
        self.normalization_form = config.pop("normalization_form", "NFKC")

        self.corpus_files = file_util.preprocess_paths(config.pop("corpus_files", []))
        self.output_path_prefix = file_util.preprocess_paths(config.pop("output_path_prefix", None))
        self.model_type = config.pop("model_type", None)

        for k, v in config.items():
            setattr(self, k, v)


class DatasetConfig:
    def __init__(self, config: dict = None):
        if not config:
            config = {}
        self.enabled = config.pop("enabled", True)
        self.stage = config.pop("stage", None)
        self.data_paths = file_util.preprocess_paths(config.pop("data_paths", None), enabled=self.enabled)
        self.tfrecords_dir = file_util.preprocess_paths(config.pop("tfrecords_dir", None), isdir=True, enabled=self.enabled)
        self.tfrecords_shards = config.pop("tfrecords_shards", 16)
        self.shuffle = config.pop("shuffle", False)
        self.cache = config.pop("cache", False)
        self.drop_remainder = config.pop("drop_remainder", True)
        self.buffer_size = config.pop("buffer_size", 1000)
        self.use_tf = config.pop("use_tf", False)
        self.augmentations = Augmentation(config.pop("augmentation_config", {}))
        self.metadata = config.pop("metadata", None)
        for k, v in config.items():
            setattr(self, k, v)


class RunningConfig:
    def __init__(self, config: dict = None):
        if not config:
            config = {}
        self.batch_size = config.pop("batch_size", 1)
        self.accumulation_steps = config.pop("accumulation_steps", 1)
        self.num_epochs = config.pop("num_epochs", 20)
        for k, v in config.items():
            setattr(self, k, v)
            if k == "checkpoint":
                if v and v.get("filepath"):
                    file_util.preprocess_paths(v.get("filepath"))
            elif k == "states_dir" and v:
                file_util.preprocess_paths(v, isdir=True)
            elif k == "tensorboard":
                if v and v.get("log_dir"):
                    file_util.preprocess_paths(v.get("log_dir"), isdir=True)


class LearningConfig:
    def __init__(self, config: dict = None):
        if not config:
            config = {}
        self.train_dataset_config = DatasetConfig(config.pop("train_dataset_config", {}))
        self.eval_dataset_config = DatasetConfig(config.pop("eval_dataset_config", {}))
        self.test_dataset_config = DatasetConfig(config.pop("test_dataset_config", {}))
        self.optimizer_config = config.pop("optimizer_config", {})
        self.learning_rate_config = config.pop("learning_rate_config", {})
        self.running_config = RunningConfig(config.pop("running_config", {}))
        for k, v in config.items():
            setattr(self, k, v)


class Config:
    """User config class for training, testing or infering"""

    def __init__(self, data: Union[str, dict]):
        config = data if isinstance(data, dict) else file_util.load_yaml(file_util.preprocess_paths(data))
        self.speech_config = config.pop("speech_config", {})
        self.decoder_config = config.pop("decoder_config", {})
        self.model_config = config.pop("model_config", {})
        self.learning_config = LearningConfig(config.pop("learning_config", {}))
        for k, v in config.items():
            setattr(self, k, v)
