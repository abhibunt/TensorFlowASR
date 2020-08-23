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
import os
import argparse
from tiramisu_asr.utils import setup_environment

setup_environment()
import tensorflow as tf

from tiramisu_asr.runners.segan_runners import SeganTrainer
from tiramisu_asr.datasets.segan_dataset import SeganAugTrainDataset
from tiramisu_asr.configs.user_config import UserConfig
from tiramisu_asr.models.segan import Generator, Discriminator

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")


def main():
    parser = argparse.ArgumentParser(prog="SEGAN")

    parser.add_argument("--config", "-c", type=str, default=DEFAULT_YAML,
                        help="The file path of model configuration file")

    parser.add_argument("--max_ckpts", type=int, default=10,
                        help="Max number of checkpoints to keep")

    args = parser.parse_args()

    config = UserConfig(DEFAULT_YAML, args.config, learning=True)

    tf.random.set_seed(2020)

    dataset = SeganAugTrainDataset(
        stage="train", clean_dir=config["learning_config"]["dataset_config"]["train_paths"],
        noises_config=config["learning_config"]["dataset_config"]["noise_config"],
        speech_config=config["speech_config"], shuffle=True
    )

    segan_trainer = SeganTrainer(config["learning_config"]["running_config"])

    with segan_trainer.strategy.scope():
        generator = Generator(
            window_size=config["speech_config"]["window_size"],
            **config["model_config"]
        )
        generator._build()
        generator.summary(line_length=150)
        discriminator = Discriminator(
            window_size=config["speech_config"]["window_size"],
            **config["model_config"]
        )
        discriminator._build()
        discriminator.summary(line_length=150)

    segan_trainer.compile(generator, discriminator,
                          config["learning_config"]["optimizer_config"],
                          max_to_keep=args.max_ckpts)
    segan_trainer.fit(train_dataset=dataset)


if __name__ == "__main__":
    main()