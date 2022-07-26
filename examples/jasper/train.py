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

import fire
import tensorflow as tf

from tensorflow_asr.configs.config import Config
from tensorflow_asr.helpers import dataset_helpers, featurizer_helpers
from tensorflow_asr.models.ctc.jasper import Jasper
from tensorflow_asr.utils import env_util

logger = env_util.setup_environment()

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")


def main(
    config_path: str = DEFAULT_YAML,
    tfrecords: bool = False,
    sentence_piece: bool = False,
    subwords: bool = False,
    bs: int = None,
    spx: int = 1,
    metadata: str = None,
    static_length: bool = False,
    devices: list = [0],
    mxp: bool = False,
    pretrained: str = None,
):
    tf.keras.backend.clear_session()
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": mxp})
    strategy = env_util.setup_strategy(devices)

    config = Config(config_path)

    speech_featurizer, text_featurizer = featurizer_helpers.prepare_featurizers(
        config=config,
        subwords=subwords,
        sentence_piece=sentence_piece,
    )

    train_dataset, eval_dataset = dataset_helpers.prepare_training_datasets(
        config=config,
        speech_featurizer=speech_featurizer,
        text_featurizer=text_featurizer,
        tfrecords=tfrecords,
        metadata=metadata,
    )

    if not static_length:
        speech_featurizer.reset_length()
        text_featurizer.reset_length()

    train_data_loader, eval_data_loader, global_batch_size = dataset_helpers.prepare_training_data_loaders(
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        strategy=strategy,
        batch_size=bs,
    )

    with strategy.scope():
        jasper = Jasper(**config.model_config, vocab_size=text_featurizer.num_classes)
        jasper.make(speech_featurizer.shape, batch_size=global_batch_size)
        if pretrained:
            jasper.load_weights(pretrained, by_name=True, skip_mismatch=True)
        jasper.summary()
        jasper.compile(
            optimizer=config.learning_config.optimizer_config,
            experimental_steps_per_execution=spx,
            global_batch_size=global_batch_size,
            blank=text_featurizer.blank,
        )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(**config.learning_config.running_config.checkpoint),
        tf.keras.callbacks.experimental.BackupAndRestore(config.learning_config.running_config.states_dir),
        tf.keras.callbacks.TensorBoard(**config.learning_config.running_config.tensorboard),
    ]

    jasper.fit(
        train_data_loader,
        epochs=config.learning_config.running_config.num_epochs,
        validation_data=eval_data_loader,
        callbacks=callbacks,
        steps_per_epoch=train_dataset.total_steps,
        validation_steps=eval_dataset.total_steps if eval_data_loader else None,
    )


if __name__ == "__main__":
    fire.Fire(main)
