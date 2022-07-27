# coding=utf-8
# Copyright 2022 TF.Text Authors.
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

from typing import List

import fire

from tensorflow_asr.utils import env_util

logger = env_util.setup_environment()

import tensorflow as tf
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab


def write_vocab_file(filepath, vocab):
    with open(filepath, "w") as f:
        for token in vocab:
            print(token, file=f)


def create_transcript_generator(
    transcripts: List[str],
):
    def gen():
        for file_path in transcripts:
            logger.info(f"Reading {file_path} ...")
            with tf.io.gfile.GFile(file_path, "r") as f:
                temp_lines = f.read().splitlines()
                for line in temp_lines[1:]:  # Skip the header of tsv file
                    data = line.split("\t", 2)[-1]  # get only transcript
                    yield data

    return gen


def main(
    *transcripts,
    output: str = None,
    vocab_size: int = 1000,
    max_token_length: int = 50,
    max_unique_chars: int = 1000,
    num_iterations: int = 4,
    reserved_tokens: list = ["[PAD]", "[UNK]"],
):
    bert_tokenizer_params = dict(lower_case=True)

    bert_vocab_args = dict(
        vocab_size=vocab_size,
        reserved_tokens=reserved_tokens,
        bert_tokenizer_params=bert_tokenizer_params,
        learn_params={
            "max_token_length": max_token_length,
            "max_unique_chars": max_unique_chars,
            "num_iterations": num_iterations,
        },
    )

    transcript_generator = create_transcript_generator(transcripts=transcripts)
    dataset = tf.data.Dataset.from_generator(transcript_generator, output_signature=tf.TensorSpec(shape=(), dtype=tf.string))

    vocab = bert_vocab.bert_vocab_from_dataset(dataset.batch(1000).prefetch(2), **bert_vocab_args)

    write_vocab_file(output, vocab)


if __name__ == "__main__":
    fire.Fire(main)
