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

import codecs
import os
import unicodedata
from multiprocessing import cpu_count

import numpy as np
import sentencepiece as sp
import tensorflow as tf
import tensorflow_datasets as tds
import tensorflow_text as tft
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

from tensorflow_asr.configs.config import DecoderConfig
from tensorflow_asr.utils import file_util

logger = tf.get_logger()

ENGLISH_CHARACTERS = [
    " ",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "'",
]


class TextFeaturizer:
    def __init__(
        self,
        decoder_config: dict,
    ):
        self.scorer = None
        self.decoder_config = DecoderConfig(decoder_config)
        self.blank = None
        self.tokens2indices = {}
        self.tokens = []
        self.num_classes = None
        self.max_length = 0

    @property
    def shape(self) -> list:
        return [self.max_length if self.max_length > 0 else None]

    @property
    def prepand_shape(self) -> list:
        return [self.max_length + 1 if self.max_length > 0 else None]

    def update_length(
        self,
        length: int,
    ):
        self.max_length = max(self.max_length, length)

    def reset_length(self):
        self.max_length = 0

    def preprocess_text(self, text):
        text = unicodedata.normalize(self.decoder_config.normalization_form, text.lower())
        return text.strip("\n").strip()  # remove trailing newline

    def tf_preprocess_text(self, text: tf.Tensor):
        text = tft.normalize_utf8(text, self.decoder_config.normalization_form)
        text = tf.strings.regex_replace(text, r"\p{Cc}|\p{Cf}", " ")
        text = tf.strings.lower(text, encoding="utf-8")
        text = tf.strings.strip(text)  # remove trailing whitespace
        return text

    def add_scorer(
        self,
        scorer: any = None,
    ):
        """Add scorer to this instance"""
        self.scorer = scorer

    def normalize_indices(
        self,
        indices: tf.Tensor,
    ) -> tf.Tensor:
        """
        Remove -1 in indices by replacing them with blanks
        Args:
            indices (tf.Tensor): shape any

        Returns:
            tf.Tensor: normalized indices with shape same as indices
        """
        with tf.name_scope("normalize_indices"):
            minus_one = -1 * tf.ones_like(indices, dtype=tf.int32)
            blank_like = self.blank * tf.ones_like(indices, dtype=tf.int32)
            return tf.where(indices == minus_one, blank_like, indices)

    def prepand_blank(
        self,
        text: tf.Tensor,
    ) -> tf.Tensor:
        """Prepand blank index for transducer models"""
        return tf.concat([[self.blank], text], axis=0)

    def extract(self, text):
        raise NotImplementedError()

    def tf_extract(self, text):
        raise NotImplementedError()

    def iextract(self, indices):
        raise NotImplementedError()

    def indices2upoints(self, indices):
        raise NotImplementedError()


class CharFeaturizer(TextFeaturizer):
    """
    Extract text feature based on char-level granularity.
    By looking up the vocabulary table, each line of transcript will be
    converted to a sequence of integer indexes.
    """

    def __init__(
        self,
        decoder_config: dict,
    ):
        """
        decoder_config = {
            "vocabulary": str,
            "blank_at_zero": bool,
            "beam_width": int,
            "lm_config": {
                ...
            }
        }
        """
        super(CharFeaturizer, self).__init__(decoder_config)
        self.__init_vocabulary()

    def __init_vocabulary(self):
        lines = []
        if self.decoder_config.vocabulary is not None:
            with codecs.open(self.decoder_config.vocabulary, "r", "utf-8") as fin:
                lines.extend(fin.readlines())
        else:
            lines = ENGLISH_CHARACTERS
        self.blank = self.decoder_config.blank_index
        self.tokens2indices = {}
        self.tokens = []
        index = 1 if self.blank == 0 else 0
        for line in lines:
            line = self.preprocess_text(line)
            if line.startswith("#") or not line:
                continue
            self.tokens2indices[line[0]] = index
            self.tokens.append(line[0])
            index += 1
        if self.blank is None:
            self.blank = len(self.tokens)  # blank not at zero
        self.non_blank_tokens = self.tokens.copy()
        self.tokens.insert(self.blank, "")  # add blank token to tokens
        self.num_classes = len(self.tokens)
        self.tokens = tf.convert_to_tensor(self.tokens, dtype=tf.string)
        self.upoints = tf.strings.unicode_decode(self.tokens, "UTF-8").to_tensor(shape=[None, 1])

    def extract(
        self,
        text: str,
    ) -> tf.Tensor:
        """
        Convert string to a list of integers
        Args:
            text: string (sequence of characters)

        Returns:
            sequence of ints in tf.Tensor
        """
        text = self.preprocess_text(text)
        text = list(text)
        indices = [self.tokens2indices[token] for token in text]
        return tf.convert_to_tensor(indices, dtype=tf.int32)

    def iextract(
        self,
        indices: tf.Tensor,
    ) -> tf.Tensor:
        """
        Convert list of indices to string
        Args:
            indices: tf.Tensor with dim [B, None]

        Returns:
            transcripts: tf.Tensor of dtype tf.string with dim [B]
        """
        indices = self.normalize_indices(indices)
        tokens = tf.gather_nd(self.tokens, tf.expand_dims(indices, axis=-1))
        with tf.device("/CPU:0"):  # string data is not supported on GPU
            tokens = tf.strings.reduce_join(tokens, axis=-1)
        return tokens

    @tf.function(input_signature=[tf.TensorSpec([None], dtype=tf.int32)])
    def indices2upoints(
        self,
        indices: tf.Tensor,
    ) -> tf.Tensor:
        """
        Transform Predicted Indices to Unicode Code Points (for using tflite)
        Args:
            indices: tf.Tensor of Classes in shape [None]

        Returns:
            unicode code points transcript with dtype tf.int32 and shape [None]
        """
        with tf.name_scope("indices2upoints"):
            indices = self.normalize_indices(indices)
            upoints = tf.gather_nd(self.upoints, tf.expand_dims(indices, axis=-1))
            return tf.gather_nd(upoints, tf.where(tf.not_equal(upoints, 0)))


class SubwordFeaturizer(TextFeaturizer):
    """
    Extract text feature based on char-level granularity.
    By looking up the vocabulary table, each line of transcript will be
    converted to a sequence of integer indexes.
    """

    def __init__(
        self,
        decoder_config: dict,
        subwords=None,
    ):
        """
        decoder_config = {
            "vocab_size": int,
            "max_subword_length": 4,
            "max_corpus_chars": None,
            "reserved_tokens": None,
            "beam_width": int,
            "lm_config": {
                ...
            }
        }
        """
        super(SubwordFeaturizer, self).__init__(decoder_config)
        self.subwords = self.__load_subwords() if subwords is None else subwords
        self.blank = 0  # subword treats blank as 0
        self.num_classes = self.subwords.vocab_size
        # create upoints
        self.__init_vocabulary()

    def __init_vocabulary(self):
        self.tokens = []
        for idx in np.arange(1, self.num_classes, dtype=np.int32):
            self.tokens.append(self.subwords.decode([idx]))
        self.non_blank_tokens = self.tokens.copy()
        self.tokens.insert(0, "")
        self.upoints = tf.strings.unicode_decode(self.tokens, "UTF-8")
        self.upoints = self.upoints.to_tensor()  # [num_classes, max_subword_length]

    def __load_subwords(self):
        filename_prefix = os.path.splitext(self.decoder_config.vocabulary)[0]
        return tds.deprecated.text.SubwordTextEncoder.load_from_file(filename_prefix)

    @classmethod
    def build_from_corpus(
        cls,
        decoder_config: dict,
        corpus_files: list = None,
    ):
        dconf = DecoderConfig(decoder_config.copy())
        corpus_files = dconf.corpus_files if corpus_files is None or len(corpus_files) == 0 else corpus_files

        def corpus_generator():
            for file in corpus_files:
                with open(file, "r", encoding="utf-8") as f:
                    lines = f.read().splitlines()
                    lines = lines[1:]
                for line in lines:
                    line = line.split("\t")
                    yield line[-1]

        subwords = tds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            corpus_generator(),
            dconf.vocab_size,
            dconf.max_token_length,
            dconf.max_unique_chars,
            dconf.reserved_tokens,
        )
        return cls(decoder_config, subwords)

    @classmethod
    def load_from_file(
        cls,
        decoder_config: dict,
        filename: str = None,
    ):
        dconf = DecoderConfig(decoder_config.copy())
        filename = dconf.vocabulary if filename is None else file_util.preprocess_paths(filename)
        filename_prefix = os.path.splitext(filename)[0]
        subwords = tds.deprecated.text.SubwordTextEncoder.load_from_file(filename_prefix)
        return cls(decoder_config, subwords)

    def save_to_file(
        self,
        filename: str = None,
    ):
        filename = self.decoder_config.vocabulary if filename is None else file_util.preprocess_paths(filename)
        filename_prefix = os.path.splitext(filename)[0]
        return self.subwords.save_to_file(filename_prefix)

    def extract(
        self,
        text: str,
    ) -> tf.Tensor:
        """
        Convert string to a list of integers
        Args:
            text: string (sequence of characters)

        Returns:
            sequence of ints in tf.Tensor
        """
        text = self.preprocess_text(text)
        indices = self.subwords.encode(text)
        return tf.convert_to_tensor(indices, dtype=tf.int32)

    def iextract(
        self,
        indices: tf.Tensor,
    ) -> tf.Tensor:
        """
        Convert list of indices to string
        Args:
            indices: tf.Tensor with dim [B, None]

        Returns:
            transcripts: tf.Tensor of dtype tf.string with dim [B]
        """
        with tf.device("/CPU:0"):  # string data is not supported on GPU
            total = tf.shape(indices)[0]
            batch = tf.constant(0, dtype=tf.int32)
            transcripts = tf.TensorArray(
                dtype=tf.string,
                size=total,
                dynamic_size=False,
                infer_shape=False,
                clear_after_read=False,
                element_shape=tf.TensorShape([]),
            )

            def cond(_batch, _total, _):
                return tf.less(_batch, _total)

            def body(_batch, _total, _transcripts):
                norm_indices = self.normalize_indices(indices[_batch])
                norm_indices = tf.gather_nd(norm_indices, tf.where(tf.not_equal(norm_indices, 0)))
                decoded = tf.numpy_function(self.subwords.decode, inp=[norm_indices], Tout=tf.string)
                _transcripts = _transcripts.write(_batch, decoded)
                return _batch + 1, _total, _transcripts

            _, _, transcripts = tf.while_loop(cond, body, loop_vars=[batch, total, transcripts])

            return transcripts.stack()

    @tf.function(input_signature=[tf.TensorSpec([None], dtype=tf.int32)])
    def indices2upoints(
        self,
        indices: tf.Tensor,
    ) -> tf.Tensor:
        """
        Transform Predicted Indices to Unicode Code Points (for using tflite)
        Args:
            indices: tf.Tensor of Classes in shape [None]

        Returns:
            unicode code points transcript with dtype tf.int32 and shape [None]
        """
        with tf.name_scope("indices2upoints"):
            indices = self.normalize_indices(indices)
            upoints = tf.gather_nd(self.upoints, tf.expand_dims(indices, axis=-1))
            return tf.gather_nd(upoints, tf.where(tf.not_equal(upoints, 0)))


class SentencePieceFeaturizer(TextFeaturizer):
    """
    Extract text feature based on sentence piece package.
    """

    UNK_TOKEN, UNK_TOKEN_ID = "<unk>", 1
    BOS_TOKEN, BOS_TOKEN_ID = "<s>", 2
    EOS_TOKEN, EOS_TOKEN_ID = "</s>", 3
    PAD_TOKEN, PAD_TOKEN_ID = "<pad>", 0  # unused, by default

    def __init__(
        self,
        decoder_config: dict,
        model=None,
    ):
        super().__init__(decoder_config)
        self.model = self.__load_model() if model is None else model
        self.blank = 0  # treats blank as 0 (pad)
        # vocab size
        self.num_classes = self.model.get_piece_size()
        self.__init_vocabulary()

    def __load_model(self):
        filename_prefix = os.path.splitext(self.decoder_config.vocabulary)[0]
        processor = sp.SentencePieceProcessor()
        processor.load(filename_prefix + ".model")
        return processor

    def __init_vocabulary(self):
        self.tokens = []
        for idx in range(1, self.num_classes):
            self.tokens.append(self.model.decode_ids([idx]))
        self.non_blank_tokens = self.tokens.copy()
        self.tokens.insert(0, "")
        self.upoints = tf.strings.unicode_decode(self.tokens, "UTF-8")
        self.upoints = self.upoints.to_tensor()  # [num_classes, max_subword_length]

    @classmethod
    def build_from_corpus(
        cls,
        decoder_config: dict,
    ):
        """
        --model_prefix: output model name prefix. <model_name>.model and <model_name>.vocab are generated.
        --vocab_size: vocabulary size, e.g., 8000, 16000, or 32000
        --model_type: model type. Choose from unigram (default), bpe, char, or word.
        The input sentence must be pretokenized when using word type."""
        decoder_cfg = DecoderConfig(decoder_config)
        # Train SentencePiece Model

        def corpus_iterator():
            for file in decoder_cfg.corpus_files:
                with open(file, "r", encoding="utf-8") as f:
                    lines = f.read().splitlines()
                    lines = lines[1:]
                for line in lines:
                    line = line.split("\t")
                    yield line[-1]

        sp.SentencePieceTrainer.Train(
            sentence_iterator=corpus_iterator(),
            model_prefix=decoder_cfg.output_path_prefix,
            model_type=decoder_cfg.model_type,
            vocab_size=decoder_cfg.vocab_size,
            num_threads=cpu_count(),
            unk_id=cls.UNK_TOKEN_ID,
            bos_id=cls.BOS_TOKEN_ID,
            eos_id=cls.EOS_TOKEN_ID,
            pad_id=cls.PAD_TOKEN_ID,
            unk_surface="__UNKNOWN__",  # change default unk surface U+2047("⁇") by "__UNKNOWN__"
        )
        # Export fairseq dictionary
        processor = sp.SentencePieceProcessor()
        processor.Load(decoder_cfg.output_path_prefix + ".model")
        vocab = {i: processor.IdToPiece(i) for i in range(processor.GetPieceSize())}
        assert (
            vocab.get(cls.UNK_TOKEN_ID) == cls.UNK_TOKEN
            and vocab.get(cls.BOS_TOKEN_ID) == cls.BOS_TOKEN
            and vocab.get(cls.EOS_TOKEN_ID) == cls.EOS_TOKEN
        )
        vocab = {i: s for i, s in vocab.items() if s not in {cls.UNK_TOKEN, cls.BOS_TOKEN, cls.EOS_TOKEN, cls.PAD_TOKEN}}
        with open(decoder_cfg.output_path_prefix + ".txt", "w") as f_out:
            for _, s in sorted(vocab.items(), key=lambda x: x[0]):
                f_out.write(f"{s} 1\n")

        return cls(decoder_config, processor)

    @classmethod
    def load_from_file(
        cls,
        decoder_config: dict,
        filename: str = None,
    ):
        if filename is not None:
            filename_prefix = os.path.splitext(file_util.preprocess_paths(filename))[0]
        else:
            filename_prefix = decoder_config.get("output_path_prefix", None)
        processor = sp.SentencePieceProcessor()
        processor.load(filename_prefix + ".model")
        return cls(decoder_config, processor)

    def extract(
        self,
        text: str,
    ) -> tf.Tensor:
        """
        Convert string to a list of integers
        # encode: text => id
        sp.encode_as_pieces('This is a test') --> ['▁This', '▁is', '▁a', '▁t', 'est']
        sp.encode_as_ids('This is a test') --> [209, 31, 9, 375, 586]
        Args:
            text: string (sequence of characters)

        Returns:
            sequence of ints in tf.Tensor
        """
        text = self.preprocess_text(text)
        text = text.strip()  # remove trailing space
        indices = self.model.encode_as_ids(text)
        return tf.convert_to_tensor(indices, dtype=tf.int32)

    def iextract(
        self,
        indices: tf.Tensor,
    ) -> tf.Tensor:
        """
        Convert list of indices to string
        # decode: id => text
        sp.decode_pieces(['▁This', '▁is', '▁a', '▁t', 'est']) --> This is a test
        sp.decode_ids([209, 31, 9, 375, 586]) --> This is a test

        Args:
            indices: tf.Tensor with dim [B, None]

        Returns:
            transcripts: tf.Tensor of dtype tf.string with dim [B]
        """
        indices = self.normalize_indices(indices)
        with tf.device("/CPU:0"):  # string data is not supported on GPU

            def decode(x):
                if x[0] == self.blank:
                    x = x[1:]
                return self.model.decode_ids(x.tolist())

            text = tf.map_fn(
                lambda x: tf.numpy_function(decode, inp=[x], Tout=tf.string),
                indices,
                fn_output_signature=tf.TensorSpec([], dtype=tf.string),
            )
        return text

    @tf.function(input_signature=[tf.TensorSpec([None], dtype=tf.int32)])
    def indices2upoints(
        self,
        indices: tf.Tensor,
    ) -> tf.Tensor:
        """
        Transform Predicted Indices to Unicode Code Points (for using tflite)
        Args:
            indices: tf.Tensor of Classes in shape [None]

        Returns:
            unicode code points transcript with dtype tf.int32 and shape [None]
        """
        with tf.name_scope("indices2upoints"):
            indices = self.normalize_indices(indices)
            upoints = tf.gather_nd(self.upoints, tf.expand_dims(indices, axis=-1))
            return tf.gather_nd(upoints, tf.where(tf.not_equal(upoints, 0)))


class WordPieceFeaturizer(TextFeaturizer):
    def __init__(
        self,
        decoder_config: dict,
    ):
        super().__init__(decoder_config)
        self.blank = self.decoder_config.blank_index  # treat [PAD] as blank
        self.vocab = None
        with tf.io.gfile.GFile(self.decoder_config.vocabulary, "r") as voc:
            self.vocab = voc.read().splitlines()
        if not self.vocab:
            raise ValueError("Unable to read vocabulary")
        self.tokenizer = tft.FastWordpieceTokenizer(
            vocab=self.vocab,
            token_out_type=tf.int32,
            unknown_token=self.decoder_config.unknown_token,
            no_pretokenization=True,  # False is limited, so we manually do pretokenization
            support_detokenization=True,
        )
        self.num_classes = self.decoder_config.vocab_size

    @classmethod
    def build_from_corpus(
        cls,
        decoder_config: dict,
    ):
        dconf = DecoderConfig(decoder_config.copy())

        def corpus_generator():
            for file_path in dconf.corpus_files:
                logger.info(f"Reading {file_path} ...")
                with tf.io.gfile.GFile(file_path, "r") as f:
                    temp_lines = f.read().splitlines()
                    for line in temp_lines[1:]:  # Skip the header of tsv file
                        data = line.split("\t", 2)[-1]  # get only transcript
                        yield data

        def write_vocab_file(filepath, vocab):
            with open(filepath, "w") as f:
                for token in vocab:
                    print(token, file=f)

        dataset = tf.data.Dataset.from_generator(corpus_generator, output_signature=tf.TensorSpec(shape=(), dtype=tf.string))
        vocab = bert_vocab.bert_vocab_from_dataset(
            dataset.batch(1000).prefetch(2),
            vocab_size=dconf.vocab_size,
            reserved_tokens=dconf.reserved_tokens,
            bert_tokenizer_params=dict(
                lower_case=False,  # keep original from dataset
                keep_whitespace=True,  # according to papers
                normalization_form=dconf.normalization_form,
                preserve_unused_token=False,
            ),
            learn_params=dict(
                max_token_length=dconf.max_token_length,
                max_unique_chars=dconf.max_unique_chars,
                num_iterations=dconf.num_iterations,
            ),
        )
        write_vocab_file(dconf.vocabulary, vocab)

        return cls(decoder_config)

    def extract(
        self,
        text: str,
    ) -> tf.Tensor:
        """
        Convert string to a list of integers
        Args:
            text: string (sequence of characters)

        Returns:
            sequence of ints in tf.Tensor
        """
        return self.tf_extract(text)

    def tf_extract(
        self,
        text: tf.Tensor,
    ) -> tf.Tensor:
        text = self.tf_preprocess_text(text)
        text = tf.strings.regex_replace(text, "\\s", "| |")
        text = tf.strings.split(text, "|")
        indices = self.tokenizer.tokenize(text).merge_dims(0, 1)
        return indices

    def iextract(
        self,
        indices: tf.Tensor,
    ) -> tf.Tensor:
        """
        Convert list of indices to string
        Args:
            indices: tf.Tensor with dim [B, None]

        Returns:
            transcripts: tf.Tensor of dtype tf.string with dim [B]
        """
        indices = tf.ragged.boolean_mask(indices, tf.not_equal(indices, self.blank))
        indices = tf.ragged.boolean_mask(indices, tf.not_equal(indices, self.decoder_config.unknown_index))
        transcripts = self.tokenizer.detokenize(indices)
        transcripts = tf.strings.regex_replace(transcripts, "\\s+", " ")  # trim "  " to " "
        return transcripts

    @tf.function(input_signature=[tf.TensorSpec([None], dtype=tf.int32)])
    def indices2upoints(
        self,
        indices: tf.Tensor,
    ) -> tf.Tensor:
        """
        Transform Predicted Indices to Unicode Code Points (for using tflite)
        Args:
            indices: tf.Tensor of Classes in shape [None]

        Returns:
            unicode code points transcript with dtype tf.int32 and shape [None]
        """
        with tf.name_scope("indices2upoints"):
            transcripts = self.iextract(tf.reshape(indices, [1, -1]))
            upoints = tf.strings.unicode_decode(transcripts, "UTF-8").to_tensor()
            return tf.reshape(upoints, [-1])
