import os

import tensorflow as tf

from tensorflow_asr.featurizers.text_featurizers import WordPieceFeaturizer

decoder_config = {
    "vocabulary": f"{os.path.dirname(__file__)}/../vocabularies/librispeech/librispeech_train_4_1000.wordpiece",
    "max_subword_length": 4,
}

text = "but it would have broken down after ten miles of that hard trail dawn came while they wound over the crest of the range and with the sun in their faces they took the downgrade it was well into the morning before nash reached logan"


def test_wordpiece_featurizer():
    featurizer = WordPieceFeaturizer(decoder_config=decoder_config)
    indices = featurizer.extract(text)
    print(indices.numpy())
    reversed_text = featurizer.iextract(tf.reshape(indices, (1, -1)))
    print(reversed_text.numpy())
    upoints = featurizer.indices2upoints(indices)
    print(upoints.numpy())
