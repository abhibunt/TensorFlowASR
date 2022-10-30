import fire
import tensorflow as tf

from tensorflow_asr.configs.config import Config
from tensorflow_asr.featurizers.text_featurizers import SentencePieceFeaturizer
from tensorflow_asr.utils.env_util import setup_environment

logger = setup_environment()


def main(
    config_path: str,
):
    tf.keras.backend.clear_session()
    config = Config(config_path)
    SentencePieceFeaturizer.build_from_corpus(config.decoder_config)


if __name__ == "__main__":
    fire.Fire(main)
