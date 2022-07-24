# %% Imports
import tensorflow as tf

from tensorflow_asr.configs.config import Config
from tensorflow_asr.helpers import featurizer_helpers
from tensorflow_asr.models.transducer.conformer import Conformer
from tensorflow_asr.utils import env_util

logger = env_util.setup_environment()

# %% Load model

config_path = "/Users/nlhuy/Paraphernalia/TensorFlowASR/models/subword-conformer/config.yml"

config = Config(config_path)
tf.random.set_seed(0)
tf.keras.backend.clear_session()

speech_featurizer, text_featurizer = featurizer_helpers.prepare_featurizers(
    config=config,
    subwords=True,
    sentence_piece=False,
    wordpiece=False,
)

h5 = "/Users/nlhuy/Paraphernalia/TensorFlowASR/models/subword-conformer/latest.h5"

# build model
conformer = Conformer(**config.model_config, vocabulary_size=text_featurizer.num_classes)
conformer.make(speech_featurizer.shape)
conformer.summary()
# conformer.load_weights(h5, by_name=True, skip_mismatch=True)

# %% Gen bp

output_bp = "/Users/nlhuy/Paraphernalia/TensorFlowASR/models/subword-conformer/bp"

tf.saved_model.save(conformer, output_bp)

# %% Load bp
loaded_conformer = tf.saved_model.load(output_bp)

# %%
