# %%
import os

from tensorflow_asr.configs.config import Config
from tensorflow_asr.helpers import featurizer_helpers
from tensorflow_asr.models.transducer.conformer import Conformer
from tensorflow_asr.utils import env_util

logger = env_util.setup_environment()

config_path = os.path.join(os.path.dirname(__file__), "..", "config.yml")

env_util.setup_seed()
config = Config(config_path)

speech_featurizer, text_featurizer = featurizer_helpers.prepare_featurizers(config=config)

global_batch_size = 32
speech_featurizer.update_length(1200)
text_featurizer.update_length(700)

conformer = Conformer(
    **config.model_config,
    blank=text_featurizer.blank,
    vocab_size=text_featurizer.num_classes,
)
conformer.make(
    speech_featurizer.shape,
    prediction_shape=text_featurizer.prepand_shape,
    batch_size=global_batch_size,
)
conformer.add_featurizers(speech_featurizer, text_featurizer)
conformer.summary()
conformer.save_weights("./conformer.h5")
conformer.load_weights("./conformer.h5")
# %%
conformer.save("./saved_model")

# %%

import tf2onnx

tf2onnx.convert.from_keras(conformer, output_path="./conformer.onnx")

# %%
