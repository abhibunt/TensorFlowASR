<h1 align="center">
<p>TensorFlowASR :zap:</p>
<p align="center">
<a href="https://github.com/TensorSpeech/TensorFlowASR/blob/main/LICENSE">
  <img alt="GitHub" src="https://img.shields.io/github/license/TensorSpeech/TensorFlowASR?logo=apache&logoColor=green">
</a>
<img alt="python" src="https://img.shields.io/badge/python-%3E%3D3.6-blue?logo=python">
<img alt="tensorflow" src="https://img.shields.io/badge/tensorflow-%3E%3D2.8.0-orange?logo=tensorflow">
<a href="https://pypi.org/project/TensorFlowASR/">
  <img alt="PyPI" src="https://img.shields.io/pypi/v/TensorFlowASR?color=%234285F4&label=release&logo=pypi&logoColor=%234285F4">
</a>
</p>
</h1>
<h2 align="center">
<p>Almost State-of-the-art Automatic Speech Recognition in Tensorflow 2</p>
</h2>

<p align="center">
TensorFlowASR implements some automatic speech recognition architectures such as DeepSpeech2, Jasper, RNN Transducer, ContextNet, Conformer, etc. These models can be converted to TFLite to reduce memory and computation for deployment :smile:
</p>

## What's New?

- (9/4/2022) Breaking changes release v1.1.x

## Table of Contents

<!-- TOC -->

- [What's New?](#whats-new)
- [Table of Contents](#table-of-contents)
- [:yum: Supported Models](#yum-supported-models)
  - [Baselines](#baselines)
  - [Publications](#publications)
- [Installation](#installation)
  - [Installing from source (recommended)](#installing-from-source-recommended)
  - [Installing via PyPi](#installing-via-pypi)
  - [Installing for development](#installing-for-development)
  - [Running in a container](#running-in-a-container)
- [Training & Testing Tutorial](#training--testing-tutorial)
- [Features Extraction](#features-extraction)
- [Augmentations](#augmentations)
- [TFLite Convertion](#tflite-convertion)
- [Pretrained Models](#pretrained-models)
- [Corpus Sources](#corpus-sources)
  - [English](#english)
  - [Vietnamese](#vietnamese)
- [How to contribute](#how-to-contribute)
- [References & Credits](#references--credits)
- [Contact](#contact)

<!-- /TOC -->

## :yum: Supported Models

### Baselines

- **Transducer Models** (End2end models using RNNT Loss for training, currently supported Conformer, ContextNet, Streaming Transducer)
- **CTCModel** (End2end models using CTC Loss for training, currently supported DeepSpeech2, Jasper)

### Publications

- **Conformer Transducer** (Reference: [https://arxiv.org/abs/2005.08100](https://arxiv.org/abs/2005.08100))
  See [examples/conformer](./examples/conformer)
- **ContextNet** (Reference: [http://arxiv.org/abs/2005.03191](http://arxiv.org/abs/2005.03191))
  See [examples/contextnet](./examples/contextnet)
- **RNN Transducer** (Reference: [https://arxiv.org/abs/1811.06621](https://arxiv.org/abs/1811.06621))
  See [examples/rnn_transducer](./examples/rnn_transducer)
- **Deep Speech 2** (Reference: [https://arxiv.org/abs/1512.02595](https://arxiv.org/abs/1512.02595))
  See [examples/deepspeech2](./examples/deepspeech2)
- **Jasper** (Reference: [https://arxiv.org/abs/1904.03288](https://arxiv.org/abs/1904.03288))
  See [examples/jasper](./examples/jasper)

## Installation

For training and testing, you should use `git clone` for installing necessary packages from other authors (`ctc_decoders`, `rnnt_loss`, etc.)

### Installing from source (recommended)

```bash
git clone https://github.com/TensorSpeech/TensorFlowASR.git
cd TensorFlowASR
# Tensorflow 2.x (with 2.x.x >= 2.5.1)
pip3 install ".[tf2.x]" # or ".[tf2.x-gpu]"
```

For anaconda3:

```bash
conda create -y -n tfasr tensorflow-gpu python=3.8 # tensorflow if using CPU, this makes sure conda install all dependencies for tensorflow
conda activate tfasr
pip install -U tensorflow-gpu # upgrade to latest version of tensorflow
git clone https://github.com/TensorSpeech/TensorFlowASR.git
cd TensorFlowASR
# Tensorflow 2.x (with 2.x.x >= 2.5.1)
pip3 install ".[tf2.x]" # or ".[tf2.x-gpu]"
```

### Installing via PyPi

```bash
# Tensorflow 2.x (with 2.x >= 2.3)
pip3 install "TensorFlowASR[tf2.x]" # or pip3 install "TensorFlowASR[tf2.x-gpu]"
```

### Installing for development

```bash
git clone https://github.com/TensorSpeech/TensorFlowASR.git
cd TensorFlowASR
pip3 install -e ".[dev]"
pip3 install -e ".[tf2.x]" # or ".[tf2.x-gpu]"
```

### Running in a container

```bash
docker-compose up -d
```



## Training & Testing Tutorial

1. Define config YAML file, see the `config.yml` files in the [example folder](./examples) for reference (you can copy and modify values such as parameters, paths, etc.. to match your local machine configuration)
2. Download your corpus (a.k.a datasets) and create a script to generate `transcripts.tsv` files from your corpus (this is general format used in this project because each dataset has different format). For more detail, see [datasets](./tensorflow_asr/datasets/README.md). **Note:** Make sure your data contain only characters in your language, for example, english has `a` to `z` and `'`. **Do not use `cache` if your dataset size is not fit in the RAM**.
3. [Optional] Generate TFRecords to use `tf.data.TFRecordDataset` for better performance by using the script [create_tfrecords.py](./scripts/create_tfrecords.py)
4. Create vocabulary file (characters or subwords/wordpieces) by defining `language.characters`, using the scripts [generate_vocab_subwords.py](./scripts/generate_vocab_subwords.py) or [generate_vocab_sentencepiece.py](./scripts/generate_vocab_sentencepiece.py). There're predefined ones in [vocabularies](./vocabularies)
5. [Optional, required if you use --static-length] Generate metadata file for your dataset by using script [generate_metadata.py](./scripts/generate_metadata.py). This metadata file contains maximum lengths calculated with your `config.yml` and total number of elements in each dataset, for static shape training and precalculated steps per epoch.
6. For training, see `train.py` files in the [example folder](./examples) to see the options
7. For testing, see `test.py` files in the [example folder](./examples) to see the options. **Note:** Testing is currently not supported for TPUs. It will print nothing other than the progress bar in the console, but it will store the predicted transcripts to the file `output.tsv` and then calculate the metrics from that file.

**FYI**: Keras builtin training uses **infinite dataset**, which avoids the potential last partial batch.

See [examples](./examples/) for some predefined ASR models and results

## Features Extraction

See [features_extraction](./tensorflow_asr/featurizers/README.md)

## Augmentations

See [augmentations](./tensorflow_asr/augmentations/README.md)

## TFLite Convertion

After converting to tflite, the tflite model is like a function that transforms directly from an **audio signal** to **unicode code points**, then we can convert unicode points to string.

1. Install `tf-nightly` using `pip install tf-nightly`
2. Build a model with the same architecture as the trained model _(if model has tflite argument, you must set it to True)_, then load the weights from trained model to the built model
3. Load `TFSpeechFeaturizer` and `TextFeaturizer` to model using function `add_featurizers`
4. Convert model's function to tflite as follows:

```python
func = model.make_tflite_function(**options) # options are the arguments of the function
concrete_func = func.get_concrete_function()
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
```

5. Save the converted tflite model as follows:

```python
if not os.path.exists(os.path.dirname(tflite_path)):
    os.makedirs(os.path.dirname(tflite_path))
with open(tflite_path, "wb") as tflite_out:
    tflite_out.write(tflite_model)
```

5. Then the `.tflite` model is ready to be deployed

## Pretrained Models

Go to [drive](https://drive.google.com/drive/folders/1BD0AK30n8hc-yR28C5FW3LqzZxtLOQfl?usp=sharing)

## Corpus Sources

### English

| **Name**     | **Source**                                                         | **Hours** |
| :----------- | :----------------------------------------------------------------- | :-------- |
| LibriSpeech  | [LibriSpeech](http://www.openslr.org/12)                           | 970h      |
| Common Voice | [https://commonvoice.mozilla.org](https://commonvoice.mozilla.org) | 1932h     |

### Vietnamese

| **Name**                               | **Source**                                                                             | **Hours** |
| :------------------------------------- | :------------------------------------------------------------------------------------- | :-------- |
| Vivos                                  | [https://ailab.hcmus.edu.vn/vivos](https://ailab.hcmus.edu.vn/vivos)                   | 15h       |
| InfoRe Technology 1                    | [InfoRe1 (passwd: BroughtToYouByInfoRe)](https://files.huylenguyen.com/25hours.zip)    | 25h       |
| InfoRe Technology 2 (used in VLSP2019) | [InfoRe2 (passwd: BroughtToYouByInfoRe)](https://files.huylenguyen.com/audiobooks.zip) | 415h      |

## How to contribute

1. Fork the project
2. [Install for development](#installing-for-development)
3. Create a branch
4. Make a pull request to this repo

## References & Credits

1. [NVIDIA OpenSeq2Seq Toolkit](https://github.com/NVIDIA/OpenSeq2Seq)
2. [https://github.com/noahchalifour/warp-transducer](https://github.com/noahchalifour/warp-transducer)
3. [Sequence Transduction with Recurrent Neural Network](https://arxiv.org/abs/1211.3711)
4. [End-to-End Speech Processing Toolkit in PyTorch](https://github.com/espnet/espnet)
5. [https://github.com/iankur/ContextNet](https://github.com/iankur/ContextNet)

## Contact

Huy Le Nguyen

Email: nlhuy.cs.16@gmail.com
