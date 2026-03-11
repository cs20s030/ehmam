
# [EH-MAM: Easy-to-Hard Masked Acoustic Modeling for Self-Supervised Speech Representation Learning](https://openreview.net/pdf?id=N06hbHULIP)

## Overview

EH-MAM is a self-supervised speech representation learning framework that introduces an **Easy-to-Hard masking strategy** during pre-training. Built on top of [fairseq](https://github.com/facebookresearch/fairseq) and the data2vec architecture, it uses a teacher model (EMA) to predict reconstruction difficulty at each position and progressively shifts masking from easy to hard positions during training.

### Project Structure

```
ehmam/
├── config/v2/              # Hydra configuration files
│   ├── base_audio_only_task.yaml   # Base audio pre-training config
│   └── ...
├── data/                   # Dataset classes
│   ├── mae_image_dataset.py
│   └── image_dataset.py
├── models/
│   ├── data2vec2.py        # Main model: Data2VecMultiModel
│   ├── data2vec_audio.py   # Standalone audio model
│   ├── data2vec_vision.py  # Vision model
│   ├── modalities/
│   │   ├── base.py         # ModalitySpecificEncoder base class
│   │   ├── audio.py        # AudioEncoder (CNN + conv positional encoding)
│   │   ├── images.py       # ImageEncoder (patch embedding)
│   │   ├── text.py         # TextEncoder (token embedding)
│   │   └── modules.py      # Decoder1d, PredDecoder1d
│   └── utils.py
├── tasks/                  # Fairseq task definitions
│   ├── multimodal.py       # Multi-modal pre-training task
│   ├── image_pretraining.py
│   └── audio_classification.py
├── scripts/                # Utility scripts
├── data_utils.py           # Core: compute_mask_indices_ema_loss()
└── README.md
```

## Setup

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- CUDA-capable GPU (recommended)

### Installation

```bash
# 1. Clone and install fairseq
git clone https://github.com/facebookresearch/fairseq
cd fairseq
pip install --editable ./

# 2. Integrate EH-MAM into fairseq
# Copy all EH-MAM files into fairseq's data2vec example directory:
cp -r /path/to/ehmam/* examples/data2vec/

# 3. Add Easy-to-Hard masking to fairseq core
# Copy the compute_mask_indices_ema_loss function from ehmam/data_utils.py
# into fairseq's own fairseq/data/data_utils.py
```

### Data Preparation

Follow the [wav2vec 2.0 data preparation guide](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec) for pre-training and fine-tuning data preprocessing.

For LibriSpeech, you will need to generate manifest files (`.tsv`) pointing to your audio data.

## Usage

### Pre-training

For the full list of hyper-parameters, see the [config file](config/v2/base_audio_only_task.yaml).

```bash
python fairseq_cli/hydra_train.py -m \
    --config-dir examples/data2vec/config/v2 \
    --config-name base_audio_only_task \
    task.data=/path/to/manifests
```

### Loading a Pre-trained Model

```python
import fairseq
import argparse

code_path = "examples/data2vec"
fairseq.utils.import_user_module(argparse.Namespace(user_dir=code_path))

ckpt_path = "/path/to/the/checkpoint.pt"
models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
model = models[0]
```

### Fine-tuning for ASR

```bash
# Fine-tuning with 100hr labeled LibriSpeech data
python fairseq_cli/hydra_train.py -m \
    --config-dir examples/wav2vec/config/finetuning \
    --config-name base_100h \
    common.user_dir=examples/data2vec \
    task.data=/path/to/labeled/librispeech/ \
    model.w2v_path=/path/to/ehmam.ckpt \
    task.normalize=True
```

## Pre-trained Checkpoint

Pre-trained checkpoint (without fine-tuning) can be downloaded [here](https://drive.google.com/file/d/1Rx4MpeN1-0xjjKXx5zbJMCCvGLdVe1nr/view?usp=sharing).
