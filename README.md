
# [EH-MAM: Easy-to-Hard Masked Acoustic Modeling for Self-Supervised Speech Representation Learning](https://openreview.net/pdf?id=N06hbHULIP)


### Setup

- Codebase preparation (based on [`fairseq`](https://github.com/facebookresearch/fairseq))
```
# we use fairseq to build the model
git clone https://github.com/facebookresearch/fairseq
cd fairseq
pip install --editable ./

# plug in for EH-MAM
replace all the files in examples/data2vec with ehmam
```

- Data preparation:
please follow [`instruction provided by wav2vec2`](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec) for pre-training/fine-tuning data preprocessing


### Usage

- Training

    For the list of hyper-parameters, see [`config file`](config/v2/base_audio_only_task.yaml). 

```
# minimal example to reproduce model
$ python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/v2 \
--config-name base_audio_only_task task.data=/path/to/manifests &
```

- Loading pre-trained model as python object

```
import fairseq
import argparse
code_path = "examples/data2vec"
fairseq.utils.import_user_module(argparse.Namespace(user_dir=code_path))
ckpt_path = "/path/to/the/checkpoint.pt"
models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
model = models[0]
```

- Fine-tuning pre-trained checkpoint as ASR

```
# minimal example for fine-tuning with 100hr data
python fairseq_cli/hydra_train.py -m \
        --config-dir examples/wav2vec/config/finetuning \
        --config-name base_100h \
        common.user_dir=examples/data2vec \
        task.data=/path/to/labeled/librispeech/ \
        model.w2v_path=/path/to/ehmam.ckpt \
        task.normalize=True
```

### Pre-trained checkpoint

Pre-trained checkpoint without fine-tuning can be downloaded [here](https://drive.google.com/file/d/1Rx4MpeN1-0xjjKXx5zbJMCCvGLdVe1nr/view?usp=sharing).
