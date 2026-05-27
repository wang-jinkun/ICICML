---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
dataset_info:
  features:
  - name: image
    dtype: image
  - name: label
    dtype: image
  splits:
  - name: train
    num_bytes: 2203724361.79
    num_examples: 2594
  - name: validation
    num_bytes: 241025351.0
    num_examples: 100
  - name: test
    num_bytes: 2389508202.0
    num_examples: 1000
  download_size: 13874599089
  dataset_size: 4834257914.79
---
# Dataset Card for "ISIC2018"

[More Information needed](https://github.com/huggingface/datasets/blob/main/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)