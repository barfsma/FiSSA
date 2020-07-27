# FiSSA
This repository contains the code used for FiSSA at SemEval-2020 Task 9: Fine-tuned For Feelings.

Our code provides small adaptations to two files from [the Huggingface Transformers library](https://huggingface.co/transformers/), needed to reproduce our results. First of all: clone [the correct version of Huggingface-Transformers](https://github.com/huggingface/transformers/tree/v2.5.0) into your clone of FiSSA. Then, copy the contents of `transformers-master-sent` to `transformers-master`. The code can then be executed using our Jupyter notebook. Note: the notebook installs Transformers, do not manually install beforehand.

Detailed changes compared to the original files:
* `metrics/__init__.py`: `all_metrics` function added
* `processors/glue.py`: `SentProcessor` class added