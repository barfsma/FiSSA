# FiSSA
This repository contains the code used for FiSSA at SemEval-2020 Task 9: Fine-tuned For Feelings.

Our code provides small adaptations to two files from [the Huggingface Transformers library](https://huggingface.co/transformers/), needed to reproduce our results. You must first install the correct version, [namely 2.5.0](https://huggingface.co/transformers/v2.5.0/), and then copy the contents of our transformers-master directory to your clone of the Transformers repository. The code can then be executed using our Jupyter notebook.

Detailed changes compared to the original files:
* `metrics/__init__.py`: `all_metrics` function added
* `processors/glue.py`: `SentProcessor` class added