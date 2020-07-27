# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    import numpy as np

    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import accuracy_score, classification_report, f1_score, matthews_corrcoef, precision_score, recall_score

    _has_sklearn = True
except (AttributeError, ImportError):
    _has_sklearn = False


def is_sklearn_available():
    return _has_sklearn


if _has_sklearn:

    def simple_accuracy(preds, labels):
        return (preds == labels).mean()

    def acc_and_f1(preds, labels):
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }

    def pearson_and_spearman(preds, labels):
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }

    def all_metrics(preds, labels, eval_output_dir):
        acc = accuracy_score(y_true=labels, y_pred=preds)
        f1 = f1_score(y_true=labels, y_pred=preds, average="macro")
        f2 = f1_score(y_true=labels, y_pred=preds, average="micro")
        f3 = f1_score(y_true=labels, y_pred=preds, average="weighted")
        precision1 = precision_score(y_true=labels, y_pred=preds, average="macro")
        precision2 = precision_score(y_true=labels, y_pred=preds, average="weighted")
        recall1 = recall_score(y_true=labels, y_pred=preds, average="macro")
        recall2 = recall_score(y_true=labels, y_pred=preds, average="weighted")
        report = classification_report(y_true=labels, y_pred=preds)
        np.savetxt("{}/preds.txt".format(eval_output_dir), preds)

        return {
            "acc": acc,
            "macro-f1": f1,
            "micro-f1": f2,
            "weighted f1": f3,
            "macro precision": precision1,
            "macro recall": recall1,
            "report": report,
            "codalab score": round(f3, 6),
            "codalab precision": round(precision2, 6),
            "codalab recall": round(recall2, 6),
        }

    def glue_compute_metrics(task_name, preds, labels, eval_output_dir ):
        assert len(preds) == len(labels)
        if task_name == "cola":
            return {"mcc": matthews_corrcoef(labels, preds)}
        elif task_name == "sst-2":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "mrpc":
            return acc_and_f1(preds, labels)
        elif task_name == "sts-b":
            return pearson_and_spearman(preds, labels)
        elif task_name == "qqp":
            return acc_and_f1(preds, labels)
        elif task_name == "mnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "mnli-mm":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "qnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "rte":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "wnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "hans":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "sent":
            return all_metrics(preds, labels, eval_output_dir)
        else:
            raise KeyError(task_name)

    def xnli_compute_metrics(task_name, preds, labels):
        assert len(preds) == len(labels)
        if task_name == "xnli":
            return {"acc": simple_accuracy(preds, labels)}
        else:
            raise KeyError(task_name)
