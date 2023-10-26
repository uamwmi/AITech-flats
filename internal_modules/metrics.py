# SPDX-FileCopyrightText: 2023 Adam Wojdyła <adam.wojdyla@outlook.com>
#
# SPDX-License-Identifier: MIT

from statistics import mean
import json
import mlflow

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


class Metrics:
    def __init__(self, use_mlflow: bool = False, set_name: str = "train"):
        self.loss = []
        self.accuracy = []
        self.precision = []
        self.recall = []
        self.f_score = []
        self.use_mlflow = use_mlflow
        self.set_name = set_name

    def add_new(self, predictions: np.ndarray, trues: np.ndarray, losses):
        self.loss.append(mean(losses))
        precision, recall, f_scr, _ = precision_recall_fscore_support(
            trues, predictions, average="weighted", zero_division=1
        )
        self.precision.append(precision)
        self.recall.append(recall)
        self.f_score.append(f_scr)
        self.accuracy.append(accuracy_score(trues, predictions))
        if self.use_mlflow:
            self.log_metrics_to_mlflow(set_name=self.set_name)

    def as_dict(self, current_state=False):
        return {
            "loss": self.loss if not current_state else self.loss[-1],
            "acc": self.accuracy if not current_state else self.accuracy[-1],
            "precision": self.precision if not current_state else self.precision[-1],
            "recall": self.recall if not current_state else self.recall[-1],
            "f1": self.f_score if not current_state else self.f_score[-1],
        }

    def log_metrics_to_mlflow(self, set_name: str = "train"):
        """
        This method will log the latest metrics to mlflow. Should be invoked after each epoch.
        :param set_name: Default is `train`. Can be `test` or `train`.
        """
        _ = [
            mlflow.log_metric(f"{set_name}_{key}", value)
            for key, value in self.as_dict(current_state=True).items()
        ]

    def __str__(self):
        metrics_dict = self.as_dict(current_state=True)
        return json.dumps(metrics_dict, indent=2)
