# SPDX-FileCopyrightText: 2023 Adam Wojdyła <adam.wojdyla@outlook.com>
#
# SPDX-License-Identifier: MIT

import mlflow
from omegaconf import DictConfig


def log_plots_mlflow(plots_dict: dict):
    """
    Log plots to mlflow
    :param plots_dict: dict with plots
    """
    for key, value in plots_dict.items():
        if value is not None:
            mlflow.log_figure(value, key + ".png")
        else:
            pass

def log_run_mlflow(
        full_dataset, dataset_info, device, augmentations, config: DictConfig
):
    """
    Log run to mlflow: set tags, log params, log dataset info.
    :param full_dataset: full loaded dataset
    :param dataset_info: info about dataset
    :param device: device
    :param augmentations: list of augmentations
    :param config: configuration for experiment
    """

    mlflow.set_tag("learning_rate", config.params.lr)
    mlflow.set_tag("epochs", config.params.epoch_count)
    mlflow.log_param("device", device)
    mlflow.log_param("dataset_size", len(full_dataset))
    if config.dataset.subset_size is not None:
        mlflow.log_param("subset_size", config.dataset.subset_size)
    mlflow.log_dict(dataset_info, "dataset_info.json")
    if augmentations is not None:
        mlflow.set_tag("augmentation", ", ".join([a.__name__ for a in augmentations]))
    mlflow.log_params(config)
