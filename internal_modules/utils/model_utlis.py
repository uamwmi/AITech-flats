# SPDX-FileCopyrightText: 2023 Adam Wojdyła <adam.wojdyla@outlook.com>
#
# SPDX-License-Identifier: MIT

from hydra.utils import instantiate
from torch import nn


def freeze_model(model, model_name, classes_count, device):
    """
    Freezes all layers except the last one and replaces it with a new one with
    the specified number of classes.

    :param model: Model to freeze
    :param model_name: Name of the model. Supported values are:
                       "resnet34", "vgg16", "vgg19"
    :param classes_count: Number of classes to predict
    :param device: Device to operate on

    :return: Frozen model
    """
    if model_name.lower() == "resnet34":
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, classes_count)
    elif model_name.lower() in ["vgg16", "vgg19"]:
        for param in model.features.parameters():
            param.require_grad = False
        num_features = model.classifier[6].in_features
        features = list(model.classifier.children())[:-1]
        features.extend([nn.Linear(num_features, classes_count)])
        model.classifier = nn.Sequential(*features)
    elif model_name.lower() == "vitb16":
        for param in model.parameters():
            param.requires_grad = False
        model.heads.head = nn.Linear(model.heads.head.in_features, classes_count)
    else:
        raise NotImplementedError(f"Model {model_name} is not supported")

    return model.to(device)


def get_optimizer(model, config):
    """
    Resolves appropriate optimizer based on the config.

    :param model: Model to optimize
    :param config: Config with optimizer parameters

    :return: Optimizer instance
    """
    optimizer_name = config.optimizer.name.lower()
    if optimizer_name == "sgd":
        return instantiate(
            config.optimizer.target,
            params=model.parameters(),
            lr=config.params.lr,
            momentum=config.params.momentum,
        )
    if optimizer_name == "adam":
        return instantiate(
            config.optimizer.target, params=model.parameters(), lr=config.params.lr
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")
