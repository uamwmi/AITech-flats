# SPDX-FileCopyrightText: 2023 Adam Wojdyła <adam.wojdyla@outlook.com>
#
# SPDX-License-Identifier: MIT

import mlflow
from azureml.core import Workspace


def get_azure_workspace(config):
    azure_workspace = Workspace.get(
        name=config.mlflow.azure.workspace_name,
        subscription_id=config.mlflow.azure.subscription_id,
        resource_group=config.mlflow.azure.resource_group,
    )
    return azure_workspace


def set_mlflow_experiment(config):
    workspace = get_azure_workspace(config)
    mlflow.set_tracking_uri(workspace.get_mlflow_tracking_uri())
