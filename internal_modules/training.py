# SPDX-FileCopyrightText: 2023 Adam Wojdyła <adam.wojdyla@outlook.com>
#
# SPDX-License-Identifier: MIT

# pylint: disable=broad-except

import copy
import datetime
import os
import sys
import time

import mlflow
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from tqdm import tqdm

from data.loaders import FlatsDatasetLoader
from internal_modules import plots
from internal_modules.inference import evaluate, infer
from internal_modules.metrics import Metrics

ROOT_FOLDER_PATH = os.path.abspath(os.path.join(os.path.realpath(__file__), "../"))
MODELS_FOLDER_PATH = os.path.abspath(
    os.path.join(ROOT_FOLDER_PATH, "../experiments/models")
)

if ROOT_FOLDER_PATH not in sys.path:
    sys.path.append(ROOT_FOLDER_PATH)


def train(
    model: nn.Module,
    flats_dataset_loader: FlatsDatasetLoader,
    optimizer_fn: Optimizer,
    loss_fn,
    device: str,
    epochs: int,
    use_mlflow: bool = False,
) -> tuple[Metrics, Metrics]:
    train_data = flats_dataset_loader.get_train_loader()
    test_data = flats_dataset_loader.get_test_loader()
    test_metrics = Metrics(use_mlflow=use_mlflow, set_name="test")
    train_metrics = Metrics(use_mlflow=use_mlflow, set_name="train")

    model.train()
    epoch = 0
    for _ in tqdm(range(epochs), total=epochs):
        epoch += 1
        print(f"Starting epoch: {epoch}...")
        train_outs, train_losses = np.array([]), []
        labels = np.array([])

        for batch_idx, data in enumerate(train_data):
            optimizer_fn.zero_grad()
            output, loss = infer(data, model, loss_fn, device)
            labels = np.concatenate((labels, data[1].data.numpy()))
            train_outs = np.concatenate(
                (
                    train_outs,
                    torch.nn.functional.softmax(output, dim=1)
                    .cpu()
                    .data.numpy()
                    .argmax(axis=1),
                )
            )
            loss.backward()
            train_losses.append(loss.item())
            optimizer_fn.step()

            if batch_idx % 50 == 0:
                print(
                    f"Training set ["
                    f"{batch_idx * len(data[0])}/{len(train_data.dataset)} "
                    f"({100.0 * batch_idx / len(train_data):.0f}%)] "
                    f"Loss: {loss.item():.6f}"
                )
        print(f"Training finished in epoch: {epoch}...")
        train_metrics.add_new(train_outs, labels, train_losses)
        print("Evaluating model...")
        test_trues, test_predictions, test_losses = evaluate(
            model, test_data, loss_fn, device
        )
        print("Evaluation finished.")
        test_metrics.add_new(test_predictions, test_trues, test_losses)
        print(f"Metrics after epoch: {epoch}: ", end="")
        print(test_metrics)

    return test_metrics, train_metrics


def conduct_experiment(
    title,
    model,
    n_epochs,
    optimizer,
    flats_dataset_loader: FlatsDatasetLoader,
    device: str,
    criterion=nn.CrossEntropyLoss(),
    use_mlflow=False,
    register_model=False,
):
    # Train model
    start = time.time()
    test_metrics, train_metrics = train(
        model, flats_dataset_loader, optimizer, criterion, device, n_epochs, use_mlflow
    )
    end = time.time()
    print(f"Training finished in {end - start} seconds")

    # Save model
    local_model_name = f"{datetime.datetime.now().strftime('%y-%b-%d-%H-%M')}.pt"
    model_path = os.path.join(
        MODELS_FOLDER_PATH,
        title,
        local_model_name,
    )
    save_model(model, model_path)
    model_cpu = save_model_torchscript(model, model_path)

    # Evaluate model and plot results
    combined_plots = plots.plot_metrics(
        title, test_metrics, train_metrics, n_epochs, time=end - start, device=device
    )
    test_loader = flats_dataset_loader.get_test_loader()
    print("Evaluating model on test dataset...")
    _, predictions, _ = evaluate(model, test_loader, criterion, device)
    misclassified_figure = plots.show_misclassified(
        test_loader.dataset, predictions, flats_dataset_loader.get_label_names()
    )

    if use_mlflow:
        # azure_workspace.set_mlflow_experiment(title)
        artifact_model_path = os.path.join("models", title, local_model_name)
        num_of_tries = 0

        while num_of_tries < 5:
            try:
                print("Logging images to mlflow...")
                mlflow.log_figure(misclassified_figure, "misclassified.png")
                mlflow.log_figure(combined_plots, "combined_plots.png")
                print("Logging model to mlflow...")
                mlflow.pytorch.log_model(
                    pytorch_model=model_cpu,
                    artifact_path=artifact_model_path,
                    registered_model_name=title if register_model else None,
                )
                break
            except Exception as exception:
                print(str(exception))
                if "already exists." in str(exception):
                    break
                num_of_tries += 1
                print("Waiting 10 seconds before trying again...")
                time.sleep(10)

        print("Done logging model to mlflow")

    return test_metrics, train_metrics


def save_model(model, model_path):
    """Saves model to given path in .pt format

    :param model: Model to save
    :param model_path: Path to save model
    """
    model_path = model_path.replace(".pt", "-standard.pt")
    model_directory = os.path.abspath(
        os.path.join(os.path.abspath(model_path), os.pardir)
    )
    try:
        os.mkdir(model_directory)
    except FileExistsError:
        pass
    torch.save(model, model_path)
    print(f"Model saved in {model_path}")


def save_model_torchscript(model, model_path):
    """Saves torchscript model to given path in .pt format.
    Torchscript model should be used when converting to ONNX format.

    :param model: Model to save
    :param model_path: Path to save model
    """
    model_path = model_path.replace(".pt", "-torchscript.pt")
    model_directory = os.path.abspath(
        os.path.join(os.path.abspath(model_path), os.pardir)
    )
    try:
        os.mkdir(model_directory)
    except FileExistsError:
        pass
    model_cp = copy.deepcopy(model)
    model_cp.to("cpu")
    model_scripted = torch.jit.script(model_cp)
    model_scripted.save(model_path)
    print(f"Model saved in {model_path}")
    return model_cp
