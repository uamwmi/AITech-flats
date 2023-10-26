# SPDX-FileCopyrightText: 2023 Adam Wojdyła <adam.wojdyla@outlook.com>
#
# SPDX-License-Identifier: MIT

from collections import Counter
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import RocCurveDisplay, confusion_matrix, roc_curve
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage

from internal_modules import metrics


def plot_metrics(
    title: str,
    test_metrics: metrics.Metrics,
    train_metrics: metrics.Metrics,
    n_epochs: int,
    time: float,
    image_size: int = 256,
    device: str = "cpu",
):
    """
    Shows a plot from collected metrics
    :param title: Plot title
    :param test_metrics: A dict of keyed metric scores with arrays as values.
    Each metric should have the same # of items.
    Keys:
    - l - losses
    - a - accuracy scores
    - p - precision scores
    - r - recall scores
    - f - f scores
    :param train_metrics: A dict of keyed metric scores with arrays as values.
    See `test_metrics` for details
    :param n_epochs:
    :param time: Time taken to train the model in seconds
    :param image_size: A number corresponding to the size of images used to train the model
    :param device: What was used to train the model
    :return:
    """
    plt.style.use("classic")
    sns.set()
    epoch_nums = np.arange(1, n_epochs + 1)

    fig, axis = plt.subplot_mosaic(
        [["l", "l"], ["a", "p"], ["r", "f"]], constrained_layout=True, figsize=(10, 10)
    )
    axis["l"].plot(epoch_nums, test_metrics.loss)
    axis["l"].plot(epoch_nums, train_metrics.loss)
    # axis['l'].set_yscale('log')
    axis["l"].set_title("Loss")

    axis["a"].plot(epoch_nums, test_metrics.accuracy)
    axis["a"].plot(epoch_nums, train_metrics.accuracy)
    axis["a"].set_title("Accuracy")

    axis["p"].plot(epoch_nums, test_metrics.precision)
    axis["p"].plot(epoch_nums, train_metrics.precision)
    axis["p"].set_title("Precision")

    axis["r"].plot(epoch_nums, test_metrics.recall)
    axis["r"].plot(epoch_nums, train_metrics.recall)
    axis["r"].set_title("Recall")

    axis["f"].plot(epoch_nums, test_metrics.f_score)
    axis["f"].plot(epoch_nums, train_metrics.f_score)
    axis["f"].set_title("F-score")
    fig.tight_layout()
    fig.subplots_adjust(top=0.90, bottom=0.05)
    fig.suptitle(title, fontsize=24)
    fig.legend(axis, labels=["test", "train"], loc="lower center")

    plt.text(
        0.30,
        0.93,
        f"{device}, {image_size}x{image_size}, {n_epochs} iteracji, czas treningu: "
        f'{str(timedelta(seconds=time)).split(".", maxsplit=1)[0]}',
        fontsize=14,
        transform=plt.gcf().transFigure,
    )

    return fig


def show_misclassified(
    dataset: Dataset, predictions: np.ndarray, label_names: dict, count_per_class: int = 5
):
    results = {}
    for i in label_names.keys():
        results[i] = []

    indexes = np.random.permutation(len(predictions))
    for i in indexes:
        pred = predictions[i]
        image_tensor, true = dataset[i]
        if len(results[pred]) < count_per_class and pred != int(true):
            results[pred].append(
                {"image": ToPILImage()(image_tensor), "actual": int(true)}
            )
    sns.reset_orig()
    fig = plt.figure(figsize=[20, 30])
    for row, (label, images) in enumerate(results.items()):
        for i, image in enumerate(images):
            plt.subplot(
                len(label_names.keys()), count_per_class, row * count_per_class + i + 1
            )
            plt.imshow(image["image"], interpolation="bicubic")
            plt.title(f'{label_names[label]}, expected {label_names[image["actual"]]}')
            plt.axis("off")

    return fig


def plot_confusion_matrix(
    model: nn.Module,
    device: str,
    dataset: Dataset,
    classes: list[str],
    title: str = "Confusion matrix",
):
    """
    Plots a confusion matrix - summary of prediction results on a classification problem.
    :param model: Model to use for predictions
    :param device: Device to use for predictions (mps, cuda, cpu)
    :param dataset: Dataset to use for predictions
    :param classes: List of class names to use in the matrix
    :param title: The title of plot
    :return:
    """
    model.eval()
    model.to(device)

    truelabels = []
    predictions = []
    i = 0
    for data, target in dataset:
        data, target = data.to(device), target.to(device)
        print(f"{i*len(data)}/{len(dataset.dataset)}", end="\r")

        for label in target.cpu().data.numpy():
            truelabels.append(label)
        for prediction in model(data).cpu().data.numpy().argmax(1):
            predictions.append(prediction)
        i += 1

    conf_matrix = confusion_matrix(truelabels, predictions)
    fig = plt.figure()
    plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    plt.xlabel("Predicted architectural style")
    plt.ylabel("Actual architectural style")
    plt.title(title)

    return fig


def plot_roc_curve_multiclass(
    model: nn.Module,
    device: str,
    dataset: Dataset,
    classes: list[str],
    title: str = "Multiclass ROC curve",
):
    """
    Plots a multiclass ROC curve.
    :param model: Model to use for predictions
    :param device: Device to use for predictions (mps, cuda, cpu)
    :param dataset: Dataset to use for predictions
    :param classes: List of class names to use in the matrix
    :param title: The title of plot
    :return:
    """
    truelabels = []
    predictions = []
    i = 0
    for data, target in dataset:
        data, target = data.to(device), target.to(device)
        print(f"{i*len(data)}/{len(dataset.dataset)}", end="\r")
        for label in target.cpu().data.numpy():
            truelabels.append(label)
        for prediction in model(data).cpu().data.numpy().argmax(1):
            pred = torch.nn.functional.one_hot(
                torch.tensor(prediction), num_classes=len(classes)
            )
            pred = pred.numpy()
            predictions.append(pred)
        i += 1

    # Get ROC metrics for each class
    fpr = {}
    tpr = {}
    thresh = {}
    colors = [
        "orange",
        "lightgreen",
        "blue",
        "red",
        "olive",
        "azure",
        "purple",
        "aqua",
        "darkgreen",
    ]
    plt.figure(figsize=(7, 7))

    for i, _ in enumerate(classes):
        fpr[i], tpr[i], thresh[i] = roc_curve(
            np.array(truelabels), np.array(predictions)[:, i], pos_label=i
        )
        plt.plot(
            fpr[i],
            tpr[i],
            linestyle="--",
            color=colors[i],
            label=classes[i] + " vs Rest",
        )

    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.title(title)
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive rate", fontsize=11)
    plt.legend(prop={"size": 11}, loc="best")
    fig = plt.gcf()

    return fig


def plot_roc_curve_one_vs_rest(
    model: nn.Module,
    device: str,
    dataset: Dataset,
    classes: list[str],
    class_id: int = 0,
):
    """
    Plots a multiclass ROC curve.
    :param model: Model to use for predictions
    :param device: Device to use for predictions (mps, cuda, cpu)
    :param dataset: Dataset to use for predictions
    :param classes: List of class names to use in the matrix
    :param class_id: Id of class to be used for ROC calculation
    :return:
    """

    truelabels = []
    predictions = []
    i = 0
    for data, target in dataset:
        data, target = data.to(device), target.to(device)
        print(f"{i*len(data)}/{len(dataset.dataset)}", end="\r")

        for label in target.cpu().data.numpy():
            tru = torch.nn.functional.one_hot(
                torch.tensor(label), num_classes=len(classes)
            )
            tru = tru.numpy()
            truelabels.append(tru)
        for prediction in model(data).cpu().data.numpy().argmax(1):
            pred = torch.nn.functional.one_hot(
                torch.tensor(prediction), num_classes=len(classes)
            )
            pred = pred.numpy()
            predictions.append(pred)
        i += 1

    truelabels = np.array(truelabels)
    predictions = np.array(predictions)
    plot_title = f"{classes[class_id]} vs the rest"
    RocCurveDisplay.from_predictions(
        truelabels[:, class_id],
        predictions[:, class_id],
        name=plot_title,
        color="darkorange",
    )
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(plot_title)
    plt.legend()
    fig = plt.gcf()

    return fig


def plot_data_distribution(full_dataset: Dataset, classes: list[str]):
    """
    Shows a bar plot with the distribution of classes in the dataset
    :param full_dataset: The full dataset.
    :param classes: List of class names
    """
    my_dict = dict(Counter(full_dataset.targets))
    y = [v for k, v in list(my_dict.items())]
    fig, _ = plt.subplots(figsize=(15, 5), dpi=100)
    plt.title("Architectural style classes distribution")
    classes.sort()
    plt.bar(classes, height=y)

    return fig


def create_plots_from_experiment(
    test_dataset, full_dataset, dataset_subset, classes, model, device
):
    """
    Create plots for the experiment
    Args:
        test_dataset: test dataset loader
        full_dataset: full dataset loader
        classes: list of classes
        model: trained model
        device: device
        dataset_subset: subset of dataset

    Returns(dict): dict with plot name as key and plot as value
    """
    confusion_matrix_plot = plot_confusion_matrix(
        model,
        device,
        test_dataset,
        classes,
        title="Normalized confusion matrix",
    )
    roc_multiclass = plot_roc_curve_multiclass(model, device, test_dataset, classes)
    roc_figure = plot_roc_curve_one_vs_rest(model, device, test_dataset, classes, 1)

    plots_dict = {
        "confusion_matrix": confusion_matrix_plot,
        "roc_multiclass": roc_multiclass,
        "roc_figure": roc_figure,
    }

    if dataset_subset is None:
        distribution_figure = plot_data_distribution(full_dataset, classes)
        plots_dict["distribution_figure"] = distribution_figure

    return plots_dict
