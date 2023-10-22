# SPDX-FileCopyrightText: 2023 Adam Wojdyła <adam.wojdyla@outlook.com>
#
# SPDX-License-Identifier: MIT

import argparse
import json

import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as nnf
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


def infer(data, network, loss_fn, device_type):
    x_cpu, y_cpu = data
    x = x_cpu.to(device_type).float()
    y = y_cpu.to(device_type).long()
    output = network(x)
    loss = loss_fn(output, y)
    return output, loss


def evaluate(
    network: nn.Module, test_data: DataLoader, loss_fn, device_type: str
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    Test a given model and return true, predicted values and loss
    """
    network.eval()
    predictions, losses = np.array([]), []
    trues = np.array([])
    with torch.no_grad():
        for data in test_data:
            output, loss = infer(data, network, loss_fn, device_type)
            trues = np.concatenate((trues, data[1].data.numpy()))
            predictions = np.concatenate(
                (
                    predictions,
                    torch.nn.functional.softmax(output, dim=1)
                    .cpu()
                    .data.numpy()
                    .argmax(axis=1),
                )
            )
            losses.append(loss.item())
    return trues, predictions, losses


def predict_image(classifier: nn.Module, classes: list[str], device: str, image) -> int:
    """
    Predict the class of image. Returns class index with top confidence.
    """

    classifier.eval()

    # Apply the same transformations as we did for the training images
    image_transformation = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]
    )

    image_tensor = image_transformation(image).to(device)
    image_tensor = image_tensor.unsqueeze_(0)

    classifier = classifier.to("cpu")
    output = classifier(image_tensor)

    prob = nnf.softmax(output, dim=1)
    prob_list = prob.numpy(force=True)[0].tolist()

    results = zip(classes, prob_list)
    results = list(results)
    results.sort(key=lambda tup: tup[1], reverse=True)

    mydict = {}
    for result in results:
        mydict[result[0]] = result[1]

    print("Top 3 classification results:")
    for i in range(3):
        print(f"Class: {results[i][0]}: Confidence: {int(100*results[i][1])}%")

    print(json.dumps(mydict, indent=2))
    index = output.cpu().data.numpy().argmax()

    return index


if __name__ == "__main__":
    infer_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(
        description="Load a model and run inference on a source image"
    )
    parser.add_argument(
        "-m", "--model", required=True, type=str, help="path to a pickled model to load"
    )
    parser.add_argument(
        "-i", "--image", required=True, type=str, help="path to an image to load"
    )
    args = parser.parse_args()
    model = torch.load(args.model)
    infer_image = cv.imread(args.image)
    infer_image = infer_image / 255
    processed = transforms.ToTensor()(infer_image).to(infer_device)
    predicted = model(processed.float().unsqueeze(0))
    labels = {
        "0": "ArtDeco",
        "1": "Classic",
        "2": "Glamour",
        "3": "Industrial",
        "4": "Minimalistic",
        "5": "Modern",
        "6": "Rustic",
        "7": "Scandinavian",
        "8": "Vintage",
    }
    print(
        labels[str(nnf.softmax(predicted, dim=1).cpu().data.numpy().argmax(axis=1)[0])]
    )
