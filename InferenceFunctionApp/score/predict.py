# SPDX-FileCopyrightText: 2023 Adam Wojdyła <adam.wojdyla@outlook.com>
#
# SPDX-License-Identifier: MIT

"""Ignore import error for pylint."""
# pylint: disable=import-error
import base64
import json
import logging
import os
from io import BytesIO

import numpy as np
import onnxruntime
from PIL import Image
from scipy.special import softmax
from torchvision.transforms import transforms

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(FILE_DIR, "models/classificationModel.onnx")
LABELS_PATH = os.path.join(FILE_DIR, "labels.json")


def load_labels(path):
    """Loads labels from a json file.

    Args:
        path (str): Path to label json file

    Returns:
        ndarray: Array of labels
    """
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    return np.asarray(data)


logging.info("Loading model from: %s", MODEL_PATH)

session = onnxruntime.InferenceSession(MODEL_PATH, None)
input_name = session.get_inputs()[0].name

labels = load_labels(LABELS_PATH)


def preprocess(base64_image):
    """Returns a tensor of an image after preprocessing.

    Args:
        base64_image: Image in base64 format

    Returns:
        numpy.ndarray: Transformed image tensor in numpy array format
    """
    transformation = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    logging.info("Preprocessing image")
    image = Image.open(BytesIO(base64.b64decode(base64_image))).convert("RGB")
    image_tensor = transformation(image)
    image_tensor = image_tensor.unsqueeze_(0)
    image_tensor = image_tensor.numpy()

    return image_tensor


def predict_classes_from_base64_image(base64_image):
    """Returns a dictionary of classes and their probabilities.

    Args:
        base64_image (str): image in base64 format

    Returns:
        dict: Dictionary of classes and their probabilities
    """
    input_data = preprocess(base64_image)
    logging.info("Running model")
    raw_result = session.run([], {input_name: input_data})
    prob_list = softmax(raw_result[0][0])
    results = zip(labels, prob_list)
    results = list(results)
    results.sort(key=lambda tup: tup[1], reverse=True)

    mydict = {}
    for result in results:
        mydict[result[0]] = float(result[1])

    return mydict
