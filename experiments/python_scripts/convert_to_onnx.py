#!/usr/bin/env python

# SPDX-FileCopyrightText: 2023 Adam Wojdyła <adam.wojdyla@outlook.com>
#
# SPDX-License-Identifier: MIT

import argparse
import torch
import torch.onnx


def convert_onnx(model, destination_path):
    """
    Converts model to ONNX format
    :param model: model to be converted
    :param destination_path: path to save the model with .onnx extension
    :return: None
    """
    batch_size = 1
    input_size = torch.randn(batch_size, 3, 224, 224, requires_grad=True)

    model.to("cpu")
    model.eval()

    torch.onnx.export(
        model,
        input_size,
        destination_path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["modelInput"],
        output_names=["modelOutput"],
        dynamic_axes={
            "modelInput": {0: "batch_size"},
            "modelOutput": {0: "batch_size"},
        },
    )
    print("Model has been converted to ONNX")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX format")
    parser.add_argument("-s", "--source_path", type=str, help="Path to the PyTorch model file")
    parser.add_argument("-d", "--dest_path", type=str, help="Path to save the ONNX model file")
    args = parser.parse_args()

    loaded_model = torch.load(args.source_path)
    convert_onnx(loaded_model, args.dest_path)
    