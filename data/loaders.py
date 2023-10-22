# SPDX-FileCopyrightText: 2023 Adam Wojdyła <adam.wojdyla@outlook.com>
#
# SPDX-License-Identifier: MIT

import json
import multiprocessing as mp
import os
import sys
from abc import ABC

ROOT_FOLDER_PATH = os.path.abspath(os.path.join(os.path.realpath(__file__), "../../"))
if ROOT_FOLDER_PATH not in sys.path:
    sys.path.append(ROOT_FOLDER_PATH)

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from internal_modules.utils.data_utils import remove_ds


def _prepare_directory(input_dir: str):
    remove_ds(input_dir)


def _get_default_workers_count():
    return int(mp.cpu_count() / 2)


class RamDataset(Dataset):
    def __init__(self, disk_dataset: torchvision.datasets.ImageFolder, targets):
        self.targets = targets
        self.x, self.y = [], []
        for data in tqdm(disk_dataset):
            self.x.append(data[0])
            self.y.append(data[1])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, ind):
        return self.x[ind], self.y[ind]


class FlatsDatasetLoader(Dataset, ABC):
    def __init__(
        self,
        images_dir: str,
        resize_to: int or None = None,
        batch_size: int = 512,
        device: str = "cpu",
    ):
        self.images_dir = images_dir
        self.resize_to = resize_to
        self.batch_size = batch_size
        self.device = device
        self.train_loader = None
        self.test_loader = None
        self.classes_count = 0
        self.label_names = {}
        _prepare_directory(self.images_dir)

    def load(
        self,
        train_set_ratio=0.8,
        workers_num=_get_default_workers_count(),
        verbose: bool = True,
        subset_size: int = None,
        load_to_ram: bool = False,
    ):
        transformation = transforms.Compose(
            [
                transforms.Resize(self.resize_to),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

        print(f"Workers count: {workers_num}")

        if verbose:
            print("Loading dataset from files...")
        full_dataset = torchvision.datasets.ImageFolder(
            root=self.images_dir, transform=transformation
        )
        full_dataset_targets = full_dataset.targets

        self.classes_count = len(full_dataset.classes)
        self.label_names = {
            value: key for key, value in full_dataset.class_to_idx.items()
        }

        if subset_size:
            full_dataset = torch.utils.data.Subset(
                full_dataset,
                np.random.choice(len(full_dataset), subset_size, replace=False),
            )
        if load_to_ram:
            full_dataset = RamDataset(full_dataset, targets=full_dataset_targets)

        train_size = int(train_set_ratio * len(full_dataset))
        test_size = len(full_dataset) - train_size

        generator = torch.Generator()
        generator.manual_seed(0)
        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size], generator=generator
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=workers_num,
            shuffle=True,
        )

        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=workers_num,
            shuffle=False,
        )

        print("Done.")
        return full_dataset

    def get_train_loader(self) -> DataLoader:
        return self.train_loader

    def get_test_loader(self) -> DataLoader:
        return self.test_loader

    def get_label_names(self) -> dict:
        return self.label_names

    def get_classes_count(self) -> int:
        return self.classes_count

    def as_dict(self):
        return {
            "FullDatasetCount": len(self.train_loader.dataset)
            + len(self.test_loader.dataset),
            "TrainDatasetCount": len(self.train_loader.dataset),
            "TestDatasetCount": len(self.test_loader.dataset),
            "ClassesCount": self.classes_count,
            "LabelNames": self.get_label_names(),
        }

    def __str__(self):
        json_obj = self.as_dict()
        json_formatted_str = json.dumps(json_obj, indent=2)
        return json_formatted_str
