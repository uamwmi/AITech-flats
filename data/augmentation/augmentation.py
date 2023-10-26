# SPDX-FileCopyrightText: 2023 Adam Wojdyła <adam.wojdyla@outlook.com>
#
# SPDX-License-Identifier: MIT

import os
import sys

ROOT_FOLDER_PATH = os.path.abspath(
    os.path.join(os.path.realpath(__file__), "../../../")
)
if ROOT_FOLDER_PATH not in sys.path:
    sys.path.append(ROOT_FOLDER_PATH)

from data.augmentation.basic_augmentation import BasicAugmentation
from data.augmentation.grayscale_augmentation import GrayScaleAugmentation
from internal_modules.utils import data_utils

if __name__ == "__main__":
    dataset_path = os.path.join(ROOT_FOLDER_PATH, "data/images/raw/houzz")
    dataset_subset_path = os.path.join(ROOT_FOLDER_PATH, "data/images/houzz_subset")
    dataset_subset_augmented_path = os.path.join(
        ROOT_FOLDER_PATH, "data/images/houzz_subset_augmented"
    )

    # Create subset
    data_utils.create_subset(
        source_dir=dataset_path,
        dest_dir=dataset_subset_path,
        subset_size=800,
        shuffle=False,
    )

    # Create augmentators
    grayscale_augmentation = GrayScaleAugmentation(
        source_dir=dataset_subset_path,
        dest_dir=dataset_subset_augmented_path
    )
    basic_augmentation = BasicAugmentation(
        source_dir=dataset_subset_path,
        dest_dir=dataset_subset_augmented_path
    )
    augmentations = [basic_augmentation]


    # Preview augmented images (optional)
    # augmentations[0].visualize_dataset(5, augmented=True, randomize=False)

    # Run augmentation
    # for augmentation in augmentations:
    #     augmentation.run_augmentation(overwrite=True, visualize=False)
