# SPDX-FileCopyrightText: 2023 Adam Wojdyła <adam.wojdyla@outlook.com>
#
# SPDX-License-Identifier: MIT

import os
import sys
import shutil
from abc import ABC, abstractmethod

ROOT_FOLDER_PATH = os.path.abspath(
    os.path.join(os.path.realpath(__file__), "../../../")
)
if ROOT_FOLDER_PATH not in sys.path:
    sys.path.append(ROOT_FOLDER_PATH)

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from internal_modules.utils import data_utils

class AugmentationTemplate(ABC):
    def __init__(self, source_dir, dest_dir=None):
        self.source_dir = source_dir
        self.dest_dir = dest_dir
        self._class_name = self.__class__.__name__

    @abstractmethod
    def apply_image_augmentation(self, image) -> Image.Image:
        """Apply augmentation on single image.

        Args:
            image (Image.pyi): input image

        Returns: An :py:class:`~PIL.Image.Image` augmented image.
        """

    def run_augmentation(self, overwrite=False, visualize=False):
        """
        Run augmentation on dataset, save augmented images to dest_dir.
        """
        data_utils.remove_ds(self.source_dir)
        directories = os.listdir(self.source_dir)
        aug_flag = "_aug_"
        aug_appendix = f"{aug_flag}{self._class_name}.jpg"

        print(
            f"Running augmentation: {self._class_name} on {self.source_dir} to {self.dest_dir}"
        )

        for directory in directories:
            directory_path = os.path.join(self.source_dir, directory)
            files = os.listdir(directory_path)

            if self.dest_dir:
                data_utils.safe_mkdir(os.path.join(self.dest_dir, directory))

            print(f"Augmenting {directory} with {len(files)} images")
            for file in tqdm(files):
                img_raw_source_path = os.path.join(directory_path, file)
                img_raw = Image.open(img_raw_source_path)
                img_raw_dest_path = os.path.join(self.dest_dir, directory, file)
                img_augmented_name = file.replace(".jpg", aug_appendix)
                img_augmented_dest_path = os.path.join(
                    self.dest_dir, directory, img_augmented_name
                )
                img_augmented = self.apply_image_augmentation(img_raw)

                if not overwrite and (
                    aug_flag in file or os.path.exists(img_augmented_dest_path)
                ):
                    continue

                if not self.dest_dir or self.source_dir == self.dest_dir:
                    img_augmented.save(
                        os.path.join(self.source_dir, directory, img_augmented_name),
                        quality=100,
                    )
                else:
                    img_augmented.save(img_augmented_dest_path, quality=100)
                    if not overwrite and os.path.exists(img_raw_dest_path):
                        continue
                    shutil.copy(img_raw_source_path, img_raw_dest_path)

        print(f"Augmentation: {self._class_name} finished")

        if visualize:
            self.visualize_dataset(5, augmented=True, randomize=False)

    def visualize_dataset(self, samples_count, augmented=False, randomize=True):
        """Visualize dataset

        Args:
            images_folder_dir (string): Path to images folder
            samples_count (int): Number of samples to visualize
            augmented (bool, optional): Whether to apply augmentation. Defaults to False.
        """
        data_utils.remove_ds(self.source_dir)
        directories = os.listdir(self.source_dir)
        n_col = samples_count
        n_row = len(directories)

        if augmented:
            plt.figure(2, figsize=[4 * n_col, 30])
        plt.figure(1, figsize=[4 * n_col, 30])

        for dir_nr, directory in enumerate(directories):
            directory_path = os.path.join(self.source_dir, directory)
            files = os.listdir(directory_path)

            if randomize:
                images = np.random.choice(files, samples_count)
            else:
                images = files[:samples_count]

            for file_nr, file in enumerate(images):
                plt.figure(1)
                plt.subplot(n_row, n_col, file_nr + 1 + dir_nr * n_col)
                img = Image.open(directory_path + "/" + file)
                img = transforms.functional.resize(img, (256, 256))
                plt.title(directory, fontdict={"fontsize": 8})
                plt.imshow(img)

                if augmented:
                    plt.figure(2)
                    plt.subplot(n_row, n_col, file_nr + 1 + dir_nr * n_col)
                    img = self.apply_image_augmentation(img)
                    plt.title(directory, fontdict={"fontsize": 8})
                    plt.imshow(img)

        print("Saving plots...")
        fig1 = plt.figure(1)
        fig1.savefig(f"data/augmentation/plots/{self._class_name}_dataset_raw.png")

        if augmented:
            fig2 = plt.figure(2)
            fig2.savefig(f"data/augmentation/plots/{self._class_name}_dataset_aug.png")
