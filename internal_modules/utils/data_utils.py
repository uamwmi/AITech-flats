# SPDX-FileCopyrightText: 2023 Adam Wojdyła <adam.wojdyla@outlook.com>
#
# SPDX-License-Identifier: MIT

import math
import os
import random
import shutil

from hydra.utils import instantiate


def create_subset(
    source_dir, dest_dir, subset_size, shuffle=True, sub_folder_name=None
):
    """Creates a subset of the dataset. Source_dir should contain sub folders with images.

    Args:
        source_dir (string): Source directory to dataset folder
        dest_dir (string): Destination directory to new dataset folder
        subset_size (int): Size of the subset
        sub_folder_name (string, optional): Single subfolder name.
        Create subset only for given folder.
    """
    remove_ds(source_dir)
    directories = os.listdir(source_dir)

    # round up to equalise the number of images in each class
    single_folder_subset_size = math.ceil(subset_size / len(directories))
    subset_size = single_folder_subset_size * len(directories)

    print(f"Creating subset of {subset_size} images from {source_dir} to {dest_dir}")

    for _, directory in enumerate(directories):
        if sub_folder_name is not None and directory != sub_folder_name:
            continue

        safe_mkdir(os.path.join(dest_dir, directory))
        source_files_path = os.path.join(source_dir, directory)
        dest_files_path = os.path.join(dest_dir, directory)
        files = os.listdir(source_files_path)

        if shuffle:
            random.shuffle(files)

        print(
            f"Copying {len(files[:single_folder_subset_size])} files {source_files_path} \
              to {dest_files_path}"
        )

        for _, file in enumerate(files[:single_folder_subset_size]):
            if os.path.isfile(os.path.join(dest_files_path, file)):
                print(f"File {file} already exists in {dest_files_path}")
                continue
            shutil.copy(
                os.path.join(source_files_path, file),
                os.path.join(dest_files_path, file),
            )

    print("Subset created")


def run_augmentation(subset_size, source_dir, dest_dir, augmentation):
    """
    Runs data augmentation on a given dataset.

    Args:
        subset_size (int): The size of the subset to create before running augmentation.
            If subset size is provided, the script will create a subset of the source dataset
            and run augmentations in this subset.
            If None, the entire dataset will be used.
        source_dir (str): The path to the directory containing the source dataset.
        dest_dir (str): The path to the directory where the augmented dataset will be saved.
        augmentation (str): The name of the augmentation technique to use.

    Returns:
        None
    """
    if subset_size:
        print("Creating subset of dataset before augmentation")
        create_subset(
            source_dir=source_dir,
            dest_dir=dest_dir,
            subset_size=subset_size,
            shuffle=False,
        )
        augmentation_instance = instantiate(
            augmentation, source_dir=dest_dir, dest_dir=dest_dir
        )
    else:
        augmentation_instance = instantiate(
            augmentation, source_dir=source_dir, dest_dir=dest_dir
        )

    augmentation_instance.run_augmentation()


def remove_ds(root_path):
    ds_store_file_name = ".DS_Store"
    for root, _, files in os.walk(root_path):
        if ds_store_file_name in files:
            ds_store_file_path = os.path.join(root, ds_store_file_name)
            os.remove(ds_store_file_path)
            print(f"Removed {ds_store_file_path}")


def safe_mkdir(path):
    """Safe mkdir
    Args:
        path (string): Path to leaf directory to be created
    """
    if not os.path.exists(path):
        os.makedirs(path)
