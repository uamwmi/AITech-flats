# SPDX-FileCopyrightText: 2023 Adam Wojdyła <adam.wojdyla@outlook.com>
#
# SPDX-License-Identifier: MIT

import os
import shutil
from pathlib import Path


def mkdir_if_not_exists(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


def bulk_copy(file_names: list[str], input_dir, output):
    for i, file in enumerate(file_names):
        shutil.copy(
            os.path.join(input_dir, file), os.path.join(output, str(i) + '.' + file.split('.')[1])
        )


def split_houzz_dataset(
        raw_path: str,
        train_out_folder: str,
        test_out_folder: str,
        train_test_ratio: float = 0.8
):
    image_dir = Path(raw_path)

    classes = []
    for maybe_dir in os.listdir(image_dir):
        class_dir = os.path.join(image_dir, maybe_dir)
        if os.path.isdir(class_dir):
            classes.append(maybe_dir)

    print(f'Found {len(classes)} classes')
    for cls in classes:
        mkdir_if_not_exists(os.path.join(train_out_folder, str(cls)))
        mkdir_if_not_exists(os.path.join(test_out_folder, str(cls)))

    raw_folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    for raw_directory, cls in zip(raw_folders, classes):
        raw_files = os.listdir(raw_directory)
        print(f'{raw_directory}: {len(raw_files)}')
        split_point = round(len(raw_files) * train_test_ratio)
        train_files = raw_files[:split_point]
        print(f'\tTrain files: {len(train_files)}')
        test_files = raw_files[split_point + 1:]
        print(f'\tTest files: {len(test_files)}')
        print('Copying... ', end='')
        bulk_copy(test_files, raw_directory, os.path.join(TEST_OUTPUT, str(cls)))
        bulk_copy(train_files, raw_directory, os.path.join(TRAIN_OUTPUT, str(cls)))
        print('Done.')


if __name__ == '__main__':
    HOUZZ_DATASET_PATH = 'images/raw/houzz'
    TRAIN_OUTPUT = 'images/train'
    TEST_OUTPUT = 'images/test'
    split_houzz_dataset(HOUZZ_DATASET_PATH, TRAIN_OUTPUT, TEST_OUTPUT)
