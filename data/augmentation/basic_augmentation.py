# SPDX-FileCopyrightText: 2023 Adam Wojdyła <adam.wojdyla@outlook.com>
#
# SPDX-License-Identifier: MIT

from PIL import Image
from torchvision import transforms

from data.augmentation.augmentation_template import AugmentationTemplate


class BasicAugmentation(AugmentationTemplate):
    def __init__(self, source_dir, dest_dir):
        super().__init__(source_dir, dest_dir)

    def apply_image_augmentation(self, image) -> Image:
        image = transforms.functional.adjust_sharpness(image, 2)
        image = transforms.functional.adjust_contrast(image, 1.2)
        image = transforms.functional.adjust_saturation(image, 1.5)
        image = transforms.functional.equalize(image)
        return image
