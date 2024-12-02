"""
Custom dataset class for the prostate dataset. Up to two classes will exist, prostate and bladder.
"""
import os

import numpy as np
import torch
from PIL import Image
from natsort import natsorted
from torchvision import tv_tensors


class ProstateDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None, augmentation_count=1):
        """
        Set the root and transforms variables. Do some minor dataset validation.

        :param root: Directory containing images and labels folders.
        :param transforms: Transformers to be used on this dataset.
        :param augmentation_count: Increase dataset size by this amount (oversampling).
        """
        self.root = root
        self.transforms = transforms
        self.augmentation_count = augmentation_count
        self.imgs = list(natsorted(os.listdir(os.path.join(root, "images"))))
        self.labels = list(natsorted(os.listdir(os.path.join(root, "labels"))))

        self.validate_dataset()

    def __getitem__(self, idx):
        original_idx = idx // self.augmentation_count
        img_path = os.path.join(self.root, "images", self.imgs[original_idx])
        label_path = os.path.join(self.root, "labels", self.labels[original_idx])
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        # Read the label file
        boxes = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, w, h = map(float, parts[1:])
                # Convert normalized coordinates to absolute coordinates.
                x_center *= width
                y_center *= height
                w *= width
                h *= height

                # Convert to (x_min, y_min, x_max, y_max) format.
                x_min = x_center - w / 2
                y_min = y_center - h / 2
                x_max = x_center + w / 2
                y_max = y_center + h / 2

                boxes.append([x_min, y_min, x_max, y_max, class_id])
        # Convert boxes to the format expected by the model.
        boxes = torch.tensor(boxes, dtype=torch.float32)
        target = {'boxes': boxes[:, :4], 'labels': boxes[:, 4].long()}

        if self.transforms is not None:
            """
            This was problematic, so be careful about this. All box classes must be transformed in the same
            manner as the input image. Can check that transformations are operating as expected by plotting
            before and after transforms are applied. AutoAugment does unwanted things.
            """
            img_array = np.array(img)
            bboxes = tv_tensors.BoundingBoxes(target['boxes'], format="XYXY", canvas_size=img_array.shape[:2])
            img, bboxes_out = self.transforms(img, bboxes)

            target['boxes'] = bboxes_out

        return img, target

    def __len__(self):
        return len(self.imgs) * self.augmentation_count

    def get_image_count(self):
        return len(self.imgs)

    def validate_dataset(self):
        """
        Minor dataset validation:
            1. Ensure there are the same amount of images as labels.
            2. Ensure the labels and images have the same naming convention.
        """
        # 1
        assert len(self.imgs) == len(self.labels), "Dataset input images and labels are of different length."
        # 2
        for i in range(len(self.imgs)):
            if self.imgs[i].split('.')[0] != self.labels[i].split('.')[0]:
                assert False, f"There is a mismatch between imgs and labels at {self.imgs[i]} and {self.labels[i]}."
