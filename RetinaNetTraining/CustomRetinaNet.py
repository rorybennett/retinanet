"""
Custom RetinaNet (retinanet_resnet50_fpn_v2) for training prostate and bladder detection on abdominal ultrasound scans
of the prostate.

Makes use of retinanet_resnet50_fpn_v2, with a custom forward() pass that can return losses if required in
validation mode. Normally validation does not have ground truths (targets), but for this model it is assumed that
cross-fold validation is done, so the validation dataset has ground truths, and therefore losses can be calculated
and returned.
"""

from collections import OrderedDict
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.transforms import v2


def check_degen_boxes(targets):
    """
    Checks for degenerate boxes within the targets dictionary. If degenerate boxes are found, an assertion error
    is raised.

    :param targets: Dictionary of target boxes.
    """
    for target_idx, target in enumerate(targets):
        boxes = target["boxes"]
        degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
        if degenerate_boxes.any():
            # print the first degenerate box
            bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
            degen_bb: List[float] = boxes[bb_idx].tolist()
            torch._assert(
                False,
                "All bounding boxes should have positive height and width."
                f" Found invalid box {degen_bb} for target at index {target_idx}.",
            )


def get_training_transforms(image_size):
    return v2.Compose([
        v2.Resize((image_size, image_size)),
        v2.RandomAffine(degrees=30, shear=15, translate=(0.1, 0.1), scale=(0.6, 1.2)),
        v2.RandomHorizontalFlip(p=0.2),
        v2.RandomErasing(0.5, scale=(0.02, 0.08)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])


def get_validation_transforms(image_size):
    return v2.Compose([
        v2.Resize((image_size, image_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])


class CustomRetinaNet:
    def __init__(self, weights=None):
        self.model = retinanet_resnet50_fpn_v2(weights=weights)

    def forward(self, images, targets=None):
        # If model.train() call standard training function.
        if self.model.training:
            return self.model(images, targets)
        else:
            # Set the model to training mode temporarily to get losses
            with torch.no_grad():
                self.model.train()
                losses = self.model.forward(images, targets)
                # Set the model back to evaluation mode
                self.model.eval()
                detections = self.model.forward(images)
            return losses, detections
