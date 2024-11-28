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
        v2.RandomAffine(degrees=30, shear=15, translate=(0.1, 0.1), scale=(0.3, 1)),
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

    def forward(self, images, targets=None, return_losses=False):
        # If model.train() call standard training function.
        if self.model.training:
            return self.model(images, targets)
        ################################################################################################################
        # This section is for validation (must do what the original model does, plus allow for returning losses).
        ################################################################################################################
        # Get the original image sizes
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))
        # Transform the input.
        images, targets = self.model.transform(images, targets)
        # Check for degenerate boxes.
        if targets is not None:
            check_degen_boxes(targets)
        # Get the features from the backbone.
        features = self.model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        features = list(features.values())

        # Compute the retinanet heads outputs using the features.
        head_outputs = self.model.head(features)

        # Create the set of anchors.
        anchors = self.model.anchor_generator(images, features)

        # Compute the losses if requested.
        if return_losses and targets is not None:
            losses = self.model.compute_loss(targets, head_outputs, anchors)
        else:
            losses = {}

        # Recover level sizes.
        num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
        HW = 0
        for v in num_anchors_per_level:
            HW += v
        HWA = head_outputs["cls_logits"].size(1)
        A = HWA // HW
        num_anchors_per_level = [hw * A for hw in num_anchors_per_level]

        # Split outputs per level.
        split_head_outputs: Dict[str, List[Tensor]] = {}
        for k in head_outputs:
            split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
        split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

        # compute the detections
        detections = self.model.postprocess_detections(split_head_outputs, split_anchors, images.image_sizes)
        detections = self.model.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        return losses, detections
