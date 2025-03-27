from project_root import PROJECT_ROOT
import transformers

from pathlib import Path
import torch
from scripts.datasets.segmentation_utils import bbox_xyxy_from_mask_torch


class Mask2FormerAdapter(torch.nn.Module):
    def __init__(self, mask2former: transformers.Mask2FormerForUniversalSegmentation):
        super().__init__()
        self.mask2former = mask2former

    def forward(self, x: torch.Tensor):
        y = self.mask2former(x)
        masks_queries_logits = y["masks_queries_logits"][0]
        class_queries_logits = y["class_queries_logits"][0]

        scores = torch.nn.functional.softmax(class_queries_logits, dim=-1)
        scores = scores[:, :-1].flatten(0, 1)

        masks_u8 = (masks_queries_logits > 0).to(torch.uint8)
        bboxes = bbox_xyxy_from_mask_torch(masks_u8)

        return {
            "scores": scores,
            "masks": masks_u8,
            "boxes": bboxes,
            "labels": torch.zeros((bboxes.shape[0],), dtype=torch.uint8),
        }
