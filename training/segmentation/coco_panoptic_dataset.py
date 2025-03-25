import torch
import json
import PIL
import numpy as np
from pathlib import Path


class CocoPanopticDataset(torch.utils.data.Dataset):
    """Image segmentation dataset."""

    def __init__(self, annotations_path: Path, processor, transform=None):
        self.root_path = annotations_path.parent
        with annotations_path.open() as f:
            annotations_json = json.load(f)

        image_from_id: dict[int, str] = {
            image["id"]: image["file_name"] for image in annotations_json["images"]
        }
        annotations = annotations_json["annotations"]

        self.annotations = annotations
        self.image_files = [
            image_from_id[annotation["image_id"]] for annotation in annotations
        ]
        self.segmentation_files = [
            annotation["file_name"] for annotation in annotations
        ]
        self.inst2class = [
            {np.int32(s["id"]): s["category_id"] for s in annotation["segments_info"]}
            for annotation in annotations
        ]

        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image = PIL.Image.open(self.root_path / self.image_files[idx]).convert("RGB")
        image = np.asarray(image)

        segmentation = PIL.Image.open(self.root_path / self.segmentation_files[idx])
        segmentation = segmentation.convert("RGB")
        segmentation = np.asarray(segmentation).astype(np.int32)

        instance_seg = (
            segmentation[:, :, 0]
            + 256 * segmentation[:, :, 1]
            + 256 * 256 * segmentation[:, :, 2]
        )

        class_from_instance = self.inst2class[idx]

        # apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image, mask=instance_seg)
            image, instance_seg = transformed["image"], transformed["mask"]

        # convert to C, H, W
        image = image.transpose(2, 0, 1)

        # instance_list = np.unique(instance_seg)
        # if instance_list.shape == (1, 1) and instance_list[0] == 0:
        #     # Some image does not have annotation (all ignored)
        #     inputs = self.processor([image], return_tensors="pt")
        #     inputs = {k: v.squeeze() for k, v in inputs.items()}
        #     inputs["class_labels"] = torch.tensor([0])
        #     inputs["mask_labels"] = torch.zeros(
        #         (0, inputs["pixel_values"].shape[-2], inputs["pixel_values"].shape[-1])
        #     )
        # else:
        inputs = self.processor(
            [image],
            [instance_seg],
            instance_id_to_semantic_id=class_from_instance,
            return_tensors="pt",
        )
        inputs = {
            k: v.squeeze() if isinstance(v, torch.Tensor) else v[0]
            for k, v in inputs.items()
        }

        return inputs
