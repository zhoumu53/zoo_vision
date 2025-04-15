import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class Sam2Processor:
    def __init__(self) -> None:
        if torch.cuda.is_available():
            self.device_ = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device_ = torch.device("mps")
        else:
            self.device_ = torch.device("cpu")

        print(f"Sam2 device: {self.device_.type}")

        if self.device_.type == "cuda":
            # use bfloat16 for the entire notebook
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif self.device_.type == "mps":
            print(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )

        print("Using sam2 device type: " + self.device_.type)

        sam2_checkpoint = "../models/sam2/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device_)
        self.predictor_ = SAM2ImagePredictor(sam2_model)

    def process_click(
        self, image: np.ndarray, positive: np.ndarray, negative: np.ndarray
    ) -> np.ndarray:
        self.predictor_.set_image(image)

        point_coords = np.concatenate([positive, negative])
        point_labels = np.zeros((positive.shape[0] + negative.shape[0]))
        point_labels[0 : positive.shape[0]] = 1

        masks, scores, _ = self.predictor_.predict(
            point_coords,
            point_labels,
            multimask_output=True,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        return masks[0]
