from project_root import PROJECT_ROOT
from zoo_vision.training.identity.model import get_model

from pathlib import Path
import torch
import torchvision as tv


class ModelWithTransforms(torch.nn.Module):
    def __init__(self, model, transforms):
        super().__init__()
        self.model = model
        self.transforms = transforms

    def forward(self, x: torch.Tensor):
        xn = self.transforms(x)
        y = self.model.forward(xn)
        return y


class StateModelWithTransforms(torch.nn.Module):
    def __init__(self, model, transforms):
        super().__init__()
        self.model = model
        self.transforms = transforms

    def forward(self, x: torch.Tensor, state0: torch.Tensor | None = None):
        xn = self.transforms(x)
        y = self.model.forward(x=xn, gru_state0=state0)
        return y


def load_model(path: Path):
    if path.suffix == ".ptc":
        model = torch.jit.load(path)
    else:
        extra_transforms = None
        state_in_forward = False

        print("Loading weights from disk...")
        checkpoint = torch.load(PROJECT_ROOT / path, weights_only=False)
        model_name = (
            checkpoint["args"].model
            if hasattr(checkpoint["args"], "model")
            else checkpoint["args"]["model"]
        )

        print("Loading empty model...")
        if model_name == "maskrcnn_resnet50_fpn_v2":
            num_classes = checkpoint["model"]["rpn.head.cls_logits.bias"].shape[0]
            model = tv.models.detection.maskrcnn_resnet50_fpn_v2(
                weights=None,
                weights_backbone=None,
                num_classes=num_classes,
            )
        elif model_name == "densenet121":
            num_classes = checkpoint["model"]["classifier.weight"].shape[0]
            model = tv.models.densenet121(
                num_classes=num_classes,
            )
            extra_transforms = tv.models.DenseNet121_Weights.IMAGENET1K_V1.transforms(
                antialias=True
            )
        elif model_name == "zoo_id_gru":
            num_classes = checkpoint["model"]["classifier.weight"].shape[0]
            model = get_model("zoo_id_gru", num_classes=num_classes)
            model.gru.flatten_parameters()
            state_in_forward = True

            extra_transforms = tv.models.DenseNet121_Weights.IMAGENET1K_V1.transforms(
                antialias=True
            )
        else:
            raise RuntimeError("Unknown model")

        print("Restoring weights...")
        model.load_state_dict(checkpoint["model"])

        if extra_transforms is not None:
            if state_in_forward:
                model = StateModelWithTransforms(model, extra_transforms)
            else:
                model = ModelWithTransforms(model, extra_transforms)

    return model
