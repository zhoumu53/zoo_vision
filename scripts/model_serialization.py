from pathlib import Path
import torch
import torchvision as tv

from project_root import PROJECT_ROOT


class ModelWithTransforms(torch.nn.Module):
    def __init__(self, model, transforms):
        super().__init__()
        self.model = model
        self.transforms = transforms

    def forward(self, x):
        xn = self.transforms(x)
        y = self.model.forward(xn)
        return y


def load_model(path: Path):
    if path.suffix == ".ptc":
        model = torch.jit.load(path)
    else:
        extra_transforms = None

        print("Loading empty model...")
        if path.name.startswith("maskrcnn_c2_"):
            model = tv.models.detection.maskrcnn_resnet50_fpn_v2(
                weights=None,
                weights_backbone=None,
                num_classes=2,
            )
        elif path.name.startswith("dense121_c5_"):
            model = tv.models.densenet121(
                num_classes=5,
            )
            extra_transforms = tv.models.DenseNet121_Weights.IMAGENET1K_V1.transforms(
                antialias=True
            )
        else:
            raise RuntimeError("Unknown model")

        print("Loading weights from disk...")
        checkpoint = torch.load(PROJECT_ROOT / path, weights_only=False)

        print("Restoring weights...")
        model.load_state_dict(checkpoint["model"])

        if extra_transforms is not None:
            model = model = ModelWithTransforms(model, extra_transforms)

    return model
