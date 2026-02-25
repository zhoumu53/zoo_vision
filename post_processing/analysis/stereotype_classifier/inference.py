import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

ImageInput = Union[str, Path, Image.Image, np.ndarray, torch.Tensor]


@dataclass
class InferenceBundle:
    model: nn.Module
    transform: transforms.Compose
    idx_to_class: Dict[int, str]
    device: torch.device


def build_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_model_for_inference(
    checkpoint_path: Union[str, Path], device: str = None
) -> InferenceBundle:
    checkpoint_path = Path(checkpoint_path)
    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    checkpoint = torch.load(checkpoint_path, map_location=resolved_device, weights_only=True)
    idx_to_class = {int(k): v for k, v in checkpoint["idx_to_class"].items()}
    image_size = int(checkpoint.get("image_size", 224))

    model = build_model(num_classes=len(idx_to_class))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved_device)
    model.eval()

    return InferenceBundle(
        model=model,
        transform=build_transforms(image_size=image_size),
        idx_to_class=idx_to_class,
        device=resolved_device,
    )


def _to_pil_image(image: ImageInput, input_is_bgr: bool = True) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")

    if isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
            tensor = tensor.permute(1, 2, 0)
        array = tensor.numpy()
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return _to_pil_image(array, input_is_bgr=input_is_bgr)

    if isinstance(image, np.ndarray):
        array = image
        if array.ndim == 2:
            return Image.fromarray(array).convert("RGB")
        if array.ndim == 3 and array.shape[2] == 3:
            if input_is_bgr:
                array = array[:, :, ::-1]
            return Image.fromarray(array.astype(np.uint8)).convert("RGB")
        raise ValueError(f"Unsupported numpy image shape: {array.shape}")

    raise TypeError(f"Unsupported image input type: {type(image)}")


def preprocess_image(
    image: ImageInput, bundle: InferenceBundle, input_is_bgr: bool = True
) -> torch.Tensor:
    pil_image = _to_pil_image(image, input_is_bgr=input_is_bgr)
    tensor = bundle.transform(pil_image).unsqueeze(0).to(bundle.device)
    return tensor


@torch.inference_mode()
def predict_logits_from_image(
    image: ImageInput, bundle: InferenceBundle, input_is_bgr: bool = True
) -> torch.Tensor:
    input_tensor = preprocess_image(image, bundle, input_is_bgr=input_is_bgr)
    return bundle.model(input_tensor)


@torch.inference_mode()
def predict_topk_from_image(
    image: ImageInput,
    bundle: InferenceBundle,
    k: int = 3,
    input_is_bgr: bool = True,
) -> List[Tuple[str, float]]:
    logits = predict_logits_from_image(image, bundle, input_is_bgr=input_is_bgr)
    probs = torch.softmax(logits, dim=1)
    top_probs, top_indices = torch.topk(probs, k=min(k, probs.shape[1]), dim=1)

    results: List[Tuple[str, float]] = []
    for p, idx in zip(top_probs[0].tolist(), top_indices[0].tolist()):
        label = bundle.idx_to_class[int(idx)]
        results.append((label, float(p)))
    return results


@torch.inference_mode()
def predict_label_from_image(
    image: ImageInput, bundle: InferenceBundle, input_is_bgr: bool = True
) -> str:
    logits = predict_logits_from_image(image, bundle, input_is_bgr=input_is_bgr)
    pred_idx = int(torch.argmax(logits, dim=1).item())
    return bundle.idx_to_class[pred_idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference for stereotype classifier")
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path(
            "/media/mu/zoo_vision/post_processing/analysis/stereotype_classifier/model.pt"
        ),
    )
    parser.add_argument("--image_path", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument(
        "--input_is_bgr",
        action="store_true",
        help="Set this if image arrays are BGR (OpenCV style).",
    )
    args = parser.parse_args()

    bundle = load_model_for_inference(args.checkpoint_path, device=args.device)
    pred_label = predict_label_from_image(
        args.image_path, bundle, input_is_bgr=args.input_is_bgr
    )
    topk = predict_topk_from_image(
        args.image_path, bundle, k=args.topk, input_is_bgr=args.input_is_bgr
    )

    print(f"Prediction: {pred_label}")
    print("Top-k:")
    for label, prob in topk:
        print(f"  {label}: {prob:.4f}")


if __name__ == "__main__":
    main()
