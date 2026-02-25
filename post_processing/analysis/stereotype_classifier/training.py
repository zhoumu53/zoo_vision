import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

GT_CSV = Path("/media/mu/zoo_vision/data/stereotype/gt.csv")
IMAGE_DIR = Path("/media/mu/zoo_vision/data/stereotype/images")

DATE_PATTERN = re.compile(r"(\d{8})")
TRAIN_YEAR = "2025"
TEST_YEAR = "2026"


def parse_year_from_filename(filename: str) -> str:
    match = DATE_PATTERN.search(filename)
    if not match:
        raise ValueError(f"Could not parse date from filename: {filename}")
    return match.group(1)[:4]


def load_split_samples(
    gt_csv: Path, image_dir: Path, train_year: str, test_year: str
) -> Tuple[List[Tuple[Path, str]], List[Tuple[Path, str]]]:
    train_samples: List[Tuple[Path, str]] = []
    test_samples: List[Tuple[Path, str]] = []

    with gt_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"].strip()
            label = row["label"].strip()
            year = parse_year_from_filename(filename)
            image_path = image_dir / filename

            if not image_path.exists():
                continue

            if year == train_year:
                train_samples.append((image_path, label))
            elif year == test_year:
                test_samples.append((image_path, label))

    if not train_samples:
        raise RuntimeError(f"No training samples found for year={train_year}")
    if not test_samples:
        raise RuntimeError(f"No test samples found for year={test_year}")

    return train_samples, test_samples


class StereotypeDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Tuple[Path, str]],
        class_to_idx: Dict[str, int],
        transform: transforms.Compose,
    ) -> None:
        self.samples = list(samples)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        label_idx = self.class_to_idx[label]
        return image, label_idx


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
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return correct / max(1, targets.size(0))


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy_from_logits(logits, targets)

    return running_loss / len(dataloader), running_acc / len(dataloader)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)

        running_loss += loss.item()
        running_acc += accuracy_from_logits(logits, targets)

    return running_loss / len(dataloader), running_acc / len(dataloader)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train stereotype classifier with strict year split."
    )
    parser.add_argument("--gt_csv", type=Path, default=GT_CSV)
    parser.add_argument("--image_dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--train_year", type=str, default=TRAIN_YEAR)
    parser.add_argument("--test_year", type=str, default=TEST_YEAR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path(
            "/media/mu/zoo_vision/post_processing/analysis/stereotype_classifier/model.pt"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_samples, test_samples = load_split_samples(
        args.gt_csv, args.image_dir, args.train_year, args.test_year
    )

    class_names = sorted({label for _, label in train_samples + test_samples})
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    transform = build_transforms(image_size=args.image_size)
    train_dataset = StereotypeDataset(train_samples, class_to_idx, transform)
    test_dataset = StereotypeDataset(test_samples, class_to_idx, transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_test_acc = -1.0
    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.checkpoint_path.with_suffix(".json")

    print(f"Train samples ({args.train_year}): {len(train_dataset)}")
    print(f"Test samples ({args.test_year}): {len(test_dataset)}")
    print(f"Classes: {class_names}")
    print(f"Device: {device}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"| train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"| test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "idx_to_class": idx_to_class,
                    "image_size": args.image_size,
                    "train_year": args.train_year,
                    "test_year": args.test_year,
                },
                args.checkpoint_path,
            )
            with metadata_path.open("w") as f:
                json.dump(
                    {
                        "checkpoint_path": str(args.checkpoint_path),
                        "best_test_acc": best_test_acc,
                        "class_names": class_names,
                        "train_samples": len(train_dataset),
                        "test_samples": len(test_dataset),
                        "train_year": args.train_year,
                        "test_year": args.test_year,
                    },
                    f,
                    indent=2,
                )
            print(f"Saved best checkpoint to {args.checkpoint_path}")

    print(f"Best test accuracy: {best_test_acc:.4f}")


if __name__ == "__main__":
    main()
