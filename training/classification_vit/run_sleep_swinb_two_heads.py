#!/usr/bin/env python3
# run_sleep_swinb_two_heads.py
#
# Single-stage training with Swin-B and dual-head architecture:
#   - Head 1: behavior classification (4 classes: 00_invalid, 01_standing, 02_sleeping_left, 03_sleeping_right)
#   - Head 2: image quality classification (2 classes: good, bad)
#   - Swin-B pretrained (timm)
#   - noise augmentations
#   - Class-weighted CrossEntropy for both heads
#
# Data structure:
#   train_root/
#     00_invalid/       -> behavior: invalid (0), quality: bad (1)
#     01_standing/      -> behavior: standing (1), quality: good (0)
#     02_sleeping_left/ -> behavior: sleeping_left (2), quality: good (0)
#     03_sleeping_right/-> behavior: sleeping_right (3), quality: good (0)
#
# Install:
#   pip install torch torchvision timm albumentations opencv-python scikit-learn tqdm numpy
#
# Example:
#   python run_sleep_swinb_two_heads.py \
#     --train_root /media/mu/zoo_vision/data/behaviour/sleep_v2 \
#     --eval_root  /media/mu/zoo_vision/data/behaviour/sleep_v4 \
#     --out_dir    runs_sleep/two_heads_swinb \
#     --model      swin_base_patch4_window7_224 \
#     --img_size   224 --batch_size 32 --epochs 25 \
#     --lr  3e-5 --weight_decay 0.05 \
#     --val_ratio 0.2 --seed 42

import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Sequence, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

import timm
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


# -----------------------------
# Labels
# -----------------------------
# Behavior classes (4 classes)
BEHAVIOR_CLASSES = {
    "00_invalid": 0,
    "01_standing": 1,
    "02_sleeping_left": 2,
    "03_sleeping_right": 3,
}
IDX_TO_BEHAVIOR = {v: k for k, v in BEHAVIOR_CLASSES.items()}
NUM_BEHAVIOR_CLASSES = len(BEHAVIOR_CLASSES)

# Quality classes (2 classes)
QUALITY_CLASSES = {
    "good": 0,
    "bad": 1,
}
IDX_TO_QUALITY = {v: k for k, v in QUALITY_CLASSES.items()}
NUM_QUALITY_CLASSES = len(QUALITY_CLASSES)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def sanitize_sampler_weights(w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Ensure weights are finite and strictly positive for torch.multinomial.
    """
    w = np.asarray(w, dtype=np.float64)

    # Replace NaN/Inf with 0
    bad = ~np.isfinite(w)
    if bad.any():
        w[bad] = 0.0

    # Clamp negatives to 0
    w[w < 0] = 0.0

    # If everything is zero, fallback to uniform
    s = float(w.sum())
    if s <= 0:
        w[:] = 1.0

    # Ensure strictly positive (avoid exact zeros if you want)
    w = w + eps
    return w

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()

# -----------------------------
# IO
# -----------------------------
def collect_items_dual_label(
    root: Union[str, Path, Sequence[Union[str, Path]]]
) -> List[Tuple[str, int, int]]:
    """
    Collect items with dual labels (behavior, quality).
    Expected structure:
      root/
        00_invalid/       -> behavior: invalid (0), quality: bad (1)
        01_standing/      -> behavior: standing (1), quality: good (0)
        02_sleeping_left/ -> behavior: sleeping_left (2), quality: good (0)
        03_sleeping_right/-> behavior: sleeping_right (3), quality: good (0)
    
    Returns:
      List of (path, behavior_label, quality_label)
    """
    if isinstance(root, (str, Path)):
        roots = [Path(root)]
    else:
        roots = [Path(r) for r in root]
    
    items: List[Tuple[str, int, int]] = []
    
    for root_p in roots:
        for behavior_name, behavior_label in BEHAVIOR_CLASSES.items():
            behavior_dir = root_p / behavior_name
            if not behavior_dir.is_dir():
                continue
                #raise FileNotFoundError(f"Missing behavior folder: {behavior_dir}")
            
            quality_label = QUALITY_CLASSES["bad"] if behavior_name == "00_invalid" else QUALITY_CLASSES["good"]
            
            for p in behavior_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    items.append((str(p), behavior_label, quality_label))
    
    if not items:
        raise RuntimeError(f"No images found under: {root}")
    return items


# -----------------------------
# Aug / Dataset
# -----------------------------
class HorizontalFlipSwapSleepLR:
    """Horizontal flip that swaps left/right behavior label (sleeping_left <-> sleeping_right)."""
    def __init__(self, p: float):
        self.p = float(p)

    def __call__(self, img_rgb: np.ndarray, behavior_label: int) -> Tuple[np.ndarray, int]:
        if self.p <= 0:
            return img_rgb, behavior_label
        if random.random() < self.p:
            img_rgb = cv2.flip(img_rgb, 1)
            # Swap left/right sleeping labels
            if behavior_label == BEHAVIOR_CLASSES["02_sleeping_left"]:
                behavior_label = BEHAVIOR_CLASSES["03_sleeping_right"]
            elif behavior_label == BEHAVIOR_CLASSES["03_sleeping_right"]:
                behavior_label = BEHAVIOR_CLASSES["02_sleeping_left"]
        return img_rgb, behavior_label


def build_train_aug(img_size: int, noise_level: float) -> A.Compose:
    # noise_level in [0,1] controls upper noise variance
    # var_limit in albumentations is in pixel-value space roughly; use a conservative range.
    var_hi = 60.0 * float(noise_level) + 10.0  # 10..70
    var_lo = 10.0
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA),
            _pad_if_needed(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=1.0,
            ),
            A.RandomResizedCrop(
                size=(img_size, img_size),
                scale=(0.80, 1.0),
                ratio=(0.90, 1.10),
                interpolation=cv2.INTER_AREA,
                p=1.0,
            ),
            _shift_scale_rotate(
                shift_limit=0.02,
                scale_limit=0.08,
                rotate_limit=5,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=0.6,
            ),

            # low-light robustness
            A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.20, p=0.7),
            A.RandomGamma(gamma_limit=(80, 120), p=0.4),

            # noise / blur
            A.GaussNoise(var_limit=(var_lo, var_hi), p=0.45),
            A.MotionBlur(blur_limit=3, p=0.15),
            A.GaussianBlur(blur_limit=3, p=0.15),
            _image_compression(quality_range=(40, 95), p=0.35),

            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def build_eval_aug(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA),
            _pad_if_needed(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=1.0,
            ),
            A.CenterCrop(height=img_size, width=img_size, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def _pad_if_needed(
    *,
    min_height: int,
    min_width: int,
    border_mode: int,
    fill: int,
    p: float,
) -> A.PadIfNeeded:
    try:
        return A.PadIfNeeded(
            min_height=min_height,
            min_width=min_width,
            border_mode=border_mode,
            fill=fill,
            p=p,
        )
    except TypeError:
        return A.PadIfNeeded(
            min_height=min_height,
            min_width=min_width,
            border_mode=border_mode,
            value=fill,
            p=p,
        )


def _shift_scale_rotate(
    *,
    shift_limit: float,
    scale_limit: float,
    rotate_limit: int,
    border_mode: int,
    fill: int,
    p: float,
) -> A.ShiftScaleRotate:
    try:
        return A.ShiftScaleRotate(
            shift_limit=shift_limit,
            scale_limit=scale_limit,
            rotate_limit=rotate_limit,
            border_mode=border_mode,
            fill=fill,
            p=p,
        )
    except TypeError:
        return A.ShiftScaleRotate(
            shift_limit=shift_limit,
            scale_limit=scale_limit,
            rotate_limit=rotate_limit,
            border_mode=border_mode,
            value=fill,
            p=p,
        )


def _image_compression(
    *,
    quality_range: Tuple[int, int],
    p: float,
) -> A.ImageCompression:
    try:
        return A.ImageCompression(quality_range=quality_range, p=p)
    except TypeError:
        return A.ImageCompression(
            quality_lower=quality_range[0],
            quality_upper=quality_range[1],
            p=p,
        )


class ImageDataset(Dataset):
    def __init__(
        self,
        items: List[Tuple[str, int, int]],  # (path, behavior_label, quality_label)
        img_size: int,
        train: bool,
        noise_level: float = 0.7,
        hflip_swap_p: float = 0.0,
    ):
        self.items = items
        self.train = bool(train)
        self.aug = build_train_aug(img_size, noise_level) if self.train else build_eval_aug(img_size)
        self.hflip_swap = HorizontalFlipSwapSleepLR(hflip_swap_p) if (self.train and hflip_swap_p > 0) else None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        path, behavior_label, quality_label = self.items[idx]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.hflip_swap is not None:
            img, behavior_label = self.hflip_swap(img, behavior_label)

        x = self.aug(image=img)["image"].float()
        return (
            x,
            torch.tensor(int(behavior_label), dtype=torch.long),
            torch.tensor(int(quality_label), dtype=torch.long),
            idx
        )


# -----------------------------
# Loss / Sampler
# -----------------------------
def make_weighted_sampler(behavior_labels: List[int], num_behavior_classes: int) -> WeightedRandomSampler:
    """Create weighted sampler based on behavior class distribution."""
    counts = np.bincount(np.array(behavior_labels), minlength=num_behavior_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    class_w = 1.0 / counts
    sample_w = torch.tensor([class_w[y] for y in behavior_labels], dtype=torch.double)
    return WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)


def make_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(np.array(labels), minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (num_classes * counts)  # inverse frequency normalized
    w = torch.tensor(w, dtype=torch.float32)
    return w


# -----------------------------
# Metrics
# -----------------------------
@torch.no_grad()
def eval_dual_head(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, object]:
    """Evaluate dual-head model on behavior and quality tasks."""
    model.eval()
    
    # Confusion matrices
    conf_behavior = torch.zeros((NUM_BEHAVIOR_CLASSES, NUM_BEHAVIOR_CLASSES), dtype=torch.long)
    conf_quality = torch.zeros((NUM_QUALITY_CLASSES, NUM_QUALITY_CLASSES), dtype=torch.long)
    
    total, correct_behavior, correct_quality = 0, 0, 0

    for x, y_behavior, y_quality, _ in loader:
        x = x.to(device, non_blocking=True)
        y_behavior = y_behavior.to(device, non_blocking=True)
        y_quality = y_quality.to(device, non_blocking=True)
        
        logits_behavior, logits_quality = model(x)
        pred_behavior = logits_behavior.argmax(1)
        pred_quality = logits_quality.argmax(1)
        
        total += y_behavior.numel()
        correct_behavior += (pred_behavior == y_behavior).sum().item()
        correct_quality += (pred_quality == y_quality).sum().item()
        
        for t, p in zip(y_behavior.view(-1), pred_behavior.view(-1)):
            conf_behavior[int(t), int(p)] += 1
        for t, p in zip(y_quality.view(-1), pred_quality.view(-1)):
            conf_quality[int(t), int(p)] += 1

    # Behavior metrics
    acc_behavior = correct_behavior / max(1, total)
    recalls_behavior, f1s_behavior = [], []
    for c in range(NUM_BEHAVIOR_CLASSES):
        tp = conf_behavior[c, c].item()
        fn = conf_behavior[c, :].sum().item() - tp
        fp = conf_behavior[:, c].sum().item() - tp
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        recalls_behavior.append(rec)
        f1s_behavior.append(f1)

    # Quality metrics
    acc_quality = correct_quality / max(1, total)
    recalls_quality, f1s_quality = [], []
    for c in range(NUM_QUALITY_CLASSES):
        tp = conf_quality[c, c].item()
        fn = conf_quality[c, :].sum().item() - tp
        fp = conf_quality[:, c].sum().item() - tp
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        recalls_quality.append(rec)
        f1s_quality.append(f1)

    return {
        "behavior": {
            "acc": float(acc_behavior),
            "bal_acc": float(np.mean(recalls_behavior)),
            "macro_f1": float(np.mean(f1s_behavior)),
            "confusion": conf_behavior.cpu().numpy(),
        },
        "quality": {
            "acc": float(acc_quality),
            "bal_acc": float(np.mean(recalls_quality)),
            "macro_f1": float(np.mean(f1s_quality)),
            "confusion": conf_quality.cpu().numpy(),
        },
    }


# -----------------------------
# Training
# -----------------------------
class DualHeadModel(nn.Module):
    """
    Dual-head model with shared backbone.
    - Head 1: behavior classification (4 classes)
    - Head 2: quality classification (2 classes)
    """
    def __init__(self, backbone_name: str, pretrained: bool = True, img_size: int | None = None):
        super().__init__()
        
        # Create backbone (without classifier head)
        kwargs = {}
        if img_size is not None:
            kwargs["img_size"] = img_size
        
        # Load pretrained model with dummy num_classes, then remove head
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # No classifier head
            **kwargs
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, img_size or 224, img_size or 224)
            feat_dim = self.backbone(dummy).shape[1]
        
        # Create two classifier heads
        self.head_behavior = nn.Linear(feat_dim, NUM_BEHAVIOR_CLASSES)
        self.head_quality = nn.Linear(feat_dim, NUM_QUALITY_CLASSES)
    
    def forward(self, x):
        features = self.backbone(x)
        logits_behavior = self.head_behavior(features)
        logits_quality = self.head_quality(features)
        return logits_behavior, logits_quality


def train_dual_head(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    behavior_weights: torch.Tensor,
    quality_weights: torch.Tensor,
    out_dir: str,
    amp: bool = True,
    behavior_loss_weight: float = 1.0,
    quality_loss_weight: float = 1.0,
    hard_mining: bool = False,
    hard_mining_power: float = 1.5,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "best_dual_head.pt")
    last_path = os.path.join(out_dir, "last_dual_head.pt")
    log_path = os.path.join(out_dir, "training_log.txt")

    # Open log file
    log_file = open(log_path, "w")
    log_file.write("Epoch,Train_Loss_Behavior,Train_Loss_Quality,Train_Loss_Total,")
    log_file.write("Val_Behavior_Acc,Val_Behavior_BalAcc,Val_Behavior_F1,")
    log_file.write("Val_Quality_Acc,Val_Quality_BalAcc,Val_Quality_F1,Avg_F1\n")
    log_file.flush()

    # Loss functions for both heads (use reduction="none" for hard mining)
    reduction = "none" if hard_mining else "mean"
    criterion_behavior = nn.CrossEntropyLoss(weight=behavior_weights.to(device), reduction=reduction)
    criterion_quality = nn.CrossEntropyLoss(weight=quality_weights.to(device), reduction=reduction)

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device == "cuda"))
    best_score = -1.0

    # Hard mining only works if sampler has "weights" and dataset returns idx
    do_hm = hard_mining and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "weights")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss_behavior, running_loss_quality, running_loss_total = 0.0, 0.0, 0.0
        num_batches = 0

        # Track per-sample losses for hard mining
        epoch_losses = None
        if do_hm:
            epoch_losses = np.zeros(len(train_loader.dataset), dtype=np.float32)
            epoch_counts = np.zeros(len(train_loader.dataset), dtype=np.int32)

        for batch in train_loader:
            x, y_behavior, y_quality, idxs = batch
            x = x.to(device, non_blocking=True)
            y_behavior = y_behavior.to(device, non_blocking=True)
            y_quality = y_quality.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(amp and device == "cuda")):
                logits_behavior, logits_quality = model(x)
                loss_behavior_vec = criterion_behavior(logits_behavior, y_behavior)
                loss_quality_vec = criterion_quality(logits_quality, y_quality)
                
                if hard_mining:
                    # Combine losses per sample
                    loss_vec = behavior_loss_weight * loss_behavior_vec + quality_loss_weight * loss_quality_vec
                    loss = loss_vec.mean()
                    loss_behavior = loss_behavior_vec.mean()
                    loss_quality = loss_quality_vec.mean()
                else:
                    loss_behavior = loss_behavior_vec
                    loss_quality = loss_quality_vec
                    loss = behavior_loss_weight * loss_behavior + quality_loss_weight * loss_quality

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss_behavior += float(loss_behavior.item())
            running_loss_quality += float(loss_quality.item())
            running_loss_total += float(loss.item())
            num_batches += 1

            # Collect per-sample loss for hard mining
            if do_hm:
                lv = loss_vec.detach().float().cpu().numpy()
                ii = idxs.detach().cpu().numpy()
                for k in range(len(ii)):
                    epoch_losses[ii[k]] += lv[k]
                    epoch_counts[ii[k]] += 1

        # Validation
        val_m = eval_dual_head(model, val_loader, device)
        
        # Combined score (average of both macro F1s)
        score = (val_m["behavior"]["macro_f1"] + val_m["quality"]["macro_f1"]) / 2.0

        ckpt = {
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "val_metrics": val_m,
        }
        torch.save(ckpt, last_path)

        avg_loss_behavior = running_loss_behavior / max(1, num_batches)
        avg_loss_quality = running_loss_quality / max(1, num_batches)
        avg_loss_total = running_loss_total / max(1, num_batches)

        print(f"\n[Epoch {epoch}/{epochs}]")
        print(f"Train Loss - Behavior: {avg_loss_behavior:.4f}, Quality: {avg_loss_quality:.4f}, Total: {avg_loss_total:.4f}")
        print(f"[Behavior VAL] acc={val_m['behavior']['acc']:.4f} bal_acc={val_m['behavior']['bal_acc']:.4f} macro_f1={val_m['behavior']['macro_f1']:.4f}")
        print(val_m["behavior"]["confusion"])
        print(f"[Quality VAL] acc={val_m['quality']['acc']:.4f} bal_acc={val_m['quality']['bal_acc']:.4f} macro_f1={val_m['quality']['macro_f1']:.4f}")
        print(val_m["quality"]["confusion"])

        # Write to log file
        log_file.write(f"{epoch},{avg_loss_behavior:.6f},{avg_loss_quality:.6f},{avg_loss_total:.6f},")
        log_file.write(f"{val_m['behavior']['acc']:.6f},{val_m['behavior']['bal_acc']:.6f},{val_m['behavior']['macro_f1']:.6f},")
        log_file.write(f"{val_m['quality']['acc']:.6f},{val_m['quality']['bal_acc']:.6f},{val_m['quality']['macro_f1']:.6f},{score:.6f}\n")
        log_file.flush()

        if score > best_score:
            best_score = score
            torch.save(ckpt, best_path)
            print(f"Saved best -> {best_path} (avg_macro_f1={best_score:.4f})")

        # Update sampler weights (hard mining)
        if do_hm:
            mask = epoch_counts > 0

            # If some samples weren't seen (can happen with weighted sampling), give them median loss
            if mask.any():
                epoch_losses[mask] = epoch_losses[mask] / np.maximum(epoch_counts[mask], 1)
                fill_value = float(np.median(epoch_losses[mask]))
            else:
                fill_value = 1.0

            epoch_losses[~mask] = fill_value

            # Replace NaN/Inf early
            epoch_losses = np.nan_to_num(epoch_losses, nan=fill_value, posinf=fill_value, neginf=fill_value)

            mn = float(epoch_losses.min())
            mx = float(epoch_losses.max())
            denom = (mx - mn) if (mx > mn) else 1.0
            norm = (epoch_losses - mn) / denom  # should be in [0,1]

            # Emphasize hard samples; keep non-negative
            new_w = (norm + 1e-3) ** float(hard_mining_power)

            # Critical: sanitize for multinomial
            new_w = sanitize_sampler_weights(new_w, eps=1e-6)

            # Update sampler weights in-place
            train_loader.sampler.weights = torch.tensor(new_w, dtype=torch.double)
            print(f"[Hard Mining] Updated sampler weights: min={new_w.min():.6g} max={new_w.max():.6g} mean={new_w.mean():.6g}")

    log_file.close()
    print(f"\nTraining log saved to: {log_path}")
    return best_path


def torch_load_trusted(path: str, map_location="cpu"):
    """
    Load a checkpoint that YOU created (trusted).
    We intentionally use weights_only=False because our checkpoint is a dict
    with numpy objects/metrics; weights_only=True will fail.
    """
    return torch.load(path, map_location=map_location, weights_only=False)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_root", required=True, help="Path or list of paths to training data root (quality/behavior folders)", nargs='+')
    ap.add_argument("--eval_root", required=True, help="Path to evaluation data root (quality/behavior folders)")
    ap.add_argument("--out_dir", required=True, help="Output directory for results")

    ap.add_argument("--model", default="swin_base_patch4_window7_224")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)

    # Training configs
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight_decay", type=float, default=0.05)

    # Aug
    ap.add_argument("--noise_level", type=float, default=0.7)  # 0..1
    ap.add_argument("--hflip_swap_p", type=float, default=0.0, help="Probability of horizontal flip with left/right swap for sleeping poses")

    # Loss weighting
    ap.add_argument("--behavior_loss_weight", type=float, default=1.0)
    ap.add_argument("--quality_loss_weight", type=float, default=1.0)

    # Hard mining
    ap.add_argument("--hard_mining", action="store_true", help="Enable hard example mining")
    ap.add_argument("--hard_mining_power", type=float, default=1.5, help="Power for emphasizing hard samples (>1 emphasizes harder)")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = (device == "cuda")

    # -----------------------------
    # Prepare data (dual labels)
    # -----------------------------
    train_items = collect_items_dual_label(args.train_root)
    val_items = collect_items_dual_label(args.eval_root)

    print(f"[Data] train={len(train_items)}  val={len(val_items)}")

    # Extract labels for class weighting
    train_behavior_labels = [b for _, b, _ in train_items]
    train_quality_labels = [q for _, _, q in train_items]

    # Create sampler based on behavior class distribution
    sampler = make_weighted_sampler(train_behavior_labels, num_behavior_classes=NUM_BEHAVIOR_CLASSES)
    
    # Create class weights
    behavior_weights = make_class_weights(train_behavior_labels, num_classes=NUM_BEHAVIOR_CLASSES)
    quality_weights = make_class_weights(train_quality_labels, num_classes=NUM_QUALITY_CLASSES)

    print(f"Behavior class weights: {behavior_weights}")
    print(f"Quality class weights: {quality_weights}")

    # Create datasets
    ds_train = ImageDataset(
        train_items,
        img_size=args.img_size,
        train=True,
        noise_level=args.noise_level,
        hflip_swap_p=args.hflip_swap_p,
    )

    ds_val = ImageDataset(
        val_items,
        img_size=args.img_size,
        train=False,
        hflip_swap_p=0.0,
    )

    # Create data loaders
    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # -----------------------------
    # Create dual-head model
    # -----------------------------
    model = DualHeadModel(
        backbone_name=args.model,
        pretrained=True,
        img_size=args.img_size
    ).to(device)

    print(f"Created dual-head model with backbone: {args.model}")

    # -----------------------------
    # Train
    # -----------------------------
    best_ckpt = train_dual_head(
        model=model,
        train_loader=dl_train,
        val_loader=dl_val,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        behavior_weights=behavior_weights,
        quality_weights=quality_weights,
        out_dir=args.out_dir,
        amp=amp,
        behavior_loss_weight=args.behavior_loss_weight,
        quality_loss_weight=args.quality_loss_weight,
        hard_mining=args.hard_mining,
        hard_mining_power=args.hard_mining_power,
    )

    # -----------------------------
    # Final evaluation
    # -----------------------------
    print("\n==== Final Evaluation ====")
    ckpt = torch_load_trusted(best_ckpt, map_location="cpu")
    model = DualHeadModel(
        backbone_name=args.model,
        pretrained=False,
        img_size=args.img_size
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)

    final_metrics = eval_dual_head(model, dl_val, device)
    
    print(f"\n[Behavior] acc={final_metrics['behavior']['acc']:.4f} "
          f"bal_acc={final_metrics['behavior']['bal_acc']:.4f} "
          f"macro_f1={final_metrics['behavior']['macro_f1']:.4f}")
    print("Behavior confusion matrix:")
    print(final_metrics["behavior"]["confusion"])
    
    print(f"\n[Quality] acc={final_metrics['quality']['acc']:.4f} "
          f"bal_acc={final_metrics['quality']['bal_acc']:.4f} "
          f"macro_f1={final_metrics['quality']['macro_f1']:.4f}")
    print("Quality confusion matrix:")
    print(final_metrics["quality"]["confusion"])

    # Write results
    out_txt = os.path.join(args.out_dir, "eval_summary.txt")
    with open(out_txt, "w") as f:
        f.write("=== Behavior Classification ===\n")
        f.write(f"acc={final_metrics['behavior']['acc']:.6f}\n")
        f.write(f"bal_acc={final_metrics['behavior']['bal_acc']:.6f}\n")
        f.write(f"macro_f1={final_metrics['behavior']['macro_f1']:.6f}\n")
        f.write("confusion:\n")
        f.write(np.array2string(final_metrics["behavior"]["confusion"]) + "\n\n")

        f.write("=== Quality Classification ===\n")
        f.write(f"acc={final_metrics['quality']['acc']:.6f}\n")
        f.write(f"bal_acc={final_metrics['quality']['bal_acc']:.6f}\n")
        f.write(f"macro_f1={final_metrics['quality']['macro_f1']:.6f}\n")
        f.write("confusion:\n")
        f.write(np.array2string(final_metrics["quality"]["confusion"]) + "\n")

    print(f"\nWrote: {out_txt}")
    print(f"Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
