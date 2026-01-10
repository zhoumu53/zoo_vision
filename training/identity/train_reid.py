"""
Re-Identification Training Script for Elephant Identity

This script trains a ReID model using metric learning (triplet loss) instead of
classification loss. The model learns to embed elephant images into a feature
space where images of the same individual are close together and images of
different individuals are far apart.

Usage:
    python train_reid.py --data-path /path/to/data --model resnet50 --output-dir ./runs/reid_exp1
"""

import argparse
import datetime
import os
import random
import time
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

import utils


class RandomErasing:
    """Random Erasing augmentation from https://arxiv.org/abs/1708.04896"""

    def __init__(
        self,
        probability=0.5,
        sl=0.02,
        sh=0.4,
        r1=0.3,
        mean=(0.4914, 0.4822, 0.4465),
        mode="pixel",
        device="cpu",
    ):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.mode = mode
        self.device = device

    def __call__(self, img):
        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if self.mode == "pixel":
                    img[0, x1 : x1 + h, y1 : y1 + w] = self.mean[0]
                    img[1, x1 : x1 + h, y1 : y1 + w] = self.mean[1]
                    img[2, x1 : x1 + h, y1 : y1 + w] = self.mean[2]
                else:
                    img[:, x1 : x1 + h, y1 : y1 + w] = torch.from_numpy(
                        np.random.rand(3, h, w)
                    )
                return img

        return img


def get_transforms(
    img_size=(224, 224),
    pixel_mean=(0.485, 0.456, 0.406),
    pixel_std=(0.229, 0.224, 0.225),
    is_train=True,
    random_flip_prob=0.5,
    random_erase_prob=0.5,
    padding=10,
):
    """
    Get data transforms for ReID training.

    Args:
        img_size: Target image size (height, width)
        pixel_mean: Normalization mean
        pixel_std: Normalization std
        is_train: Whether training or validation
        random_flip_prob: Probability of random horizontal flip
        random_erase_prob: Probability of random erasing
        padding: Padding for random crop
    """
    if is_train:
        transforms = T.Compose(
            [
                T.Resize(img_size),
                T.RandomHorizontalFlip(p=random_flip_prob),
                T.Pad(padding),
                T.RandomCrop(img_size),
                T.ToTensor(),
                T.Normalize(mean=pixel_mean, std=pixel_std),
                RandomErasing(
                    probability=random_erase_prob, mode="pixel", device="cpu"
                ),
            ]
        )
    else:
        transforms = T.Compose(
            [
                T.Resize(img_size),
                T.ToTensor(),
                T.Normalize(mean=pixel_mean, std=pixel_std),
            ]
        )

    return transforms


class TripletLoss(nn.Module):
    """
    Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    """

    def __init__(self, margin=0.3, distance="euclidean"):
        """
        Args:
            margin: Margin for triplet loss
            distance: 'euclidean' or 'cosine'
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance = distance
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: Feature embeddings, shape [batch_size, embedding_dim]
            labels: Ground truth labels, shape [batch_size]

        Returns:
            Triplet loss value
        """
        if self.distance == "euclidean":
            # Compute pairwise distance matrix
            dist_mat = self._euclidean_dist(embeddings, embeddings)
        elif self.distance == "cosine":
            dist_mat = self._cosine_dist(embeddings, embeddings)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance}")

        # For each anchor, find the hardest positive and negative
        dist_ap, dist_an = self._hard_example_mining(dist_mat, labels)

        # Compute ranking loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss, dist_ap.mean(), dist_an.mean()

    def _euclidean_dist(self, x, y):
        """
        Compute euclidean distance between two tensors.

        Args:
            x: pytorch Variable, with shape [m, d]
            y: pytorch Variable, with shape [n, d]
        Returns:
            dist: pytorch Variable, with shape [m, n]
        """
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(x, y.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

    def _cosine_dist(self, x, y):
        """Compute cosine distance"""
        x_norm = F.normalize(x, p=2, dim=1)
        y_norm = F.normalize(y, p=2, dim=1)
        dist = 1 - torch.mm(x_norm, y_norm.t())
        return dist

    def _hard_example_mining(self, dist_mat, labels):
        """
        For each anchor, find the hardest positive and negative sample.

        Args:
            dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
            labels: pytorch LongTensor, with shape [N]

        Returns:
            dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
            dist_an: pytorch Variable, distance(anchor, negative); shape [N]
        """
        assert len(dist_mat.size()) == 2
        assert dist_mat.size(0) == dist_mat.size(1)
        N = dist_mat.size(0)

        # shape [N, N]
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N]
        dist_ap, _ = torch.max(
            dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True
        )
        # `dist_an` means distance(anchor, negative)
        # both `dist_an` and `relative_n_inds` with shape [N]
        dist_an, _ = torch.min(
            dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True
        )

        dist_ap = dist_ap.squeeze(1)
        dist_an = dist_an.squeeze(1)

        return dist_ap, dist_an


class CenterLoss(nn.Module):
    """
    Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    """

    def __init__(self, num_classes=10, feat_dim=2048, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(
                torch.randn(self.num_classes, self.feat_dim).cuda()
            )
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = (
            torch.pow(x, 2)
            .sum(dim=1, keepdim=True)
            .expand(batch_size, self.num_classes)
            + torch.pow(self.centers, 2)
            .sum(dim=1, keepdim=True)
            .expand(self.num_classes, batch_size)
            .t()
        )
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e12).sum() / batch_size

        return loss


class ReIDModel(nn.Module):
    """
    ReID model with embedding layer.
    """

    def __init__(
        self,
        backbone_name="resnet50",
        embedding_dim=512,
        num_classes=5,
        pretrained=True,
    ):
        """
        Args:
            backbone_name: Name of the backbone (resnet50, densenet121, etc.)
            embedding_dim: Dimension of the embedding space
            num_classes: Number of identities (for auxiliary classification loss)
            pretrained: Whether to use pretrained weights
        """
        super(ReIDModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # Load backbone
        if backbone_name == "resnet50":
            backbone = torchvision.models.resnet50(pretrained=pretrained)
            self.backbone_dim = backbone.fc.in_features
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        elif backbone_name == "resnet101":
            backbone = torchvision.models.resnet101(pretrained=pretrained)
            self.backbone_dim = backbone.fc.in_features
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        elif backbone_name == "densenet121":
            backbone = torchvision.models.densenet121(pretrained=pretrained)
            self.backbone_dim = backbone.classifier.in_features
            self.backbone = backbone.features
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.backbone_name = backbone_name

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Batch normalization neck
        self.bottleneck = nn.BatchNorm1d(self.backbone_dim)
        self.bottleneck.bias.requires_grad_(False)

        # Embedding layer
        self.embedding = nn.Linear(self.backbone_dim, embedding_dim, bias=False)
        nn.init.kaiming_normal_(self.embedding.weight, mode="fan_out")

        # Classification head (auxiliary task)
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, x, return_features=False):
        """
        Args:
            x: Input images, shape [batch_size, 3, H, W]
            return_features: Whether to return intermediate features

        Returns:
            embeddings: L2-normalized embeddings for metric learning
            logits: Classification logits (if training)
        """
        # Extract features
        features = self.backbone(x)

        # Global pooling
        if self.backbone_name.startswith("densenet"):
            features = F.relu(features, inplace=True)
        features = self.gap(features)
        features = features.view(features.size(0), -1)

        # Bottleneck
        bn_features = self.bottleneck(features)

        # Embedding
        embeddings = self.embedding(bn_features)

        # L2 normalize embeddings
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)

        if self.training:
            # Classification logits
            logits = self.classifier(embeddings_norm)

            if return_features:
                return embeddings_norm, logits, features
            return embeddings_norm, logits
        else:
            if return_features:
                return embeddings_norm, features
            return embeddings_norm


def train_one_epoch(
    model,
    triplet_criterion,
    ce_criterion,
    optimizer,
    data_loader,
    device,
    epoch,
    args,
    tb_writer=None,
):
    """Train for one epoch."""
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("triplet_loss", utils.SmoothedValue(fmt="{value:.4f}"))
    metric_logger.add_meter("ce_loss", utils.SmoothedValue(fmt="{value:.4f}"))
    metric_logger.add_meter("dist_ap", utils.SmoothedValue(fmt="{value:.4f}"))
    metric_logger.add_meter("dist_an", utils.SmoothedValue(fmt="{value:.4f}"))

    header = f"Epoch: [{epoch}]"

    for batch_idx, (images, labels) in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        embeddings, logits = model(images)

        # Compute losses
        triplet_loss, dist_ap, dist_an = triplet_criterion(embeddings, labels)
        ce_loss = ce_criterion(logits, labels)

        # Combined loss
        loss = triplet_loss * args.triplet_weight + ce_loss * args.ce_weight

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        batch_size = images.size(0)
        metric_logger.update(triplet_loss=triplet_loss.item())
        metric_logger.update(ce_loss=ce_loss.item())
        metric_logger.update(dist_ap=dist_ap.item())
        metric_logger.update(dist_an=dist_an.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # Accuracy
        acc1 = (logits.argmax(dim=1) == labels).float().mean() * 100
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)

        # TensorBoard logging
        if tb_writer and batch_idx % args.print_freq == 0:
            global_step = epoch * len(data_loader) + batch_idx
            tb_writer.add_scalar("Train/triplet_loss", triplet_loss.item(), global_step)
            tb_writer.add_scalar("Train/ce_loss", ce_loss.item(), global_step)
            tb_writer.add_scalar("Train/total_loss", loss.item(), global_step)
            tb_writer.add_scalar("Train/dist_ap", dist_ap.item(), global_step)
            tb_writer.add_scalar("Train/dist_an", dist_an.item(), global_step)
            tb_writer.add_scalar("Train/acc1", acc1.item(), global_step)
            tb_writer.add_scalar(
                "Train/lr", optimizer.param_groups[0]["lr"], global_step
            )

    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print(f"Averaged stats: {metric_logger}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, tb_writer=None):
    """
    Evaluate model using k-NN accuracy in embedding space.
    """
    model.eval()

    all_embeddings = []
    all_labels = []

    print("Extracting features...")
    for images, labels in data_loader:
        images = images.to(device)
        embeddings = model(images)

        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Compute pairwise distances
    print("Computing distances...")
    dist_mat = torch.cdist(all_embeddings, all_embeddings, p=2)

    # Compute Rank-1, Rank-5 accuracy
    print("Computing rank accuracy...")
    ranks = [1, 5, 10]
    rank_accs = {}

    num_samples = dist_mat.size(0)
    matches = all_labels.expand(num_samples, num_samples).eq(
        all_labels.expand(num_samples, num_samples).t()
    )

    for rank in ranks:
        correct = 0
        for i in range(num_samples):
            # Get distances for this query
            dist = dist_mat[i]
            # Set distance to self to infinity
            dist[i] = float("inf")
            # Get top-k indices
            _, indices = torch.topk(dist, rank, largest=False)
            # Check if any of top-k match the query label
            if matches[i, indices].any():
                correct += 1

        rank_acc = correct / num_samples * 100
        rank_accs[f"Rank-{rank}"] = rank_acc
        print(f"Rank-{rank} Accuracy: {rank_acc:.2f}%")

    if tb_writer:
        for rank_name, acc in rank_accs.items():
            tb_writer.add_scalar(f"Val/{rank_name}", acc, epoch)

    return rank_accs


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="ReID Training for Elephant Identity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data parameters
    parser.add_argument(
        "--data-path",
        required=True,
        type=str,
        help="Path to dataset directory with train/ and val/ subdirectories",
    )
    parser.add_argument(
        "--workers", default=4, type=int, help="Number of data loading workers"
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="resnet50",
        choices=["resnet50", "resnet101", "densenet121"],
        help="Backbone model architecture",
    )
    parser.add_argument(
        "--embedding-dim", default=512, type=int, help="Dimension of embedding space"
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use pretrained backbone",
    )

    # Training parameters
    parser.add_argument("--batch-size", default=32, type=int, help="Batch size per GPU")
    parser.add_argument(
        "--epochs", default=120, type=int, help="Number of total epochs to run"
    )
    parser.add_argument(
        "--lr", default=0.00035, type=float, help="Initial learning rate"
    )
    parser.add_argument(
        "--weight-decay", default=0.0005, type=float, help="Weight decay"
    )
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum for SGD")

    # Loss parameters
    parser.add_argument(
        "--triplet-margin", default=0.3, type=float, help="Margin for triplet loss"
    )
    parser.add_argument(
        "--triplet-weight", default=1.0, type=float, help="Weight for triplet loss"
    )
    parser.add_argument(
        "--ce-weight", default=1.0, type=float, help="Weight for cross-entropy loss"
    )
    parser.add_argument(
        "--distance-metric",
        default="euclidean",
        choices=["euclidean", "cosine"],
        help="Distance metric for triplet loss",
    )

    # Data augmentation
    parser.add_argument("--img-size", default=224, type=int, help="Input image size")
    parser.add_argument(
        "--random-flip-prob",
        default=0.5,
        type=float,
        help="Probability of random horizontal flip",
    )
    parser.add_argument(
        "--random-erase-prob",
        default=0.5,
        type=float,
        help="Probability of random erasing",
    )
    parser.add_argument(
        "--padding", default=10, type=int, help="Padding for random crop"
    )

    # Scheduler parameters
    parser.add_argument(
        "--lr-scheduler",
        default="step",
        choices=["step", "cosine"],
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--lr-step-size",
        default=40,
        type=int,
        help="Period of learning rate decay for StepLR",
    )
    parser.add_argument(
        "--lr-gamma",
        default=0.1,
        type=float,
        help="Multiplicative factor of learning rate decay",
    )
    parser.add_argument(
        "--warmup-epochs", default=10, type=int, help="Number of warmup epochs"
    )

    # Output parameters
    parser.add_argument(
        "--output-dir", default="./runs/reid", type=str, help="Path to save outputs"
    )
    parser.add_argument("--print-freq", default=10, type=int, help="Print frequency")
    parser.add_argument(
        "--save-freq", default=10, type=int, help="Save checkpoint every N epochs"
    )

    # Resume training
    parser.add_argument(
        "--resume", default="", type=str, help="Path to checkpoint to resume from"
    )
    parser.add_argument("--start-epoch", default=0, type=int, help="Start epoch")

    # Device
    parser.add_argument(
        "--device", default="cuda", type=str, help="Device to use (cuda or cpu)"
    )

    # Distributed training
    parser.add_argument(
        "--world-size", default=1, type=int, help="Number of distributed processes"
    )
    parser.add_argument(
        "--dist-url", default="env://", type=str, help="URL for distributed training"
    )

    return parser


def main(args):
    # Setup
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Create output directory
    if args.output_dir:
        utils.mkdir(args.output_dir)

    # Setup TensorBoard
    tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))

    # Create transforms
    train_transform = get_transforms(
        img_size=(args.img_size, args.img_size),
        is_train=True,
        random_flip_prob=args.random_flip_prob,
        random_erase_prob=args.random_erase_prob,
        padding=args.padding,
    )

    val_transform = get_transforms(
        img_size=(args.img_size, args.img_size),
        is_train=False,
    )

    # Load datasets
    print("Loading data...")
    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")

    train_dataset = ImageFolder(train_dir, transform=train_transform)
    val_dataset = ImageFolder(val_dir, transform=val_transform)

    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {train_dataset.classes}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create data loaders
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False
        )
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )

    # Create model
    print(f"Creating model: {args.model}")
    model = ReIDModel(
        backbone_name=args.model,
        embedding_dim=args.embedding_dim,
        num_classes=num_classes,
        pretrained=args.pretrained,
    )
    model.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # Create loss functions
    triplet_criterion = TripletLoss(
        margin=args.triplet_margin, distance=args.distance_metric
    )
    ce_criterion = nn.CrossEntropyLoss()

    # Create optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # Create learning rate scheduler
    if args.lr_scheduler == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma,
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs - args.warmup_epochs,
        )

    # Warmup scheduler
    if args.warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=args.warmup_epochs,
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, lr_scheduler],
            milestones=[args.warmup_epochs],
        )

    # Resume from checkpoint
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1

    # Training loop
    print("Start training...")
    start_time = time.time()
    best_rank1 = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # Train
        train_stats = train_one_epoch(
            model,
            triplet_criterion,
            ce_criterion,
            optimizer,
            train_loader,
            device,
            epoch,
            args,
            tb_writer,
        )

        lr_scheduler.step()

        # Evaluate
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            val_stats = evaluate(model, val_loader, device, epoch, tb_writer)
            rank1_acc = val_stats["Rank-1"]

            # Save best model
            if rank1_acc > best_rank1:
                best_rank1 = rank1_acc
                if args.output_dir:
                    checkpoint = {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": vars(args),
                        "rank1_acc": rank1_acc,
                    }
                    utils.save_on_master(
                        checkpoint,
                        os.path.join(args.output_dir, "model_best.pth"),
                    )
                    print(f"Saved best model with Rank-1: {rank1_acc:.2f}%")

        # Save checkpoint
        if args.output_dir and (epoch + 1) % args.save_freq == 0:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": vars(args),
            }
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth"),
            )
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, "checkpoint_latest.pth"),
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time: {total_time_str}")
    print(f"Best Rank-1 Accuracy: {best_rank1:.2f}%")

    tb_writer.close()


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
