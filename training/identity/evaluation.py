"""
Evaluation script for elephant identity ReID models.

This script evaluates a trained identity model on the test/validation dataset
and provides comprehensive metrics including:
- Top-1 and Top-5 accuracy
- Per-class accuracy and confusion matrix
- Precision, Recall, F1-score per class
- Model inference speed
- Feature embedding visualization (optional)

Usage:
    # Evaluate a single image model
    python evaluation.py --data-path /path/to/data --model densenet121 --checkpoint /path/to/model.pth

    # Evaluate a GRU sequence model
    python evaluation.py --data-path /path/to/data --model zoo_id_gru --checkpoint /path/to/model.pth --sequence_length 10

    # Generate confusion matrix and save results
    python evaluation.py --data-path /home/dherrera/data/elephants/identity/dataset/certainty/val --checkpoint ../../models/identity/vit/v4/config.ptc --save-results --output-dir ./results
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
)
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
from model import get_model
from elephant_identity_dataset import ElephantIdentityDataset
from identity_sequence_dataset import IdentitySequenceDataset
from transforms import get_mixup_cutmix
import presets
import utils


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    use_sequences: bool = False,
) -> dict:
    """
    Evaluate model on dataset and return metrics.

    Args:
        model: The model to evaluate
        data_loader: DataLoader for test/validation data
        device: Device to run evaluation on
        class_names: List of class names
        use_sequences: Whether the model processes sequences

    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()

    all_predictions = []
    all_targets = []
    all_probabilities = []
    inference_times = []

    num_classes = len(class_names)

    print("Starting evaluation...")
    with torch.inference_mode():
        for batch_idx, (images, targets) in enumerate(data_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            batch_size = images.shape[0]

            # Time inference
            start_time = time.time()

            if use_sequences:
                # Process sequences: images.shape = [b, t, c, h, w]
                time_count = images.shape[1]
                prediction = torch.zeros(
                    [batch_size, time_count, num_classes],
                    dtype=torch.float32,
                    device=device,
                )
                hidden_state = None

                for t in range(time_count):
                    image_t = images[:, [t], ...]
                    output_t = model(image_t, hidden_state)
                    prediction_t = output_t["logits"]
                    hidden_state = output_t["gru_state"]
                    prediction[:, [t], :] = prediction_t

                prediction = prediction.reshape([batch_size * time_count, num_classes])

                # Handle targets
                if len(targets.shape) == 3:
                    targets = targets.reshape([batch_size * time_count, num_classes])
                else:
                    targets = targets.reshape([batch_size * time_count])
            else:
                # Single image: images.shape = [b, c, h, w]
                output = model(images)
                # Handle both dict outputs (from custom models) and tensor outputs (from standard models)
                if isinstance(output, dict):
                    prediction = output["logits"] if "logits" in output else output
                else:
                    prediction = output

            inference_time = time.time() - start_time
            inference_times.append(inference_time / batch_size)  # Per image

            # Get probabilities and predictions
            probabilities = torch.softmax(prediction, dim=1)
            _, predicted_classes = torch.max(prediction, dim=1)

            # Handle soft targets (from uncertainty dataset)
            if len(targets.shape) == 2:
                _, target_classes = torch.max(targets, dim=1)
            else:
                target_classes = targets

            all_predictions.extend(predicted_classes.cpu().numpy())
            all_targets.extend(target_classes.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(data_loader)} batches")

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)

    # Calculate metrics
    results = {}

    # Overall accuracy
    results["accuracy"] = accuracy_score(all_targets, all_predictions)

    # Top-5 accuracy
    top5_correct = 0
    for i, target in enumerate(all_targets):
        top5_preds = np.argsort(all_probabilities[i])[-5:]
        if target in top5_preds:
            top5_correct += 1
    results["top5_accuracy"] = top5_correct / len(all_targets)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_predictions, average=None, labels=range(num_classes)
    )

    results["per_class"] = {}
    for i, class_name in enumerate(class_names):
        results["per_class"][class_name] = {
            "precision": precision[i],
            "recall": recall[i],
            "f1_score": f1[i],
            "support": support[i],
            "accuracy": np.sum((all_targets == i) & (all_predictions == i))
            / np.sum(all_targets == i),
        }

    # Confusion matrix
    results["confusion_matrix"] = confusion_matrix(all_targets, all_predictions)

    # Classification report
    results["classification_report"] = classification_report(
        all_targets, all_predictions, target_names=class_names
    )

    # Inference speed
    results["avg_inference_time"] = np.mean(inference_times)
    results["std_inference_time"] = np.std(inference_times)

    # Store raw predictions for further analysis
    results["predictions"] = all_predictions
    results["targets"] = all_targets
    results["probabilities"] = all_probabilities

    return results


def plot_confusion_matrix(
    cm: np.ndarray, class_names: list[str], save_path: str = None
):
    """Plot and optionally save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_per_class_metrics(results: dict, save_path: str = None):
    """Plot per-class precision, recall, and F1-score."""
    class_names = list(results["per_class"].keys())
    metrics = ["precision", "recall", "f1_score", "accuracy"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        values = [results["per_class"][cls][metric] for cls in class_names]

        axes[idx].bar(class_names, values, color="steelblue")
        axes[idx].set_ylabel(metric.replace("_", " ").title())
        axes[idx].set_ylim([0, 1.05])
        axes[idx].set_title(f'Per-Class {metric.replace("_", " ").title()}')
        axes[idx].grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Per-class metrics plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def save_results_to_file(results: dict, output_path: str):
    """Save evaluation results to a text file."""
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("ELEPHANT IDENTITY MODEL EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Overall Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Top-5 Accuracy: {results['top5_accuracy']:.4f}\n\n")

        f.write(
            f"Average Inference Time: {results['avg_inference_time']*1000:.2f} ms/image\n"
        )
        f.write(f"Std Inference Time: {results['std_inference_time']*1000:.2f} ms\n\n")

        f.write("-" * 80 + "\n")
        f.write("PER-CLASS METRICS\n")
        f.write("-" * 80 + "\n\n")

        for class_name, metrics in results["per_class"].items():
            f.write(f"{class_name}:\n")
            f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall:    {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score:  {metrics['f1_score']:.4f}\n")
            f.write(f"  Support:   {metrics['support']}\n\n")

        f.write("-" * 80 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("-" * 80 + "\n\n")
        f.write(results["classification_report"])
        f.write("\n")

        f.write("-" * 80 + "\n")
        f.write("CONFUSION MATRIX\n")
        f.write("-" * 80 + "\n\n")
        f.write(str(results["confusion_matrix"]))
        f.write("\n")

    print(f"Results saved to {output_path}")


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate elephant identity ReID model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data parameters
    parser.add_argument(
        "--data-path",
        required=True,
        type=str,
        help="Path to validation/test dataset directory",
    )
    parser.add_argument(
        "--use_uncertainty_data",
        action="store_true",
        help="Expect a dataset that has (good, bad, terrible) certainty hyper-classes",
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="densenet121",
        type=str,
        help="Model architecture (densenet121, zoo_id_gru)",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Path to model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--sequence_length",
        default=10,
        type=int,
        help="Number of images in sequence (for GRU model)",
    )
    parser.add_argument(
        "--uncertain_rate",
        default=0,
        type=int,
        help="Rate of uncertain images in sequence (for GRU model)",
    )

    # Evaluation parameters
    parser.add_argument(
        "--batch-size", default=32, type=int, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--workers", default=4, type=int, help="Number of data loading workers"
    )
    parser.add_argument(
        "--device", default="cuda", type=str, help="Device to use (cuda or cpu)"
    )

    # Output parameters
    parser.add_argument(
        "--output-dir",
        default="./eval_results",
        type=str,
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--save-results", action="store_true", help="Save evaluation results and plots"
    )
    parser.add_argument(
        "--val-resize-size", default=256, type=int, help="Resize size for validation"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="Crop size for validation"
    )

    return parser


def main(args):
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {args.data_path}")

    # Get validation preprocessing
    preprocessing = presets.ClassificationPresetEval(
        crop_size=args.val_crop_size,
        resize_size=args.val_resize_size,
    )

    if args.use_uncertainty_data:
        dataset = ElephantIdentityDataset(
            root=args.data_path,
            transform=preprocessing,
        )
    else:
        from torchvision.datasets import ImageFolder

        dataset = ImageFolder(
            root=args.data_path,
            transform=preprocessing,
        )

    class_names = dataset.classes
    num_classes = len(class_names)
    print(f"Classes: {class_names}")
    print(f"Number of samples: {len(dataset)}")

    # Wrap in sequence dataset if using GRU model
    use_sequences = "gru" in args.model
    if use_sequences:
        print(f"Using sequence dataset with length {args.sequence_length}")
        dataset = IdentitySequenceDataset(
            args.sequence_length, args.uncertain_rate, dataset
        )

    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")

    # Check if it's a TorchScript model (.ptc) or regular checkpoint (.pth)
    if args.checkpoint.endswith(".ptc"):
        # Load TorchScript compiled model directly
        print("Detected TorchScript model (.ptc)")
        model = torch.jit.load(args.checkpoint, map_location=device)
        model.to(device)
        model.eval()
        print(f"\nTorchScript model loaded successfully!")
    else:
        # Load regular PyTorch checkpoint
        print("Loading regular PyTorch checkpoint")
        model = get_model(model_name=args.model, weights=None, num_classes=num_classes)

        checkpoint = torch.load(
            args.checkpoint, map_location=device, weights_only=False
        )

        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            else:
                model.load_state_dict(checkpoint)

            # Print checkpoint info if available
            if "epoch" in checkpoint:
                print(f"Checkpoint epoch: {checkpoint['epoch']}")
            if "best_acc1" in checkpoint:
                print(f"Best training accuracy: {checkpoint['best_acc1']:.4f}")
        else:
            # Checkpoint is just the state dict
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()
        print(f"\nModel loaded successfully!")

    # Run evaluation
    print("\n" + "=" * 80)
    print("STARTING EVALUATION")
    print("=" * 80 + "\n")

    results = evaluate_model(
        model=model,
        data_loader=data_loader,
        device=device,
        class_names=class_names,
        use_sequences=use_sequences,
    )

    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80 + "\n")

    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    print(f"Top-5 Accuracy: {results['top5_accuracy']:.4f}")
    print(f"Average Inference Time: {results['avg_inference_time']*1000:.2f} ms/image")
    print(f"\nPer-Class Results:")

    for class_name, metrics in results["per_class"].items():
        print(f"\n{class_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  Support:   {metrics['support']}")

    print("\n" + "-" * 80)
    print(results["classification_report"])

    # Save results if requested
    if args.save_results:
        print("\nSaving results...")

        # Save text results
        results_file = os.path.join(args.output_dir, "evaluation_results.txt")
        save_results_to_file(results, results_file)

        # Save confusion matrix plot
        cm_plot_path = os.path.join(args.output_dir, "confusion_matrix.png")
        plot_confusion_matrix(results["confusion_matrix"], class_names, cm_plot_path)

        # Save per-class metrics plot
        metrics_plot_path = os.path.join(args.output_dir, "per_class_metrics.png")
        plot_per_class_metrics(results, metrics_plot_path)

        # Save raw predictions for further analysis
        np.savez(
            os.path.join(args.output_dir, "predictions.npz"),
            predictions=results["predictions"],
            targets=results["targets"],
            probabilities=results["probabilities"],
        )
        print(f"Raw predictions saved to {args.output_dir}/predictions.npz")

        print(f"\nAll results saved to {args.output_dir}")

    print("\nEvaluation complete!")

    return results


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
