"""
Test script for ReID evaluation - classification vs feature similarity comparison
"""
import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report

# Add project paths
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
POSE_REID_ROOT = PROJECT_ROOT / "training" / "PoseGuidedReID"

for path in (PROJECT_ROOT, POSE_REID_ROOT):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from post_processing.core.reid_inference import ReIDInference


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_images_from_directory(image_dir: Path, transform, logger):
    """
    Load images from directory structure: image_dir/{ID}/*.jpg
    
    Returns:
        images: List of transformed image tensors
        labels: List of identity labels (folder names)
        paths: List of image paths
    """
    image_dir = Path(image_dir)
    
    images = []
    labels = []
    paths = []
    
    # Get all subdirectories (each representing an ID)
    id_dirs = sorted([d for d in image_dir.iterdir() if d.is_dir()])
    
    if not id_dirs:
        raise ValueError(f"No subdirectories found in {image_dir}")
    
    logger.info(f"Found {len(id_dirs)} identity directories")
    
    for id_dir in id_dirs:
        label = id_dir.name
        
        # Get all jpg images in this directory
        image_files = sorted(list(id_dir.glob("*.jpg")) + list(id_dir.glob("*.JPG")))
        
        if not image_files:
            logger.warning(f"No images found in {id_dir}")
            continue
        
        logger.info(f"Loading {len(image_files)} images for ID: {label}")
        
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img)
                
                images.append(img_tensor)
                labels.append(label)
                paths.append(str(img_path))
            except Exception as e:
                logger.warning(f"Failed to load {img_path}: {e}")
    
    logger.info(f"Loaded {len(images)} total images from {len(set(labels))} identities")
    
    return images, labels, paths


def create_label_mapping(labels):
    """Create mapping from string labels to integer IDs."""
    unique_labels = sorted(set(labels))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    label_ids = np.array([label_to_id[label] for label in labels])
    return label_ids, label_to_id


def load_data_from_source(source_path: str, transform, logger, data_type="query"):
    """
    Load data from either image directory or pre-extracted features (.npz).
    
    Args:
        source_path: Path to either image directory or .npz feature file
        transform: Image transform (only used if loading from directory)
        logger: Logger instance
        data_type: "query" or "gallery" for logging
        
    Returns:
        images: List of image tensors (None if loading from .npz)
        labels: List of string labels
        paths: List of image paths
        features: Tensor of features (None if loading from directory)
    """
    source_path = Path(source_path)
    
    # Check if it's a .npz file
    if source_path.suffix == '.npz':
        logger.info(f"Loading pre-extracted {data_type} features from {source_path}")
        data = np.load(source_path, allow_pickle=True)

        features = torch.from_numpy(data['feature']).float()
        labels = data['label'].tolist()
        paths = data.get('path', data.get('paths', [str(i) for i in range(len(labels))]))
        
        # Handle different path storage formats
        if isinstance(paths, np.ndarray):
            paths = paths.tolist()
        
        logger.info(f"Loaded {len(features)} {data_type} features, shape: {features.shape}")
        return None, labels, paths, features
    
    # Otherwise, load from image directory
    else:
        logger.info(f"Loading {data_type} images from directory: {source_path}")
        images, labels, paths = load_images_from_directory(source_path, transform, logger)
        return images, labels, paths, None


def evaluate_classification(
    reid_model: ReIDInference,
    images: list,
    label_ids: np.ndarray,
    batch_size: int,
    logger,
):
    """
    Evaluate classification accuracy using the classifier head.
    
    Args:
        reid_model: ReIDInference instance
        images: List of preprocessed image tensors
        label_ids: Integer labels
        batch_size: Batch size for inference
        logger: Logger instance
        
    Returns:
        predictions: Predicted labels
        accuracy: Overall classification accuracy
        metrics: Dictionary with confusion matrix, precision, recall, F1
    """
    logger.info("=== Classification Evaluation ===")
    
    all_predictions = []
    all_gt = []
    
    num_batches = (len(images) + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc="Classification inference"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(images))
        
        batch_images = torch.stack(images[start_idx:end_idx]).to(reid_model.device)
        batch_labels = label_ids[start_idx:end_idx]
        
        predictions, batch_acc = reid_model.run_prediction(
            batch_images,
            gt_labels=batch_labels,
            is_evaluate=False,  # We'll compute overall accuracy later
        )
        
        all_predictions.extend(predictions)
        all_gt.extend(batch_labels)
    
    all_predictions = np.array(all_predictions)
    all_gt = np.array(all_gt)
    
    # Compute overall accuracy
    correct = np.sum(all_predictions == all_gt)
    accuracy = correct / len(all_gt)
    
    logger.info(f"\nClassification Accuracy: {accuracy*100:.2f}% ({correct}/{len(all_gt)})")
    
    # Compute confusion matrix
    unique_labels = np.unique(all_gt)
    cm = confusion_matrix(all_gt, all_predictions, labels=unique_labels)
    
    logger.info("\nConfusion Matrix:")
    logger.info(f"{'':>10}" + "".join([f"Pred {i:>6}" for i in unique_labels]))
    for i, label_id in enumerate(unique_labels):
        row_str = f"True {label_id:>4} |" + "".join([f"{cm[i, j]:>10}" for j in range(len(unique_labels))])
        logger.info(row_str)
    
    # Compute precision, recall, F1 per class
    precision, recall, f1, support = precision_recall_fscore_support(
        all_gt, all_predictions, labels=unique_labels, average=None, zero_division=0
    )
    
    logger.info("\nPer-class Metrics:")
    logger.info(f"{'Class':>8} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>10}")
    logger.info("-" * 60)
    for i, label_id in enumerate(unique_labels):
        logger.info(
            f"{label_id:>8} {precision[i]*100:>11.2f}% {recall[i]*100:>11.2f}% "
            f"{f1[i]*100:>11.2f}% {support[i]:>10}"
        )
    
    # Compute macro and weighted averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_gt, all_predictions, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_gt, all_predictions, average='weighted', zero_division=0
    )
    
    logger.info("-" * 60)
    logger.info(
        f"{'Macro':>8} {precision_macro*100:>11.2f}% {recall_macro*100:>11.2f}% "
        f"{f1_macro*100:>11.2f}%"
    )
    logger.info(
        f"{'Weighted':>8} {precision_weighted*100:>11.2f}% {recall_weighted*100:>11.2f}% "
        f"{f1_weighted*100:>11.2f}%"
    )
    
    # Store metrics
    metrics = {
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
    }
    
    return all_predictions, accuracy, metrics


def evaluate_feature_similarity(
    reid_model: ReIDInference,
    query_images: list,
    query_label_ids: np.ndarray,
    query_paths: list,
    gallery_images: list,
    gallery_label_ids: np.ndarray,
    gallery_paths: list,
    batch_size: int,
    max_rank: int,
    logger,
    query_features: torch.Tensor = None,
    gallery_features: torch.Tensor = None,
):
    """
    Evaluate using feature similarity (CMC ranking accuracy).
    
    Args:
        reid_model: ReIDInference instance
        query_images: List of preprocessed query image tensors (or None if using pre-extracted features)
        query_label_ids: Query integer labels
        query_paths: Query image paths
        gallery_images: List of preprocessed gallery image tensors (or None if using pre-extracted features)
        gallery_label_ids: Gallery integer labels
        gallery_paths: Gallery image paths
        batch_size: Batch size for feature extraction
        max_rank: Maximum rank for CMC evaluation
        logger: Logger instance
        query_features: Pre-extracted query features (optional)
        gallery_features: Pre-extracted gallery features (optional)
        
    Returns:
        cmc: CMC curve
        mAP: Mean Average Precision
    """
    logger.info("=== Feature Similarity Evaluation (CMC) ===")
    
    # Extract or use pre-loaded query features
    if query_features is None:
        logger.info("Extracting query features...")
        query_features = []
        
        num_batches = (len(query_images) + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc="Query feature extraction"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(query_images))
            
            batch_images = torch.stack(query_images[start_idx:end_idx]).to(reid_model.device)
            features = reid_model.extract_features(batch_images)
            query_features.append(features)
        
        query_features = torch.cat(query_features, dim=0)
    else:
        logger.info("Using pre-extracted query features")
        query_features = query_features.to(reid_model.device)
    
    logger.info(f"Query features shape: {query_features.shape}")
    
    # Extract or use pre-loaded gallery features
    if gallery_features is None:
        logger.info("Extracting gallery features...")
        gallery_features = []
        
        num_batches = (len(gallery_images) + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc="Gallery feature extraction"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(gallery_images))
            
            batch_images = torch.stack(gallery_images[start_idx:end_idx]).to(reid_model.device)
            features = reid_model.extract_features(batch_images)
            gallery_features.append(features)
        
        gallery_features = torch.cat(gallery_features, dim=0)
    else:
        logger.info("Using pre-extracted gallery features")
        gallery_features = gallery_features.to(reid_model.device)
    
    logger.info(f"Gallery features shape: {gallery_features.shape}")
    
    logger.info(f"Query set: {len(query_features)} samples")
    logger.info(f"Gallery set: {len(gallery_features)} samples")
    
    # Convert paths to numpy arrays
    query_paths = np.array(query_paths)
    gallery_paths = np.array(gallery_paths)
    
    # Compute CMC
    logger.info("Computing CMC curve...")
    cmc, mAP, all_AP = reid_model.compute_cmc(
        query_features=query_features,
        gallery_features=gallery_features,
        query_labels=query_label_ids,
        gallery_labels=gallery_label_ids,
        query_paths=query_paths,
        gallery_paths=gallery_paths,
        max_rank=max_rank,
        filter_date=False,
        
    )
    
    # Print CMC results
    logger.info("\nCMC Results:")
    logger.info(f"  mAP: {mAP*100:.2f}%")
    for rank in [1, 5, 10, 20]:
        if rank <= len(cmc):
            logger.info(f"  Rank-{rank}: {cmc[rank-1]*100:.2f}%")
    
    return cmc, mAP


def main():
    parser = argparse.ArgumentParser(description="Test ReID model evaluation")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model config file (e.g., configs/elephant_resnet.yml)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Path to query image directory ({ID}/*.jpg) or pre-extracted features (.npz)",
    )
    parser.add_argument(
        "--gallery_image_dir",
        type=str,
        required=True,
        help="Path to gallery image directory ({ID}/*.jpg) or pre-extracted features (.npz)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        required=True,
        help="Number of identity classes",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--max_rank",
        type=int,
        default=50,
        help="Maximum rank for CMC evaluation",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    logger = setup_logging(args.verbose)
    
    # Initialize ReID models - one for classification, one for features
    logger.info("Initializing ReID models...")
    logger.info(f"  Config: {args.config}")
    logger.info(f"  Checkpoint: {args.checkpoint}")
    logger.info(f"  Device: {args.device}")
    
    # Model for classification (with FC layer)
    logger.info("\nLoading model for classification...")
    reid_model_cls = ReIDInference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        num_classes=args.num_classes,
        logger=logger,
        mode="classification",
    )
    
    # Model for feature extraction (without FC layer)
    logger.info("\nLoading model for feature extraction...")
    reid_model_feat = ReIDInference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        num_classes=args.num_classes,
        logger=logger,
        mode="feature",
    )
    
    # Load query data (images or features)
    logger.info(f"\nLoading query data from {args.image_dir}...")
    query_images, query_labels, query_paths, query_features = load_data_from_source(
        args.image_dir,
        reid_model_cls.transform,
        logger,
        data_type="query",
    )
    
    # Load gallery data (images or features)
    logger.info(f"\nLoading gallery data from {args.gallery_image_dir}...")
    gallery_images, gallery_labels, gallery_paths, gallery_features = load_data_from_source(
        args.gallery_image_dir,
        reid_model_cls.transform,
        logger,
        data_type="gallery",
    )
    
    # Create label mapping (combine query and gallery labels for consistent mapping)
    all_labels = query_labels + gallery_labels
    _, label_to_id = create_label_mapping(all_labels)
    
    # Map both query and gallery labels using the same mapping
    query_label_ids = np.array([label_to_id[label] for label in query_labels])
    gallery_label_ids = np.array([label_to_id[label] for label in gallery_labels])
    
    logger.info(f"\nLabel mapping: {label_to_id}")
    logger.info(f"Query: {len(query_labels)} samples from {len(set(query_labels))} identities")
    logger.info(f"Gallery: {len(gallery_labels)} samples from {len(set(gallery_labels))} identities")
    
    # Evaluation 1: Classification accuracy (only if images are loaded)
    if query_images is not None:
        logger.info("\n" + "="*60)
        predictions, cls_accuracy, cls_metrics = evaluate_classification(
            reid_model=reid_model_cls,
            images=query_images,
            label_ids=query_label_ids,
            batch_size=args.batch_size,
            logger=logger,
        )
    else:
        logger.info("\n" + "="*60)
        logger.info("Skipping classification evaluation (using pre-extracted features)")
        cls_accuracy = None
        cls_metrics = None
    
    # Evaluation 2: Feature similarity (CMC)
    logger.info("\n" + "="*60)
    cmc, mAP = evaluate_feature_similarity(
        reid_model=reid_model_feat,
        query_images=query_images,
        query_label_ids=query_label_ids,
        query_paths=query_paths,
        gallery_images=gallery_images,
        gallery_label_ids=gallery_label_ids,
        gallery_paths=gallery_paths,
        batch_size=args.batch_size,
        max_rank=args.max_rank,
        logger=logger,
        query_features=query_features,
        gallery_features=gallery_features,
    )
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("=== EVALUATION SUMMARY ===")
    if cls_accuracy is not None:
        logger.info(f"\nClassification Metrics:")
        logger.info(f"  Accuracy: {cls_accuracy*100:.2f}%")
        logger.info(f"  Precision (macro): {cls_metrics['precision_macro']*100:.2f}%")
        logger.info(f"  Recall (macro): {cls_metrics['recall_macro']*100:.2f}%")
        logger.info(f"  F1-Score (macro): {cls_metrics['f1_macro']*100:.2f}%")
    logger.info(f"\nFeature Similarity Metrics:")
    logger.info(f"  mAP: {mAP*100:.2f}%")
    logger.info(f"  Rank-1: {cmc[0]*100:.2f}%")
    if len(cmc) > 4:
        logger.info(f"  Rank-5: {cmc[4]*100:.2f}%")
    if len(cmc) > 9:
        logger.info(f"  Rank-10: {cmc[9]*100:.2f}%")
    logger.info("="*60)


if __name__ == "__main__":
    main()
