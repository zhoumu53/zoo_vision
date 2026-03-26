"""
Galaxy API endpoints: list elephants, upload & match.

Adapted from brown-bear-server/apps/api/app/api/galaxy.py and galaxy_upload.py
"""

import logging
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from config import (
    ELEPHANT_INFO,
    MAX_UPLOAD_SIZE_MB,
    UPLOAD_DIR,
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_TARGET_LABELS,
    YOLO_WEIGHTS_PATH,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/galaxy", tags=["galaxy"])

# Lazy-loaded singletons
_gallery = None
_detector = None
_reid_model = None

# In-memory job store
_jobs: dict[str, dict] = {}


def get_gallery():
    global _gallery
    if _gallery is None:
        from reid import GalleryManager
        from config import GALLERY_NPZ_PATH
        _gallery = GalleryManager(GALLERY_NPZ_PATH)
        _gallery.load()
    return _gallery


def get_detector():
    global _detector
    if _detector is None:
        from detection import ElephantDetector
        _detector = ElephantDetector(
            weights_path=YOLO_WEIGHTS_PATH,
            target_labels=YOLO_TARGET_LABELS,
            confidence_threshold=YOLO_CONFIDENCE_THRESHOLD,
        )
    return _detector


def label_to_info(label: str) -> dict:
    """Convert folder label (e.g. '01_Chandra') to display info."""
    for folder_name, info in ELEPHANT_INFO.items():
        if label == folder_name or label == info["name"]:
            return info
    # Fallback: extract name from label
    parts = label.split("_", 1)
    name = parts[1] if len(parts) > 1 else label
    return {"id": 0, "name": name, "color": "#FFFFFF"}


@router.get("/elephants")
async def list_elephants():
    """List all elephants with 3D positions and image counts."""
    gallery = get_gallery()
    raw = gallery.get_elephants()
    elephants = []
    for e in raw:
        info = label_to_info(e["elephant_label"])
        elephants.append({
            "elephant_id": info["id"],
            "elephant_name": info["name"],
            "color": info["color"],
            "image_count": e["image_count"],
            "sample_crop_path": e["sample_crop_path"],
            "x": e["x"],
            "y": e["y"],
            "z": e["z"],
        })
    return {"elephants": elephants}


@router.post("/upload")
async def upload_photo(photo: UploadFile = File(...)):
    """Upload a photo for elephant identification."""
    if not photo.content_type or not photo.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    contents = await photo.read()
    if len(contents) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(400, f"File too large (max {MAX_UPLOAD_SIZE_MB}MB)")

    job_uuid = str(uuid.uuid4())[:8]
    job_dir = UPLOAD_DIR / job_uuid
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save original
    orig_path = job_dir / "original.jpg"
    with open(orig_path, "wb") as f:
        f.write(contents)

    job_id = job_uuid
    _jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "stage": "Queued",
        "result": None,
        "partial": None,
        "image_path": str(orig_path),
        "job_dir": str(job_dir),
    }

    # Process in background (simple synchronous for now)
    import threading
    t = threading.Thread(target=_process_upload, args=(job_id,))
    t.start()

    return {"job_id": job_id, "uuid": job_uuid}


@router.get("/upload/{job_id}/status")
async def upload_status(job_id: str):
    """Poll upload processing status."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "status": job["status"],
        "progress": job["progress"],
        "stage": job["stage"],
        "result": job["result"],
        "partial": job["partial"],
    }


def _process_upload(job_id: str):
    """Process an uploaded photo: detect, extract features, match."""
    job = _jobs[job_id]
    try:
        job["status"] = "running"
        job["progress"] = 10
        job["stage"] = "Loading image..."

        image = cv2.imread(job["image_path"])
        if image is None:
            raise ValueError("Failed to read image")

        job_dir = Path(job["job_dir"])

        # Step 1: Detect elephants
        job["progress"] = 20
        job["stage"] = "Detecting elephants..."
        detector = get_detector()
        detections = detector.detect_and_crop(image)

        if not detections:
            job["status"] = "done"
            job["progress"] = 100
            job["stage"] = "Complete"
            job["result"] = {
                "outcome": "no_elephant_detected",
                "original_url": f"/storage/uploads/{Path(job_dir).name}/original.jpg",
                "detected_elephants": [],
            }
            return

        # Provide partial detection data for animation
        partial_elephants = []
        for i, det in enumerate(detections):
            crop_path = job_dir / f"elephant_{i}.jpg"
            cv2.imwrite(str(crop_path), det["crop"])
            partial_elephants.append({
                "index": i,
                "bbox": det["bbox"],
                "crop_url": f"/storage/uploads/{job_dir.name}/elephant_{i}.jpg",
            })

        job["progress"] = 40
        job["partial"] = {
            "detected_elephants_partial": partial_elephants,
            "original_url": f"/storage/uploads/{job_dir.name}/original.jpg",
        }

        # Step 2: Extract features (using ReID model)
        job["progress"] = 60
        job["stage"] = "Extracting features..."
        gallery = get_gallery()

        # For each detected elephant, extract ReID features
        detected_elephants = []
        for i, det in enumerate(detections):
            crop = det["crop"]

            # Resize crop to 224x224 for ReID
            crop_resized = cv2.resize(crop, (224, 224))
            # Convert BGR to RGB, normalize
            crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            # ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            crop_normalized = (crop_rgb - mean) / std
            # To tensor format (C, H, W)
            crop_tensor = np.transpose(crop_normalized, (2, 0, 1)).astype(np.float32)

            # Use gallery's centroids for matching (feature already L2-normalized)
            # For now, use the resized crop as a simple feature proxy
            # In production, this would use the actual ReID model
            # feature = reid_model.extract_features(crop_tensor)
            # For demo: use random feature matching against gallery
            feature = crop_tensor.flatten()[:gallery.features.shape[1]]
            norm = np.linalg.norm(feature)
            if norm > 1e-9:
                feature = feature / norm

            # Find nearest elephants
            nearest = gallery.find_nearest(feature)
            position = gallery.project_to_3d(feature)

            nearest_elephants = []
            for match in nearest:
                match_info = label_to_info(match["elephant_label"])
                pos = match.get("position")
                nearest_elephants.append({
                    "elephant_id": match_info["id"],
                    "elephant_name": match_info["name"],
                    "similarity": match["similarity"],
                    "sample_crop_path": match["sample_crop_path"],
                    "image_count": match["image_count"],
                    "position": pos,
                })

            detected_elephants.append({
                "index": i,
                "crop_url": f"/storage/uploads/{job_dir.name}/elephant_{i}.jpg",
                "bbox": det["bbox"],
                "nearest_elephants": nearest_elephants,
                "uploaded_position": {"x": position[0], "y": position[1], "z": position[2]} if position else None,
                "possibly_new": nearest[0]["possibly_new"] if nearest else True,
            })

        job["progress"] = 100
        job["stage"] = "Complete"
        job["status"] = "done"
        job["result"] = {
            "outcome": "match_found",
            "original_url": f"/storage/uploads/{job_dir.name}/original.jpg",
            "detected_elephants": detected_elephants,
        }

    except Exception as e:
        logger.exception("Upload processing failed for job %s", job_id)
        job["status"] = "failed"
        job["stage"] = str(e)
        job["progress"] = 0
