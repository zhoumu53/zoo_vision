"""
Galaxy API endpoints: list elephants, upload & match.
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
_reid_extractor = None

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


def get_reid_extractor():
    global _reid_extractor
    if _reid_extractor is None:
        from reid import ReIDExtractor
        _reid_extractor = ReIDExtractor()
    return _reid_extractor


def label_to_info(label: str) -> dict:
    """Convert label to display info."""
    if label in ELEPHANT_INFO:
        return ELEPHANT_INFO[label]
    # Try matching by folder-style name like "01_Chandra"
    for key, info in ELEPHANT_INFO.items():
        if label.endswith(key) or label == info["name"]:
            return info
    parts = label.split("_", 1)
    name = parts[1] if len(parts) > 1 else label
    return {"id": 0, "name": name, "color": "#FFFFFF", "profile": None}


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
            "profile": info.get("profile"),
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
        h, w = image.shape[:2]
        detections = []
        try:
            detector = get_detector()
            detections = detector.detect_and_crop(image)
        except Exception as e:
            logger.warning("YOLO detection failed: %s, using full image", e)

        if not detections:
            # No detection or detection failed - use the whole image as crop
            logger.info("Using full image as single crop")
            detections = [{
                "bbox": {"x": 0.5, "y": 0.5, "w": 1.0, "h": 1.0},
                "bbox_abs": [0, 0, w, h],
                "crop": image,
                "confidence": 0.0,
            }]

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

        # Step 2: Extract real ReID features
        job["progress"] = 60
        job["stage"] = "Extracting features..."
        reid = get_reid_extractor()
        gallery = get_gallery()

        detected_elephants = []
        for i, det in enumerate(detections):
            crop = det["crop"]

            # Extract real feature using ReID model
            feature = reid.extract(crop)

            # Find nearest elephants in gallery
            nearest = gallery.find_nearest(feature)
            position = gallery.project_to_3d(feature)

            nearest_elephants = []
            for match in nearest:
                match_info = label_to_info(match["elephant_label"])
                pos = match.get("position")
                nearest_elephants.append({
                    "elephant_id": match_info["id"],
                    "elephant_name": match_info["name"],
                    "similarity": round(match["similarity"], 4),
                    "mean_similarity": match.get("mean_similarity"),
                    "cosine_distance": match["cosine_distance"],
                    "match_level": match["match_level"],
                    "vote_count": match.get("vote_count"),
                    "vote_ratio": match.get("vote_ratio"),
                    "margin": match.get("margin"),
                    "sample_crop_path": match["sample_crop_path"],
                    "image_count": match["image_count"],
                    "position": pos,
                    "profile": match_info.get("profile"),
                })

            detected_elephants.append({
                "index": i,
                "crop_url": f"/storage/uploads/{job_dir.name}/elephant_{i}.jpg",
                "bbox": det["bbox"],
                "nearest_elephants": nearest_elephants,
                "uploaded_position": {"x": position[0], "y": position[1], "z": position[2]} if position else None,
                "match_level": nearest[0]["match_level"] if nearest else "unknown",
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
