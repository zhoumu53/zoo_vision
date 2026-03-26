"""Configuration for the Elephant Game backend."""

from pathlib import Path

# Gallery features NPZ (pre-computed ReID features for all training elephants)
GALLERY_NPZ_PATH = Path(
    "/media/ElephantsWD/elephants/reid_models/"
    "swin_adamw_lr0003_bs128_softmax_triplet_Fulldata/"
    "pred_features/train_iid/pytorch_result_e.npz"
)

# Training image root (ImageFolder structure)
TRAIN_DATA_PATH = Path("/media/mu/zoo_vision/data/full_data/train")

# YOLO detection model
YOLO_WEIGHTS_PATH: str | None = None  # None = use default yolov8n.pt
YOLO_TARGET_LABELS = ("elephant",)
YOLO_CONFIDENCE_THRESHOLD = 0.65

# ReID model config
REID_CONFIG_PATH = Path(
    "/media/mu/zoo_vision/training/PoseGuidedReID/configs/"
    "swim_transformer/swin/swin_base_patch4_window7_224_22k.yaml"
)
REID_CHECKPOINT_PATH = Path(
    "/media/ElephantsWD/elephants/reid_models/"
    "swin_adamw_lr0003_bs128_softmax_triplet_Fulldata/net_best.pth"
)

# Elephant metadata
ELEPHANT_INFO = {
    "01_Chandra": {"id": 1, "name": "Chandra", "color": "#F5FFC6"},
    "02_Indi": {"id": 2, "name": "Indi", "color": "#B4E1FF"},
    "03_Fahra": {"id": 3, "name": "Fahra", "color": "#AB87FF"},
    "04_Panang": {"id": 4, "name": "Panang", "color": "#EDBBB4"},
    "05_Thai": {"id": 5, "name": "Thai", "color": "#C1FF9B"},
}

# Matching thresholds
NEW_ELEPHANT_SIMILARITY_THRESHOLD = 0.55
TOP_K_MATCHES = 5

# Galaxy visualization
GALAXY_POSITION_SCALE = 40.0

# Upload settings
MAX_UPLOAD_SIZE_MB = 10
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
