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
REID_NUM_CLASSES = 6

# Elephant metadata
ELEPHANT_INFO = {
    "Chandra": {"id": 1, "name": "Chandra", "color": "#F5FFC6", "profile": "/profiles/Chandra.jpg"},
    "Indi":    {"id": 2, "name": "Indi",    "color": "#B4E1FF", "profile": "/profiles/Indi.jpg"},
    "Fahra":   {"id": 3, "name": "Fahra",   "color": "#AB87FF", "profile": "/profiles/Fahra.jpg"},
    "Panang":  {"id": 4, "name": "Panang",  "color": "#EDBBB4", "profile": "/profiles/Panang.jpg"},
    "Thai":    {"id": 5, "name": "Thai",     "color": "#C1FF9B", "profile": "/profiles/Thai.jpg"},
}

# Top-k voting parameters
# We retrieve the top VOTE_K nearest individual images, count per-elephant
# votes, and use the vote distribution + mean similarity to classify.
VOTE_K = 100               # Number of nearest images to vote
SAME_VOTE_RATIO = 0.60     # >= 60% of top-k must be this elephant for "same"
SAME_MEAN_SIM = 0.90       # Mean similarity of the winning votes must exceed this
SIMILAR_VOTE_RATIO = 0.30  # >= 30% for "similar"
SIMILAR_MEAN_SIM = 0.70    # Mean similarity threshold for "similar"
TOP_K_MATCHES = 5

# Galaxy visualization
GALAXY_POSITION_SCALE = 40.0

# Upload settings
MAX_UPLOAD_SIZE_MB = 10
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
