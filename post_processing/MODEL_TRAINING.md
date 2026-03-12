# Model Training Guide

This document summarizes how we train the three models used by the post-processing pipeline:

1. ReID model (PoseGuidedReID)
2. Behavior classification model (sleep + quality, two-head Swin)
3. Stereotype classifier

---

## 1) ReID Model

### Purpose
Train identity embeddings for elephant re-identification used in:
- per-track identity voting
- track stitching support
- cross-camera identity matching

### Code location
- Training package: `/media/mu/zoo_vision/training/PoseGuidedReID`
- Main train entrypoint: `training/PoseGuidedReID/tools/train.py`
- Elephant dataset loader: `training/PoseGuidedReID/project/datasets/elephant.py`
- Main Swin config (used in production-style script):
  - `training/PoseGuidedReID/configs/swim_transformer/swin/swin_base_patch4_window7_224_22k.yaml`
- Main script used for Swin softmax+triplet setup:
  - `training/PoseGuidedReID/scripts/train_elephant_swin_softmax_triplet.sh`

### Dataset
Current config/script defaults point to:
- time-split dataset root: `/media/mu/zoo_vision/data/reid_time_split`
- optional full dataset root (script override): `/media/mu/zoo_vision/data/full_data`

Expected structure for elephant loader:

```text
<ROOT_DIR>/
  train/
    01_Chandra/
    02_Indi/
    03_Fahra/
    04_Panang/
    05_Thai/
    06_Zali/        # optional, supported in loader
  val/
    01_Chandra/
    ...
```

Notes:
- If `train/` and `val/` are missing, loader can merge subdirectories and create an internal split (`merge_all=True` behavior in loader).
- Camera IDs are parsed from filenames containing `zag_elp_cam_016/017/018/019` patterns.

### Trainer settings (Swin, from real script)
The Swin script uses:
- Config file:
  - `configs/swim_transformer/swin/swin_base_patch4_window7_224_22k.yaml`
- Script-level parameters in `train_elephant_swin_softmax_triplet.sh`:
  - `BATCH_SIZE=128`
  - `NUM_EPOCHS=100`
  - `LR=0.0003`
  - `DATALOADER.SAMPLER='softmax_triplet'`
  - `DATALOADER.NUM_INSTANCE=8`
  - `SOLVER.OPTIMIZER_NAME='AdamW'`
  - `SOLVER.STEPS='(40, 70)'`
  - `SOLVER.WARMUP_EPOCHS=5`, `SOLVER.WARMUP_FACTOR=0.01`, `SOLVER.WARMUP_METHOD='linear'`
  - `SOLVER.WEIGHT_DECAY=0.0001`, `SOLVER.WEIGHT_DECAY_BIAS=0.0001`
  - `MODEL.PRETRAIN_PATH=./checkpoints/swin_base_patch4_window7_224_22k.pth`
  - `MODEL.RESUME=True`
  - `DATASETS.ROOT_DIR=/media/mu/zoo_vision/data/reid_time_split`
  - `DATASETS.IMG_DIR=/media/mu/zoo_vision/data/reid_time_split`
  - `DATASETS.NAMES='elephant'`

Base defaults from the YAML include:
- `MODEL.NAME=swin_base_patch4_window7_224_22k`
- `MODEL.TYPE=swin`
- `SOLVER.MAX_EPOCHS=90` (overridden to 100 by script)
- `SOLVER.BASE_LR=1.25e-4` (overridden to 3e-4 by script)

### Run examples

Swin + softmax_triplet (main script in use):

```bash
cd /media/mu/zoo_vision/training/PoseGuidedReID
bash scripts/train_elephant_swin_softmax_triplet.sh
```

### Outputs
Under `OUTPUT_DIR`:
- checkpoints:
  - `net_best.pth`
  - `net_last.pth`
  - periodic `net_<epoch>.pth`
- extracted features (when `--do_inference`):
  - `pred_features/train_iid/...`
  - `pred_features/val_iid/...`

### Environment / dependencies
- `training/PoseGuidedReID/requirements.txt`
- Also requires a valid `wandb` login (`wandb.login()` is called in `tools/train.py`).

---

## 2) Behavior Classification Model (Sleep + Quality, Two-Head)

### Purpose
Frame-level behavior classification used during post-processing feature extraction:
- behavior head: `00_invalid`, `01_standing`, `02_sleeping_left`, `03_sleeping_right`
- quality head: `good`, `bad`

### Code location
- Training script: `/media/mu/zoo_vision/training/classification_vit/run_sleep_swinb_two_heads.py`
- Example launcher: `training/classification_vit/train_behavior_new.sh`

### Dataset
The script expects directory-by-class data:

```text
<train_root>/
  00_invalid/
  01_standing/
  02_sleeping_left/
  03_sleeping_right/

<eval_root>/
  00_invalid/
  01_standing/
  02_sleeping_left/
  03_sleeping_right/
```

Key behavior:
- `00_invalid` is mapped to quality label `bad`
- all other behavior classes map to quality label `good`
- `--train_root` accepts one or multiple roots

Common dataset paths used in this repo:
- `/media/mu/zoo_vision/data/behaviour/sleep_v1`
- `/media/mu/zoo_vision/data/behaviour/sleep_v2`
- `/media/mu/zoo_vision/data/behaviour/sleep_v5`
- eval set: `/media/mu/zoo_vision/data/behaviour/sleep_v4`

### Trainer settings (default / typical)
From `run_sleep_swinb_two_heads.py`:
- Backbone: `swin_base_patch4_window7_224` (timm)
- Input size: 224
- Optimizer: AdamW
- Loss:
  - CrossEntropy on behavior head (class-weighted)
  - CrossEntropy on quality head (class-weighted)
  - combined weighted sum (`--behavior_loss_weight`, `--quality_loss_weight`)
- Sampler: weighted random sampler (by behavior class distribution)
- Augmentation:
  - random resized crop, shift/scale/rotate
  - brightness/contrast/gamma
  - blur + noise + compression
  - horizontal flip with sleeping left/right label swap (`--hflip_swap_p`)
- Defaults:
  - epochs: 25
  - batch size: 32
  - lr: 3e-5
  - weight_decay: 0.05
  - num_workers: 6
  - noise_level: 0.7

### Run example

```bash
cd /media/mu/zoo_vision/training/classification_vit
python run_sleep_swinb_two_heads.py \
  --train_root /media/mu/zoo_vision/data/behaviour/sleep_v1 \
               /media/mu/zoo_vision/data/behaviour/sleep_v2 \
               /media/mu/zoo_vision/data/behaviour/sleep_v5 \
  --eval_root  /media/mu/zoo_vision/data/behaviour/sleep_v4 \
  --out_dir    runs_sleep/two_heads_swinb_flip_cleandata \
  --model      swin_base_patch4_window7_224 \
  --hflip_swap_p 0.5 \
  --img_size   224 \
  --batch_size 32 \
  --epochs     100 \
  --lr         3e-5 \
  --weight_decay 0.05 \
  --seed 42
```

### Outputs
In `--out_dir`:
- `best_dual_head.pt`
- `last_dual_head.pt`
- `training_log.txt`
- `eval_summary.txt`

---

## 3) Stereotype Classifier

### Purpose
Classify stereotype-related labels from images for downstream stereotype analysis.

### Code location
- Training: `/media/mu/zoo_vision/post_processing/analysis/stereotype_classifier/training.py`
- Inference: `/media/mu/zoo_vision/post_processing/analysis/stereotype_classifier/inference.py`

### Dataset
Default paths in script:
- GT CSV: `/media/mu/zoo_vision/data/stereotype/gt.csv`
- Images: `/media/mu/zoo_vision/data/stereotype/images`

CSV assumptions:
- contains `filename` and `label` columns
- year is parsed from date token in filename (`YYYYMMDD...`)

Split strategy:
- strict year split
- train on `--train_years` (default: `2025`)
- evaluate on `--test_year` (default: `2026`)

### Trainer settings
From `training.py`:
- Model: `resnet18` with ImageNet pretrained weights
- Final layer replaced by number of classes found in data
- Transform:
  - resize to square (`--image_size`, default 224)
  - grayscale to 3 channels
  - normalization mean/std = 0.5
- Optimizer: AdamW
- Loss: CrossEntropy
- Defaults:
  - epochs: 20
  - batch size: 32
  - lr: 1e-4
  - weight_decay: 1e-4
  - num_workers: 4
  - seed: 42
- Best model selection:
  - by best test accuracy (if test split exists)
  - otherwise by lowest training loss

### Run example

```bash
python /media/mu/zoo_vision/post_processing/analysis/stereotype_classifier/training.py \
  --gt_csv /media/mu/zoo_vision/data/stereotype/gt.csv \
  --image_dir /media/mu/zoo_vision/data/stereotype/images \
  --train_years 2025 \
  --test_year 2026 \
  --epochs 20 \
  --batch_size 32 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --checkpoint_path /media/mu/zoo_vision/post_processing/analysis/stereotype_classifier/model.pt
```

### Outputs
- checkpoint: `model.pt`
- metadata: `model.json` (same basename)
  - class names
  - train/test sample counts
  - train/test years
  - best metric

---

## Model Registry (for post-processing)

Current model paths used by post-processing configs/scripts:

- ReID:
  - config: `/media/mu/zoo_vision/training/PoseGuidedReID/configs/swim_transformer/swin/swin_base_patch4_window7_224_22k.yaml`
  - checkpoint example: `/media/ElephantsWD/elephants/reid_models/swin_adamw_lr0003_bs128_softmax_triplet_Fulldata/net_best.pth`
- Behavior (two-head sleep classifier):
  - directory example: `/media/mu/zoo_vision/training/classification_vit/runs_sleep/two_heads_swinb_flip`
  - model file used by inference wrapper: `best_dual_head.pt`
- Stereotype classifier:
  - `/media/mu/zoo_vision/post_processing/analysis/stereotype_classifier/model.pt`

---

## Reproducibility Checklist

Before launching a new training run:
1. Fix dataset roots and split policy.
2. Set output directory with experiment name (include model/lr/batchsize/sampler).
3. Record exact command in a shell script or log file.
4. Keep `config` + `checkpoint` + summary metrics together.
5. For ReID, verify `wandb` login and GPU selection.
