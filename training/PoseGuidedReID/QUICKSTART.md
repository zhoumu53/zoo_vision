# Quick Setup Guide - Elephant ReID Training

## ✅ What's Been Done

1. **Created Elephant Dataset Loader** (`project/datasets/elephant.py`)
   - Reads from `/media/mu/zoo_vision/data/reid_time_split/`
   - Uses `train/` folder for training
   - Uses `val/` folder for testing
   - Automatically extracts camera IDs and years from filenames

2. **Created Configuration** (`configs/elephant_resnet.yml`)
   - ResNet50 backbone
   - Triplet + Softmax loss (proper ReID approach)
   - 256x256 input images
   - 120 epochs training

3. **Created Training Script** (`train_elephant_resnet.sh`)
   - Single GPU training
   - WandB logging enabled
   - Saves checkpoints every 20 epochs

4. **Created Evaluation Script** (`eval_elephant_resnet.sh`)
   - Tests trained models
   - Computes mAP, CMC, Rank-1/5/10

## 📦 Installation

Run the installation script:
```bash
cd /media/mu/zoo_vision/training/PoseGuidedReID
bash install_deps.sh
```

This installs:
- yacs, timm, opencv-python
- albumentations (data augmentation)
- pytorch_metric_learning (triplet loss)
- tensorboard, wandb (logging)

## 🚀 Usage

### Test Dataset Loading
```bash
cd /media/mu/zoo_vision/training/PoseGuidedReID
python test_elephant_dataset.py
```

### Start Training
```bash
cd /media/mu/zoo_vision/training/PoseGuidedReID
bash train_elephant_resnet.sh
```

### Monitor Training
- **Terminal**: Watch the training logs
- **WandB**: Check your dashboard at wandb.ai
- **TensorBoard**: `tensorboard --logdir ./logs/elephant_resnet50`

### Evaluate Model
```bash
bash eval_elephant_resnet.sh ./logs/elephant_resnet50/resnet50_120.pth
```

## 📊 Expected Output

During training you'll see:
- **Loss**: Should decrease from ~3.0 to <0.5
- **mAP**: Mean Average Precision (higher is better, target >80%)
- **Rank-1**: Top-1 accuracy (target >90%)
- **Rank-5**: Top-5 accuracy (target >95%)

## 🔧 Adjusting Parameters

Edit `configs/elephant_resnet.yml` or override via command line:

```bash
python tools/train.py \
    --config_file configs/elephant_resnet.yml \
    --do_training \
    SOLVER.IMS_PER_BATCH 32 \          # Reduce if OOM
    SOLVER.MAX_EPOCHS 200 \            # Train longer
    SOLVER.BASE_LR 0.0001 \            # Adjust learning rate
    MODEL.DEVICE_ID "(0,1)" \          # Multi-GPU
    INPUT.IMG_SIZE "[384,384]"         # Larger images
```

## 📁 Output Structure

```
logs/elephant_resnet50/
├── resnet50_20.pth      # Checkpoint at epoch 20
├── resnet50_40.pth      # Checkpoint at epoch 40
├── ...
├── resnet50_120.pth     # Final checkpoint
├── train.log            # Training logs
└── events.out.tfevents  # TensorBoard logs
```

## 🐘 Key Differences from Old System

| Feature | Old (Classification) | New (ReID) |
|---------|---------------------|------------|
| Loss | CrossEntropy only | **Triplet + CrossEntropy** |
| Output | 5 class probabilities | **512-d embedding vector** |
| Matching | argmax(probabilities) | **cosine_similarity(embeddings)** |
| Adding new elephant | Retrain entire model | **Add to gallery** |
| Confidence | No direct measure | **Distance-based** |

## 🔍 Troubleshooting

**Out of Memory?**
- Reduce batch size: `SOLVER.IMS_PER_BATCH 32`
- Reduce image size: `INPUT.IMG_SIZE "[224,224]"`

**Poor performance?**
- Train longer: `SOLVER.MAX_EPOCHS 200`
- Check if data is balanced (each elephant has similar # images)
- Try different learning rate

**Installation issues?**
- Make sure you ran `bash install_deps.sh`
- Check Python version: `python --version` (should be 3.11+)

## 📚 Next Steps

1. **Run baseline training** with current settings
2. **Analyze results** - check which elephants are confused
3. **Tune hyperparameters** if needed
4. **Export to C++** once satisfied with performance
5. **Add temporal model** (GRU) on top of embeddings for sequence-based ID

## 💡 Tips

- Start with default config, don't change too many things at once
- WandB is your friend - compare different runs easily
- Save your best checkpoint separately
- Document what config produced your best results

Ready to train! 🎯
