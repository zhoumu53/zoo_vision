# Elephant ReID Training with PoseGuidedReID

This codebase has been adapted to train Re-Identification (ReID) models for the ZooVision elephant dataset.

## Dataset Structure

Your dataset is located at: `/media/mu/zoo_vision/data/reid_time_split`

```
reid_time_split/
├── train/
│   ├── 01_Chandra/
│   ├── 02_Indi/
│   ├── 03_Fahra/
│   ├── 04_Panang/
│   └── 05_Thai/
└── val/
    ├── 01_Chandra/
    ├── 02_Indi/
    ├── 03_Fahra/
    ├── 04_Panang/
    └── 05_Thai/
```

## Key Differences from Classification

This is a **proper ReID system** that uses:
- ✅ **Metric Learning**: Triplet Loss for learning discriminative embeddings
- ✅ **Combined Loss**: Softmax (ID classification) + Triplet Loss (metric learning)
- ✅ **Embedding Space**: 512-d feature vectors (not just class probabilities)
- ✅ **Gallery-Probe Matching**: Uses cosine similarity for inference
- ✅ **Better Generalization**: Learns invariant features across viewpoints

## Quick Start

### 1. Training

```bash
cd /media/mu/zoo_vision/training/PoseGuidedReID
bash train_elephant_resnet.sh
```

This will train a ResNet50 model with:
- **Batch Size**: 64
- **Epochs**: 120
- **Learning Rate**: 0.00035
- **Loss**: Triplet + Softmax
- **Input Size**: 256x256

### 2. Monitor Training

Logs and checkpoints will be saved to:
```
./logs/elephant_resnet50/
```

The training uses WandB for visualization. Check your WandB dashboard for:
- Training/validation loss curves
- mAP (mean Average Precision)
- CMC (Cumulative Matching Characteristics)
- Rank-1, Rank-5, Rank-10 accuracy

### 3. Evaluation

After training, evaluate a checkpoint:

```bash
bash eval_elephant_resnet.sh ./logs/elephant_resnet50/resnet50_120.pth
```

### 4. Custom Training

Modify parameters in `configs/elephant_resnet.yml` or override via command line:

```bash
python tools/train.py \
    --config_file configs/elephant_resnet.yml \
    --do_training \
    SOLVER.IMS_PER_BATCH 32 \
    SOLVER.MAX_EPOCHS 200 \
    SOLVER.BASE_LR 0.0001
```

## Configuration Details

### Model Architecture (ResNet50)
- **Backbone**: ResNet50 pretrained on ImageNet
- **Neck**: BNNeck (Batch Normalization Neck)
- **Output**: 512-d embedding + 5-class classifier

### Data Augmentation
- Random horizontal flip (p=0.5)
- Random erasing (p=0.5)
- Random crop with padding
- Normalization (ImageNet stats)

### Loss Function
- **ID Loss**: Cross-Entropy with label smoothing
- **Triplet Loss**: Batch-hard triplet mining
- **Weights**: ID=1.0, Triplet=1.0

### Sampler
- **Type**: RandomIdentitySampler
- **Instances per ID**: 4
- Ensures each batch has multiple images per identity for triplet mining

## Output Format

The model outputs **embeddings**, not class probabilities:
- **Shape**: [batch_size, 512]
- **Normalization**: L2-normalized for cosine similarity
- **Inference**: Compare embeddings using cosine distance

## Adapting for C++ Inference

To use the trained model in your C++ pipeline:

1. **Export to TorchScript**:
```python
import torch
model = torch.load('resnet50_120.pth')
model.eval()
scripted = torch.jit.script(model)
scripted.save('elephant_reid.pt')
```

2. **Extract embeddings** (not logits):
```cpp
at::Tensor embeddings = model.forward(images);  // [N, 512]
embeddings = F::normalize(embeddings, F::NormalizeFuncOptions().p(2).dim(1));
```

3. **Match tracks** using cosine similarity:
```cpp
float similarity = F::cosine_similarity(embedding1, embedding2);
if (similarity > THRESHOLD) {
    // Same elephant
}
```

## Key Advantages over Your Current System

| Aspect | Old (Classification) | New (ReID) |
|--------|---------------------|------------|
| Loss | CrossEntropy only | Triplet + CrossEntropy |
| Output | Class probabilities | Feature embeddings |
| Matching | Argmax | Cosine similarity |
| New individuals | Requires retraining | Just add to gallery |
| Uncertainty | Hard to measure | Distance-based confidence |
| Viewpoint invariance | Limited | Better generalization |

## Troubleshooting

### Out of Memory
Reduce batch size in config:
```yaml
SOLVER:
  IMS_PER_BATCH: 32
```

### Poor Performance
- Increase training epochs
- Try different learning rates
- Add more data augmentation
- Use larger input size (384x384)

### Dataset Issues
Check if images load correctly:
```python
from project.datasets import make_dataloader
from project.config import cfg

cfg.merge_from_file('configs/elephant_resnet.yml')
train_loader, *_ = make_dataloader(cfg)

for imgs, pids, cids, meta in train_loader:
    print(f"Batch: {imgs.shape}, IDs: {pids.unique()}")
    break
```

## Next Steps

1. **Baseline Training**: Run the provided script first
2. **Hyperparameter Tuning**: Adjust LR, batch size, etc.
3. **Model Variations**: Try ViT or Swin Transformer
4. **Integration**: Export and integrate into C++ pipeline
5. **Temporal Model**: Add GRU on top of embeddings (like your old system)

## References

- Original PoseGuidedReID: https://github.com/haoni0812/PoseGuided
- Metric Learning: "In Defense of the Triplet Loss for Person Re-Identification"
- BNNeck: "Bag of Tricks and A Strong Baseline for Deep Person Re-identification"
