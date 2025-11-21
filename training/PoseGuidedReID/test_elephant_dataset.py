#!/usr/bin/env python
"""
Quick test to verify the elephant dataset loads correctly
"""

import sys
sys.path.insert(0, '/media/mu/zoo_vision/training/PoseGuidedReID')

from project.config import cfg
from project.datasets import make_dataloader

def test_dataset():
    print("=" * 60)
    print("Testing Elephant ReID Dataset")
    print("=" * 60)
    
    # Load config
    config_file = '/media/mu/zoo_vision/training/PoseGuidedReID/configs/elephant_resnet.yml'
    cfg.merge_from_file(config_file)
    
    print(f"\nDataset: {cfg.DATASETS.NAMES}")
    print(f"Root Dir: {cfg.DATASETS.ROOT_DIR}")
    print(f"Image Size: {cfg.INPUT.IMG_SIZE}")
    print(f"Batch Size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"Num Workers: {cfg.DATALOADER.NUM_WORKERS}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    try:
        train_loader, train_loader_normal, test_iid_loader, test_ood_loader, val_loader, \
        pid_container, num_cams, num_train_classes, num_test_iid_classes, num_test_classes, num_val_classes = make_dataloader(cfg)
        
        print("\n✓ Dataloaders created successfully!")
        
        # Print statistics
        print("\nDataset Statistics:")
        print(f"  Train classes: {num_train_classes}")
        print(f"  Val classes: {num_val_classes}")
        print(f"  Number of cameras: {num_cams}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        
        # Test loading a batch
        print("\nTesting batch loading...")
        for imgs, pids, cids, meta in train_loader:
            print(f"\n✓ Successfully loaded batch!")
            print(f"  Images shape: {imgs.shape}")
            print(f"  PIDs: {pids.tolist()}")
            print(f"  Camera IDs: {cids.tolist()}")
            print(f"  Unique IDs in batch: {len(pids.unique())}")
            break
        
        # Print PID container
        print("\nIdentity Mapping:")
        for split, mapping in pid_container.items():
            if mapping:
                print(f"  {split}: {mapping}")
        
        print("\n" + "=" * 60)
        print("✓ All tests passed! Ready for training.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == '__main__':
    success = test_dataset()
    sys.exit(0 if success else 1)
