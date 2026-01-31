# encoding: utf-8
"""
Elephant ReID Dataset
@author: Adapted for ZooVision Elephants
"""

import os
import os.path as osp
from .bases import BaseDataset
from glob import glob
import re


class Elephant(BaseDataset):
    """
    Elephant ReID Dataset
    
    Dataset statistics:
    # identities: 5 (Chandra, Indi, Fahra, Panang, Thai)
    # cameras: 4 (zag_elp_cam_016, 017, 018, 019)
    
    Dataset structure:
        reid_time_split/
            train/
                01_Chandra/
                02_Indi/
                03_Fahra/
                04_Panang/
                05_Thai/
            val/
                01_Chandra/
                ...
    """

    def __init__(self, 
                 cfg=None,
                 root='', 
                 verbose=True,
                 merge_all=True,  # If True, merge all subdirectories for training
                 val_samples=100,  # Number of random samples for validation
                 **kwargs):
        super(Elephant, self).__init__()

        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = self.root
        self.cfg = cfg
        self.merge_all = merge_all
        self.val_samples = val_samples
        
        # Camera mapping from filename
        self.camera_pattern = re.compile(r'(zag_elp_cam_\d{3})')
        self.camera2id = {
            'zag_elp_cam_016': 0,
            'zag_elp_cam_017': 1,
            'zag_elp_cam_018': 2,
            'zag_elp_cam_019': 3
        }
        
        # Individual mapping - PIDs must start from 0 for PyTorch CrossEntropyLoss
        self.name2id = {
            'Chandra': 0,
            'Indi': 1,
            'Fahra': 2,
            'Panang': 3,
            'Thai': 4,
            'Zali': 5
        }

        self._check_before_run()

        self.pid_container = {'train':{}, 'iid':{}}
        self.cid_container = set()
        self.year_container = set()
        
        # Check if we should merge all data or use train/val split
        train_dir = osp.join(self.dataset_dir, 'train')
        val_dir = osp.join(self.dataset_dir, 'val')
        
        if self.merge_all or not (osp.exists(train_dir) and osp.exists(val_dir)):
            if verbose:
                print("=> Merging all subdirectories for training data")
            # Process all subdirectories and create train/val split
            self.train, self.test_iid, self.train_year_list, self.test_iid_year_list, self.n_train_classes, self.n_test_iid_classes = \
                self.process_merged_dir(self.dataset_dir)
        else:
            if verbose:
                print("=> Using existing train/val split")
            self.train, self.train_year_list, self.n_train_classes = self.process_dir(train_dir, data_type='train')
            self.test_iid, self.test_iid_year_list, self.n_test_iid_classes = self.process_dir(val_dir, data_type='iid')
        
        # Not used, but required by the base dataloader interface
        self.test = self.test_iid
        self.test_year_list = self.test_iid_year_list
        self.n_test_classes = self.n_test_iid_classes
        self.val = self.test_iid
        self.val_year_list = self.test_iid_year_list
        self.n_val_classes = self.n_test_iid_classes

        if verbose:
            print("=> Elephant ReID data loaded")
            self.print_dataset_statistics()

    def _extract_camera_id(self, img_path):
        """Extract camera ID from image filename."""
        match = self.camera_pattern.search(img_path)
        if match:
            camera_name = match.group(1)
            return self.camera2id.get(camera_name, 0)
        return 0

    def _extract_year(self, img_path):
        """Extract year from image filename (format: cam_YYYYMMDD_...)."""
        match = re.search(r'_(\d{4})\d{4}_', img_path)
        if match:
            return int(match.group(1))
        return 2025  # default year

    def process_dir(self, dir_path, data_type=None):
        """
        Process directory containing elephant images organized by ID.
        
        Args:
            dir_path: Path to directory (train/val)
            data_type: Type of dataset (train/iid/ood/val)
            
        Returns:
            tuple: (processed_data, year_set, num_classes)
        """
        print(f'Processing elephant directory: {dir_path}')
        
        if not osp.exists(dir_path):
            print(f"Warning: {dir_path} does not exist, returning empty dataset")
            return [], set([2025]), 0
        
        processed_data = []
        pid_set = set()
        year_set = set()
        
        # Get all identity folders (e.g., 01_Chandra, 02_Indi, etc.)
        id_folders = sorted([d for d in os.listdir(dir_path) 
                           if osp.isdir(osp.join(dir_path, d))])
        
        for id_folder in id_folders:
            # Extract name from folder (e.g., "01_Chandra" -> "Chandra")
            pid_str, name = id_folder.split('_', 1)
            
            # Map name to 0-based PID
            if name not in self.name2id:
                print(f"Warning: Unknown elephant name '{name}', skipping folder {id_folder}")
                continue
            
            pid = self.name2id[name]
            
            # Store in container
            if data_type is not None:
                self.pid_container[data_type][pid] = name
            
            # Get all images in this identity folder
            id_path = osp.join(dir_path, id_folder)
            img_paths = glob(osp.join(id_path, '*.jpg')) + glob(osp.join(id_path, '*.png'))
            
            for img_path in img_paths:
                
                img_name = osp.basename(img_path)
                
                # Extract metadata from filename
                camid = self._extract_camera_id(img_name)
                year = self._extract_year(img_name)
                
                # Update containers
                self.cid_container.add(camid)
                year_set.add(year)
                pid_set.add(pid)
                
                # Create metadata
                meta_info = [
                    pid,
                    name,
                    camid,
                    -1,  # timestamp (not available)
                    1,   # is_unique
                    img_name,
                    -1   # sex (not available)
                ]
                
                processed_data.append((img_path, pid, camid, meta_info))
        
        self.year_container.update(year_set)
        
        return processed_data, year_set, len(pid_set)

    def process_merged_dir(self, root_dir):
        """
        Process all subdirectories, merge data, and create train/val split.
        
        Args:
            root_dir: Root directory containing all subdirectories
            
        Returns:
            tuple: (train_data, val_data, train_years, val_years, n_train_classes, n_val_classes)
        """
        print(f'Processing merged elephant directory: {root_dir}')
        
        all_data = []
        pid_set = set()
        year_set = set()
        
        # Get all subdirectories that contain identity folders
        # Exclude common non-data directories
        exclude_dirs = {'.git', '__pycache__', 'models', 'logs', 'checkpoints'}
        
        for subdir in os.listdir(root_dir):
            subdir_path = osp.join(root_dir, subdir)
            
            # Skip files and excluded directories
            if not osp.isdir(subdir_path) or subdir in exclude_dirs:
                continue
            
            # Check if this directory contains identity folders (e.g., 01_Chandra, 02_Indi)
            id_folders = [d for d in os.listdir(subdir_path) 
                         if osp.isdir(osp.join(subdir_path, d)) and '_' in d]
            
            if not id_folders:
                continue
                
            print(f'  Processing subdirectory: {subdir}')
            
            # Process each identity folder in this subdirectory
            for id_folder in id_folders:
                # Extract name from folder (e.g., "01_Chandra" -> "Chandra")
                parts = id_folder.split('_', 1)
                if len(parts) != 2:
                    continue
                    
                pid_str, name = parts
                
                # Map name to 0-based PID
                if name not in self.name2id:
                    print(f"Warning: Unknown elephant name '{name}', skipping folder {id_folder}")
                    continue
                
                pid = self.name2id[name]
                pid_set.add(pid)
                
                # Get all images in this identity folder
                id_path = osp.join(subdir_path, id_folder)
                
                # Recursively find all images (some might be in subdirectories)
                img_paths = []
                for root, dirs, files in os.walk(id_path):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                            img_paths.append(osp.join(root, file))
                
                for img_path in img_paths:
                    img_name = osp.basename(img_path)
                    
                    # Extract metadata from filename
                    camid = self._extract_camera_id(img_name)
                    year = self._extract_year(img_name)
                    
                    # Update containers
                    self.cid_container.add(camid)
                    year_set.add(year)
                    
                    # Create metadata
                    meta_info = [
                        pid,
                        name,
                        camid,
                        -1,  # timestamp (not available)
                        1,   # is_unique
                        img_name,
                        -1   # sex (not available)
                    ]
                    
                    all_data.append((img_path, pid, camid, meta_info))
        
        self.year_container.update(year_set)
        
        if not all_data:
            print("Warning: No data found, returning empty datasets")
            return [], [], set([2025]), set([2025]), 0, 0
        
        print(f'Total images found: {len(all_data)}')
        
        # Shuffle and split into train/val
        import random
        random.seed(42)  # For reproducibility
        random.shuffle(all_data)
        
        # Take val_samples for validation, rest for training
        val_size = min(self.val_samples, len(all_data) // 10)  # At most 10% for val
        val_data = all_data[:val_size]
        train_data = all_data[val_size:]
        
        print(f'Split: {len(train_data)} train, {len(val_data)} val')
        
        # Update pid containers
        for data in train_data:
            pid = data[1]
            name = data[3][1]
            self.pid_container['train'][pid] = name
        
        for data in val_data:
            pid = data[1]
            name = data[3][1]
            self.pid_container['iid'][pid] = name
        
        # Calculate unique classes in each split
        train_pids = set([d[1] for d in train_data])
        val_pids = set([d[1] for d in val_data])
        
        return train_data, val_data, year_set, year_set, len(train_pids), len(val_pids)

    def print_dataset_statistics(self):
        """Print dataset statistics."""
        print("Dataset statistics:")
        print("  --------------------------------------------------------------------")
        print("  subset    | year(min-max) | #ids| #cameras| #images ")
        print("  --------------------------------------------------------------------")

        if self.year_container:
            total_years = f"{min(self.year_container)}-{max(self.year_container)}"
        else:
            total_years = "N/A"
        print(f"  ==== Time period: {total_years} ==== ")
        
        def safe_year_range(year_list):
            if year_list:
                return f"{min(year_list)}-{max(year_list)}"
            return "N/A"
        
        train_years = safe_year_range(self.train_year_list)
        test_years = safe_year_range(self.test_iid_year_list)

        train_info = "  |  ".join(map(str, self.get_imagedata_info(self.train)))
        test_info = "  |  ".join(map(str, self.get_imagedata_info(self.test_iid)))

        print(f"  {'train':<9} | {train_years:<13} | {train_info}")
        print(f"  {'test':<9} | {test_years:<13} | {test_info}")
        print("  --------------------------------------------------------------------")
