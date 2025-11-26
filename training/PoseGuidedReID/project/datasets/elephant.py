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
                 **kwargs):
        super(Elephant, self).__init__()

        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = self.root
        self.cfg = cfg
        
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.val_dir = osp.join(self.dataset_dir, 'val')
        
        # Simple setup: train for training, val for testing
        self.test_iid_dir = self.val_dir
        
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
            'Thai': 4
        }

        self._check_before_run()

        self.pid_container = {'train':{}, 'iid':{}}
        self.cid_container = set()
        self.year_container = set()
        
        self.train, self.train_year_list, self.n_train_classes = self.process_dir(self.train_dir, data_type='train')
        self.test_iid, self.test_iid_year_list, self.n_test_iid_classes = self.process_dir(self.test_iid_dir, data_type='iid')
        
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
