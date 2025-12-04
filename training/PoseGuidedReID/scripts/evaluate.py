import numpy as np
import os
import torch
import json


def evaluate_reid_performance(output_dir):


    torch.cuda.empty_cache()
    query_feature_path = f'{output_dir}/pred_features/val_iid/pytorch_result_e60.npz'
        
    gallery_path = f'{output_dir}/pred_features/train_iid/pytorch_result_e60.npz'
    if os.path.exists(gallery_path) == False:
        gallery_path = f'{output_dir}/pred_features/train_iid/pytorch_result_elast.npz'

    mAP_b, mAP_20_b, r1_b, r5_b, r10_b, r20_b, sorted_qg_matrix_b, distmat_b, all_AP, all_mean_AP = get_results(query_feature_path, gallery_path=gallery_path, filter_date=split_type!='val_iid')
    
    print(f'ReID results: mAP: {mAP_b:.1%}, mAP@20: {mAP_20_b:.1%}, R1: {r1_b:.1%}, R5: {r5_b:.1%}, R10: {r10_b:.1%}, R20: {r20_b:.1%}')