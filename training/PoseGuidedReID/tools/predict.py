from project.utils.logger import setup_logger
from project.datasets import make_base_dataloader
from project.models import make_model

import random
import torch
import numpy as np
import os
import argparse
from project.config import cfg
from project.utils.tools import load_model, setup_ddp_training, fuse_all_conv_bn
from project.processor.processor import do_prediction
import wandb
wandb.login()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--out_dim", default=512, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    local_rank = args.local_rank

    output_dir = cfg.OUTPUT_DIR
    
    logger = setup_logger("project", output_dir, if_train=False)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        local_rank, world_size, device = setup_ddp_training()
    else:
        device_ids = cfg.MODEL.DEVICE_ID[0]
        device = torch.device("cuda:{}".format(device_ids))

    root_dir = cfg.DATASETS.TEST_ROOT_DIR if 'train' not in args.data_type else cfg.DATASETS.ROOT_DIR
    img_dir = cfg.DATASETS.TEST_IMG_DIR if 'train' not in args.data_type else cfg.DATASETS.IMG_DIR

    if cfg.DATASETS.NAMES == 'base':
        dataloader, num_classes, camera_num = make_base_dataloader(cfg, 
                                                                   is_train=args.do_training,
                                                                   root_dir=root_dir,
                                                                   img_dir=img_dir
                                                                   )
    else:
        raise ValueError(f"Please use the make_base_dataloader for prediction")

    model_weights_path = cfg.TEST.WEIGHT
    if model_weights_path is None or model_weights_path == '':
        model_weights_path = os.path.join(cfg.OUTPUT_DIR, 'net_best.pth')
        
    model_structure = make_model(cfg, num_classes=num_classes, logger=logger, return_feature=True, device=device)
    model = load_model(model_structure, model_weights_path, logger=logger, remove_fc=False, local_rank=local_rank)
    model.eval()
    model = fuse_all_conv_bn(model).to(device)
    predictions = do_prediction(cfg, 
                                model, 
                                dataloader, 
                                data_type='val', 
                                out_dir=cfg.OUTPUT_DIR, 
                                local_rank=local_rank, 
                                device=device)