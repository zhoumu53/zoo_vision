from project.utils.logger import setup_logger
from project.datasets import make_dataloader, make_base_dataloader
from project.model import make_model
from project.solver import make_optimizer
from project.solver.scheduler_factory import create_scheduler
from project.losses import make_loss

import random
import torch
import torch.optim as optim
import numpy as np
import os
import argparse
from project.config import cfg
import sys
import glob
from project.utils.tools import load_model, setup_ddp_training, fuse_all_conv_bn
from project.processor.processor import train_model, do_inference
from project.utils.wandb_tools import WandbLogger
import shutil
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

def main():

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--notes", default="training...", type=str)
    parser.add_argument("--out_dim", default=512, type=int)
    parser.add_argument("--do_inference", action='store_true', help='do inference on the dataset (train/test_iid/test_ood)')
    parser.add_argument("--do_evaluation", action='store_true', help='evaluate the model on the dataset (ood/iid vs gallery)')
    parser.add_argument("--do_prediction", action='store_true', help='do inference on the dataset (train/test_iid/test_ood)')
    parser.add_argument("--do_training", action='store_true', help='train the model')
    parser.add_argument("--data_type", default=None, type=str)
    parser.add_argument("--model_name", default="", type=str)
    parser.add_argument("--wb_run_id", default=None, type=str)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    local_rank = args.local_rank

    output_dir = cfg.OUTPUT_DIR
    
    # Extract project_name and exp_name from output_dir path
    output_parts = output_dir.rstrip('/').split('/')
    project_name = output_parts[-2] if len(output_parts) >= 2 else cfg.DATASETS.NAMES
    exp_name = output_parts[-1]

    print("Project Name:", project_name)
    print("Experiment Name:", exp_name)

    wb_run = None
    wb_run_id = args.wb_run_id

    print("WandB Run ID:", wb_run_id)
    if local_rank == 0:

        wb = WandbLogger(cfg, project_name=project_name)
        wb_config = wb.build_config(cfg, args.config_file, exp_name, project_name, output_dir)
        
        # Check if wb_run_id is None or empty string
        if not wb_run_id:
            print(f"Initializing new WandB run with name: {exp_name}")
            wb_run = wb.init_wandb(project_name, exp_name, wb_config, notes=args.notes)
            wb_run_id = wb_run.id
        else:
            print(f"Resuming WandB run with ID: {wb_run_id}")   
            wb_run = wb.resume(run_id=wb_run_id, resume='allow')
        
    if_train = args.do_training
    
    logger = setup_logger("project", output_dir, if_train=if_train)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)
    logger.info("Running with config:\n{}".format(cfg))
    # device = 'cuda'

    # if os.path.exists(output_dir):
    #     files = glob.glob(output_dir + '/*.pth')
    #     if len(files) > 0:
    #         print('You already trained the model. Now loading the model from the trained ckp.')
    #         # sys.exit("You already trained the model. Please change another output_dir for logging.")
    # else:
    #     os.makedirs(output_dir)

    if cfg.MODEL.DIST_TRAIN:
        local_rank, world_size, device = setup_ddp_training()
    else:
        device_ids = cfg.MODEL.DEVICE_ID[0]
        device = torch.device("cuda:{}".format(device_ids))

    if cfg.DATASETS.NAMES == 'bear' or cfg.DATASETS.NAMES == 'macaque' or cfg.DATASETS.NAMES == 'elephant':
        train_loader, train_loader_normal, \
            test_iid_loader, test_ood_loader, val_iid_loader, \
                all_classes, camera_num, \
                    train_num_classes, test_iid_num_classes, test_ood_num_classes, val_iid_num_classes= make_dataloader(cfg)
    elif cfg.DATASETS.NAMES == 'base':
        train_loader, train_num_classes, camera_num = make_base_dataloader(cfg, 
                                                                           is_train=args.do_training, 
                                                                           root_dir=cfg.DATASETS.ROOT_DIR,
                                                                           img_dir=cfg.DATASETS.IMG_DIR)
    

    model = make_model(cfg, 
                       num_classes=train_num_classes, 
                       logger=logger, 
                       return_feature=True, 
                       device=device, 
                       camera_num=camera_num)

    criterion, criterion_triplet = make_loss()
    optimizer = make_optimizer(cfg, model)
    scheduler = create_scheduler(cfg, optimizer)

    if args.do_training:
        train_model(cfg, 
                    train_loader,
                    model, 
                    criterion, 
                    criterion_triplet,
                    optimizer, 
                    scheduler, 
                    local_rank=local_rank,
                    device=device,
                    wb_run=wb_run)
    

    ################### INFERENCE & SAVE IMAGE FEATURES FOR EVALUATION #####################

    if args.do_inference:
        
        ### run inference on val_iid and val_ood (save feature only)
        model_weights_path = cfg.TEST.WEIGHT
        if model_weights_path is None or model_weights_path == '':
            model_weights_path = os.path.join(cfg.OUTPUT_DIR, 'net_last.pth')
            
        if cfg.DATASETS.NAMES == 'bear' or cfg.DATASETS.NAMES == 'macaque' or cfg.DATASETS.NAMES == 'elephant':
            
            is_swin = True if 'swin' in cfg.MODEL.TYPE else False

            data_type = 'train_iid'
            model_structure = make_model(cfg, 
                                         num_classes=train_num_classes, 
                                         logger=logger, 
                                         return_feature=True, 
                                         device=device,
                                         camera_num=camera_num)
            model = load_model(model_structure, 
                              model_weights_path, 
                              logger=logger, 
                              remove_fc=True, 
                              local_rank=local_rank, 
                              is_swin=is_swin)
            model.eval()
            model = fuse_all_conv_bn(model).to(device)
            train_feature_path = do_inference(cfg, 
                                                model, 
                                                train_loader_normal, 
                                                data_type=data_type, 
                                                out_dir=cfg.OUTPUT_DIR, 
                                                local_rank=local_rank, 
                                                device=device)
            
            data_type = 'val_iid'
            model_structure = make_model(cfg, 
                                         num_classes=val_iid_num_classes, 
                                         logger=logger, 
                                         return_feature=True, 
                                         device=device, 
                                         camera_num=camera_num)
            model = load_model(model_structure, 
                              model_weights_path, 
                              logger=logger, 
                              remove_fc=True, 
                              local_rank=local_rank, 
                              is_swin=is_swin)
            model.eval()
            model = fuse_all_conv_bn(model).to(device)
            iid_feature_path = do_inference(cfg, 
                                            model, 
                                            val_iid_loader, 
                                            data_type=data_type, 
                                            out_dir=cfg.OUTPUT_DIR, 
                                            local_rank=local_rank, 
                                            device=device)
            
            
            if cfg.DATASETS.NAMES == 'elephant':
                return 
            
            data_type = 'test_iid'
            model_structure = make_model(cfg, 
                                         num_classes=test_iid_num_classes, 
                                         logger=logger, 
                                         return_feature=True, 
                                         device=device, 
                                         camera_num=camera_num)
            model = load_model(model_structure, 
                              model_weights_path, 
                              logger=logger, 
                              remove_fc=True, local_rank=local_rank, is_swin=is_swin)
            model.eval()
            model = fuse_all_conv_bn(model).to(device)
            iid_feature_path = do_inference(cfg, 
                                            model, 
                                            test_iid_loader, 
                                            data_type=data_type, 
                                            out_dir=cfg.OUTPUT_DIR, 
                                            local_rank=local_rank, 
                                            device=device,
                                            camera_num=camera_num)

            data_type = 'test_ood'
            model_structure = make_model(cfg, 
                                         num_classes=test_ood_num_classes, 
                                         logger=logger, 
                                         return_feature=True, 
                                         device=device,
                                         camera_num=camera_num)
            model = load_model(model_structure, 
                              model_weights_path, 
                              logger=logger, remove_fc=True, local_rank=local_rank, is_swin=is_swin)
            model.eval()
            model = fuse_all_conv_bn(model).to(device)
            ood_feature_path = do_inference(cfg, 
                                            model, 
                                            test_ood_loader, 
                                            data_type=data_type, 
                                            out_dir=cfg.OUTPUT_DIR, 
                                            local_rank=local_rank, 
                                            device=device)
            
        elif cfg.DATASETS.NAMES == 'base':
            
            ## run inference on train dataset
            data_type = 'train_iid'
            model_structure = make_model(cfg, num_classes=train_num_classes, logger=logger, return_feature=True, device=device, camera_num=camera_num)
            model = load_model(model_structure, model_weights_path, logger=logger, remove_fc=True, local_rank=local_rank)
            model.eval()
            model = fuse_all_conv_bn(model).to(device)
            train_feature_path = do_inference(cfg, 
                                                model, 
                                                train_loader, 
                                                data_type=data_type, 
                                                out_dir=cfg.OUTPUT_DIR, 
                                                local_rank=local_rank, 
                                                device=device)
            
            data_type = 'test_katmai' if args.data_type == 'katmai' else 'test'
            print("cfg.DATASETS.TEST_ROOT_DIR, cfg.DATASETS.TEST_IMG_DIR", cfg.DATASETS.TEST_ROOT_DIR, cfg.DATASETS.TEST_IMG_DIR)
            dataloader, n_classes, camera_num = make_base_dataloader(cfg, 
                                                         is_train=False, 
                                                         root_dir=cfg.DATASETS.TEST_ROOT_DIR, 
                                                         img_dir=cfg.DATASETS.TEST_IMG_DIR
                                                         )
            model_structure = make_model(cfg, num_classes=n_classes, logger=logger, return_feature=True, device=device, camera_num=camera_num)
            model = load_model(model_structure, model_weights_path, logger=logger, remove_fc=True, local_rank=local_rank)
            model.eval()
            model = fuse_all_conv_bn(model).to(device)
            test_feature_path = do_inference(cfg, 
                                                model, 
                                                dataloader, 
                                                data_type=data_type, 
                                                out_dir=cfg.OUTPUT_DIR, 
                                                local_rank=local_rank, 
                                                device=device)
            

if __name__ == '__main__':
    main()