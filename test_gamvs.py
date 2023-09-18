import argparse
import os
import os.path as osp
import logging
import time
import subprocess
import sys
import shutil

import random 
import numpy as np

import torch
import torch.nn as nn
import torch.optim

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import find_dataset_def
from models import find_model_def, find_loss_def
from utils import utils

from utils.logger import setup_logger
from utils.config import load_config
from utils.functions import *
from datasets.data_io import read_pfm, save_pfm, write_cam
import cv2
import matplotlib.pyplot as plt
from utils.xy_fusion import filter_depth

import pandas as pd

def test_model(model,
                data_loader,
                max_tree_depth,
                depth2stage,
                num_d,
                out_depths,
                output_dir,
                logger,
                prob_depth=None,
                color_mode=None
                ):
    avg_test_scalars = {"depth{}".format(i): DictAverageMeter() for i in out_depths}
    all_test_scalars_dict = {"depth{}".format(i): [] for i in out_depths}
    
    model.eval()
    total_iteration = data_loader.__len__()
    max_stage_id = max(depth2stage.values())
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for iteration, sample in enumerate(data_loader):
            torch.cuda.reset_peak_memory_stats()
            sample_cuda = tocuda(sample)

            for curr_tree_depth in range(1, max_tree_depth + 1):


                stage_id = depth2stage[str(curr_tree_depth)]
                outputs = model(data_batch=sample_cuda, stage_id=stage_id, is_train=False)

                preds = outputs["pred_depth"]
                exp_var = outputs["exp_var"]
                prob_map = outputs["pred_prob"]

                depth_est = preds
                next_depth_stage = depth2stage[str(curr_tree_depth + 1)]
                if next_depth_stage != stage_id:
                    next_num_d_tree = num_d[str(next_depth_stage)]
                    sample_cuda["depth"], sample_cuda["depth_min_max_updt"] = utils.update_depth(torch.unsqueeze(depth_est, 1),
                                                                            torch.unsqueeze(exp_var, 1),
                                                                            next_num_d_tree, scale_factor=2.0,
                                                                            mode='bilinear')

            forward_max_memory_allocated = torch.cuda.max_memory_allocated() / (1024.0 ** 2)

            curr_tree_depth = max_tree_depth
            scan_names = sample["scan_name"]
            img_ids = tensor2numpy(sample["img_id"])
            for batch_idx in range(len(scan_names)):
                scan_name = scan_names[batch_idx]
                img_id = img_ids[batch_idx]
                test_scalars = {"max_mem": forward_max_memory_allocated}

                avg_test_scalars["depth{}".format(curr_tree_depth)].update(test_scalars)
                test_scalars["scan_name"] = scan_name
                test_scalars["img_id"] = img_id
                all_test_scalars_dict["depth{}".format(curr_tree_depth)].append(test_scalars)

                logger.info(
                            " ".join(
                                [
                                    "Iter {}/{}".format(iteration, total_iteration),
                                    "scan_name {}".format(scan_name),
                                    "img_id {}".format(img_id),
                                    "max_mem: {:.0f}".format(forward_max_memory_allocated)
                                ]
                            )
                        )
            stage_id = depth2stage[str(curr_tree_depth)]
            depth_output_dir = osp.join(output_dir, "depth_{}".format(curr_tree_depth))
            os.makedirs(depth_output_dir, exist_ok=True)
            scan_names = sample["scan_name"]
            img_ids = tensor2numpy(sample["img_id"])
            img_cams = tensor2numpy(sample["ref_cams"][str(stage_id)])
            ref_imgs = tensor2numpy(sample["ref_imgs"][str(stage_id)])

            for batch_idx in range(len(scan_names)):
                scan_name = scan_names[batch_idx]
                scan_folder = osp.join(depth_output_dir, scan_name)
                prob_folder = osp.join(scan_folder, 'prob')

                if not osp.isdir(scan_folder):
                    os.makedirs(scan_folder, exist_ok=True)
                    os.makedirs(prob_folder, exist_ok=True)
                    logger.info("**** {} ****".format(scan_name))
                
                img_id = img_ids[batch_idx]
                img_cam = img_cams[batch_idx]
                depth_min_max = sample["depth_min_max"][batch_idx]
                mask = sample_cuda["masks"][str(stage_id)]


                depth_start = depth_min_max[0]
                depth_end = depth_min_max[1]

                init_depth_map = depth_est[batch_idx].cpu().numpy()


                init_depth_map_path = osp.join(scan_folder, "{:0>8}_init.pfm".format(img_id))
                pred_depth_img_path = osp.join(scan_folder, "{:0>8}_pred.png".format(img_id))

                save_pfm(init_depth_map_path, init_depth_map)
                plt.imsave(pred_depth_img_path, init_depth_map, cmap="Greys_r", vmin=depth_start, vmax=depth_end)


                prob_map= prob_map[batch_idx].cpu().numpy()
                prob_map_path = osp.join(scan_folder, "{:0>8}_prob.pfm".format(img_id))
                save_pfm(prob_map_path, prob_map)

                ref_image = ref_imgs[batch_idx]
                out_ref_image_path = osp.join(scan_folder, "{:0>8}.jpg".format(img_id))
                if color_mode == "BGR":
                    cv2.imwrite(out_ref_image_path, ref_image)
                else:
                    plt.imsave(out_ref_image_path, ref_image)
                out_init_cam_path = osp.join(scan_folder, "cam_{:0>8}_init.txt".format(img_id))
                write_cam(out_init_cam_path, img_cam)
        
    return avg_test_scalars, all_test_scalars_dict

def test(rank, cfg):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = cfg["master_port"]
    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=cfg["world_size"])

    synchronize()
    set_random_seed(cfg["random_seed"])

    logger = setup_logger("gamvs_test{}".format(str(rank)), cfg["log_dir"], prefix="test")
    output_dir = cfg["output_dir"]
    torch.cuda.set_device(rank)

    state_dict = None

    if os.path.exists(cfg["model_path"]):
        loadckpt = os.path.join(cfg["model_path"])
        logger.info("Loading checkpoint from {}".format(loadckpt))
        state_dict = torch.load(loadckpt, map_location=torch.device("cpu"))
    else:
        logger.info("No checkpoint found in {}.".format(cfg["model_path"]))
        return

    MVSDataset = find_dataset_def(cfg["dataset"])

    model_def = find_model_def(cfg["model_file"], cfg["model_name"])
    model = model_def(cfg).to(rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    loss_def = find_loss_def(cfg["model_file"], cfg.get("loss_name", cfg["model_name"] + "_loss"))
    
    if cfg["data"]["test"]["with_gt"]:
        model_loss = loss_def
    else:
        model_loss = None
    print('rank',rank)

    logger.info("Build model:\n{}".format(str(model)))
    model = DistributedDataParallel(model, device_ids=[rank])

    logger.info("Loading model ...")
    model.load_state_dict(state_dict['model'],strict=False)

    # stat = model.state_dict()
    # for k, v in stat.items():
    #     try:
    #         print(k, v.mean(), v.std(), v.max(), v.min())
    #     except:
    #         print(k, v)

    if cfg.get("img_mean") and cfg.get("img_std"):
        img_mean = cfg.get("img_mean")
        img_std = cfg.get("img_std")
        logger.info("Mannual img_mean {} and img_std {}\n".format(img_mean, img_std))
    else:
        logger.info("No img_mean and img_std\n")
        img_mean = None
        img_std = None
    
    test_dataset = MVSDataset(cfg["data"]["test"]["root_dir"], 
        cfg["data"]["test"]["listfile"], "test", 
        cfg["data"]["test"]["num_view"], 
        cfg["data"]["test"]["num_depth"], 
        cfg["data"]["test"]["interval_scale"],
        img_mean=img_mean, img_std=img_std, 
        self_norm=cfg["data"]["test"]["self_norm"],
        color_mode=cfg["data"]["test"]["color_mode"],
        with_gt=cfg["data"]["test"]["with_gt"], 
        max_h=cfg["data"]["test"]["max_h"], 
        max_w=cfg["data"]["test"]["max_w"], 
        base_image_size=cfg["data"]["test"]["base_image_size"],
        use_cam_max=cfg["data"]["test"].get("use_cam_max", False),
        stage_info=cfg["model"].get("stage_info", None),
        depth_num=cfg["data"]["test"].get("init_depth_num", 64))

    test_sampler = DistributedSampler(test_dataset, num_replicas=cfg["world_size"], rank=rank, shuffle=False)
    test_data_loader = DataLoader(test_dataset, cfg["test"]["batch_size"], sampler=test_sampler, num_workers=cfg["data"]["num_workers"])

    if rank == 0:
        tensorboard_logger = SummaryWriter(cfg["log_dir"])
    else:
        tensorboard_logger = None
    
    # test
    logger.info("Start testing ...")

    avg_test_scalars, all_test_scalars_dict = test_model(model,
                                                         data_loader=test_data_loader,
                                                         max_tree_depth=cfg["max_depth"],
                                                         depth2stage=cfg["model"]["stage_info"]["depth2stage"],
                                                         num_d=cfg["model"]["stage_info"]["num_d"],
                                                         out_depths=cfg["data"]["test"]["out_depths"],
                                                         output_dir=osp.join(cfg["output_dir"],"test_output"),
                                                         logger=logging.getLogger("gamvs_test{}".format(str(rank)) + ".test"),
                                                         prob_depth=cfg["data"]["test"].get("prob_depth",None),
                                                         color_mode=cfg["data"]["test"]["color_mode"],
                                                         )



    world_size = get_world_size()
    gathered_scalars = [None for i in range(world_size)]
    dist.all_gather_object(gathered_scalars, all_test_scalars_dict)

    if rank == 0:
        from itertools import chain
        out_depths = cfg["data"]["test"]["out_depths"]
        max_tree_depth = cfg["max_depth"]
        if out_depths == "all":
            out_depths = list(range(1, max_tree_depth + 1))
        for curr_tree_depth in out_depths:
            gathered_scalars_depth = [i["depth{}".format(curr_tree_depth)] for i in gathered_scalars]
            gathered_scalars_depth = list(chain.from_iterable(gathered_scalars_depth))
            test_df = pd.DataFrame(gathered_scalars_depth)
            test_df.to_csv(osp.join(cfg["log_dir"], "test_info_depth{}.csv".format(curr_tree_depth)), index=False)
            test_df_mean = test_df.mean(numeric_only=True).to_frame().T
            test_df_mean.to_csv(osp.join(cfg["log_dir"], "test_info_depth{}_mean.csv".format(curr_tree_depth)), index=False)
            print("depth {}".format(curr_tree_depth), test_df_mean)

def xy_filter(rank, cfg):
    if cfg["fusion"]["xy_filter"].get("nprocs", None) is not None:
        scans = cfg["fusion"]["xy_filter"]["scans"][rank]
    else:
        with open(cfg["data"]["test"]["listfile"]) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

    data_folder = osp.join(cfg["output_dir"], "test_output")
    output_dir = cfg["fusion"]["xy_filter"].get("output_dir", cfg["output_dir"])
    save_depths = cfg["data"]["test"]["save_depths"]
    for curr_tree_depth in save_depths:
        depth_data_folder = osp.join(data_folder, "depth_{}".format(curr_tree_depth))
        for para_id in range(len(cfg["fusion"]["xy_filter"]["prob_threshold"])):
            #prob_threshold = cfg["fusion"]["xy_filter"]["prob_threshold"][para_id]
            prob_threshold = 0
            num_consistent = cfg["fusion"]["xy_filter"]["num_consistent"][para_id]
            img_dist_thresh = cfg["fusion"]["xy_filter"]["img_dist_thresh"][para_id] if cfg["fusion"]["xy_filter"].get("img_dist_thresh", None) is not None else 1.0
            depth_thresh = cfg["fusion"]["xy_filter"]["depth_thresh"][para_id] if cfg["fusion"]["xy_filter"].get("depth_thresh", None) is not None else 0.01

            point_dir = os.path.join(output_dir, str(para_id), "depth_{}".format(curr_tree_depth), "xy_filter", "collected_points_{}".format(prob_threshold)) \
                if prob_threshold != 0.0 else os.path.join(output_dir, str(para_id), "depth_{}".format(curr_tree_depth), "xy_filter", "collected_points")
            os.makedirs(point_dir, exist_ok=True)
            for scan in scans:
                scan_folder = os.path.join(depth_data_folder, scan)
                scan_data_folder = os.path.join(cfg["data"]["test"]["root_dir"], scan)
                if cfg["fusion"]["xy_filter"]["global_pair"]:
                    pair_path = osp.join(cfg["data"]["test"]["root_dir"], "Cameras/pair.txt")
                else:
                    pair_path = osp.join(scan_data_folder, "pair.txt")
                data_name = cfg.get("data_name", "dtu")
                if "dtu" in data_name:
                    scan_id = int(scan[4:])
                    ply_name = osp.join(point_dir, "gamvs_{:03d}_l3.ply".format(scan_id))
                elif "tanks" in data_name:
                    ply_name = osp.join(point_dir, "{}.ply".format(scan))
                filter_depth(scan_folder=scan_folder, pair_path=pair_path, plyfilename=ply_name,
                    prob_threshold=prob_threshold, num_consistent=num_consistent, img_dist_thresh=img_dist_thresh, depth_thresh=depth_thresh)

def xy_filter_per(rank, cfg):
    if cfg["fusion"]["xy_filter_per"].get("nprocs", None) is not None:
        scans = cfg["fusion"]["xy_filter_per"]["scans"][rank]
    else:
        with open(cfg["data"]["test"]["listfile"]) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

    data_folder = osp.join(cfg["output_dir"], "test_output")
    output_dir = cfg["fusion"]["xy_filter_per"].get("output_dir", cfg["output_dir"])
    save_depths = cfg["data"]["test"]["save_depths"]
    for curr_tree_depth in save_depths:
        depth_data_folder = osp.join(data_folder, "depth_{}".format(curr_tree_depth))
        for para_id in range(cfg["fusion"]["xy_filter_per"]["para_num"]):
            if cfg["fusion"]["xy_filter_per"].get("para_tag", None) is not None:
                para_tag = cfg["fusion"]["xy_filter_per"].get("para_tag")[para_id]
            else:
                para_tag = para_id
            for scan in scans:
                paras = cfg["fusion"]["xy_filter_per"][scan]
                prob_threshold = paras["prob_threshold"][para_tag]
                #prob_threshold =0
                point_dir = os.path.join(output_dir, str(para_tag), "depth_{}".format(curr_tree_depth), "xy_filter_per", "collected_points_{}".format(prob_threshold)) \
                    if prob_threshold is not None else os.path.join(output_dir, str(para_tag), "depth_{}".format(curr_tree_depth), "xy_filter_per", "collected_points")
                num_consistent = paras["num_consistent"][para_tag]
                img_dist_thresh = paras["img_dist_thresh"][para_tag] if paras.get("img_dist_thresh", None) is not None else 1.0
                depth_thresh = paras["depth_thresh"][para_tag] if paras.get("depth_thresh", None) is not None else 0.01

                os.makedirs(point_dir, exist_ok=True)
                scan_folder = os.path.join(depth_data_folder, scan)
                scan_data_folder = os.path.join(cfg["data"]["test"]["root_dir"], scan)
                if cfg["fusion"]["xy_filter_per"]["global_pair"]:
                    pair_path = osp.join(cfg["data"]["test"]["root_dir"], "Cameras/pair.txt")
                else:
                    pair_path = osp.join(scan_data_folder, "pair.txt")
                data_name = cfg.get("data_name", "dtu")
                if "dtu" in data_name:
                    scan_id = int(scan[4:])
                    ply_name = osp.join(point_dir, "gamvs_{:03d}_l3.ply".format(scan_id))
                elif "tanks" in data_name:
                    ply_name = osp.join(point_dir, "{}.ply".format(scan))
                filter_depth(scan_folder=scan_folder, pair_path=pair_path, plyfilename=ply_name,
                    prob_threshold=prob_threshold, num_consistent=num_consistent, img_dist_thresh=img_dist_thresh, depth_thresh=depth_thresh)

def main():
    parser = argparse.ArgumentParser(description="PyTorch Testing")
    parser.add_argument("--cfg", dest="config_file", default="configs/test_dtu_ucs.yaml", metavar="FILE", help="path to config file", type=str)
    args = parser.parse_args()
    cfg = load_config(args.config_file)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["true_gpu"]
    output_dir = cfg["output_dir"]

    if not osp.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    num_gpus = len(cfg["gpu"])

    timestamp = time.strftime(".%m_%d_%H_%M_%S")
    log_dir = os.path.join(output_dir, "log{}".format(timestamp))

    if not osp.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    cfg["config_file"] = args.config_file
    cfg["log_dir"] = log_dir

    # copy config file to log_dir
    shutil.copy(args.config_file, log_dir)

    logger = setup_logger("gamvs", log_dir, prefix="test")

    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(str(sys.argv))
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    world_size = num_gpus
    cfg["world_size"] = world_size
    
    if not cfg.get("no_testing", False):
        mp.spawn(test,
            args=(cfg,),
            nprocs=world_size,
            join=True)

    if not cfg.get("no_fusion", False):
        if cfg["fusion"]["name"] == "gipuma_filter":
            logger.info("Gipuma filtering ...")
            gipuma_filter(cfg)
        elif cfg["fusion"]["name"] == "gipuma_filter_per":
            logger.info("Gipuma filtering ...")
            gipuma_filter_per(cfg)
        elif cfg["fusion"]["name"] == "xy_filter":
            logger.info("xy filtering ...")
            if cfg["fusion"]["xy_filter"].get("nprocs", None) is None:
                xy_filter(-1, cfg)
            else:
                with open(cfg["data"]["test"]["listfile"]) as f:
                    scans = f.readlines()
                    scans = [line.rstrip() for line in scans]
                cfg["fusion"]["xy_filter"]["scans"] = chunk_list(scans, cfg["fusion"]["xy_filter"]["nprocs"])
                mp.spawn(xy_filter,
                    args=(cfg,),
                    nprocs=cfg["fusion"]["xy_filter"]["nprocs"],
                    join=True)
        elif cfg["fusion"]["name"] == "xy_filter_per":
            logger.info("xy filtering ...")
            if cfg["fusion"]["xy_filter_per"].get("nprocs", None) is None:
                xy_filter_per(-1, cfg)
            else:
                with open(cfg["data"]["test"]["listfile"]) as f:
                    scans = f.readlines()
                    scans = [line.rstrip() for line in scans]
                cfg["fusion"]["xy_filter_per"]["scans"] = chunk_list(scans, cfg["fusion"]["xy_filter_per"]["nprocs"])
                mp.spawn(xy_filter_per,
                    args=(cfg,),
                    nprocs=cfg["fusion"]["xy_filter_per"]["nprocs"],
                    join=True)

    logger.info("All Done")


if __name__ == "__main__":
    main()
