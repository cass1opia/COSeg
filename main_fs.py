"""
COSeg Training script.

"""

import os
import time
import random
import numpy as np
import argparse
import shutil
import copy
import csv
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
from functools import partial

from util import config
from util.s3dis_fs import S3DIS_FS, S3DIS_FS_TEST, S3DIS_FSForVIS
from util.scannet_v2_fs import Scannetv2_FS, Scannetv2_FS_TEST
from util.outdoor_fs import Outdoor_FS, Outdoor_FS_TEST, Outdoor_FSForVIS
from util.common_util import (
    AverageMeter,
    find_free_port,
)
from util.data_util import (
    collate_fn_limit_fs,
    collate_fn_limit_fs_train,
)
from util import transform
from util.logger import get_logger

from util.lr import MultiStepWithWarmup, PolyLR
from util.common_util import load_pretrain_checkpoint, evaluate_metric
from model.coseg import COSeg
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re


def get_parser():
    parser = argparse.ArgumentParser(
        description="PyTorch Point Cloud Semantic Segmentation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/essen_outdoor_COSeg_fs.yaml",
        help="config file",
    )
    parser.add_argument(
        "opts",
        help="see config/essen_outdoor_COSeg_fs.yaml for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (
        args.multiprocessing_distributed
        and args.rank % args.ngpus_per_node == 0
    )


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(x) for x in args.train_gpu
    )
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # import torch.backends.mkldnn
    # ackends.mkldnn.enabled = False
    # os.environ["LRU_CACHE_CAPACITY"] = "1"
    # cudnn.deterministic = True
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(
            main_worker,
            nprocs=args.ngpus_per_node,
            args=(args.ngpus_per_node, args),
        )
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args, best_iou
    args, best_iou = argss, 0
    if args.distributed:
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        torch.cuda.set_device(gpu)
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    if main_process():
        global logger, writer
        logger = get_logger(args.save_path)
        writer = SummaryWriter(args.save_path)
        if args.vis:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            # Bestimme Split aus dem CV-Fold (s0/s1/...)
            split_tag = ""
            if hasattr(args, "cvfold"):
                try:
                    split_tag = f"s{int(args.cvfold)}"
                except Exception:
                    split_tag = f"s{args.cvfold}"
            run_name = f"{args.data_name}_{split_tag}_{args.n_way}way_{args.k_shot}shot_{timestamp}"
            
            wandb.init(
                project="COSeg",
                name=run_name,
                config=args,
            )

    # get model
    model = COSeg(args)

    # set optimizer
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.base_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "AdamW":
        defaults = {}
        defaults["lr"] = args.base_lr

        params = []
        memo = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(
                recurse=False
            ):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                hyperparams = copy.copy(defaults)

                params.append({"params": [value], **hyperparams})

        optimizer = torch.optim.AdamW(
            params, lr=args.base_lr, weight_decay=args.weight_decay
        )

    if main_process():
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info(model)
        logger.info(
            "#Model parameters: {}".format(
                sum([x.nelement() for x in model.parameters()])
            )
        )
        if args.get("max_grad_norm", None):
            logger.info("args.max_grad_norm = {}".format(args.max_grad_norm))

    if args.distributed:
        args.workers = int(
            (args.workers + ngpus_per_node - 1) / ngpus_per_node
        )
        if args.sync_bn:
            if main_process():
                logger.info("use SyncBN")
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[gpu], find_unused_parameters=True
        )
    else:
        model = model.cuda()

    if args.pretrain_backbone:
        # load pretrained backbone
        model = load_pretrain_checkpoint(model, args.pretrain_backbone, gpu)

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            pretrained_dict = checkpoint["state_dict"]
            if not isinstance(
                model, torch.nn.parallel.DistributedDataParallel
            ):
                pretrained_dict = {
                    k.replace("module.", ""): v
                    for k, v in pretrained_dict.items()
                }

            model.load_state_dict(pretrained_dict)
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(
                args.resume, map_location=lambda storage, loc: storage.cuda()
            )
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler_state_dict = checkpoint["scheduler"]
            best_iou = checkpoint["best_iou"]
            if main_process():
                logger.info(
                    "=> loaded checkpoint '{}' (epoch {})".format(
                        args.resume, checkpoint["epoch"]
                    )
                )
        else:
            if main_process():
                logger.info(
                    "=> no checkpoint found at '{}'".format(args.resume)
                )

    val_transform = None
    if args.data_name == "s3dis":
        if args.forvis:
            val_data = S3DIS_FSForVIS(
                split="test",
                data_root=args.data_root,
                voxel_size=args.voxel_size,
                voxel_max=args.voxel_max,
                transform=val_transform,
                cvfold=args.cvfold,
                num_episode=args.num_episode,
                n_way=args.n_way,
                k_shot=args.k_shot,
                n_queries=args.n_queries,
                target_class=args.target_class,
            )
        else:
            val_data = S3DIS_FS_TEST(
                split=args.eval_split,
                data_root=args.data_root,
                voxel_size=args.voxel_size,
                voxel_max=args.voxel_max,
                transform=val_transform,
                cvfold=args.cvfold,
                num_episode=args.num_episode,
                n_way=args.n_way,
                k_shot=args.k_shot,
                n_queries=args.n_queries,
                num_episode_per_comb=args.num_episode_per_comb,
            )
        valid_calsses = list(val_data.classes)

    elif args.data_name == "scannetv2":
        val_data = Scannetv2_FS_TEST(
            split=args.eval_split,
            data_root=args.data_root,
            voxel_size=args.voxel_size,
            voxel_max=args.voxel_max,
            transform=val_transform,
            cvfold=args.cvfold,
            num_episode=args.num_episode,
            n_way=args.n_way,
            k_shot=args.k_shot,
            n_queries=args.n_queries,
            num_episode_per_comb=args.num_episode_per_comb,
        )
        valid_calsses = list(val_data.classes)
    elif args.data_name == "outdoor":
        if args.forvis:
            val_data = Outdoor_FSForVIS(
                split="test",
                data_root=args.data_root,
                voxel_size=args.voxel_size,
                voxel_max=args.voxel_max,
                transform=val_transform,
                cvfold=args.cvfold,
                num_episode=args.num_episode,
                n_way=args.n_way,
                k_shot=args.k_shot,
                n_queries=args.n_queries,
                target_class=args.target_class,
            )
        else:
            val_data = Outdoor_FS_TEST(
                split=args.eval_split,
                data_root=args.data_root,
                voxel_size=args.voxel_size,
                voxel_max=args.voxel_max,
                transform=val_transform,
                cvfold=args.cvfold,
                num_episode=args.num_episode,
                n_way=args.n_way,
                k_shot=args.k_shot,
                n_queries=args.n_queries,
                num_episode_per_comb=args.num_episode_per_comb,
            )
        valid_calsses = list(val_data.classes)
    else:
        raise ValueError(
            "The dataset {} is not supported.".format(args.data_name)
        )

    if not args.forvis:
        # main process firstly call, since it will construct the dataset if not exist
        # and avoid conflicts from other processes
        if main_process():
            logger.info(
                "The main process prepares test data while other processes wait..."
            )
            val_data.prepare_test_data()

        if args.distributed:
            dist.barrier()
            val_data.prepare_test_data()

    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
    else:
        val_sampler = None
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
        collate_fn=partial(
            collate_fn_limit_fs, include_scene_names=args.forvis
        ),
    )

    if args.test:
        validate(val_loader, model, list(val_data.classes))
        if main_process():
            writer.close()
            logger.info("==>Test done!")
        return

    if args.data_name == "s3dis":
        train_transform = None
        if args.aug:
            jitter_sigma = args.get("jitter_sigma", 0.01)
            jitter_clip = args.get("jitter_clip", 0.05)
            if main_process():
                logger.info("augmentation all")
                logger.info(
                    "jitter_sigma: {}, jitter_clip: {}".format(
                        jitter_sigma, jitter_clip
                    )
                )
            train_transform = transform.Compose(
                [
                    transform.RandomRotate(
                        along_z=args.get("rotate_along_z", True)
                    ),
                    transform.RandomScale(
                        scale_low=args.get("scale_low", 0.8),
                        scale_high=args.get("scale_high", 1.2),
                    ),
                    transform.RandomJitter(
                        sigma=jitter_sigma, clip=jitter_clip
                    ),
                    transform.RandomDropColor(
                        color_augment=args.get("color_augment", 0.0)
                    ),
                ]
            )
        train_data = S3DIS_FS(
            split="train",
            data_root=args.data_root,
            voxel_size=args.voxel_size,
            voxel_max=args.voxel_max,
            transform=train_transform,
            shuffle_index=True,
            loop=args.loop,
            cvfold=args.cvfold,
            num_episode=args.num_episode,
            n_way=args.n_way,
            k_shot=args.k_shot,
            n_queries=args.n_queries,
        )
        train_calsses = list(train_data.classes)
    elif args.data_name == "scannetv2":
        train_transform = None
        if args.aug:
            if main_process():
                logger.info("use Augmentation")
            train_transform = transform.Compose(
                [
                    transform.RandomRotate(
                        along_z=args.get("rotate_along_z", True)
                    ),
                    transform.RandomScale(
                        scale_low=args.get("scale_low", 0.8),
                        scale_high=args.get("scale_high", 1.2),
                    ),
                    transform.RandomDropColor(
                        color_augment=args.get("color_augment", 0.0)
                    ),
                ]
            )

        train_data = Scannetv2_FS(
            split="train",
            data_root=args.data_root,
            voxel_size=args.voxel_size,
            voxel_max=args.voxel_max,
            transform=train_transform,
            shuffle_index=True,
            loop=args.loop,
            cvfold=args.cvfold,
            num_episode=args.num_episode,
            n_way=args.n_way,
            k_shot=args.k_shot,
            n_queries=args.n_queries,
        )
        train_calsses = list(train_data.classes)
    elif args.data_name == "outdoor":
        train_transform = None
        if args.aug:
            jitter_sigma = args.get("jitter_sigma", 0.01)
            jitter_clip = args.get("jitter_clip", 0.05)
            if main_process():
                logger.info("augmentation all")
                logger.info(
                    "jitter_sigma: {}, jitter_clip: {}".format(
                        jitter_sigma, jitter_clip
                    )
                )
            train_transform = transform.Compose(
                [
                    transform.RandomRotate(
                        along_z=args.get("rotate_along_z", True)
                    ),
                    transform.RandomScale(
                        scale_low=args.get("scale_low", 0.8),
                        scale_high=args.get("scale_high", 1.2),
                    ),
                    transform.RandomJitter(
                        sigma=jitter_sigma, clip=jitter_clip
                    ),
                    transform.RandomDropColor(
                        color_augment=args.get("color_augment", 0.0)
                    ),
                ]
            )
        train_data = Outdoor_FS(
            split="train",
            data_root=args.data_root,
            voxel_size=args.voxel_size,
            voxel_max=args.voxel_max,
            transform=train_transform,
            shuffle_index=True,
            loop=args.loop,
            cvfold=args.cvfold,
            num_episode=args.num_episode,
            n_way=args.n_way,
            k_shot=args.k_shot,
            n_queries=args.n_queries,
        )
        train_calsses = list(train_data.classes)
    else:
        raise ValueError(
            "The dataset {} is not supported.".format(args.data_name)
        )

    if main_process():
        logger.info("Train Classes: {}".format(train_calsses))
        logger.info("train_data samples: '{}'".format(len(train_data)))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data
        )
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=1,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        collate_fn=collate_fn_limit_fs_train,
    )

    # set scheduler
    if args.scheduler == "MultiStepWithWarmup":
        assert args.scheduler_update == "step"
        if main_process():
            logger.info(
                "scheduler: MultiStepWithWarmup. scheduler_update: {}".format(
                    args.scheduler_update
                )
            )
        iter_per_epoch = len(train_loader)
        milestones = [
            int(args.epochs * 0.6) * iter_per_epoch,
            int(args.epochs * 0.8) * iter_per_epoch,
        ]
        scheduler = MultiStepWithWarmup(
            optimizer,
            milestones=milestones,
            gamma=0.1,
            warmup=args.warmup,
            warmup_iters=args.warmup_iters,
            warmup_ratio=args.warmup_ratio,
        )
    elif args.scheduler == "MultiStep":
        assert args.scheduler_update == "epoch"
        milestones = (
            [int(x) for x in args.milestones.split(",")]
            if hasattr(args, "milestones")
            else [int(args.epochs * 0.6), int(args.epochs * 0.8)]
        )
        gamma = args.gamma if hasattr(args, "gamma") else 0.1
        if main_process():
            logger.info(
                "scheduler: MultiStep. scheduler_update: {}. milestones: {}, gamma: {}".format(
                    args.scheduler_update, milestones, gamma
                )
            )
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )
    elif args.scheduler == "Poly":
        if main_process():
            logger.info(
                "scheduler: Poly. scheduler_update: {}".format(
                    args.scheduler_update
                )
            )
        if args.scheduler_update == "epoch":
            scheduler = PolyLR(
                optimizer, max_iter=args.epochs, power=args.power
            )
        elif args.scheduler_update == "step":
            iter_per_epoch = len(train_loader)
            scheduler = PolyLR(
                optimizer,
                max_iter=args.epochs * iter_per_epoch,
                power=args.power,
            )
        else:
            raise ValueError(
                "No such scheduler update {}".format(args.scheduler_update)
            )
    else:
        raise ValueError("No such scheduler {}".format(args.scheduler))

    if args.resume and os.path.isfile(args.resume):
        scheduler.load_state_dict(scheduler_state_dict)
        print("resume scheduler")

    ###################
    # start training #
    ###################

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        loss_train, mIoU_train, mAcc_train, allAcc_train = train(
            train_loader,
            model,
            optimizer,
            epoch,
            scaler,
            scheduler,
            train_calsses,
        )
        if args.scheduler_update == "epoch":
            scheduler.step()
        epoch_log = epoch + 1

        if main_process():
            writer.add_scalar("loss_train", loss_train, epoch_log)
            writer.add_scalar("mIoU_train", mIoU_train, epoch_log)
            writer.add_scalar("mAcc_train", mAcc_train, epoch_log)
            writer.add_scalar("allAcc_train", allAcc_train, epoch_log)

        is_best = False
        if args.evaluate and (epoch_log % args.eval_freq == 0):
            loss_val, mIoU_val, mAcc_val, allAcc_val = validate(
                val_loader, model, list(val_data.classes)
            )
            if main_process():
                writer.add_scalar("loss_val", loss_val, epoch_log)
                writer.add_scalar("mIoU_val", mIoU_val, epoch_log)
                writer.add_scalar("mAcc_val", mAcc_val, epoch_log)
                writer.add_scalar("allAcc_val", allAcc_val, epoch_log)
                is_best = mIoU_val > best_iou
                best_iou = max(best_iou, mIoU_val)

        if (epoch_log % args.save_freq == 0) and main_process():
            if not os.path.exists(args.save_path + "/model/"):
                os.makedirs(args.save_path + "/model/")
            filename = args.save_path + "/model/model_last.pth"
            logger.info("Saving checkpoint to: " + filename)
            torch.save({
                "epoch": epoch_log,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_iou": best_iou
            }, filename)
            if is_best:
                logger.info("Is best")
                shutil.copyfile(
                    filename, args.save_path + "/model/model_best.pth"
                )

    if main_process():
        writer.close()
        logger.info("==>Training done!\nBest Iou: %.3f" % (best_iou))


def train(
    train_loader, model, optimizer, epoch, scaler, scheduler, train_calsses
):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, (
        support_x,
        support_base_y,
        support_y,
        support_offset,
        query_x,
        query_base_y,
        query_y,
        query_offset,
        sampled_classes,
    ) in enumerate(train_loader):
        data_time.update(time.time() - end)

        query_y = query_y.cuda(non_blocking=True)

        use_amp = args.use_amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            output, loss = model(
                support_offset,
                support_x,
                support_y,
                query_offset,
                query_x,
                query_y,
                epoch,
                support_base_y=support_base_y,
                query_base_y=query_base_y,
                sampled_classes=sampled_classes,
            )

        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if args.scheduler_update == "step":
            scheduler.step()

        output = output.max(1)[1].squeeze(0)  # output: 1, c, pts
        n = query_y.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = query_y.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n

        intersection, union, target = evaluate_metric(
            output, query_y, sampled_classes, train_calsses, args.ignore_label
        )
        
        # Logge Klasseninformationen für diese Training-Episode
        if main_process() and args.vis:
            # Konvertiere Klassen-IDs zu Namen für Outdoor-Dataset
            CLASS_ID_TO_NAME = {
                1: "StreetSign",
                3: "CircularTrafficSign", 
                4: "OctagonalTrafficSign",
                5: "RectangularTrafficSign",
                6: "TriangularTrafficSign",
                7: "DirectionSign",
                8: "FlippedTriangularTrafficSign",
                9: "PriorityRoad",
                11: "Zone",
                13: "DistanceMarker",
            }
            
            episode_class_names = []
            for class_id in sampled_classes:
                class_name = CLASS_ID_TO_NAME.get(class_id, f"Class_{class_id}")
                episode_class_names.append(class_name)
            
            combined_class_name = "_".join(episode_class_names)
            
            wandb.log({
                "train/episode_classes": combined_class_name,
                "train/episode_class_ids": sampled_classes.tolist(),
                "train/batch": i + 1,
                "train/episode_num_classes": len(episode_class_names)
            }, commit=False)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(
                union
            ), dist.all_reduce(target)
        intersection, union, target = (
            intersection.cpu().numpy(),
            union.cpu().numpy(),
            target.cpu().numpy(),
        )
        intersection_meter.update(intersection), union_meter.update(
            union
        ), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (
            sum(target_meter.val) + 1e-10
        )
        miou = np.mean(intersection_meter.val / (union_meter.val + 1e-10))
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = "{:02d}:{:02d}:{:02d}".format(
            int(t_h), int(t_m), int(t_s)
        )

        if (i + 1) % args.print_freq == 0 and main_process():
            lr = scheduler.get_last_lr()
            if isinstance(lr, list):
                lr = [round(x, 8) for x in lr]
                lr = max(lr)
            elif isinstance(lr, float):
                lr = round(lr, 8)
            logger.info(
                "Epoch: [{}/{}][{}/{}] "
                "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Remain {remain_time} "
                "Loss {loss_meter.val:.4f} "
                "Lr: {lr} "
                "Accuracy {accuracy:.4f} "
                "miou {miou:.4f}.".format(
                    epoch + 1,
                    args.epochs,
                    i + 1,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    remain_time=remain_time,
                    loss_meter=loss_meter,
                    lr=lr,
                    accuracy=accuracy,
                    miou=miou,
                )
            )
        if main_process():
            writer.add_scalar("loss_train_batch", loss_meter.val, current_iter)
            writer.add_scalar(
                "mIoU_train_batch",
                np.mean(intersection / (union + 1e-10)),
                current_iter,
            )
            writer.add_scalar(
                "mAcc_train_batch",
                np.mean(intersection / (target + 1e-10)),
                current_iter,
            )
            writer.add_scalar("allAcc_train_batch", accuracy, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info(
            "Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                epoch + 1, args.epochs, mIoU, mAcc, allAcc
            )
        )
    return loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, valid_calsses):
    if main_process():
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    
    # CSV-Daten sammeln
    csv_data = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    
    filename_prefix = f"{args.data_name}_S_{args.cvfold}_N_{args.n_way}_K_{args.k_shot}_test_episodes_{args.num_episode_per_comb}_pts_{args.voxel_max}_vs_{args.voxel_size:.2f}"

    if args.forvis:
        target_class = val_loader.dataset.target_class
        pred_path = os.path.join(args.vis_save_path, target_class)
        os.makedirs(pred_path, exist_ok=True)

    torch.cuda.empty_cache()
    model.eval()
    end = time.time()
    # Mapping von Klassen-ID zu Index in valid_calsses
    class_id_to_index = {int(cid): idx for idx, cid in enumerate(valid_calsses)}
    # Mapping Klassen-ID -> Name (Outdoor spezifisch; Fallback auf Class_{id})
    CLASS_ID_TO_NAME_LOG = {
        1: "StreetSign",
        3: "CircularTrafficSign",
        4: "OctagonalTrafficSign",
        5: "RectangularTrafficSign",
        6: "TriangularTrafficSign",
        7: "DirectionSign",
        8: "FlippedTriangularTrafficSign",
        9: "PriorityRoad",
        11: "Zone",
        13: "DistanceMarker",
    }
    # Sammler: per-Klasse Verteilungen der Episoden-IoU und -Accuracy
    per_class_iou_values = {int(cid): [] for cid in valid_calsses}
    per_class_acc_values = {int(cid): [] for cid in valid_calsses}
    # Tracker für beste/schlechteste Episode pro Klasse (nach IoU)
    best_iou_per_class = np.full(len(valid_calsses), -1.0, dtype=np.float64)
    worst_iou_per_class = np.full(len(valid_calsses), 2.0, dtype=np.float64)
    # Listen der Episoden-Indices (1-basiert), die den Best-/Worstwert erreichen
    best_eps_per_class = [[] for _ in range(len(valid_calsses))]
    worst_eps_per_class = [[] for _ in range(len(valid_calsses))]
    for i, batch in enumerate(val_loader):
        if args.forvis:
            (
                support_x,
                support_y,
                support_offset,
                query_x,
                query_y,
                query_offset,
                sampled_classes,
                scene_names,
            ) = batch
        else:
            (
                support_x,
                support_y,
                support_offset,
                query_x,
                query_y,
                query_offset,
                sampled_classes,
            ) = batch

        data_time.update(time.time() - end)

        query_y = query_y.cuda(non_blocking=True)

        with torch.no_grad():
            output, loss = model(
                support_offset,
                support_x,
                support_y,
                query_offset,
                query_x,
                query_y,
                5,
                sampled_classes=sampled_classes,
            )

        output = output.max(1)[1].squeeze(0)  # output: 1, c, pts
        n = query_y.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = query_y.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n

        intersection, union, target = evaluate_metric(
            output, query_y, sampled_classes, valid_calsses, args.ignore_label
        )
        
        # Logge Klasseninformationen für diese Episode
        if main_process() and args.vis:
            # Konvertiere Klassen-IDs zu Namen für Outdoor-Dataset
            CLASS_ID_TO_NAME = {
                1: "StreetSign",
                3: "CircularTrafficSign", 
                4: "OctagonalTrafficSign",
                5: "RectangularTrafficSign",
                6: "TriangularTrafficSign",
                7: "DirectionSign",
                8: "FlippedTriangularTrafficSign",
                9: "PriorityRoad",
                11: "Zone",
                13: "DistanceMarker",
            }
            
            episode_class_names = []
            for class_id in sampled_classes:
                class_name = CLASS_ID_TO_NAME.get(class_id, f"Class_{class_id}")
                episode_class_names.append(class_name)
            
            combined_class_name = "_".join(episode_class_names)
            
            wandb.log({
                "eval/episode_classes": combined_class_name,
                "eval/episode_class_ids": sampled_classes.tolist(),
                "eval/batch": i + 1,
                "eval/episode_num_classes": len(episode_class_names)
            }, commit=False)

        if args.forvis:
            query_name = scene_names[0]
            support_name = scene_names[1]
            save_dir = os.path.join(pred_path, f"{query_name}_{support_name}")
            os.makedirs(save_dir, exist_ok=True)
            np.save(
                os.path.join(save_dir, "query.npy"),
                query_x.cpu().numpy(),
            )
            np.save(
                os.path.join(save_dir, "querylb.npy"),
                query_y.cpu().numpy(),
            )
            np.save(
                os.path.join(save_dir, "sup.npy"),
                support_x.cpu().numpy(),
            )
            np.save(
                os.path.join(save_dir, "suplb.npy"),
                support_y.cpu().numpy(),
            )
            np.save(
                os.path.join(save_dir, "pred.npy"),
                output.cpu().numpy(),
            )
            torch.cuda.empty_cache()

        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(
                union
            ), dist.all_reduce(target)
        intersection, union, target = (
            intersection.cpu().numpy(),
            union.cpu().numpy(),
            target.cpu().numpy(),
        )
        # Episoden-IoU/Accuracy pro Klasse berechnen und Tracker aktualisieren
        iou_episode = intersection / (union + 1e-10)
        acc_episode = intersection / (target + 1e-10)
        # Update best/worst nur für Klassen, die in dieser Episode vorkommen
        eps_idx = i + 1  # 1-basierter Index
        present_indices = []
        for c in sampled_classes:
            c_int = int(c)
            if c_int in class_id_to_index:
                present_indices.append(class_id_to_index[c_int])
        for cls_idx in present_indices:
            val = float(iou_episode[cls_idx])
            # Verteilungs-Sammler befüllen
            cls_id = int(valid_calsses[cls_idx])
            per_class_iou_values[cls_id].append(val)
            per_class_acc_values[cls_id].append(float(acc_episode[cls_idx]))
            # Bestwert
            if val > best_iou_per_class[cls_idx] + 1e-6:
                best_iou_per_class[cls_idx] = val
                best_eps_per_class[cls_idx] = [eps_idx]
            elif abs(val - best_iou_per_class[cls_idx]) <= 1e-6:
                if (len(best_eps_per_class[cls_idx]) == 0) or (best_eps_per_class[cls_idx][-1] != eps_idx):
                    best_eps_per_class[cls_idx].append(eps_idx)
            # Schlechtester Wert
            if val < worst_iou_per_class[cls_idx] - 1e-6:
                worst_iou_per_class[cls_idx] = val
                worst_eps_per_class[cls_idx] = [eps_idx]
            elif abs(val - worst_iou_per_class[cls_idx]) <= 1e-6:
                if (len(worst_eps_per_class[cls_idx]) == 0) or (worst_eps_per_class[cls_idx][-1] != eps_idx):
                    worst_eps_per_class[cls_idx].append(eps_idx)
        intersection_meter.update(intersection), union_meter.update(
            union
        ), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (
            sum(target_meter.val) + 1e-10
        )
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()
        
        # CSV-Daten für jeden Batch sammeln
        csv_data.append({
            'batch': i + 1,
            'total_batches': len(val_loader),
            'data_time_val': data_time.val,
            'data_time_avg': data_time.avg,
            'batch_time_val': batch_time.val,
            'batch_time_avg': batch_time.avg,
            'loss_val': loss_meter.val,
            'loss_avg': loss_meter.avg,
            'accuracy': accuracy,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Wandb logging für Batch-Level-Daten
        if main_process() and args.vis:
            wandb.log({
                "eval/batch": i + 1,
                "eval/data_time_val": data_time.val,
                "eval/data_time_avg": data_time.avg,
                "eval/batch_time_val": batch_time.val,
                "eval/batch_time_avg": batch_time.avg,
                "eval/loss_val": loss_meter.val,
                "eval/loss_avg": loss_meter.avg,
                "eval/accuracy": accuracy
            }, commit=True)
        
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info(
                "Test: [{}/{}] "
                "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) "
                "Accuracy {accuracy:.4f}.".format(
                    i + 1,
                    len(val_loader),
                    data_time=data_time,
                    batch_time=batch_time,
                    loss_meter=loss_meter,
                    accuracy=accuracy,
                )
            )

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    
    # CSV-Datei schreiben
    if main_process():
        # Erstelle Unterordner für diesen Testlauf
        results_dir = os.path.join("stats", "results", filename_prefix)
        os.makedirs(results_dir, exist_ok=True)
        
        # Batch-Level-Daten speichern
        batch_csv_path = os.path.join(results_dir, f"batches_{timestamp}.csv")
        
        with open(batch_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['batch', 'total_batches', 'data_time_val', 'data_time_avg', 
                         'batch_time_val', 'batch_time_avg', 'loss_val', 'loss_avg', 
                         'accuracy', 'timestamp']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        
        # Finale Ergebnisse speichern
        final_csv_path = os.path.join(results_dir, f"results_{timestamp}.csv")
        
        with open(final_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['timestamp', 'mIoU', 'mAcc', 'allAcc', 'final_loss_avg', 
                         'total_batches', 'valid_classes']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Finale Ergebnisse
            final_data = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'mIoU': mIoU,
                'mAcc': mAcc,
                'allAcc': allAcc,
                'final_loss_avg': loss_meter.avg,
                'total_batches': len(val_loader),
                'valid_classes': ','.join(map(str, valid_calsses))
            }
            writer.writerow(final_data)
            
            # Klassen-spezifische Ergebnisse
            for i in range(len(valid_calsses)):
                class_data = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'mIoU': iou_class[i],
                    'mAcc': accuracy_class[i],
                    'allAcc': accuracy_class[i],  # Für Klassen ist allAcc = mAcc
                    'final_loss_avg': loss_meter.avg,
                    'total_batches': len(val_loader),
                    'valid_classes': f"Class_{valid_calsses[i]}"
                }
                writer.writerow(class_data)
        
        logger.info(f"Test-Statistiken gespeichert in: {batch_csv_path} und {final_csv_path}")
        
        # Wandb logging für finale Evaluationsergebnisse
        if args.vis:
            wandb.log({
                "eval/final_mIoU": mIoU,
                "eval/final_mAcc": mAcc,
                "eval/final_allAcc": allAcc,
                "eval/final_loss_avg": loss_meter.avg,
                "eval/total_batches": len(val_loader)
            })
            
            # Klassen-spezifische Metriken zu wandb loggen
            for i in range(len(valid_calsses)):
                wandb.log({
                    f"eval/class_{valid_calsses[i]}_iou": iou_class[i],
                    f"eval/class_{valid_calsses[i]}_accuracy": accuracy_class[i]
                })
        
        logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                mIoU, mAcc, allAcc
            )
        )
        for i in range(len(valid_calsses)):
            cid = int(valid_calsses[i])
            cname = CLASS_ID_TO_NAME_LOG.get(cid, f"Class_{cid}")
            logger.info(
                "Class_{} ({}): iou/accuracy {:.4f}/{:.4f}.".format(
                    valid_calsses[i], cname, iou_class[i], accuracy_class[i]
                )
            )
        # Beste/schlechteste Episoden pro Klasse loggen
        for idx, cls in enumerate(valid_calsses):
            best_eps_list = best_eps_per_class[idx]
            worst_eps_list = worst_eps_per_class[idx]
            cid = int(cls)
            cname = CLASS_ID_TO_NAME_LOG.get(cid, f"Class_{cid}")
            logger.info(
                "Class_{} ({}) BestEpisodes: eps={} (IoU {:.4f}), WorstEpisodes: eps={} (IoU {:.4f}).".format(
                    cls, cname,
                    best_eps_list if len(best_eps_list) > 0 else [],
                    best_iou_per_class[idx] if len(best_eps_list) > 0 else -1.0,
                    worst_eps_list if len(worst_eps_list) > 0 else [],
                    worst_iou_per_class[idx] if len(worst_eps_list) > 0 else -1.0,
                )
            )
        # Aggregation: Worst-Episoden über alle Klassen
        from collections import Counter
        worst_counter = Counter()
        for eps_list in worst_eps_per_class:
            worst_counter.update(eps_list)
        if len(worst_counter) > 0:
            total_classes = len(valid_calsses)
            sorted_worst = sorted(worst_counter.items(), key=lambda x: (-x[1], x[0]))
            top_n = 20 if len(sorted_worst) > 20 else len(sorted_worst)
            summary_items = [
                f"{ep}: {cnt}/{total_classes} ({cnt/total_classes:.2%})" for ep, cnt in sorted_worst[:top_n]
            ]
            logger.info(
                "Worst episodes across classes (Top {}): {}".format(
                    top_n, ", ".join(summary_items)
                )
            )
            # CSV: vollständige Aggregation speichern
            results_dir = os.path.join("stats", "results", filename_prefix)
            os.makedirs(results_dir, exist_ok=True)
            agg_csv_path = os.path.join(results_dir, f"worst_episodes_aggregate_{timestamp}.csv")
            with open(agg_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['episode', 'count', 'fraction']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for ep, cnt in sorted_worst:
                    writer.writerow({'episode': ep, 'count': cnt, 'fraction': cnt / total_classes})
            logger.info(f"Worst-episoden-Aggregation gespeichert in: {agg_csv_path}")
            # wandb Kurz-Zusammenfassung
            if args.vis:
                wandb.log({
                    "eval/worst_episodes_unique": len(worst_counter),
                    "eval/worst_episodes_max_count": max(worst_counter.values()),
                })
        # Optional: wandb-Logging
        if args.vis:
            log_payload = {}
            for idx, cls in enumerate(valid_calsses):
                log_payload[f"eval/class_{cls}_best_iou"] = float(best_iou_per_class[idx]) if len(best_eps_per_class[idx]) > 0 else -1.0
                log_payload[f"eval/class_{cls}_best_episodes"] = ",".join(map(str, best_eps_per_class[idx])) if len(best_eps_per_class[idx]) > 0 else ""
                log_payload[f"eval/class_{cls}_worst_iou"] = float(worst_iou_per_class[idx]) if len(worst_eps_per_class[idx]) > 0 else -1.0
                log_payload[f"eval/class_{cls}_worst_episodes"] = ",".join(map(str, worst_eps_per_class[idx])) if len(worst_eps_per_class[idx]) > 0 else ""
            wandb.log(log_payload)

        # Verteilungen als Matplotlib-Histogramme speichern (mit Klassennamen)
        hist_dir = os.path.join("stats", "histograms")
        os.makedirs(hist_dir, exist_ok=True)
        CLASS_ID_TO_NAME = {
            1: "StreetSign",
            3: "CircularTrafficSign",
            4: "OctagonalTrafficSign",
            5: "RectangularTrafficSign",
            6: "TriangularTrafficSign",
            7: "DirectionSign",
            8: "FlippedTriangularTrafficSign",
            9: "PriorityRoad",
            11: "Zone",
            13: "DistanceMarker",
        }
        def sanitize(name: str) -> str:
            return re.sub(r"[^A-Za-z0-9_-]+", "_", name)

        all_iou_series = []
        all_acc_series = []
        for cls in valid_calsses:
            cls_int = int(cls)
            cls_name = CLASS_ID_TO_NAME.get(cls_int, f"Class_{cls_int}")
            file_stub = sanitize(cls_name)
            iou_vals = per_class_iou_values.get(cls_int, [])
            acc_vals = per_class_acc_values.get(cls_int, [])
            if len(iou_vals) > 0:
                plt.figure()
                plt.hist(iou_vals, bins=20, range=(0.0, 1.0))
                plt.title(f"{cls_name} iou hist")
                plt.xlabel("IoU")
                plt.ylabel("episodes")
                plt.tight_layout()
                plt.savefig(os.path.join(hist_dir, f"{file_stub}_iou_hist.png"))
                plt.close()
                all_iou_series.append((cls_name, iou_vals))
            if len(acc_vals) > 0:
                plt.figure()
                plt.hist(acc_vals, bins=20, range=(0.0, 1.0))
                plt.title(f"{cls_name} acc hist")
                plt.xlabel("Accuracy")
                plt.ylabel("episodes")
                plt.tight_layout()
                plt.savefig(os.path.join(hist_dir, f"{file_stub}_acc_hist.png"))
                plt.close()
                all_acc_series.append((cls_name, acc_vals))

        # Sammelplot: alle IoU-Histogramme
        if len(all_iou_series) > 0:
            plt.figure()
            for name, values in all_iou_series:
                plt.hist(values, bins=20, range=(0.0, 1.0), alpha=0.35, label=name, histtype='stepfilled')
            plt.title("All classes IoU hist")
            plt.xlabel("IoU")
            plt.ylabel("episodes")
            plt.legend(fontsize=8, loc='upper right', ncol=1)
            plt.tight_layout()
            plt.savefig(os.path.join(hist_dir, "all_iou_hist.png"))
            plt.close()

        # Sammelplot: alle Accuracy-Histogramme
        if len(all_acc_series) > 0:
            plt.figure()
            for name, values in all_acc_series:
                plt.hist(values, bins=20, range=(0.0, 1.0), alpha=0.35, label=name, histtype='stepfilled')
            plt.title("All classes Accuracy hist")
            plt.xlabel("Accuracy")
            plt.ylabel("episodes")
            plt.legend(fontsize=8, loc='upper right', ncol=1)
            plt.tight_layout()
            plt.savefig(os.path.join(hist_dir, "all_acc_hist.png"))
            plt.close()

        # Sammelplot: IoU und Accuracy zusammen
        if (len(all_iou_series) > 0) or (len(all_acc_series) > 0):
            plt.figure()
            for name, values in all_iou_series:
                plt.hist(values, bins=20, range=(0.0, 1.0), alpha=0.25, label=f"{name} IoU", histtype='stepfilled')
            for name, values in all_acc_series:
                plt.hist(values, bins=20, range=(0.0, 1.0), alpha=0.25, label=f"{name} Acc", histtype='stepfilled')
            plt.title("All classes IoU + Acc hist")
            plt.xlabel("Metric value")
            plt.ylabel("episodes")
            plt.legend(fontsize=7, loc='upper right', ncol=1)
            plt.tight_layout()
            plt.savefig(os.path.join(hist_dir, "all_iou_acc_hist.png"))
            plt.close()
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    return loss_meter.avg, mIoU, mAcc, allAcc


if __name__ == "__main__":
    import gc

    gc.collect()
    main()
