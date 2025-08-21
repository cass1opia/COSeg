from functools import partial
import os
import time

import yaml
from main_fs import validate
from model.coseg import COSeg
import torch 
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
from util import logger
from util.common_util import AverageMeter, evaluate_metric, load_pretrain_checkpoint
from util.data_util import collate_fn_limit_fs
from util.outdoor_fs import Outdoor_FS_TEST

def main_process():
    return args["rank"] % args["ngpus_per_node"] == 0
def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

def get_args(path_to_config):
    with open(path_to_config, "r") as f:
        return yaml.safe_load(f)

def main():
    args = get_args("config/example.yaml")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in args["gpu_ids"])
    port = find_free_port()
    dist_url = f"tcp://127.0.0.1:{port}"
    ngpus_per_node = len(args["gpu_ids"])
    mp.spawn(
        main_worker, 
        nprocs=ngpus_per_node,
        args=(ngpus_per_node, args, dist_url),
        join=True,
    )

def main_worker(gpu, nprocs, args_local, dist_url):
    global args
    args = args_local
    args["rank"] = gpu
    args["ngpus_per_node"] = nprocs
    print(args)
    if main_process():
        global logger
        logger = logger.get_logger(args.save_path)

    torch.cuda.set_device(gpu)

    dist.init_process_group(
        backend=args["dist_backend"],
        init_method=dist_url,
        world_size=nprocs,
        rank=gpu,
    )
        
    model = COSeg(args)

    args.workers = int((args.workers + nprocs - 1) / nprocs)
    
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[gpu], find_unused_parameters=True
    )

    model = load_pretrain_checkpoint(model, args.pretrain_backbone, gpu)

    checkpoint = torch.load(args.weight)
    pretrained_dict = checkpoint["state_dict"]
    model.load_state_dict(pretrained_dict)

    val_data = Outdoor_FS_TEST(
                split=args.eval_split,
                data_root=args.data_root,
                voxel_size=args.voxel_size,
                voxel_max=args.voxel_max,
                transform=None,
                cvfold=args.cvfold,
                num_episode=args.num_episode,
                n_way=args.n_way,
                k_shot=args.k_shot,
                n_queries=args.n_queries,
                num_episode_per_comb=args.num_episode_per_comb,
            )
    if main_process():
        val_data.prepare_test_data()

    dist.barrier()
    val_data.prepare_test_data()

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)

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

    validate(val_loader, model, list(val_data.classes), args)

def validate(val_loader, model, valid_classes, args):
    if main_process():
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    torch.cuda.empty_cache()
    model.eval()
    end = time.time()
    for i, batch in enumerate(val_loader):
    
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
        loss *= n
        count = query_y.new_tensor([n], dtype=torch.long)
        dist.all_reduce(loss), dist.all_reduce(count)
        n = count.item()
        loss /= n

        intersection, union, target = evaluate_metric(
            output, query_y, sampled_classes, valid_classes, args.ignore_label
        )

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
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()
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
    if main_process():
        logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                mIoU, mAcc, allAcc
            )
        )
        for i in range(len(valid_classes)):
            logger.info(
                "Class_{} Result: iou/accuracy {:.4f}/{:.4f}.".format(
                    valid_classes[i], iou_class[i], accuracy_class[i]
                )   
            )
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    with open(os.path.join(args.save_path, "results.csv"), "w") as f:
        f.write("class,iou,accuracy\n")
        for i in range(len(valid_classes)):
            f.write(f"{valid_classes[i]},{iou_class[i]},{accuracy_class[i]}\n")

    return loss_meter.avg, mIoU, mAcc, allAcc

if __name__ == "__main__":
    main()