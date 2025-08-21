from functools import partial
import os
import time

import yaml
from model.coseg import COSeg
import torch
import numpy as np
from util import logger
from util.common_util import AverageMeter, evaluate_metric, load_pretrain_checkpoint
from util.data_util import collate_fn_limit_fs
from util.outdoor_fs import Outdoor_FS_TEST


def get_args(path_to_config):
    with open(path_to_config, "r") as f:
        return yaml.safe_load(f)


def main():
    args = get_args("config/example.yaml")

    # Limit to a single GPU (use the first ID from the list if provided)
    if isinstance(args.get("gpu_ids", []), (list, tuple)) and len(args["gpu_ids"]) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args["gpu_ids"][0])
    else:
        # Fall back to GPU 0 if no list provided
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    global logger
    logger = logger.get_logger(args.save_path)

    device_index = 0
    torch.cuda.set_device(device_index)

    # Build model
    model = COSeg(args)

    if args.get("sync_bn"):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = model.cuda()

    # Load optional pretrained backbone
    model = load_pretrain_checkpoint(model, args.get("pretrain_backbone"), device_index)

    # Load full weights if provided
    if args.get("weight"):
        checkpoint = torch.load(args["weight"], map_location=f"cuda:{device_index}")
        pretrained_dict = checkpoint.get("state_dict", checkpoint)

        try:
            model.load_state_dict(pretrained_dict, strict=True)
        except RuntimeError:
            # Handle potential 'module.' prefixes from DDP checkpoints
            cleaned = {(
                k.replace("module.", "", 1) if k.startswith("module.") else k
            ): v for k, v in pretrained_dict.items()}
            model.load_state_dict(cleaned, strict=False)

    # Dataset
    val_data = Outdoor_FS_TEST(
        split=args.get("eval_split"),
        data_root=args.get("data_root"),
        voxel_size=args.get("voxel_size"),
        voxel_max=args.get("voxel_max"),
        transform=None,
        cvfold=args.get("cvfold"),
        num_episode=args.get("num_episode"),
        n_way=args.get("n_way"),
        k_shot=args.get("k_shot"),
        n_queries=args.get("n_queries"),
        num_episode_per_comb=args.get("num_episode_per_comb"),
    )

    # Prepare once (no distributed barriers needed)
    val_data.prepare_test_data()

    # DataLoader (no DistributedSampler)
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.get("workers", 4),
        pin_memory=True,
        collate_fn=partial(
            collate_fn_limit_fs, include_scene_names=args.get("forvis")
        ),
    )

    validate(val_loader, model, list(val_data.classes), args)


def validate(val_loader, model, valid_classes, args):
    logger.info(">>>>>>>>>>>>>> Start Evaluation (Single GPU) >>>>>>>>>>>>>>>>")
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
        loss /= n

        intersection, union, target = evaluate_metric(
            output, query_y, sampled_classes, valid_classes, args.get("ignore_label")
        )

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
        if (i + 1) % args.get("print_freq", 1) == 0:
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

    os.makedirs(args.get("save_path", "."), exist_ok=True)
    with open(os.path.join(args.get("save_path"), "results.csv"), "w") as f:
        f.write("class,iou,accuracy\n")
        for i in range(len(valid_classes)):
            f.write(f"{valid_classes[i]},{iou_class[i]},{accuracy_class[i]}\n")

    return loss_meter.avg, mIoU, mAcc, allAcc


if __name__ == "__main__":
    main()


