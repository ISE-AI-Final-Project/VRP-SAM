# Train GSO

import argparse
import datetime
import math
import os
import pdb
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import model.DETR.util.misc as detr_utils
from common import utils
from common.evaluation import Evaluator
from common.logger import AverageMeter, Logger
from data.dataset import FSSDataset
from model.VRP_encoder_SEN import VRP_encoder_SEN, build_SEN
from SAM2pred import SAM_pred

# Set the environment variables for distributed training
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"


def train_nshot(
    args, epoch, model, dataloader, criterion, optimizer, scheduler, nshot=5, max_norm=0
):
    r"""Train VRP_encoder model"""

    # pdb.set_trace()
    utils.fix_randseed(args.seed + epoch)
    model.module.train_mode()
    criterion.train()

    metric_logger = detr_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", detr_utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "class_error", detr_utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 100

    batch_num = 1
    total_batch = len(dataloader)

    # for batch in metric_logger.log_every(dataloader, print_freq, header):

    for batch in dataloader:
        # print("Batch----------------------")

        batch = utils.to_cuda(batch)

        outputs = model.module.forward_nshot(
            args.condition,
            batch["query_img"],
            batch["support_imgs"],
            batch["support_masks"],
            training=True,
            nshot=nshot,
        )

        targets = []

        for bbox, label in zip(batch["bboxes"], batch["unique_obj_id"]):
            target = {}
            target["boxes"] = torch.Tensor(bbox)
            target["labels"] = torch.zeros_like(torch.Tensor(label)).to(torch.int64)
            targets.append(utils.to_cuda(target))

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )
        # print("Losses", losses)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = detr_utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        scheduler.step()

        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if (batch_num - 1) % print_freq == 0 or batch_num == total_batch:
            log_dict = {
                k: meter.global_avg for k, meter in metric_logger.meters.items()
            }
            Logger.info(f"\t[{batch_num}/{total_batch}] : {log_dict}")

        batch_num += 1

    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def init_process(args):
    dist.init_process_group(backend="nccl")

    local_rank = dist.get_rank()
    print("Num cuda", torch.cuda.device_count(), "Local Rank", local_rank)

    # local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if utils.is_main_process():
        Logger.initialize(args, training=True)
    utils.fix_randseed(args.seed)

    return device, local_rank


def main(args):
    device, local_rank = init_process(args)

    # Model initialization
    model, criterion = build_SEN(args=args, device=device)

    if utils.is_main_process():
        Logger.log_params(model)

    # Load Weight
    if args.load_weight != "":
        if os.path.exists(args.load_weight):
            model.load_state_dict(torch.load(args.load_weight, map_location=device))
            print(f"Model loaded from {args.load_weight}")
        else:
            print(f"No saved model found at {args.load_weight}")

    # Dataset initialization
    FSSDataset.initialize(
        img_size=512, datapath=args.datapath, use_original_imgsize=False
    )
    dataloader_trn = FSSDataset.build_dataloader(
        args.benchmark, args.bsz, args.nworker, args.fold, "trn", shot=args.nshot
    )

    optimizer = optim.AdamW(
        [
            {"params": model.module.transformer_decoder.parameters()},
            {"params": model.module.downsample_query.parameters(), "lr": args.lr},
            {"params": model.module.merge_1.parameters(), "lr": args.lr},
            {"params": model.module.class_embed.parameters(), "lr": args.lr},
            {"params": model.module.bbox_embed.parameters(), "lr": args.lr},
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    # Evaluator.initialize(args)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(dataloader_trn)
    )

    print("Start training")
    start_time = time.time()

    # Training
    # best_val_miou = float("-inf")
    # best_val_loss = float("inf")
    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        if utils.is_main_process():
            Logger.info(f"[EPOCH {epoch+1}/{args.epochs}]{'='*25}")

        train_stats = train_nshot(
            args=args,
            epoch=epoch,
            model=model,
            dataloader=dataloader_trn,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            nshot=args.nshot,
            max_norm=args.clip_max_norm,
        )

        if utils.is_main_process():

            total_epoch_time = time.time() - epoch_start_time
            total_epoch_time_str = str(
                datetime.timedelta(seconds=int(total_epoch_time))
            )
            Logger.info(
                f"[Finished EPOCH {epoch+1}/{args.epochs} | {total_epoch_time_str} s. : {train_stats}]\n{'='*40}\n"
            )

            if epoch % 20 == 0:
                Logger.save_model(model, epoch)

    if utils.is_main_process():
        Logger.tbd_writer.close()
        Logger.info("==================== Finished Training ====================")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":

    # sys.argv = [
    #     "run",
    #     "--datapath",
    #     ".",
    #     "--logpath",
    #     "gso_train_detr",
    #     "--benchmark",
    #     "gso_detr",
    #     "--backbone",
    #     "resnet50",
    #     "--fold",
    #     "0",
    #     "--condition",
    #     "mask",
    #     "--num_query",
    #     "50",
    #     "--epochs",
    #     "300",
    #     "--lr",
    #     "1e-4",
    #     "--bsz",
    #     "1",
    #     "--local_rank",
    #     "0",
    #     "--no_aux_loss",
    #     # '--load_weight', "checkpoints/gso_train1/best_model_ep14.ptrom",
    #     "--sam_weight",
    #     "/home/icetenny/senior-1/segment-anything/model/sam_vit_h_4b8939.pth",
    # ]

    # Arguments parsing
    parser = argparse.ArgumentParser(
        description="Visual Prompt Encoder Pytorch Implementation"
    )
    parser.add_argument("--datapath", type=str, default="ice")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="coco",
        choices=["pascal", "coco", "fss", "gso", "gso_detr"],
    )
    parser.add_argument("--logpath", type=str, default="")
    parser.add_argument(
        "--bsz", type=int, default=2
    )  # batch size = num_gpu * bsz default num_gpu = 4
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--nworker", type=int, default=8)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--fold", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument(
        "--condition",
        type=str,
        default="scribble",
        choices=["point", "scribble", "box", "mask"],
    )
    parser.add_argument(
        "--use_ignore",
        type=bool,
        default=True,
        help="Boundaries are not considered during pascal training",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="number of cpu threads to use during batch generation",
    )
    parser.add_argument("--num_query", type=int, default=50)
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        choices=["vgg16", "resnet50", "resnet101"],
    )
    parser.add_argument(
        "--nshot",
        type=int,
        default=14,
    )
    parser.add_argument(
        "--load_weight",
        type=str,
        default="",
    )
    parser.add_argument(
        "--sam_weight",
        type=str,
        default="/home/icetenny/senior-1/segment-anything/model/sam_vit_h_4b8939.pth",
    )

    # DETR
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )
    # * Segmentation
    parser.add_argument(
        "--masks",
        action="store_true",
        help="Train segmentation head if the flag is provided",
    )
    # Loss
    parser.add_argument(
        "--no_aux_loss",
        dest="aux_loss",
        action="store_false",
        help="Disables auxiliary decoding losses (loss at each layer)",
    )
    # * Matcher
    parser.add_argument(
        "--set_cost_class",
        default=1,
        type=float,
        help="Class coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_bbox",
        default=5,
        type=float,
        help="L1 box coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_giou",
        default=2,
        type=float,
        help="giou box coefficient in the matching cost",
    )
    # * Loss coefficients
    parser.add_argument("--mask_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument(
        "--eos_coef",
        default=0.1,
        type=float,
        help="Relative classification weight of the no-object class",
    )

    args = parser.parse_args()
    print(args)

    main(args=args)
