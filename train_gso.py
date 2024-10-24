r""" Visual Prompt Encoder training (validation) code """

import argparse
import os
import pdb

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from common import utils
from common.evaluation import Evaluator
from common.logger import AverageMeter, Logger
from data.dataset import FSSDataset
from model.VRP_encoder import VRP_encoder
from SAM2pred import SAM_pred


def train_nshot(
    args, epoch, model, sam_model, dataloader, optimizer, scheduler, training, nshot=5
):
    r"""Train VRP_encoder model"""

    # pdb.set_trace()
    utils.fix_randseed(args.seed + epoch) if training else utils.fix_randseed(args.seed)
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):

        batch = utils.to_cuda(batch)
        protos = model.module.forward_nshot(
            args.condition,
            batch["query_img"],
            batch["support_imgs"],
            batch["support_masks"],
            training,
            nshot=nshot,
        )

        low_masks, pred_mask = sam_model(
            batch["query_img"], batch["query_name"], protos
        )
        logit_mask = low_masks

        pred_mask = torch.sigmoid(logit_mask) > 0.5
        pred_mask = pred_mask.float()

        loss = model.module.compute_objective(logit_mask, batch["query_mask"])
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            # print(loss)

        area_inter, area_union = Evaluator.classify_prediction(
            pred_mask.squeeze(1), batch
        )
        average_meter.update(
            area_inter, area_union, batch["class_id"], loss.detach().clone()
        )
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    average_meter.write_result("Training" if training else "Validation", epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


if __name__ == "__main__":

    # Arguments parsing
    parser = argparse.ArgumentParser(
        description="Visual Prompt Encoder Pytorch Implementation"
    )
    parser.add_argument("--datapath", type=str, default="ice")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="coco",
        choices=["pascal", "coco", "fss", "gso"],
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
        default=-1,
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

    args = parser.parse_args()
    # Distributed setting

    dist.init_process_group(backend="nccl")

    local_rank = dist.get_rank()
    print("Num cuda", torch.cuda.device_count(), "Local Rank", local_rank)

    # local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if utils.is_main_process():
        Logger.initialize(args, training=True)
    utils.fix_randseed(args.seed)
    # Model initialization
    model = VRP_encoder(args, args.backbone, False)
    if utils.is_main_process():
        Logger.log_params(model)

    sam_model = SAM_pred()
    sam_model.to(device)
    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # Device setup
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True,
    )

    for param in model.module.layer0.parameters():
        param.requires_grad = False
    for param in model.module.layer1.parameters():
        param.requires_grad = False
    for param in model.module.layer2.parameters():
        param.requires_grad = False
    for param in model.module.layer3.parameters():
        param.requires_grad = False
    for param in model.module.layer4.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(
        [
            {"params": model.module.transformer_decoder.parameters()},
            {"params": model.module.downsample_query.parameters(), "lr": args.lr},
            {"params": model.module.merge_1.parameters(), "lr": args.lr},
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    Evaluator.initialize(args)

    # Load Weight
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

    # dataloader_val = FSSDataset.build_dataloader(
    #     args.benchmark, args.bsz, args.nworker, args.fold, "val"
    # )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(dataloader_trn)
    )
    # Training
    best_val_miou = float("-inf")
    best_val_loss = float("inf")
    for epoch in range(args.epochs):

        trn_loss, trn_miou, trn_fb_iou = train_nshot(
            args,
            epoch,
            model,
            sam_model,
            dataloader_trn,
            optimizer,
            scheduler,
            training=True,
            nshot=args.nshot,
        )
        # with torch.no_grad():
        #     val_loss, val_miou, val_fb_iou = train_nshot(
        #         args,
        #         epoch,
        #         model,
        #         sam_model,
        #         dataloader_val,
        #         optimizer,
        #         scheduler,
        #         training=False,
        #     )

        # Save the best model
        if trn_miou > best_val_miou:
            best_val_miou = trn_miou
            if utils.is_main_process():
                Logger.save_model_miou(model, epoch, trn_miou)
        if utils.is_main_process():
            Logger.tbd_writer.add_scalars(
                "data/loss", {"trn_loss": trn_loss, "val_loss": 0}, epoch
            )
            Logger.tbd_writer.add_scalars(
                "data/miou", {"trn_miou": trn_miou, "val_miou": 0}, epoch
            )
            Logger.tbd_writer.add_scalars(
                "data/fb_iou",
                {"trn_fb_iou": trn_fb_iou, "val_fb_iou": 0},
                epoch,
            )
            Logger.tbd_writer.flush()
    Logger.tbd_writer.close()
    Logger.info("==================== Finished Training ====================")
