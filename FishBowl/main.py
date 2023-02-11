# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
import sys
import argparse
import time
from tqdm import tqdm
import numpy as np
import multiprocessing as mp

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import pandas as pd

sys.path.append("../")
from FishBowl_dataset.dataloader import get_dataloader
from utils.test_subset_eval import *
from utils.utils import *
from models.savos import SavosModel



os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"


def train(train_dataloader, val_dataloader, model, args):

    args.log_path = os.path.join(args.log_path, args.data_path)

    if dist.get_rank() == 0:
        if not os.path.exists(args.log_path):
            os.makedirs(args.log_path)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=args.lr_decay, gamma=args.gamma)
    num_epochs = args.epochs
    device = args.device

    best_model_path = os.path.join(args.log_path, "best_model.pt")
    best_val_iou = - np.inf
    writer1 = SummaryWriter(os.path.join(args.log_path, 'train'))
    writer2 = SummaryWriter(os.path.join(args.log_path, 'val'))
    batch_count = 0
    val_batch_count = 0

    # torch.save(model, best_model_path)
    for epoch in tqdm(range(num_epochs)):
        rank = dist.get_rank()
        if rank == 0:
            start = time.time()
        model.train()
        model.mode = "train"
        train_loss = torch.tensor(
            0, device=device, requires_grad=False).float()
        train_consist_loss = torch.tensor(
            0, device=device, requires_grad=False).float()
        train_photomatric_loss = torch.tensor(
            0, device=device, requires_grad=False).float()
        train_iou = torch.tensor(0, device=device, requires_grad=False).float()
        train_iou_count = torch.tensor(
            0, device=device, requires_grad=False).float()

        for obj_patches, infos in train_dataloader:
            # obj_patches = obj_patches.to(args.device)
            obj_patches = obj_patches.cuda(non_blocking=True)
            batch_count += 1
            # obj_patches, list of patch for each sample in batch
            # each of the sample: [seq_len, patch_h, patch_w, 6]
            bz = obj_patches.shape[0]
            loss_eval = model(obj_patches, infos, writer1, batch_count)
            loss_total = loss_eval["loss_total"]

            optimizer.zero_grad()
            loss_total.backward()
            average_gradients(model)
            optimizer.step()

            train_loss += reduce_tensors(loss_eval["loss_total"]).item() * bz
            train_consist_loss += reduce_tensors(
                loss_eval["loss_consist"]).item() * bz
            train_photomatric_loss += reduce_tensors(
                loss_eval["loss_photomatric"]).item() * bz
            train_iou += reduce_tensors(loss_eval["iou"]).item()
            train_iou_count += reduce_tensors(loss_eval["iou_count"]).item()

            if rank == 0:
                if args.verbose:
                    print(
                        "\t".join([k + ":" + f"{v.item():.3f}" for k, v in loss_eval.items()]))
                    print("iou", loss_eval["iou"].item() /
                          loss_eval["iou_count"].item())

                writer1.add_scalar(
                    "batch-loss", loss_total.item(), batch_count)
            # break

        if rank == 0:
            train_data_num = len(train_dataloader.dataset)
            writer1.add_scalar("loss", train_loss/train_data_num, epoch)
            writer1.add_scalar(
                "loss_consist", train_consist_loss/train_data_num, epoch)
            writer1.add_scalar("loss_photomatric",
                               train_photomatric_loss/train_data_num, epoch)
            writer1.add_scalar("iou", train_iou/train_iou_count, epoch)
            print("===training time===", time.time()-start)
            start = time.time()

        if epoch % 1 == 0:
            # test the model on val dataset
            model.eval()
            model.mode = "valid"
            val_loss = torch.tensor(0, device=device).float()
            val_iou = torch.tensor(0, device=device).float()
            val_iou_count = torch.tensor(0, device=device).float()
            rank = dist.get_rank()

            with torch.no_grad():
                for obj_patches, infos in val_dataloader:
                    val_batch_count += 1
                    obj_patches = obj_patches.to(args.device)
                    bz = len(obj_patches)
                    loss_eval = model(obj_patches, infos,
                                      writer2, val_batch_count)
                    loss_total = loss_eval["loss_total"]

                    val_loss += reduce_tensors(
                        loss_eval["loss_total"]).item() * bz
                    val_iou += reduce_tensors(loss_eval["iou"]).item()
                    val_iou_count += reduce_tensors(
                        loss_eval["iou_count"]).item()

                    if rank == 0 and args.verbose:
                        print(
                            "\t".join([k + ":" + f"{v.item():.3f}" for k, v in loss_eval.items()]))

                if rank == 0:
                    val_data_num = len(val_dataloader.dataset)
                    writer2.add_scalar("loss", val_loss/val_data_num, epoch)
                    writer2.add_scalar("iou", val_iou/val_iou_count, epoch)

                    print("Current val loss: %.4f" % (val_loss/val_data_num))
                    print("Current val iou: %.4f" % (val_iou/val_iou_count))

                    if best_val_iou < val_iou/val_iou_count:
                        best_val_iou = val_iou/val_iou_count
                        print(
                            "Impove! saving the model. Current best val iou: %.4f" % best_val_iou)
                        torch.save(model, best_model_path)
        if rank == 0:
            scheduler.step()
            print("===eval time===", time.time()-start)


def test(test_dataloader, model, args):
    test_batch_count = 0
    test_loss = torch.tensor(0, device=args.device).float()
    test_iou = np.array(0).astype(np.float32)
    test_iou_count = np.array(0).astype(np.float32)

    with torch.no_grad():
        loop = tqdm(test_dataloader) if not args.verbose and dist.get_rank(
        ) == 0 else test_dataloader
        for obj_patches, infos in loop:
            test_batch_count += 1
            obj_patches = obj_patches.to(args.device)
            bz = len(obj_patches)
            loss_eval = model(obj_patches, infos, None, test_batch_count)

            test_loss += reduce_tensors(loss_eval["loss_total"]).item() * bz
            test_iou += reduce_tensors(loss_eval["iou"]).item()
            test_iou_count += reduce_tensors(loss_eval["iou_count"]).item()

            if args.verbose:
                print(
                    "\t".join([k + ":" + f"{v.item():.3f}" for k, v in loss_eval.items()]))
                print("iou: %.4f" %
                      (loss_eval["iou"].item() / loss_eval["iou_count"].item()))

        print("IoU: %.4f" % (test_iou / test_iou_count))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="")
    parser.add_argument('--dataset', type=str, default="FishBowl")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--log_path", type=str, default="debug_log")

    parser.add_argument('--enlarge_coef', type=float, required=True)
    parser.add_argument('--patch_h', type=int, default=64)
    parser.add_argument('--patch_w', type=int, default=128)
    parser.add_argument('--loss_type', type=str,
                        choices=["FocalLoss", "BCE"], required=True)
    parser.add_argument('--Image_H', type=int, default=320)
    parser.add_argument('--Image_W', type=int, default=480)
    parser.add_argument('--rnn_dim', type=int, default=256)
    parser.add_argument('--z_dim', type=int, default=256)
    parser.add_argument('--in_channels', type=int, default=6)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default=torch.float32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--training_method", type=str, required=True)

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lr-decay', type=int, default=20,
                        help='After how epochs to decay LR by a factor of gamma.')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='LR decay factor.')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--local_rank", default=-1, type=int,
                        help="node rank for distributed training")

    args = parser.parse_args()

    set_seed(args.seed)

    if args.mode == "train":
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        model = SavosModel(z_dim=args.z_dim, rnn_dim=args.rnn_dim,
                           in_channels=args.in_channels, mode="train",
                           args=args)
        model = model.to(args.dtype).to(args.device)
        if args.load_model:
            model.load_state_dict(torch.load(os.path.join(
                args.log_path, "best_model.pt"), map_location=args.device).state_dict())
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank])

        train_loader, val_loader = get_dataloader(args, "train")
        train(train_loader, val_loader, model, args)

    elif args.mode == "test":
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        test_loader = get_dataloader(args, "test")

        vis_path = os.path.join(args.log_path, "png_log")
        if not os.path.exists(vis_path) and dist.get_rank() == 0:
            os.mkdir(vis_path)

        model = SavosModel(z_dim=args.z_dim, rnn_dim=args.rnn_dim,
                           in_channels=args.in_channels, vis_log=vis_path, mode="test",
                           args=args, save_eval_dict={})
        model = model.to(args.dtype).to(args.device)

        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank])
        model.load_state_dict(torch.load(
            os.path.join(args.log_path, "best_model.pt")))
        # model = torch.load(os.path.join(args.log_path,"best_model.pt"))

        test(test_loader, model, args)
        dataframe = pd.DataFrame.from_dict(model.module.save_eval_dict)
        dataframe.to_csv(os.path.join(args.log_path, '%s_%s.csv' %
                         (args.trainning_method, dist.get_rank())))

        if dist.get_rank() == 0:
            dfs = []
            for i in range(4):
                dfs.append(pd.read_csv(os.path.join(args.log_path,
                           '%s_%s.csv' % (args.training_method, i))))
            dfs = pd.concat(dfs, axis=0)
            print("====Evaluation====")
            print("====Metric1: (in paper)====")
            print_res(dfs, "IoU", "metric1")
            print_res(dfs, "invisible_IoU", "metric1")
            print("====Metric2====")
            print_res(dfs, "IoU", "metric2")
            print_res(dfs, "invisible_IoU", "metric2")
