# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import torch

from .dataset import Kitti
from torch.utils.data import DataLoader


def Kitti_collate_fn(batch):
    keys_infos = batch[0].keys()
    new_batch_infos = {}

    for k in keys_infos:
        tmp = [val[k] for val in batch]
        new_batch_infos[k] = torch.stack(tmp, dim=0)

    return new_batch_infos["input_obj_patches"], new_batch_infos


def get_dataloader(args, mode):
    if mode == "train":
        train_set = Kitti(data_dir="dataset/data/car_data",
                          args=args, mode="train")
        val_set = Kitti(data_dir="dataset/data/car_data",
                        args=args, mode="val")

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set)
        train_loader = DataLoader(train_set, args.batch_size, shuffle=False,
                                  sampler=train_sampler, collate_fn=Kitti_collate_fn,
                                  num_workers=args.num_workers)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
        val_loader = DataLoader(val_set, args.batch_size, shuffle=False,
                                sampler=val_sampler, collate_fn=Kitti_collate_fn,
                                num_workers=args.num_workers)

        return train_loader, val_loader
    elif mode == "test":
        test_set = Kitti(data_dir="dataset/data/car_data",
                         args=args, mode="test")
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_set)
        test_loader = DataLoader(test_set, args.batch_size, shuffle=False,
                                 sampler=test_sampler, collate_fn=Kitti_collate_fn)
        return test_loader
    else:
        raise
