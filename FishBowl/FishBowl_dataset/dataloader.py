# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import numpy as np
import torch
from torch import distributed
from torch.utils import data
from tqdm import utils
from torch.utils.data import DataLoader

from .dataset import FishBowl


# one batch contains bz videos data
# each video data contains num_obj fish object
# each object is a dict with patches(6 channels),
# visible mask crop, flow crop, obj_position, fm_labels, occlued ratios
def fishbowl_collate_fn(batch):
    keys_infos = batch[0].keys()
    new_batch_infos = {}

    for k in keys_infos:
        tmp = [val[k] for val in batch]
        new_batch_infos[k] = torch.stack(tmp, dim=0)

    return new_batch_infos["input_obj_patches"], new_batch_infos


def get_dataloader(args, mode):
    if mode == "train":
        train_set = FishBowl(
            data_dir="FishBowl_dataset/data/train_data", args=args, mode="train")
        val_set = FishBowl(
            data_dir="FishBowl_dataset/data/val_data", args=args, mode="valid")

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set)
        train_loader = DataLoader(train_set, args.batch_size, shuffle=False,
                                  sampler=train_sampler, collate_fn=fishbowl_collate_fn,
                                  num_workers=args.num_workers)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
        val_loader = DataLoader(val_set, args.batch_size, shuffle=False,
                                sampler=val_sampler, collate_fn=fishbowl_collate_fn,
                                num_workers=args.num_workers)

        return train_loader, val_loader
    elif mode == "test":
        test_set = FishBowl(
            data_dir="FishBowl_dataset/data/test_data", args=args, mode="test")
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_set)
        test_loader = DataLoader(test_set, args.batch_size, shuffle=False,
                                 sampler=test_sampler, collate_fn=fishbowl_collate_fn)
        return test_loader
    else:
        raise
