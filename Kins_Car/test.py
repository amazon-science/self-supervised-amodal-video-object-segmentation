# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os, sys, argparse
from tqdm import tqdm
import pickle as pkl

import torch
import torch.distributed as dist

sys.path.append("../")
from dataset.dataset import Kitti
from models.savos import SavosModel
from utils.utils import *


def save_eval_res(args, objects_pred_res, video_name):
    save_path = os.path.join(args.predres_path, video_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    min_timestep = min([val["st"] for val in objects_pred_res])
    max_timestep = max([val["et"] for val in objects_pred_res])
    num_of_obj_in_video = len(objects_pred_res)

    for t in range(min_timestep, max_timestep + 1):
        single_img_pred = {}
        for val in objects_pred_res:
            if val["st"] <= t and t <= val["et"]:
                idx = t - 1 - val["st"]
                cur_pred = val["pred_fm"][idx]
                cur_train_signal = val["train_signal"][idx]
                obj_id = val["obj_id"]
                vm = val["vm"][idx]
                reverse_train_signal = val["reverse_train_signal"][idx]
                # train_signal : the incremental visible mask compared to last frame
                single_img_pred[obj_id] = {"vm": vm,
                                           "pred_fm": cur_pred,
                                           "train_signal": cur_train_signal,
                                           "reverse_train_signal": reverse_train_signal,
                                           "obj_id": obj_id,
                                           "num_of_obj_in_video": num_of_obj_in_video}
        if len(single_img_pred) > 0:
            with open(os.path.join(save_path, "%d_pred_res.pkl" % t), "wb") as file_:
                pkl.dump(single_img_pred, file_)


def imagelevel_eval(model, args):
    test_set = Kitti(data_dir="dataset/data/car_data", args=args, mode="test")
    loop = tqdm(list(test_set.data_summary.keys()))
    for video_name in loop:

        objects_data = test_set.get_image_instances(video_name)
        objects_pred_res = []

        for obj_dict in objects_data:
            device = "cpu" if obj_dict["input_obj_patches"].shape[0] > 100 else args.device
            model = model.to(device)
            vm, pred_fm, train_signal, reverse_train_signal = model.kins_inference(
                obj_dict, args, device)
            pred_res = {"st": obj_dict["st"],
                        "et": obj_dict["et"],
                        "obj_id": obj_dict["obj_id"],
                        "vm": vm,
                        "pred_fm": pred_fm,
                        "train_signal": train_signal,
                        "reverse_train_signal": reverse_train_signal}

            objects_pred_res.append(pred_res)

        save_eval_res(args, objects_pred_res, video_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="")
    parser.add_argument('--dataset', type=str, default="Kins_Car")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--log_path", type=str,
                        default="log_bidirectional_consist_next_vm_label_1.5bbox_finalconsist")

    parser.add_argument('--enlarge_coef', type=float, default=1.5)
    parser.add_argument('--patch_h', type=int, default=64)
    parser.add_argument('--patch_w', type=int, default=128)
    parser.add_argument('--loss_type', type=str,
                        choices=["FocalLoss", "BCE"], default="BCE")
    parser.add_argument('--rnn_dim', type=int, default=256)
    parser.add_argument('--z_dim', type=int, default=256)
    parser.add_argument('--in_channels', type=int, default=6)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default=torch.float32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--training_method", type=str, default=None)

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lr-decay', type=int, default=20,
                        help='After how epochs to decay LR by a factor of gamma.')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='LR decay factor.')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument("--verbose", action="store_true")
    # parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--local_rank", default=-1, type=int,
                        help="node rank for distributed training")

    args = parser.parse_args()
    args.predres_path = "log_pred_res"
    set_seed(args.seed)

    # TODO delete this
    import torch.distributed as dist
    # dist_init_method = 'tcp://{master_ip}:{master_port}'.format(master_ip='127.0.0.1', master_port='10000')
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)
    model = SavosModel(z_dim=args.z_dim, rnn_dim=args.rnn_dim,
                       in_channels=args.in_channels, mode="test", args=args)
    model = model.to(args.dtype).to(args.device)

    w_dict = torch.load(os.path.join(
        args.log_path, "best_model.pt"), map_location=args.device).state_dict()
    new_w_dict = {}
    for k, v in w_dict.items():
        new_w_dict[k.replace('module.', '')] = v
    model.load_state_dict(new_w_dict)
    imagelevel_eval(model, args)
