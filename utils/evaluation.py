# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import torch

def evaluation(frame_pred, frame_label, loss_mask, counts, infos, save_dict=None):
    frame_pred = (frame_pred > 0.5).to(torch.int64)
    frame_label = frame_label.to(torch.int64)
    loss_mask = loss_mask.to(torch.int64)
    counts = counts.to(torch.int64)

    iou_ = get_IoU(frame_pred, frame_label)
    invisible_iou_ = get_IoU(frame_pred * (1-loss_mask), frame_label * (1 - loss_mask))

    if save_dict is not None:
        save_dict.setdefault("IoU", []).extend((iou_.detach().cpu().numpy()).tolist())
        save_dict.setdefault("invisible_IoU", []).extend((invisible_iou_.detach().cpu().numpy()).tolist())
        save_dict.setdefault("iou_count", []).extend((counts.detach().cpu().numpy()).tolist())

        for bz in range(infos["video_ids"].shape[0]):
            save_dict.setdefault("v_id", []).extend(infos["video_ids"][bz][1:-1].detach().cpu().numpy())
            save_dict.setdefault("obj_id", []).extend(infos["object_ids"][bz][1:-1].detach().cpu().numpy())
            save_dict.setdefault("frame_id", []).extend(infos["frame_ids"][bz][1:-1].detach().cpu().numpy())

    return (iou_ * counts).sum(),  counts.sum()


def get_IoU(pt_mask, gt_mask):
    # pred_mask  [N, Image_W, Image_H]
    # gt_mask   [N, Image_W, Image_H]
    SMOOTH = 1e-6

    intersection = (pt_mask & gt_mask).sum((-1, -2)).to(torch.float32) # [N, 1]
    union = (pt_mask | gt_mask).sum((-1, -2)).to(torch.float32) # [N, 1]

    iou = (intersection + SMOOTH) / (union + SMOOTH) # [N, 1]

    return iou