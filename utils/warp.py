# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# warping function. warp(flow, source_img)
import numpy as np
import torch
import torch.nn as nn


class Warp(nn.Module):
    def __init__(self, W, H, device="cpu"):
        super(Warp, self).__init__()
        # create a grid
        gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
        self.W = W
        self.H = H
        self.gridX = torch.tensor(gridX, requires_grad=False, device=device)
        self.gridY = torch.tensor(gridY, requires_grad=False, device=device)

    def forward(self, img, flow):
        # input should be (N, C, H, W) and (N, 2, H, W)
        # Extract horizontal and vertical flows.
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        x = self.gridX.unsqueeze(0).expand_as(u).float() + u
        y = self.gridY.unsqueeze(0).expand_as(v).float() + v
        # range -1 to 1
        x = 2*(x/self.W - 0.5)
        y = 2*(y/self.H - 0.5)
        # stacking X and Y

        grid = torch.stack((x, y), dim=3)
        # Sample pixels using bilinear interpolation.

        imgOut = torch.nn.functional.grid_sample(
            img, grid, padding_mode="border", align_corners=False)
        return imgOut


def warp(flow, cur_full_mask):
    # shape of flow: [1, 2, patch_h, patch_w]
    # shape of cur_full_mask: [1, 2, patch_h, patch_w]
    H, W = cur_full_mask.shape[-2:]
    warp_func = Warp(W, H, flow.device)
    # TODO check the function
    next_full_mask = warp_func(cur_full_mask, -flow)
    return next_full_mask
