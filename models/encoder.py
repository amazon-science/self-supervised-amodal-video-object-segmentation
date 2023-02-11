# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import torch.nn as nn


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class SpatialTemproalEncoder(nn.Module):
    def __init__(self, rnn_dim, in_channels):
        super(SpatialTemproalEncoder, self).__init__()
        self.rnn_dim = rnn_dim
        self.eps = 1e-6

        modules = []
        hidden_dim = [32, 64, 128, 256, 256, 512]
        strides = [2, 2, 2, 2, (1, 2), 1]
        kernels = [4, 4, 4, 4, (3, 4), 4]
        pads = [1, 1, 1, 1, 1, 0]
        num_layers = len(hidden_dim)
        for i in range(num_layers):
            modules.append(nn.Sequential(
                # 32 32 64
                nn.Conv2d(
                    in_channels, hidden_dim[i], kernels[i], strides[i], pads[i]),
                nn.GroupNorm(1, hidden_dim[i]),
                nn.LeakyReLU()))
            in_channels = hidden_dim[i]

        modules.append(nn.Sequential(View((-1, 512)), nn.Linear(512, rnn_dim)))

        self.encoder = nn.Sequential(*modules)
        self.lstm = nn.LSTM(rnn_dim, rnn_dim, batch_first=True)

    def forward(self, obj_patches):

        # eobj_patches: [bz, seq_len, patch_h, patch_w, 6]
        bz, seq_len = obj_patches.shape[:2]
        # bz, total_seq_length, 6, patch_h, patch_w
        obj_patches = obj_patches.permute((0, 1, 4, 2, 3))
        new_shape = [-1] + list(obj_patches.shape[-3:])
        obj_patches = obj_patches.reshape(new_shape)
        z = self.encoder(obj_patches)  # bz * total_seq_length, z_dim

        z = z.reshape(bz, seq_len, -1)
        rnn_out, _ = self.lstm(z)

        return rnn_out
