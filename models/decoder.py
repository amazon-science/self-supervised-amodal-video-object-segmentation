# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from torch.distributions import Normal, kl_divergence
import torch
import torch.nn as nn

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class MaskDecoder(nn.Module):
    def __init__(self, z_dim):
        super(MaskDecoder, self).__init__()
        self.z_dim = z_dim
        self.eps = 1e-6
        self.last_vm_feat_dim = 256

        hidden_dim = [32, 64, 128, 256, 256, 512]
        strides = [2, 2, 2, 2, (1,2), 1]
        kernels = [4, 4, 4, 4, (3,4), 4]
        pads = [1, 1, 1, 1, 1, 0]
        in_channels = 1
        self.down_networks = []
        for i in range(len(hidden_dim)):
            tmp = nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim[i], kernels[i], strides[i],  pads[i]),
                    nn.GroupNorm(1, hidden_dim[i]),
                    nn.LeakyReLU())
            in_channels = hidden_dim[i]
            self.down_networks.append(tmp)
        self.down_networks = nn.Sequential(*self.down_networks)
        self.last_vm_map = nn.Sequential(View((-1, in_channels)), nn.Linear(in_channels, self.last_vm_feat_dim))

        hidden_dim = [256, 256, 128, 64, 32, 2]
        strides = [1, (1,2), 2, 2, 2, 2]
        kernels = [4, (3,4), 4, 4, 4, 4]
        pads = [0, 1, 1, 1, 1, 1]
        in_channels = 512
        self.fm_deconv_map = nn.Sequential(nn.Linear(z_dim + self.last_vm_feat_dim, in_channels), View((-1, in_channels, 1, 1)))
        self.up_networks = []
        for i in range(len(hidden_dim)):
            tmp = nn.Sequential(
                    nn.GroupNorm(1, in_channels),
                    nn.LeakyReLU(),
                    nn.ConvTranspose2d(in_channels, hidden_dim[i], kernels[i], strides[i],  pads[i]))
            in_channels = hidden_dim[i]
            self.up_networks.append(tmp)
        self.up_networks = nn.Sequential(*self.up_networks)


    def forward(self, z_emb, cur_vm):
        bz, seq_len = z_emb.shape[:2]
        new_z_shape = [-1] + list(z_emb.shape[2:])
        z_emb = z_emb.reshape(new_z_shape)

        new_cur_vm_shape = [-1] + list(cur_vm.shape[2:])
        cur_vm = cur_vm.reshape(new_cur_vm_shape)
        cur_vm = cur_vm.unsqueeze(1)

        x = cur_vm
        tmp = []
        for net in self.down_networks:
            x = net(x)
            tmp.append(x)

        flatten_last_vm_feat = self.last_vm_map(x)
        cat_feat = torch.cat((z_emb, flatten_last_vm_feat), dim=-1) # [N, 512]

        x = self.fm_deconv_map(cat_feat) # [N, 64, 1, 1]
        for i, net in enumerate(self.up_networks[:-1]):
            x = net(x)
            # x = torch.cat([x, tmp[-i-2]], dim=1)
        x = self.up_networks[-1](x)
        full_mask = x
        full_mask[:,0] = (1-self.eps) * torch.sigmoid(full_mask[:,0]) + self.eps * 0.5
        # full_mask = full_mask.squeeze(1)
        new_full_mask_shape = [bz, seq_len] + list(full_mask.shape[1:])
        full_mask = full_mask.reshape(new_full_mask_shape)
        return full_mask


class FlowDecoder(nn.Module):
    def __init__(self, z_dim):
        super(FlowDecoder, self).__init__()
        self.z_dim = z_dim
        self.eps = 1e-6

        modules = []
        hidden_dim = [16, 32, 64, 128, 256, 256]
        strides = [2, 2, 2, 2, (1,2), 1]
        kernels = [4, 4, 4, 4, (3,4), 4]
        pads = [1, 1, 1, 1, 1, 0]
        in_channels = 2
        self.last_flow_feat_dim = 256
        num_layers = len(hidden_dim)
        for i in range(num_layers):
            modules.append(nn.Sequential(
                            nn.Conv2d(in_channels, hidden_dim[i], kernels[i], strides[i], pads[i]),      # 32 32 64
                            nn.GroupNorm(1, hidden_dim[i]),
                            nn.LeakyReLU()))
            in_channels = hidden_dim[i]

        modules.append(nn.Sequential(View((-1, 256)), nn.Linear(256, self.last_flow_feat_dim)))

        self.last_flow_encoder = nn.Sequential(*modules)


        modules = []

        hidden_dim = [256, 256, 128, 64, 32, 2]
        strides = [1, (1,2), 2, 2, 2, 2]
        kernels = [4, (3,4), 4, 4, 4, 4]
        pads = [0, 1, 1, 1, 1, 1]
        in_channels = 512

        num_layers = len(hidden_dim)
        # do not use vae for now
        modules.append(nn.Sequential(
                        nn.Linear(self.z_dim+self.last_flow_feat_dim, in_channels),
                        View((-1, in_channels, 1, 1))))

        for i in range(num_layers):
            modules.append(nn.Sequential(
                            nn.GroupNorm(1, in_channels),
                            nn.LeakyReLU(),
                            nn.ConvTranspose2d(in_channels, hidden_dim[i], kernels[i], strides[i], pads[i])))
            in_channels = hidden_dim[i]
        self.flow_decoder = nn.Sequential(*modules)


    def forward(self, z_emb, cur_flow):
        # First encode the last flow
        bz, seq_len = z_emb.shape[:2]
        new_z_emb_shape = [-1] + list(z_emb.shape[2:])
        z_emb = z_emb.reshape(new_z_emb_shape)
        new_cur_flow_shape = [-1] + list(cur_flow.shape[2:])
        cur_flow = cur_flow.reshape(new_cur_flow_shape)
        cur_flow = cur_flow.permute(0,3,1,2)
        # cat and encode
        flatten_last_flow_feat = self.last_flow_encoder(cur_flow) # total_seq_length, z_dim

        # concate with the last flow, note that we use the
        # history info from 0:t, and flow of t-1. So they are aligned.
        cat_feat = torch.cat((z_emb, flatten_last_flow_feat), dim=-1)

        flow = self.flow_decoder(cat_feat) # bz * seq_length, 2, patch_h, patch_w
        new_flow_shape = [bz, seq_len] + list(flow.shape[1:])
        flow = flow.reshape(new_flow_shape)

        return flow
