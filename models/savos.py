# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import SpatialTemproalEncoder
from .decoder import MaskDecoder, FlowDecoder
from utils.warp import warp
from utils.evaluation import *
from utils.focal_loss import FocalLoss


class SavosModel(nn.Module):
    def __init__(self, rnn_dim, z_dim, in_channels,
                 vis_log=None, mode=None,
                 save_eval_dict=None, args=None):
        super(SavosModel, self).__init__()

        self.dataset = args.dataset
        self.patch_W = args.patch_w
        self.patch_H = args.patch_h
        self.mode = mode
        self.training_method = args.training_method
        self.vis_log = vis_log

        self.rnn_dim = rnn_dim
        self.z_dim = z_dim

        self.st_encoder = SpatialTemproalEncoder(rnn_dim, in_channels)
        self.mask_decoder = MaskDecoder(rnn_dim)
        self.flow_decoder = FlowDecoder(rnn_dim)
        self.focal_loss = FocalLoss(2)
        # self.transform_params = SpatialTransformNetwork()

        self.save_eval_dict = save_eval_dict

    def predict(self, full_mask, flow, vm_anchor, flow_nocrop, vm_bx):
        bz, seq_len = full_mask.shape[:2]
        new_fm_shape = [-1] + list(full_mask.shape[2:])
        full_mask = full_mask.reshape(new_fm_shape)

        new_flow_shape = [-1] + list(flow.shape[2:])
        flow = flow.reshape(new_flow_shape)

        new_vm_shape = [-1] + list(vm_anchor.shape[2:])
        vm_anchor = vm_anchor.reshape(new_vm_shape)

        new_flow_nocrop_shape = [-1] + list(flow_nocrop.shape[2:])
        flow_nocrop = flow_nocrop.reshape(new_flow_nocrop_shape)

        vm_bx = vm_bx.reshape([-1, 4])

        predicted_frames_and_alpha, predict_flow_final = self.align_and_warp(
            flow, full_mask, vm_anchor, flow_nocrop, vm_bx)
        predicted_frames = predicted_frames_and_alpha[:, 0:1]
        alpha = predicted_frames_and_alpha[:, 1:]

        new_pred_fm_shape = [bz, seq_len] + list(predicted_frames.shape[1:])
        predicted_frames = predicted_frames.reshape(new_pred_fm_shape)
        alpha = alpha.reshape(new_pred_fm_shape)

        old_flow_shape = [bz, seq_len] + list(predict_flow_final.shape[1:])

        return predicted_frames, alpha, predict_flow_final.reshape(old_flow_shape).transpose(2, 3).transpose(3, 4)

    def transform(self, pred_bx, image):
        Image_H, Image_W = image.shape[-2:]
        device = image.device
        x1, x2, y1, y2 = torch.split(pred_bx, [1, 1, 1, 1], dim=-1)

        x_scale = abs(x1 - x2) / self.patch_H
        y_scale = abs(y1 - y2) / self.patch_W
        x_shift = x1 - (Image_H / 2 - self.patch_H / 2 * x_scale)
        y_shift = y1 - (Image_W / 2 - self.patch_W / 2 * y_scale)

        x_rat = torch.zeros_like(x_scale)
        y_rat = torch.zeros_like(y_scale)

        # TODO: If new bugs, maybe come from Image_W, H after the code clean
        x = torch.cat([1/y_scale, y_rat, -2*y_shift/Image_W/y_scale], dim=-1)
        y = torch.cat([x_rat, 1/x_scale, -2*x_shift/Image_H/x_scale], dim=-1)
        transform_mat = torch.stack([x, y], dim=1).to(device)

        grid = F.affine_grid(transform_mat, image.size(), align_corners=False)
        image2 = F.grid_sample(image, grid, align_corners=False)

        return image2

    def align_and_warp(self, flow, full_mask, vm_anchor, flow_nocrop, vm_bx):
        # flow [N, 2, patch_h, patch_w]
        # full mask [N, 2, patch_h, patch_w]
        # vm_anchor [N, Image_H, Image_W]

        # align
        Image_H, Image_W = vm_anchor.shape[-2:]
        pad_W_left = (Image_W - full_mask.shape[-1]) // 2
        pad_W_right = (Image_W - full_mask.shape[-1] + 1) // 2
        pad_H_left = (Image_H-full_mask.shape[-2]) // 2
        pad_H_right = (Image_H-full_mask.shape[-2] + 1) // 2
        # pad func do from last dim to first dim
        full_mask = F.pad(
            full_mask, (pad_W_left, pad_W_right, pad_H_left, pad_H_right))
        flow = F.pad(flow, (pad_W_left, pad_W_right, pad_H_left, pad_H_right))
        full_mask = self.transform(vm_bx, full_mask)
        flow = self.transform(vm_bx, flow)

        # warp
        flow_base = torch.zeros_like(flow_nocrop)
        flow_base[vm_anchor.to(torch.bool)
                  ] = flow_nocrop[vm_anchor.to(torch.bool)]
        flow_avg = torch.sum(flow_base, dim=(1, 2)) / \
            torch.sum(vm_anchor, dim=(1, 2)).unsqueeze(-1)
        flow_base[:, :, :, :] = flow_avg.unsqueeze(1).unsqueeze(1)
        flow_base[vm_anchor.to(torch.bool)
                  ] = flow_nocrop[vm_anchor.to(torch.bool)]
        flow += flow_base.transpose(2, 3).transpose(1, 2)
        vm_frames = warp(flow, full_mask)

        return vm_frames, flow.detach()

    def predict_loss_photomatric(self, full_mask, flow, vm_anchor, flow_nocrop, vm_bx, img_patch, count, reverse=False):
        bz, seq_len = full_mask.shape[:2]

        new_flow_shape = [-1] + list(flow.shape[2:])
        flow = flow.reshape(new_flow_shape)

        new_vm_shape = [-1] + list(vm_anchor.shape[2:])
        vm_anchor = vm_anchor.reshape(new_vm_shape)

        new_flow_nocrop_shape = [-1] + list(flow_nocrop.shape[2:])
        flow_nocrop = flow_nocrop.reshape(new_flow_nocrop_shape)

        new_img_patch_shape = [-1] + list(img_patch.shape[2:])
        img_patch = img_patch.reshape(new_img_patch_shape)

        vm_bx = vm_bx.reshape([-1, 4])

        loss_photomatric = self.align_and_warp_vm_img_and_loss(
            flow, full_mask, vm_anchor, flow_nocrop, vm_bx, img_patch, count, reverse, [bz, seq_len])

        return loss_photomatric

    def align_and_warp_vm_img_and_loss(self, flow, full_mask, vm_anchor, flow_nocrop, vm_bx, img_patch, counts, reverse, bz_seq_len):
        # flow [N, 2, patch_h, patch_w]
        # full mask [N, 2, patch_h, patch_w]
        # vm_anchor [N, Image_W, Image_H]
        bz, seq_len = bz_seq_len

        Image_H, Image_W = vm_anchor.shape[-2:]
        pad_W_left = (Image_W - full_mask.shape[-1]) // 2
        pad_W_right = (Image_W - full_mask.shape[-1] + 1) // 2
        pad_H_left = (Image_H-full_mask.shape[-2]) // 2
        pad_H_right = (Image_H-full_mask.shape[-2] + 1) // 2
        # pad func do from last dim to first dim
        img_patch = img_patch.permute(0, 3, 1, 2)
        img_patch = F.pad(
            img_patch, (pad_W_left, pad_W_right, pad_H_left, pad_H_right))

        flow = F.pad(flow, (pad_W_left, pad_W_right, pad_H_left, pad_H_right))
        img_patch = self.transform(vm_bx, img_patch)
        flow = self.transform(vm_bx, flow)

        flow_base = torch.zeros_like(flow_nocrop)
        flow_base[vm_anchor.to(torch.bool)
                  ] = flow_nocrop[vm_anchor.to(torch.bool)]
        flow_avg = torch.sum(flow_base, dim=(1, 2)) / \
            torch.sum(vm_anchor, dim=(1, 2)).unsqueeze(-1)
        flow_base[:, :, :, :] = flow_avg.unsqueeze(1).unsqueeze(1)
        flow_base[vm_anchor.to(torch.bool)
                  ] = flow_nocrop[vm_anchor.to(torch.bool)]
        flow += flow_base.transpose(2, 3).transpose(1, 2)
        img_patch_warp = warp(flow, img_patch)

        # reshape to batches
        new_img_patch_shape = [bz, seq_len] + list(img_patch_warp.shape[1:])
        img_patch = img_patch.reshape(new_img_patch_shape)
        img_patch_warp = img_patch_warp.reshape(new_img_patch_shape)

        # warp vm_anchor
        vm_anchor = vm_anchor.unsqueeze(1)
        vm_anchor_warp = warp(flow, vm_anchor)
        new_vm_anchor_shape = [bz, seq_len] + list(vm_anchor.shape[1:])
        vm_anchor = vm_anchor.reshape(new_vm_anchor_shape)
        vm_anchor_warp = vm_anchor_warp.reshape(new_vm_anchor_shape)

        # only overlapping part of the warpped and next frame vm is checked
        if not reverse:
            vm_anchor_warp = vm_anchor_warp[:, :-1]
            vm_anchor = vm_anchor[:, 1:]
            img_patch_warp = img_patch_warp[:, :-1]
            img_patch = img_patch[:, 1:]
        else:
            vm_anchor_warp = vm_anchor_warp.flip(1)[:, 1:]
            vm_anchor = vm_anchor.flip(1)[:, :-1]
            img_patch_warp = img_patch_warp[:, 1:]
            img_patch = img_patch[:, :-1]
        counts = counts[:, 1:]

        img_loss_mask = vm_anchor_warp.bool() & vm_anchor.bool()
        loss_photomatric_w = 0.1
        loss_photomatric = loss_photomatric_w * \
            (img_patch_warp - img_patch).abs()
        loss_photomatric *= img_loss_mask
        loss_photomatric = torch.sum(
            loss_photomatric, dim=(-1, -2, -3)) * counts
        loss_photomatric = loss_photomatric.sum() / counts.sum()
        return loss_photomatric

    def df_iou(self, fm_pred_warped, fm_pred):
        term1 = torch.abs(fm_pred_warped * fm_pred).sum((1, 2))
        term2 = torch.abs(fm_pred_warped + fm_pred -
                          fm_pred_warped * fm_pred).sum((1, 2))

        return term1 / (term2 + 1e-8)

    # TODO choose loss and caculate accruately
    def get_consistency(self, fm_preds, flows, counts):
        loss_consist = torch.tensor(
            0, device=fm_preds[0].device, dtype=torch.float32)
        bz = fm_preds.shape[0]

        for ii in range(bz):
            count = counts[ii][1:]
            cur_pred_fm = fm_preds[ii][:-1]
            next_pred_fm = fm_preds[ii][1:]
            v_flow = flows[ii][:-1].permute((0, 3, 1, 2))
            # shape of flow: [T-1, 2, Image_H, Image_W]
            # shape of cur_full_mask: [T-1, 1, Image_H, Image_W]
            next_pred_fm_warped = warp(v_flow, cur_pred_fm)
            loss = self.df_iou(next_pred_fm_warped.squeeze(
                1), next_pred_fm.squeeze(1))
            loss = 1 - loss
            loss *= count
            # loss: T - 1 values for a seq with T
            # / torch.arange(1, loss.shape[0] + 1, device=loss.device)
            loss = torch.cumsum(loss, dim=0)
            loss_consist += loss.sum()

        loss_consist /= bz
        return loss_consist

    def forward(self, obj_patches_all, infos, writer, batch_count):
        device = obj_patches_all[0].device
        # all only mask
        if self.dataset == "FishBowl":
            gt_objs_VM, gt_objs_FM, vm_bx, flow_nocrop, flow_reverse_nocrop, loss_mask, counts = self.get_infos(
                infos, device)
        elif self.dataset == "Kins_Car":
            gt_objs_VM, vm_bx, flow_nocrop, flow_reverse_nocrop, loss_mask, counts = self.get_infos(
                infos, device)
        obj_patches_foward = obj_patches_all[..., :6]
        img_patch_forward = obj_patches_all[..., :3]
        st_embedding = self.st_encoder(obj_patches_foward)
        # st_embedding, list of embedding for each sample in batch
        # each of the sample: [seq_len, z_dim]

        pred_fm_forward = self.mask_decoder(
            st_embedding, obj_patches_foward[..., 4])
        pred_flow_forward = self.flow_decoder(
            st_embedding, obj_patches_foward[..., -2:])
        pred_objs_FM_forward, alpha_channel_forward, pred_flow_final_forward = self.predict(
            pred_fm_forward, pred_flow_forward, gt_objs_VM, flow_nocrop, vm_bx)
        if self.dataset == "FishBowl":
            loss_photomatric_forward = self.predict_loss_photomatric(
                pred_fm_forward, pred_flow_forward, gt_objs_VM, flow_nocrop, vm_bx, img_patch_forward, counts, reverse=False)
        obj_patches_reverse = torch.cat(
            [obj_patches_all[..., :4], obj_patches_all[..., -2:]], dim=-1).flip(1)
        img_patch_reverse = obj_patches_all[..., :3].flip(1)
        flow_reverse_nocrop = flow_reverse_nocrop.flip(1)
        VM_reverse = gt_objs_VM.flip(1)
        vm_bx_reverse = vm_bx.flip(1)
        counts_reverse = counts.flip(1)
        st_embedding = self.st_encoder(obj_patches_reverse)
        # st_embedding, list of embedding for each sample in batch
        # each of the sample: [seq_len, z_dim]
        pred_fm_reverse = self.mask_decoder(
            st_embedding, obj_patches_reverse[..., 4])
        pred_flow_reverse = self.flow_decoder(
            st_embedding, obj_patches_reverse[..., -2:])

        pred_objs_FM_reverse, alpha_channel_reverse, pred_flow_final_reverse = self.predict(
            pred_fm_reverse, pred_flow_reverse, VM_reverse, flow_reverse_nocrop, vm_bx_reverse)
        if self.dataset == "FishBowl":
            loss_photomatric_backward = self.predict_loss_photomatric(
                pred_fm_reverse, pred_flow_reverse, VM_reverse, flow_reverse_nocrop, vm_bx_reverse, img_patch_reverse, counts, reverse=True)

        pred_objs_FM_reverse = pred_objs_FM_reverse.flip(1)
        alpha_channel_reverse = alpha_channel_reverse.flip(1)
        pred_objs_FM_reverse = torch.cat(
            [pred_objs_FM_reverse[:, 2:], pred_objs_FM_reverse[:, -2:]], dim=1)
        alpha_channel_reverse = torch.cat(
            [alpha_channel_reverse[:, 2:], alpha_channel_reverse[:, -2:]], dim=1)

        pred_objs_FM = torch.cat(
            [pred_objs_FM_forward, pred_objs_FM_reverse], dim=2)
        alpha = torch.cat(
            [alpha_channel_forward, alpha_channel_reverse], dim=2)
        alpha = F.softmax(alpha, dim=2)
        pred_objs_FM = torch.sum(pred_objs_FM*alpha, dim=2, keepdim=True)

        loss_consist_forward = self.get_consistency(
            pred_objs_FM, pred_flow_final_forward, counts)
        loss_consist_reverse = self.get_consistency(
            pred_objs_FM.flip(1), pred_flow_final_reverse, counts.flip(1))
        loss_consist = loss_consist_forward + loss_consist_reverse

        if self.dataset == "FishBowl":
            loss_photomatric = loss_photomatric_forward + loss_photomatric_backward
            loss_eval = self.fish_loss_and_evaluation(pred_objs_FM, gt_objs_FM,
                                                      gt_objs_VM, loss_mask,
                                                      loss_consist, counts, infos, loss_photomatric)
        elif self.dataset == "Kins_Car":
            loss_eval = self.kins_loss_and_evaluation(pred_objs_FM, gt_objs_VM,
                                                      loss_mask, loss_consist,
                                                      counts, infos)

        return loss_eval

    def get_infos(self, info, device):
        visible_masks = info["vm_nocrop"].to(device)
        flow_nocrop = info["flows_nocrop"].to(device)
        flow_reverse_nocrop = info["flows_reverse_nocrop"].to(device)
        vm_bx = info["obj_position"]
        loss_mask = info["loss_mask"].to(device)
        counts = info["counts"].to(device)

        if self.dataset == "FishBowl":
            full_masks = info["full_mask"]
            fm_bx = info["fm_bx"]
            return visible_masks, full_masks, vm_bx, flow_nocrop, flow_reverse_nocrop, loss_mask, counts
        elif self.dataset == "Kins_Car":
            return visible_masks, vm_bx, flow_nocrop, flow_reverse_nocrop, loss_mask, counts

    # For FishBowl
    def fish_loss_and_evaluation(self, frame_pred, frame_gt,
                                 frame_vm_origin, loss_mask,
                                 loss_consist, counts,
                                 infos, loss_photomatric,
                                 burn_step=5, pred_step=1, eps=1e-6):

        frame_pred = frame_pred[:, :-2]
        frame_vm = frame_vm_origin[:, :-2]

        if self.mode == "train":
            if self.training_method.find("next_vm") > -1:
                frame_label = frame_vm_origin[:, 1:-1]
                loss_mask = loss_mask[:, 1:-1]
                counts = counts[:, 1:-1]
            elif self.training_method.find("cur_vm") > -1:
                frame_label = frame_vm_origin[:, :-2]
                loss_mask = loss_mask[:, :-2]
                counts = counts[:, :-2]
            elif self.training_method.find("fm_label") > -1:
                frame_label = frame_gt[:, 1:-1]
                loss_mask = torch.ones_like(loss_mask[:, 1:-1])
                counts = counts[:, 1:-1]
            else:
                assert False, "please specify the training method"

        elif self.mode in ["test", "valid"]:
            frame_label = frame_gt[:, 1:-1]
            loss_mask = loss_mask[:, 1:-1]
            counts = counts[:, 1:-1]
            next_vm = frame_vm_origin[:, 1:-1]

        frame_pred = eps * 0.5 + (1-eps) * frame_pred.squeeze(2)
        bz, seq_len = frame_pred.shape[:2]
        new_mask_shape = [-1] + list(frame_pred.shape[2:])
        frame_pred = frame_pred.reshape(new_mask_shape)
        frame_label = frame_label.reshape(new_mask_shape).to(frame_pred.device)
        loss_mask = loss_mask.reshape(new_mask_shape).to(frame_pred.device)
        counts = counts.reshape(-1).to(frame_pred.device)

        loss_rec = - torch.log(frame_pred) * frame_label - \
            torch.log(1 - frame_pred) * (1 - frame_label)
        loss_rec *= loss_mask
        loss_rec = torch.sum(loss_rec, dim=(-1, -2)) * counts
        loss_rec = loss_rec.sum() / counts.sum()

        # Adding loss_photomatric
        # We only punish overlapping part of warpped visible mask and next visible mask
        loss_total = loss_rec + loss_consist + loss_photomatric
        loss_eval = {"loss_rec": loss_rec,
                     "loss_consist": loss_consist,
                     "loss_total": loss_total,
                     "loss_photomatric": loss_photomatric}

        if self.mode in ["test", "valid"]:
            next_vm = next_vm.reshape(new_mask_shape)
            frame_pred = frame_pred * (1 - loss_mask) + next_vm * loss_mask
        iou, iou_count = evaluation(
            frame_pred, frame_label, loss_mask, counts, infos, self.save_eval_dict)
        loss_eval["iou"] = iou
        loss_eval["iou_count"] = iou_count

        return loss_eval

    # For kins
    def kins_loss_and_evaluation(self, frame_pred,
                                 frame_vm_origin, loss_mask,
                                 loss_consist, counts,
                                 infos,
                                 burn_step=5, pred_step=1, eps=1e-6):

        frame_pred = frame_pred[:, :-2]
        frame_vm = frame_vm_origin[:, :-2]
        frame_label = frame_vm_origin[:, 1:-1]
        loss_mask = loss_mask[:, 1:-1]
        counts = counts[:, 1:-1]
        next_vm = frame_vm_origin[:, 1:-1]

        frame_pred = eps * 0.5 + (1-eps) * frame_pred.squeeze(2)

        bz, seq_len = frame_pred.shape[:2]
        new_mask_shape = [-1] + list(frame_pred.shape[2:])
        frame_pred = frame_pred.reshape(new_mask_shape)
        frame_label = frame_label.reshape(new_mask_shape).to(frame_pred.device)
        loss_mask = loss_mask.reshape(new_mask_shape).to(frame_pred.device)
        counts = counts.reshape(-1).to(frame_pred.device)

        loss_rec = - torch.log(frame_pred) * frame_label - \
            torch.log(1 - frame_pred) * (1 - frame_label)
        loss_rec *= loss_mask
        loss_rec = torch.sum(loss_rec, dim=(-1, -2)) * counts
        loss_rec = loss_rec.sum() / counts.sum()

        # Adding loss_photomatric
        # We only punish overlapping part of warpped visible mask and next visible mask
        loss_total = loss_rec + loss_consist
        loss_eval = {"loss_rec": loss_rec,
                     "loss_consist": loss_consist,
                     "loss_total": loss_total}

        if self.mode in ["test", "valid"]:
            next_vm = next_vm.reshape(new_mask_shape)
            frame_pred = frame_pred * (1 - loss_mask) + next_vm * loss_mask
        iou, iou_count = evaluation(
            frame_pred, frame_label, loss_mask, counts, infos, self.save_eval_dict)
        loss_eval["iou"] = iou
        loss_eval["iou_count"] = iou_count

        return loss_eval

    def unsqueeze_first_dim(self, val_terms):
        for i, val in enumerate(val_terms):
            val_terms[i] = val.unsqueeze(0)

        return val_terms

    def get_train_signal(self, vm, flow):
        # vm: [bz, T, H, W]
        # flow: [bz, T, H, W, 2] here bz = 1

        cur_vm = vm[0][:-1].unsqueeze(1)  # [T-1, 1, Image_H, Image_W]
        flow = flow[0][:-1].permute((0, 3, 1, 2))  # [T-1, 2, Image_H, Image_W]
        next_vm_warped = warp(flow, cur_vm)
        # the signal exists from frame 1 to T, exclude frame 0
        train_signal = F.relu(vm[0][1:] - next_vm_warped[:, 0])

        train_signal = torch.cat(
            [train_signal, torch.zeros_like(train_signal[0:1])], dim=0)
        return train_signal

    def cudatensor2numpy(self, *args):
        return [val.detach().cpu().numpy() for val in args]

    def kins_inference(self, obj_dict, args, device):
        # print(obj_dict["input_obj_patches"].shape[0])
        obj_patches_all = obj_dict["input_obj_patches"].to(device)

        gt_objs_VM, vm_bx, flow_nocrop, flow_reverse_nocrop, loss_mask, counts = self.get_infos(
            obj_dict, device)
        val_terms = [obj_patches_all, gt_objs_VM, vm_bx,
                     flow_nocrop, flow_reverse_nocrop, loss_mask, counts]
        if len(obj_patches_all.shape) == 4:
            obj_patches_all, gt_objs_VM, vm_bx, flow_nocrop, flow_reverse_nocrop, loss_mask, counts = self.unsqueeze_first_dim(
                val_terms)

        # forward direction
        obj_patches_foward = obj_patches_all[..., :6]
        st_embedding = self.st_encoder(obj_patches_foward)
        pred_fm_forward = self.mask_decoder(
            st_embedding, obj_patches_foward[..., 4])
        pred_flow_forward = self.flow_decoder(
            st_embedding, obj_patches_foward[..., -2:])
        pred_objs_FM_forward, alpha_channel_forward, _ = self.predict(
            pred_fm_forward, pred_flow_forward, gt_objs_VM, flow_nocrop, vm_bx)

        # reverse direction
        obj_patches_reverse = torch.cat(
            [obj_patches_all[..., :4], obj_patches_all[..., -2:]], dim=-1).flip(1)
        flow_reverse_nocrop = flow_reverse_nocrop.flip(1)
        VM_reverse = gt_objs_VM.flip(1)
        vm_bx_reverse = vm_bx.flip(1)
        counts_reverse = counts.flip(1)
        st_embedding = self.st_encoder(obj_patches_reverse)
        pred_fm_reverse = self.mask_decoder(
            st_embedding, obj_patches_reverse[..., 4])
        pred_flow_reverse = self.flow_decoder(
            st_embedding, obj_patches_reverse[..., -2:])
        pred_objs_FM_reverse, alpha_channel_reverse, _ = self.predict(
            pred_fm_reverse, pred_flow_reverse, VM_reverse, flow_reverse_nocrop, vm_bx_reverse)

        # merge two direction
        pred_objs_FM_reverse = pred_objs_FM_reverse.flip(1)
        alpha_channel_reverse = alpha_channel_reverse.flip(1)
        pred_objs_FM_reverse = torch.cat(
            [pred_objs_FM_reverse[:, 2:], pred_objs_FM_reverse[:, -2:]], dim=1)
        alpha_channel_reverse = torch.cat(
            [alpha_channel_reverse[:, 2:], alpha_channel_reverse[:, -2:]], dim=1)

        pred_objs_FM = torch.cat(
            [pred_objs_FM_forward, pred_objs_FM_reverse], dim=2)
        alpha = torch.cat(
            [alpha_channel_forward, alpha_channel_reverse], dim=2)
        alpha = F.softmax(alpha, dim=2)
        pred_objs_FM = torch.sum(pred_objs_FM*alpha, dim=2, keepdim=True)
        pred_objs_FM[:, -1] = pred_objs_FM_reverse[:, -1]
        pred_objs_FM[:, -2] = pred_objs_FM_forward[:, -2]

        # train_signal: the incremental vm compared to vm in last frame
        train_signal = self.get_train_signal(gt_objs_VM, flow_nocrop)
        reverse_train_signal = torch.zeros_like(train_signal)
        # the first element is T-1->T-2 signal
        tmp = self.get_train_signal(VM_reverse, flow_reverse_nocrop)
        reverse_train_signal[:-2] = tmp.flip(0)[2:]

        vm = torch.zeros_like(gt_objs_VM[0])
        vm[:-1] = gt_objs_VM[0][1:]
        return self.cudatensor2numpy(vm, pred_objs_FM[0][:, 0], train_signal, reverse_train_signal)
